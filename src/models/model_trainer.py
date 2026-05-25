"""
Enterprise training pipeline for game investment potential regression.

Builds an authoritative ``target_score`` (1–100), performs release-date temporal
validation, fits a LightGBM + XGBoost + Random Forest voting ensemble, and
persists ``{best_model, feature_names, scaler}`` via joblib for Flask inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.features.sql_feature_engineer import FEATURE_COLUMNS, IDENTIFIER_COLUMNS

logger = logging.getLogger(__name__)

TARGET_COLUMN: Final[str] = "target_score"
RELEASE_DATE_COLUMN: Final[str] = "release_date"
DEFAULT_ARTIFACT_PATH: Final[str] = "enhanced_model_artifacts.pkl"
DEFAULT_TEST_FRACTION: Final[float] = 0.2
MIN_TRAIN_ROWS: Final[int] = 10
MIN_TEST_ROWS: Final[int] = 3
_EPSILON: Final[float] = 1e-8

# 16-feature normalized weighted composite (Enhanced_Game_Investment_Analysis.ipynb).
# Weights sum to 1.0; scaled to 1–100 after min–max normalization per component.
TARGET_SCORE_SPEC: Final[tuple[tuple[str, float], ...]] = (
    ("youtube_total_views", 0.15),
    ("youtube_like_ratio", 0.10),
    ("youtube_total_likes", 0.05),
    ("youtube_total_comments", 0.05),
    ("youtube_engagement_rate", 0.10),
    ("steam_metacritic", 0.10),
    ("steam_review_count", 0.05),
    ("steam_positive_rate", 0.05),
    ("steam_wishlist_count", 0.05),
    ("reddit_total_score", 0.10),
    ("reddit_post_count", 0.05),
    ("reddit_total_comments", 0.05),
    ("twitch_total_viewers", 0.05),
    ("twitch_stream_count", 0.03),
    ("viral_potential_score", 0.02),
    ("cross_platform_engagement_rate", 0.05),
)

# Accept ``steam_*`` prefixed columns from legacy Steam merges.
_STEAM_COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "steam_metacritic": ("steam_metacritic", "metacritic"),
    "steam_review_count": ("steam_review_count", "review_count"),
    "steam_positive_rate": ("steam_positive_rate", "positive_rate"),
    "steam_wishlist_count": ("steam_wishlist_count", "wishlist_count"),
}

_EXCLUDE_FROM_FEATURES: Final[frozenset[str]] = frozenset(
    {
        *IDENTIFIER_COLUMNS,
        TARGET_COLUMN,
        RELEASE_DATE_COLUMN,
        "steam_release_date",
    }
)


@dataclass(frozen=True)
class TrainingMetrics:
    """Holdout and in-sample regression diagnostics."""

    train_mae: float
    train_r2: float
    test_mae: float
    test_r2: float
    n_train: int
    n_test: int
    release_date_cutoff: pd.Timestamp


@dataclass(frozen=True)
class ModelArtifact:
    """Serialized inference bundle consumed by Flask apps."""

    best_model: VotingRegressor
    feature_names: list[str]
    scaler: StandardScaler


class GameInvestmentPredictor:
    """
    Train and persist a voting ensemble on SQL-engineered engagement features.

    Expects a DataFrame from :class:`~src.features.sql_feature_engineer.DatabaseFeatureEngineer`
    merged with ``Game.release_date`` (column ``release_date`` or ``steam_release_date``).
    Optional Steam columns enrich the 16-component ``target_score`` when present.
    """

    def __init__(
        self,
        *,
        random_state: int = 42,
        test_fraction: float = DEFAULT_TEST_FRACTION,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    ) -> None:
        self.random_state = random_state
        self.test_fraction = test_fraction
        self.artifact_path = Path(artifact_path)
        self.best_model: Optional[VotingRegressor] = None
        self.feature_names: list[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Optional[TrainingMetrics] = None

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_target_score(
        df: pd.DataFrame,
        norm_reference: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute the authoritative ``target_score`` in [1, 100].

        Uses the 16-feature min–max normalized weighted composite from the
        investment analysis notebook. Normalization statistics are taken from
        ``norm_reference`` when provided (training split only); otherwise from
        ``df`` (single-batch inference).

        Args:
            df: Feature matrix from ``DatabaseFeatureEngineer`` (± Steam merge).
            norm_reference: Rows used to fit per-component min/max (avoid test leakage).

        Returns:
            Series aligned to ``df.index`` named ``target_score``.
        """
        if df.empty:
            return pd.Series(dtype="float64", name=TARGET_COLUMN)

        ref = norm_reference if norm_reference is not None else df
        working = df.copy()
        components: list[pd.Series] = []
        weight_sum = sum(weight for _, weight in TARGET_SCORE_SPEC)
        if weight_sum <= 0:
            raise ValueError("TARGET_SCORE_SPEC weights must sum to a positive value.")

        for logical_name, weight in TARGET_SCORE_SPEC:
            values = GameInvestmentPredictor._resolve_component(working, logical_name)
            ref_values = GameInvestmentPredictor._resolve_component(ref, logical_name)
            min_val = float(ref_values.min())
            max_val = float(ref_values.max())
            denom = max_val - min_val + _EPSILON
            normalized = (values - min_val) / denom
            # Notebook weights sum to 1.05; renormalize so composite maps to [0, 100].
            components.append(normalized * (weight / weight_sum))

        composite = sum(components) * 100.0
        return composite.clip(1.0, 100.0).rename(TARGET_COLUMN)

    @staticmethod
    def _resolve_component(df: pd.DataFrame, logical_name: str) -> pd.Series:
        """Map logical target components to concrete DataFrame columns."""
        if logical_name == "youtube_like_ratio":
            likes = GameInvestmentPredictor._column_or_zeros(
                df, "youtube_total_likes"
            )
            views = GameInvestmentPredictor._column_or_zeros(
                df, "youtube_total_views"
            )
            return likes / (views + 1.0)

        if logical_name in _STEAM_COLUMN_ALIASES:
            for candidate in _STEAM_COLUMN_ALIASES[logical_name]:
                if candidate in df.columns:
                    return pd.to_numeric(df[candidate], errors="coerce").fillna(0.0)

        if logical_name in df.columns:
            return pd.to_numeric(df[logical_name], errors="coerce").fillna(0.0)

        logger.debug(
            "Target component %r missing in DataFrame; defaulting to 0.",
            logical_name,
        )
        return pd.Series(0.0, index=df.index, dtype="float64")

    @staticmethod
    def _column_or_zeros(df: pd.DataFrame, column: str) -> pd.Series:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index, dtype="float64")

    # ------------------------------------------------------------------
    # Feature matrix preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_release_dates(df: pd.DataFrame) -> pd.Series:
        """Return parsed release dates from ``release_date`` or ``steam_release_date``."""
        for col in (RELEASE_DATE_COLUMN, "steam_release_date"):
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().any():
                    return parsed
        raise ValueError(
            "Temporal split requires 'release_date' (from Game) or 'steam_release_date'. "
            "Merge Game.release_date onto the feature matrix before calling train()."
        )

    @staticmethod
    def _dedupe_latest_batch(df: pd.DataFrame) -> pd.DataFrame:
        """Keep one row per game (latest ``batch_date`` when present)."""
        if "game_id" not in df.columns:
            return df.reset_index(drop=True)

        working = df.copy()
        if "batch_date" in working.columns:
            working["batch_date"] = pd.to_datetime(working["batch_date"], errors="coerce")
            working = working.sort_values("batch_date")
        return (
            working.drop_duplicates(subset=["game_id"], keep="last")
            .reset_index(drop=True)
        )

    def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Ordered numeric feature list for model fitting."""
        preferred = [c for c in FEATURE_COLUMNS if c in df.columns]
        extras = [
            c
            for c in df.columns
            if c not in _EXCLUDE_FROM_FEATURES
            and c not in preferred
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        return preferred + sorted(extras)

    @staticmethod
    def _temporal_masks(
        release_dates: pd.Series,
        test_fraction: float,
    ) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
        """
        Train on older releases, test on the most recent ``test_fraction`` fraction.

        Uses the (1 - test_fraction) quantile of release_date as cutoff.
        """
        valid = release_dates.dropna()
        if valid.empty:
            raise ValueError("No valid release_date values for temporal split.")

        cutoff = valid.quantile(1.0 - test_fraction)
        train_mask = release_dates < cutoff
        test_mask = release_dates >= cutoff
        return train_mask, test_mask, pd.Timestamp(cutoff)

    # ------------------------------------------------------------------
    # Model factory & persistence
    # ------------------------------------------------------------------

    def _build_ensemble(self) -> VotingRegressor:
        """VotingRegressor with LightGBM, XGBoost, and Random Forest (equal weights)."""
        return VotingRegressor(
            estimators=[
                (
                    "lgbm",
                    lgb.LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        verbosity=-1,
                    ),
                ),
                (
                    "xgb",
                    xgb.XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        verbosity=0,
                    ),
                ),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ],
        )

    def save_artifact(self, filepath: str | Path | None = None) -> Path:
        """
        Serialize ``{best_model, feature_names, scaler}`` with joblib.

        Raises:
            RuntimeError: If ``train()`` has not been called yet.
        """
        if self.best_model is None or self.scaler is None or not self.feature_names:
            raise RuntimeError("No trained model to save; call train() first.")

        path = Path(filepath) if filepath is not None else self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "best_model": self.best_model,
            "feature_names": list(self.feature_names),
            "scaler": self.scaler,
        }
        joblib.dump(payload, path)
        logger.info("Saved model artifact to %s", path.resolve())
        return path

    @classmethod
    def load_artifact(cls, filepath: str | Path = DEFAULT_ARTIFACT_PATH) -> ModelArtifact:
        """Load a persisted artifact dictionary."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        raw = joblib.load(path)
        required = {"best_model", "feature_names", "scaler"}
        missing = required - set(raw.keys())
        if missing:
            raise KeyError(f"Artifact missing keys: {sorted(missing)}")
        return ModelArtifact(
            best_model=raw["best_model"],
            feature_names=list(raw["feature_names"]),
            scaler=raw["scaler"],
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> TrainingMetrics:
        """
        End-to-end training: target, temporal split, scale, fit, log, persist.

        Args:
            df: Feature matrix (``DatabaseFeatureEngineer`` output + release dates).

        Returns:
            TrainingMetrics with MAE/R² on train and test splits.
        """
        if df.empty:
            raise ValueError("Training DataFrame is empty.")

        prepared = self._dedupe_latest_batch(df)
        release_dates = self._resolve_release_dates(prepared)
        prepared = prepared.copy()
        prepared[RELEASE_DATE_COLUMN] = release_dates

        train_mask, test_mask, cutoff = self._temporal_masks(
            release_dates,
            self.test_fraction,
        )

        if train_mask.sum() < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Temporal train split has only {int(train_mask.sum())} rows "
                f"(minimum {MIN_TRAIN_ROWS})."
            )
        if test_mask.sum() < MIN_TEST_ROWS:
            raise ValueError(
                f"Temporal test split has only {int(test_mask.sum())} rows "
                f"(minimum {MIN_TEST_ROWS})."
            )

        train_df = prepared.loc[train_mask].copy()
        test_df = prepared.loc[test_mask].copy()

        train_df[TARGET_COLUMN] = self.calculate_target_score(
            train_df,
            norm_reference=train_df,
        )
        test_df[TARGET_COLUMN] = self.calculate_target_score(
            test_df,
            norm_reference=train_df,
        )

        self.feature_names = self._select_feature_columns(prepared)
        if not self.feature_names:
            raise ValueError("No numeric feature columns found for training.")

        x_train = train_df[self.feature_names].astype("float64").fillna(0.0)
        y_train = train_df[TARGET_COLUMN].astype("float64")
        x_test = test_df[self.feature_names].astype("float64").fillna(0.0)
        y_test = test_df[TARGET_COLUMN].astype("float64")

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        self.best_model = self._build_ensemble()
        logger.info(
            "Fitting VotingRegressor (lgbm + xgb + rf) on %d train / %d test rows, "
            "%d features, release_date cutoff=%s",
            len(x_train),
            len(x_test),
            len(self.feature_names),
            cutoff.date(),
        )
        self.best_model.fit(x_train_scaled, y_train)

        train_pred = self.best_model.predict(x_train_scaled)
        test_pred = self.best_model.predict(x_test_scaled)

        self.metrics = TrainingMetrics(
            train_mae=float(mean_absolute_error(y_train, train_pred)),
            train_r2=float(r2_score(y_train, train_pred)),
            test_mae=float(mean_absolute_error(y_test, test_pred)),
            test_r2=float(r2_score(y_test, test_pred)),
            n_train=len(x_train),
            n_test=len(x_test),
            release_date_cutoff=cutoff,
        )

        logger.info(
            "Train MAE=%.4f R2=%.4f | Test MAE=%.4f R2=%.4f",
            self.metrics.train_mae,
            self.metrics.train_r2,
            self.metrics.test_mae,
            self.metrics.test_r2,
        )

        self.save_artifact()
        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict ``target_score`` for new rows using the trained artifact.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if self.best_model is None or self.scaler is None or not self.feature_names:
            raise RuntimeError("Model not trained; call train() or load from artifact.")

        x = df.reindex(columns=self.feature_names, fill_value=0.0).astype("float64")
        x_scaled = self.scaler.transform(x)
        return self.best_model.predict(x_scaled)


def build_training_frame(
    feature_df: pd.DataFrame,
    release_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ``Game.release_date`` onto a feature matrix for :meth:`GameInvestmentPredictor.train`.

    Args:
        feature_df: Output of ``DatabaseFeatureEngineer.build_feature_matrix``.
        release_dates: DataFrame with ``game_id`` and ``release_date`` columns.

    Returns:
        Merged DataFrame ready for training.
    """
    if "game_id" not in feature_df.columns:
        raise ValueError("feature_df must contain 'game_id'")
    if "game_id" not in release_dates.columns or RELEASE_DATE_COLUMN not in release_dates.columns:
        raise ValueError("release_dates must contain 'game_id' and 'release_date'")

    merged = feature_df.merge(
        release_dates[["game_id", RELEASE_DATE_COLUMN]].drop_duplicates("game_id"),
        on="game_id",
        how="left",
    )
    return merged


__all__: Sequence[str] = [
    "DEFAULT_ARTIFACT_PATH",
    "GameInvestmentPredictor",
    "ModelArtifact",
    "TARGET_COLUMN",
    "TARGET_SCORE_SPEC",
    "TrainingMetrics",
    "build_training_frame",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import datetime as dt

    from sqlalchemy import select

    from src.database.models import Game, SessionLocal
    from src.features.sql_feature_engineer import DatabaseFeatureEngineer

    batch_day = dt.date.today()
    with SessionLocal() as session:
        features = DatabaseFeatureEngineer().build_feature_matrix(session, batch_day)
        release_df = pd.DataFrame(
            session.execute(
                select(Game.id.label("game_id"), Game.release_date)
            ).mappings().all()
        )

    if features.empty:
        logger.warning("No feature rows for %s; skipping training.", batch_day)
    else:
        train_df = build_training_frame(features, release_df)
        metrics = GameInvestmentPredictor().train(train_df)
        print(
            f"Trained on {metrics.n_train} games, tested on {metrics.n_test} "
            f"(cutoff {metrics.release_date_cutoff.date()}). "
            f"Test MAE={metrics.test_mae:.3f}, R2={metrics.test_r2:.3f}"
        )
