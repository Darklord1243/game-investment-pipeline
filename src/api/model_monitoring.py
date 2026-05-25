"""
Advanced Model Monitoring and Validation System

This module provides comprehensive monitoring, validation, and alerting
for the game investment prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    """
    Comprehensive model monitoring and validation system
    """
    
    def __init__(self, db_path="model_monitor.db"):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """
        Setup SQLite database for monitoring
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                data_split TEXT,
                validation_type TEXT
            )
        ''')
        
        # Data quality monitoring
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                dataset_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                threshold_value REAL,
                status TEXT
            )
        ''')
        
        # Model predictions tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                game_title TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                confidence_score REAL,
                actual_value REAL,
                prediction_error REAL
            )
        ''')
        
        # Alerts log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_model_performance(self, model_name: str, metrics: Dict, 
                             data_split: str = "test", validation_type: str = "holdout"):
        """
        Log model performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, value in metrics.items():
            cursor.execute('''
                INSERT INTO model_performance 
                (model_name, metric_name, metric_value, data_split, validation_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_name, metric_name, value, data_split, validation_type))
        
        conn.commit()
        conn.close()
    
    def monitor_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Monitor data quality metrics
        """
        quality_metrics = {}
        
        # Basic quality checks
        quality_metrics['missing_percentage'] = (df.isnull().sum().sum() / df.size) * 100
        quality_metrics['duplicate_percentage'] = (df.duplicated().sum() / len(df)) * 100
        quality_metrics['zero_variance_features'] = (df.var() == 0).sum()
        
        # Outlier detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = []
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts.append(outliers)
        
        quality_metrics['outlier_percentage'] = (sum(outlier_counts) / len(df)) * 100 if outlier_counts else 0
        
        # Data freshness
        if 'timestamp' in df.columns or 'published_at' in df.columns:
            date_col = 'timestamp' if 'timestamp' in df.columns else 'published_at'
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                latest_date = df[date_col].max()
                days_since_latest = (datetime.now() - latest_date).days
                quality_metrics['data_freshness_days'] = days_since_latest
            except:
                quality_metrics['data_freshness_days'] = -1
        
        # Log quality metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        thresholds = {
            'missing_percentage': 10.0,
            'duplicate_percentage': 5.0,
            'outlier_percentage': 15.0,
            'data_freshness_days': 7
        }
        
        for metric_name, value in quality_metrics.items():
            threshold = thresholds.get(metric_name, None)
            status = "GOOD"
            
            if threshold is not None:
                if metric_name == 'data_freshness_days':
                    status = "POOR" if value > threshold else "GOOD"
                else:
                    status = "POOR" if value > threshold else "GOOD"
            
            cursor.execute('''
                INSERT INTO data_quality 
                (dataset_name, metric_name, metric_value, threshold_value, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (dataset_name, metric_name, value, threshold, status))
            
            # Create alert if status is poor
            if status == "POOR":
                self.create_alert(
                    "data_quality", 
                    "WARNING",
                    f"Data quality issue in {dataset_name}: {metric_name} = {value:.2f}",
                    {"metric": metric_name, "value": value, "threshold": threshold}
                )
        
        conn.commit()
        conn.close()
        
        return quality_metrics
    
    def monitor_model_drift(self, model, X_new: pd.DataFrame, 
                          X_reference: pd.DataFrame, threshold: float = 0.1):
        """
        Monitor for model drift using prediction distributions
        """
        # Make predictions on both datasets
        pred_new = model.predict(X_new)
        pred_reference = model.predict(X_reference)
        
        # Calculate distribution statistics
        from scipy import stats
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(pred_reference, pred_new)
        
        # Mean shift
        mean_shift = abs(np.mean(pred_new) - np.mean(pred_reference)) / np.std(pred_reference)
        
        # Variance shift
        var_ratio = np.var(pred_new) / np.var(pred_reference)
        
        drift_metrics = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mean_shift': mean_shift,
            'variance_ratio': var_ratio
        }
        
        # Check for significant drift
        drift_detected = (ks_pvalue < 0.05) or (mean_shift > threshold) or (var_ratio > 1.5 or var_ratio < 0.5)
        
        if drift_detected:
            self.create_alert(
                "model_drift",
                "CRITICAL",
                f"Model drift detected: KS p-value = {ks_pvalue:.4f}, Mean shift = {mean_shift:.4f}",
                drift_metrics
            )
        
        return drift_metrics
    
    def validate_predictions(self, model, X_test: pd.DataFrame, 
                           y_test: pd.Series, game_titles: List[str]):
        """
        Validate and log model predictions
        """
        predictions = model.predict(X_test)
        
        # Calculate prediction errors
        errors = np.abs(predictions - y_test)
        
        # Log predictions
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, (pred, actual, error, game) in enumerate(zip(predictions, y_test, errors, game_titles)):
            # Calculate confidence score (inverse of error)
            confidence = 1 / (1 + error / np.max(y_test))
            
            cursor.execute('''
                INSERT INTO predictions_log 
                (model_name, game_title, predicted_value, confidence_score, actual_value, prediction_error)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (str(model.__class__.__name__), game, pred, confidence, actual, error))
        
        conn.commit()
        conn.close()
        
        return {
            'predictions': predictions,
            'errors': errors,
            'mean_error': np.mean(errors),
            'median_error': np.median(errors)
        }
    
    def create_alert(self, alert_type: str, severity: str, message: str, details: Optional[Dict] = None):
        """
        Create and log an alert
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (alert_type, severity, message, details)
            VALUES (?, ?, ?, ?)
        ''', (alert_type, severity, message, json.dumps(details) if details else None))
        
        conn.commit()
        conn.close()
        
        # Print alert to console
        print(f"🚨 {severity} ALERT: {message}")
        if details:
            print(f"   Details: {details}")
    
    def get_model_performance_history(self, model_name: str, days: int = 30) -> pd.DataFrame:
        """
        Get model performance history
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, metric_name, metric_value 
            FROM model_performance 
            WHERE model_name = ? 
            AND timestamp > datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(model_name,))
        conn.close()
        
        return df
    
    def get_data_quality_report(self, dataset_name: str, days: int = 7) -> Dict:
        """
        Get data quality report
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT metric_name, metric_value, status, timestamp
            FROM data_quality 
            WHERE dataset_name = ? 
            AND timestamp > datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(dataset_name,))
        conn.close()
        
        if df.empty:
            return {}
        
        # Get latest values for each metric
        latest_metrics = df.groupby('metric_name').first()
        
        return {
            'metrics': latest_metrics.to_dict('index'),
            'overall_status': 'GOOD' if all(latest_metrics['status'] == 'GOOD') else 'POOR',
            'issues': latest_metrics[latest_metrics['status'] == 'POOR'].index.tolist()
        }
    
    def get_active_alerts(self) -> pd.DataFrame:
        """
        Get active (unresolved) alerts
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, alert_type, severity, message, details
            FROM alerts 
            WHERE resolved = FALSE
            ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def resolve_alert(self, alert_id: int):
        """
        Mark an alert as resolved
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts 
            SET resolved = TRUE 
            WHERE id = ?
        ''', (alert_id,))
        
        conn.commit()
        conn.close()
    
    def generate_monitoring_report(self) -> Dict:
        """
        Generate comprehensive monitoring report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_quality': {},
            'model_performance': {},
            'alerts': {},
            'recommendations': []
        }
        
        # Data quality summary
        datasets = ['twitch', 'youtube', 'reddit', 'master']
        for dataset in datasets:
            report['data_quality'][dataset] = self.get_data_quality_report(dataset)
        
        # Active alerts
        active_alerts = self.get_active_alerts()
        report['alerts'] = {
            'count': len(active_alerts),
            'critical': len(active_alerts[active_alerts['severity'] == 'CRITICAL']),
            'warnings': len(active_alerts[active_alerts['severity'] == 'WARNING']),
            'recent': active_alerts.head(5).to_dict('records')
        }
        
        # Generate recommendations
        if report['alerts']['critical'] > 0:
            report['recommendations'].append("🚨 Address critical alerts immediately")
        
        if any(dq.get('overall_status') == 'POOR' for dq in report['data_quality'].values()):
            report['recommendations'].append("📊 Investigate data quality issues")
        
        if not report['recommendations']:
            report['recommendations'].append("✅ All systems operating normally")
        
        return report

class BacktestValidator:
    """
    Backtesting validation for investment predictions
    """
    
    def __init__(self):
        self.results = {}
    
    def temporal_validation(self, model, data: pd.DataFrame, 
                          date_col: str = 'published_at', 
                          prediction_horizon: int = 30):
        """
        Validate model using temporal splits
        """
        # Sort by date
        data = data.sort_values(date_col)
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Create temporal splits
        unique_dates = data[date_col].dt.date.unique()
        unique_dates = sorted(unique_dates)
        
        results = []
        
        for i in range(len(unique_dates) - prediction_horizon):
            # Train on data up to date i
            train_end_date = unique_dates[i]
            test_start_date = unique_dates[i + 1]
            test_end_date = unique_dates[min(i + prediction_horizon, len(unique_dates) - 1)]
            
            # Create train/test splits
            train_data = data[data[date_col].dt.date <= train_end_date]
            test_data = data[
                (data[date_col].dt.date >= test_start_date) & 
                (data[date_col].dt.date <= test_end_date)
            ]
            
            if len(train_data) < 50 or len(test_data) < 10:
                continue
            
            # Prepare features
            feature_cols = [col for col in data.columns 
                          if col not in ['engagement_score', 'investment_potential', date_col]]
            
            X_train = train_data[feature_cols]
            y_train = train_data['engagement_score']
            X_test = test_data[feature_cols]
            y_test = test_data['engagement_score']
            
            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import r2_score, mean_absolute_error
            
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            results.append({
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date,
                'r2': r2,
                'mae': mae,
                'train_size': len(train_data),
                'test_size': len(test_data)
            })
        
        return pd.DataFrame(results)
    
    def investment_simulation(self, predictions: pd.DataFrame, 
                            actual_outcomes: pd.DataFrame,
                            investment_amount: float = 10000):
        """
        Simulate investment returns based on predictions
        """
        # Merge predictions with actual outcomes
        merged = predictions.merge(actual_outcomes, on='game_title', how='inner')
        
        # Sort by predicted value (invest in top predicted games)
        merged = merged.sort_values('predicted_engagement', ascending=False)
        
        # Simulate different investment strategies
        strategies = {
            'top_5': merged.head(5),
            'top_10': merged.head(10),
            'top_20': merged.head(20),
            'top_percentile': merged.head(int(len(merged) * 0.1))
        }
        
        results = {}
        
        for strategy_name, selected_games in strategies.items():
            if len(selected_games) == 0:
                continue
                
            # Calculate returns
            investment_per_game = investment_amount / len(selected_games)
            
            # Assume return is proportional to actual engagement
            returns = []
            for _, game in selected_games.iterrows():
                # Simple return calculation (can be made more sophisticated)
                return_multiplier = game['actual_engagement'] / merged['actual_engagement'].mean()
                game_return = investment_per_game * return_multiplier
                returns.append(game_return)
            
            total_return = sum(returns)
            roi = ((total_return - investment_amount) / investment_amount) * 100
            
            results[strategy_name] = {
                'total_investment': investment_amount,
                'total_return': total_return,
                'roi_percentage': roi,
                'games_selected': len(selected_games),
                'avg_predicted_engagement': selected_games['predicted_engagement'].mean(),
                'avg_actual_engagement': selected_games['actual_engagement'].mean()
            }
        
        return results

if __name__ == "__main__":
    print("📊 Model Monitoring System Ready!")
    print("Use ModelMonitor() to track model performance and data quality") 