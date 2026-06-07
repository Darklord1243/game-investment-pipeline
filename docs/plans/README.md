# Planning output directory

Plans written by **Claude Code (OpenRouter Opus)** live here. Implementation is done in **Cursor Agent**.

## Workflow

1. Open PowerShell in project root (proxy only needed for `irm` install, not daily CC use — proxy is in `.claude/settings.local.json`).
2. `cd` to this repo, run `claude`.
3. `/model opus` → ask for a structured plan → save to `docs/plans/FEATURE_NAME.md`.
4. In Cursor: Agent mode → `@docs/plans/FEATURE_NAME.md` → "Implement the plan step by step."

### Pipeline rebuild (in progress)

| Step | File |
|---|---|
| Initial prompt | `PIPELINE_REBUILD_PROMPT.md` |
| Round 1 SRS (decisions locked §17) | `PIPELINE_REBUILD_SRS.md` |
| Round 2 prompt (Phase 0–1 detail) | `PIPELINE_REBUILD_ROUND2_PROMPT.md` |
| Round 2 output (target) | `PIPELINE_REBUILD_PHASE0-1.md` |

## Do not commit

- `.claude/settings.local.json` (OpenRouter key + proxy) — see `.claude/settings.local.json.example`.
