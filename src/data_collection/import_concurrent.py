"""
Deprecated entrypoint — use ``steam_data_miner`` instead.

Kept for backward compatibility with scripts that invoke ``import_concurrent.py``.
"""

from __future__ import annotations

from src.data_collection.steam_data_miner import main

if __name__ == "__main__":
    main()
