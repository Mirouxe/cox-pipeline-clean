from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cox_pipeline.pipeline import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lance une pipeline Cox propre et concise.")
    parser.add_argument("--config", required=True, help="Chemin du fichier YAML de configuration")
    args = parser.parse_args()
    run_pipeline(args.config)
    print("Pipeline terminée.")
