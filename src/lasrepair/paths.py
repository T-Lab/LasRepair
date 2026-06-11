from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
