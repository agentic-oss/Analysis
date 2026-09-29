import sys
from src.pipeline import run_pipeline

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(date_arg)
