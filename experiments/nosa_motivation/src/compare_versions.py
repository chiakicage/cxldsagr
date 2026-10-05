"""Compare published, frozen-control and refactored NOSA serving traces."""

from pathlib import Path

from evaluation.serving_comparison import main
from experiments.nosa_motivation.src.report import audit_run

if __name__ == "__main__":
    main(model="nosa", experiment=Path(__file__).resolve().parents[1], auditor=audit_run)
