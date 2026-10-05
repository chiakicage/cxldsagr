"""Compare published, frozen-control and refactored DeepSeek serving traces."""

from pathlib import Path

from evaluation.serving_comparison import main
from experiments.deepseek_v32_motivation.src.report import audit_run

if __name__ == "__main__":
    main(model="deepseek_v32", experiment=Path(__file__).resolve().parents[1], auditor=audit_run)
