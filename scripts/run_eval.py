#!/usr/bin/env python
"""
Run the evaluation suite against the live (or mocked) agent pipeline.

Usage:
    USE_MOCK_LLM=true python scripts/run_eval.py
    python scripts/run_eval.py   # requires OPENAI_API_KEY and a populated vector store
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.evaluator import run_evaluation


def main() -> None:
    test_cases_path = str(Path(__file__).resolve().parent.parent / "evaluation" / "test_cases.json")
    report = run_evaluation(test_cases_path, domain_name="marketing")
    result = report.to_dict()
    print(json.dumps(result, indent=2))
    print(f"\nOverall pass rate: {result['pass_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
