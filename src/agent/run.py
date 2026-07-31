"""CLI entry point for the clinical risk-summary agent.

Usage:
    python -m src.agent.run --config configs/agent.yaml --patient 0
"""

from __future__ import annotations

import argparse

from src.agent.agent import load_config, run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clinical risk-summary agent.")
    parser.add_argument("--config", default="configs/agent.yaml")
    parser.add_argument("--patient", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"Running agent for patient {args.patient} ...\n")
    summary = run_agent(args.patient, cfg)
    print("\n=== CLINICAL RISK SUMMARY ===\n")
    print(summary)


if __name__ == "__main__":
    main()
