"""
One-command pipeline runner:
  1. Generate synthetic dataset
  2. Train all models (RF + XGBoost, early vs full)
  3. Evaluate and produce all plots
  4. Run a 30-second demo of the detector
  5. Print instructions to launch the dashboard

Run:
    python run_all.py
    python run_all.py --n 10          # use 10 early packets
    python run_all.py --skip-demo     # skip detector demo
    python run_all.py --csv path.csv  # use real dataset
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def step(title: str):
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default=None, help="Path to dataset CSV")
    parser.add_argument("--n",         default=5,    type=int)
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--cicflowmeter", action="store_true")
    args = parser.parse_args()

    # ── Step 1: dataset
    step("Step 1/4 — Generate synthetic dataset")
    if args.csv is None:
        from data.generate_synthetic import generate
        args.csv = str(ROOT / "dataset" / "raw_csv" / "synthetic_flows.csv")
        generate(args.csv)
    else:
        print(f"  Using existing CSV: {args.csv}")

    # ── Step 2: train
    step("Step 2/4 — Train models (RF + XGBoost, early vs full)")
    from ml.train_model import train
    extra = ["--cicflowmeter"] if args.cicflowmeter else []
    all_metrics = train(
        csv_path=args.csv,
        n_packets=args.n,
        use_cicflowmeter=args.cicflowmeter,
    )

    # ── Step 3: evaluate
    step("Step 3/4 — Evaluate & generate plots")
    from ml.evaluate_model import main as eval_main
    eval_main(csv_path=args.csv, use_cicflowmeter=args.cicflowmeter)

    # ── Step 4: demo
    if not args.skip_demo:
        step("Step 4/4 — Live detector demo (30 sec)")
        from realtime.detector import Detector
        det = Detector(n_packets=args.n)
        det.run_demo(duration=30)
        det.stop()
    else:
        print("\n  [skipped] detector demo")

    # ── Done
    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE")
    print("=" * 60)
    print("\n  Trained models:  apt_early_detection/models/")
    print("  Alert log:       apt_early_detection/models/alerts.log")
    print("\n  Launch dashboard:")
    print("    streamlit run apt_early_detection/dashboard/app.py")
    print()


if __name__ == "__main__":
    main()
