"""
main.py
Unified entry point for the CIC-DDoS2019 Adversarial NIDS Defense Project.

NOTE: The legacy TensorFlow prototype has been deprecated and moved to code/legacy_tf_pipeline/.
This script delegates directly to the canonical PyTorch + ART pipeline in code/nids hybrid defence/.
"""

import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_DIR = os.path.join(CURRENT_DIR, "nids hybrid defence")
RUN_SCRIPT = os.path.join(PYTORCH_DIR, "run_pipeline.py")

def main():
    print("=" * 75)
    print("  CIC-DDoS2019 ADVERSARIAL NIDS DEFENSE — UNIFIED PYTORCH PIPELINE")
    print("=" * 75)
    print("  Notice: The legacy TensorFlow prototype in code/legacy_tf_pipeline/ is DEPRECATED.")
    print(f"  Launching canonical PyTorch pipeline: {RUN_SCRIPT}")
    print("=" * 75 + "\n")

    # Delegate execution to run_pipeline.py with any passed arguments
    cmd = [sys.executable, RUN_SCRIPT] + sys.argv[1:]
    res = subprocess.run(cmd, cwd=PYTORCH_DIR)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
