#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PY_COMPILE_TARGETS = [
    "Running_LongSeq.py",
    "add_phase_to_dataset.py",
    "benchmark.py",
    "benchmarkStyle100_withStyle.py",
    "process_dataset.py",
    "train_deephase.py",
    "train_styleVAE.py",
    "train_transitionNet.py",
    "scripts/check_setup.py",
    "scripts/smoke_verify.py",
    "src/Datasets/DeepPhaseDataModule.py",
    "src/Datasets/Style100Processor.py",
    "src/Datasets/StyleVAE_DataModule.py",
    "src/Net/TransitionPhaseNet.py",
    "src/utils/lightning_compat.py",
    "src/utils/locate_model.py",
    "src/utils/torch_device.py",
    "tests/test_project_hygiene.py",
    "tests/test_lightning_compat.py",
]


def run_command(command, env=None):
    print("+ {}".format(" ".join(command)))
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def main():
    cache_env = os.environ.copy()
    cache_env["PYTHONPYCACHEPREFIX"] = "/tmp/rsmt-pycache"
    commands = [
        ([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cache_env),  # unittest discover
        ([sys.executable, "-m", "py_compile", *PY_COMPILE_TARGETS], cache_env),  # py_compile
    ]
    for command, env in commands:
        exit_code = run_command(command, env=env)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
