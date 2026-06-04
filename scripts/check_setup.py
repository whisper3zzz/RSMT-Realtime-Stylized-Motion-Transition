#!/usr/bin/env python3
import argparse
import importlib.util
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STYLE100_DIR = ROOT / "MotionData" / "100STYLE"


REQUIRED_MODULES = [
    "matplotlib",
    "numpy",
    "pandas",
    "pytorch_lightning",
    "pytorch3d",
    "scipy",
    "torch",
]


DATASET_FILES = [
    "Frame_Cuts.csv",
    "skeleton",
    "train_binary.dat",
    "test_binary.dat",
    "train_binary_agument.dat",
    "test_binary_agument.dat",
]


PHASE_DATASET_FILES = [
    "deep_phase_gv.dat",
    "train+phase_gv10.dat",
    "test+phase_gv10.dat",
    "train+phase_gv10_61_21.dat",
    "test+phase_gv10_61_21.dat",
    "train+phase_gv10_120_0.dat",
    "test+phase_gv10_120_0.dat",
    "style100_benchmark_65_25.dat",
]


def check_module(name):
    return importlib.util.find_spec(name) is not None


def check_path(path):
    return path.exists()


def status_line(ok, label, detail=""):
    mark = "OK" if ok else "MISSING"
    suffix = " - " + detail if detail else ""
    return "{} {}{}".format(mark, label, suffix)


def collect_checks(include_generated):
    checks = []
    checks.append((True, "Python", platform.python_version()))
    checks.append((sys.version_info < (3, 11), "Python < 3.11 recommended", "current runtime may not match pinned PyTorch 1.13"))
    checks.append((check_path(STYLE100_DIR), "MotionData/100STYLE", str(STYLE100_DIR)))

    for module_name in REQUIRED_MODULES:
        checks.append((check_module(module_name), module_name, "Python module"))

    for filename in DATASET_FILES:
        checks.append((check_path(STYLE100_DIR / filename), filename, "100STYLE required/preprocessed file"))

    if include_generated:
        for filename in PHASE_DATASET_FILES:
            checks.append((check_path(STYLE100_DIR / filename), filename, "generated training/benchmark artifact"))

    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check RSMT runtime dependencies and expected dataset artifacts.")
    parser.add_argument("--include-generated", action="store_true", help="Also require phase/manifold/sampler generated data files.")
    args = parser.parse_args(argv)

    checks = collect_checks(args.include_generated)
    for ok, label, detail in checks:
        print(status_line(ok, label, detail))

    failed = [label for ok, label, _ in checks if not ok]
    if failed:
        print("")
        print("Setup check failed for {} item(s).".format(len(failed)))
        return 1
    print("")
    print("Setup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
