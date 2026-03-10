"""Run the full business survival pipeline in sequence.

Pipeline steps:
1. Optionally install required Python packages
2. Run preprocess.py and its tests
3. Run cox.py and its tests
4. Run logistic.py and its tests
5. Run mapbox.py and its tests

Example:
    python pipeline.py

Optional:
    python pipeline.py --install-deps
    python pipeline.py --base-dir /path/to/project
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "lifelines",
]

PIPELINE_SCRIPTS = [
    "preprocess.py",
    "test_preprocess.py",
    "cox.py",
    "test_cox.py",
    "logistic.py",
    "test_logistic.py",
    "mapbox.py",
    "test_mapbox.py",
]


def run_command(command: list[str]) -> None:
    """Run a shell command and stop execution if it fails.

    Args:
        command: Command and arguments as a list.

    Raises:
        subprocess.CalledProcessError: If the command exits with a nonzero code.
    """
    print(f"\n[RUNNING] {' '.join(command)}")
    subprocess.run(command, check=True)


def install_dependencies() -> None:
    """Install required Python packages using pip."""
    print("\n[INFO] Installing required dependencies...")
    run_command([sys.executable, "-m", "pip", "install", *REQUIRED_PACKAGES])


def validate_scripts(base_dir: Path) -> None:
    """Ensure all required pipeline scripts exist.

    Args:
        base_dir: Directory containing the pipeline scripts.

    Raises:
        FileNotFoundError: If any required script is missing.
    """
    missing_files = [
        script_name
        for script_name in PIPELINE_SCRIPTS
        if not (base_dir / script_name).exists()
    ]

    if missing_files:
        missing_str = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing required script(s) in {base_dir}: {missing_str}"
        )


def run_pipeline(base_dir: Path) -> None:
    """Run all pipeline scripts in order.

    Args:
        base_dir: Directory containing the pipeline scripts.
    """
    validate_scripts(base_dir)

    for script_name in PIPELINE_SCRIPTS:
        script_path = base_dir / script_name
        run_command([sys.executable, str(script_path)])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the full business survival pipeline."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing preprocess.py, cox.py, logistic.py, "
             "mapbox.py, and test files.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required packages before running the pipeline.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the pipeline entry point.

    Returns:
        Exit code.
    """
    args = parse_args()
    base_dir = args.base_dir.resolve()

    print(f"[INFO] Using base directory: {base_dir}")

    try:
        if args.install_deps:
            install_dependencies()

        run_pipeline(base_dir)
    except FileNotFoundError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(
            f"[ERROR] Command failed with exit code {error.returncode}",
            file=sys.stderr,
        )
        return error.returncode

    print("\n[SUCCESS] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())