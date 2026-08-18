from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "calibration-artifacts"
LOG_PATH = RUN_DIR / "diagnostic.log"
META_PATH = RUN_DIR / "run_metadata.json"


def _safe_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if REPO_ROOT not in path.parents and path != REPO_ROOT:
        raise ValueError(f"Path escapes repository: {value}")
    return path


def _copy_artifacts(patterns: list[str]) -> list[str]:
    copied: list[str] = []
    output_root = RUN_DIR / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    for pattern in patterns:
        pattern_path = _safe_repo_path(pattern)
        matches = glob.glob(str(pattern_path), recursive=True)
        for match in matches:
            source = Path(match)
            if not source.is_file():
                continue
            try:
                relative = source.relative_to(REPO_ROOT)
            except ValueError:
                continue
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(str(relative))

    return sorted(set(copied))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        default=".github/calibration/experiment.json",
        help="Repo-relative experiment specification path",
    )
    args = parser.parse_args()

    spec_path = _safe_repo_path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    script = str(spec["script"])
    script_path = _safe_repo_path(script)
    if not script_path.is_file():
        raise FileNotFoundError(f"Diagnostic script not found: {script}")
    if script_path.suffix != ".py":
        raise ValueError("Calibration runner only executes Python scripts")

    script_args = [str(value) for value in spec.get("args", [])]
    artifact_patterns = [str(value) for value in spec.get("artifacts", [])]

    command = [sys.executable, str(script_path), *script_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    print(f"Experiment: {spec.get('name', 'unnamed')}")
    print(f"Script: {script}")
    print(f"Command: {' '.join(command)}")

    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    LOG_PATH.write_text(process.stdout or "", encoding="utf-8")
    print(process.stdout or "")

    copied = _copy_artifacts(artifact_patterns)

    metadata = {
        "name": spec.get("name", "unnamed"),
        "script": script,
        "args": script_args,
        "exit_code": process.returncode,
        "artifacts_requested": artifact_patterns,
        "artifacts_copied": copied,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_ref": os.environ.get("GITHUB_REF"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if copied:
        print("Collected artifacts:")
        for path in copied:
            print(f"  {path}")
    else:
        print("No matching diagnostic output artifacts were found.")

    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
