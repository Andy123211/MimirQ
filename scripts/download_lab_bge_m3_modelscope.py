"""Download BGE-M3 from ModelScope into a caller-selected local directory."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(r"D:\tool\model-cache\mimirq-modelscope\bge-m3"),
        help="Directory outside the repository for model weights",
    )
    args = parser.parse_args(argv)
    env_executable = Path(sys.executable).parent / "Scripts" / "modelscope.exe"
    executable = str(env_executable) if env_executable.is_file() else shutil.which("modelscope")
    if not executable:
        print(
            "ModelScope CLI is missing. Install the lab environment or run "
            "`python -m pip install modelscope==1.40.1` first.",
            file=sys.stderr,
        )
        return 2

    local_dir = args.local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "download",
        "--model",
        "BAAI/bge-m3",
        "--local_dir",
        str(local_dir),
        "--max-workers",
        "2",
        "--include",
        "*.json",
        "**/*.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "vocab.txt",
        "merges.txt",
    ]
    print(f"[lab-rag] downloading BAAI/bge-m3 from ModelScope into {local_dir}")
    result = subprocess.run(command, check=False)
    if result.returncode:
        return result.returncode
    required_files = ("config.json", "pytorch_model.bin", "modules.json")
    missing = [name for name in required_files if not (local_dir / name).is_file()]
    if missing:
        print(
            f"Model download finished but required files are missing in {local_dir}: {missing}",
            file=sys.stderr,
        )
        return 2
    print(f"[lab-rag] MODEL OK path={local_dir}")
    print("[lab-rag] Set MIMIRQ_BGE_M3_MODEL_PATH to this directory before running the lab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
