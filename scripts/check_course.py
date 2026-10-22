"""Check course sources and execute the offline lesson paths."""

import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VISION_SESSIONS = set(range(18, 26)) | set(range(30, 34))


def lesson_folders():
    folders = sorted(
        path for path in (ROOT / "lessons").iterdir()
        if path.is_dir() and re.fullmatch(r"\d{2}-.+", path.name)
    )
    numbers = [int(path.name[:2]) for path in folders]
    if numbers != list(range(1, 55)):
        raise ValueError(f"Expected exactly Sessions 1-54, found {numbers}.")
    return folders


def check_sources(folders):
    documents = [ROOT / name for name in ("README.md", "CURRICULUM.md", "SETUP.md")]
    sources = []
    for folder in folders:
        for required in ("README.md", "train.py"):
            if not (folder / required).is_file():
                raise FileNotFoundError(folder / required)
        documents.extend(folder.rglob("*.md"))
        sources.extend(folder.rglob("*.py"))
    if (ROOT / "course").is_dir():
        sources.extend((ROOT / "course").rglob("*.py"))
        documents.extend((ROOT / "course").rglob("*.md"))
    for source in sources:
        ast.parse(source.read_text(), filename=str(source))
    for document in documents:
        text = document.read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("mailto:"):
                continue
            path = target.split("#", 1)[0]
            if path and not (document.parent / path).exists():
                raise FileNotFoundError(f"{document.relative_to(ROOT)} links to {target}")
    for folder in folders[:17]:
        text = (folder / "README.md").read_text()
        tables = re.findall(
            r"^\| Line \|[^\n]+\n\|[-| ]+\|\n((?:\|[^\n]+\n?)+)",
            text,
            re.MULTILINE,
        )
        if not tables:
            raise ValueError(f"No line-by-line explanation found in {folder.name}.")
        explained = {int(value) for value in re.findall(r"^\| (\d+) \|", tables[0], re.MULTILINE)}
        executable = {
            number for number, line in enumerate((folder / "train.py").read_text().splitlines(), 1)
            if line.strip()
        }
        if explained != executable:
            raise ValueError(f"Incomplete line explanations in {folder.name}.")
    print(f"Source, local-link, and early line-explanation checks passed for {len(folders)} sessions.")


def run_lesson(folder, timeout):
    number = int(folder.name[:2])
    command = [sys.executable, "-m", f"lessons.{folder.name}.train"]
    if number in VISION_SESSIONS:
        command.append("--smoke")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if result.returncode:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"Session {number} failed with status {result.returncode}.")
    label = "synthetic pipeline smoke check" if number in VISION_SESSIONS else "offline lesson"
    print(f"PASS {folder.name} ({label})", flush=True)
    if number == 9:
        subprocess.run([sys.executable, str(folder / "predict.py")], cwd=ROOT, check=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true", help="Check sources and links without executing training.")
    parser.add_argument("--lessons", nargs="+", type=int, help="Execute selected session numbers after checking all sources.")
    parser.add_argument("--timeout", type=int, default=120, help="Maximum seconds per session.")
    args = parser.parse_args()
    folders = lesson_folders()
    check_sources(folders)
    if args.static_only:
        return
    if args.lessons and any(number not in range(1, 55) for number in args.lessons):
        parser.error("--lessons values must be from 1 through 54.")
    selected = set(args.lessons) if args.lessons else set(range(1, 55))
    for folder in folders:
        if int(folder.name[:2]) in selected:
            run_lesson(folder, args.timeout)
    print("Selected offline paths passed. Real image data and pretrained-weight paths require their documented local prerequisites.")


if __name__ == "__main__":
    main()
