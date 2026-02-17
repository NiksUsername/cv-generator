#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.render_sections import resolve_data_file, render_from_yaml


def build_cv(data_file: Path | None = None, output_dir: Path = ROOT) -> Path:
    resolved_data_file = resolve_data_file(data_file)
    render_from_yaml(resolved_data_file)
    subprocess.run(["tectonic", "main_modular.tex", "-o", str(output_dir)], check=True, cwd=ROOT)
    return output_dir / "main_modular.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render CV sections and compile PDF.")
    parser.add_argument("--data-file", type=Path, default=None, help="Path to CV YAML file.")
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="Directory for built PDF.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_cv(data_file=args.data_file, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
