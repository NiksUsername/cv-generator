#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = ROOT / "data" / "cv.yaml"
TEMPLATE_DATA_FILE = ROOT / "data" / "template.yaml"
TEMPLATE_EXAMPLE_DATA_FILE = ROOT / "data" / "template.yaml.example"
SECTIONS_DIR = ROOT / "sections"


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def resolve_data_file(data_file: Path | None = None) -> Path:
    if data_file is not None:
        return data_file
    if DEFAULT_DATA_FILE.exists():
        return DEFAULT_DATA_FILE
    if TEMPLATE_DATA_FILE.exists():
        return TEMPLATE_DATA_FILE
    if TEMPLATE_EXAMPLE_DATA_FILE.exists():
        return TEMPLATE_EXAMPLE_DATA_FILE
    raise FileNotFoundError(
        f"No CV data file found. Expected {DEFAULT_DATA_FILE}, {TEMPLATE_DATA_FILE}, or {TEMPLATE_EXAMPLE_DATA_FILE}."
    )


def write_section(filename: str, lines: list[str], sections_dir: Path, source_label: str) -> None:
    sections_dir.mkdir(parents=True, exist_ok=True)
    content = f"% Auto-generated from {source_label}. Do not edit by hand.\n" + "\n".join(lines) + "\n"
    (sections_dir / filename).write_text(content, encoding="utf-8")


def render_header(data: dict[str, Any]) -> list[str]:
    contacts = []
    for item in data["contacts"]:
        text = tex_escape(item["text"])
        if item.get("style") == "link":
            text = r"\textcolor{cvlinkblue}{" + text + "}"
        contacts.append(text)
    return [
        r"\cvName{" + tex_escape(data["name"]) + "}",
        r"\vspace{0.08em}",
        r"\cvLocation{" + tex_escape(data["location"]) + "}",
        r"\vspace{0.05em}",
        r"\cvContact{" + " | ".join(contacts) + "}",
        r"\vspace{0.12em}",
    ]


def render_summary(data: dict[str, Any]) -> list[str]:
    lines = [r"\cvSection{" + tex_escape(data["title"]) + "}"]
    for paragraph in data["paragraphs"]:
        lines.append(r"{\cvBody " + tex_escape(paragraph) + r"}\par")
    return lines


def render_skills(data: dict[str, Any]) -> list[str]:
    skills = ", ".join(tex_escape(item) for item in data["items"])
    return [
        r"\cvSection{" + tex_escape(data["title"]) + "}",
        r"{\cvBody " + skills + r"}\par",
    ]


def render_work(data: dict[str, Any]) -> list[str]:
    lines = [r"\cvSection{" + tex_escape(data["title"]) + "}", ""]
    entries = data["entries"]
    for idx, entry in enumerate(entries):
        company = tex_escape(entry["company"])
        dates = tex_escape(entry["dates"])
        role = tex_escape(entry["role"])
        location = tex_escape(entry.get("location", ""))
        if location:
            lines.append(r"\cvRole{" + company + "}{" + dates + "}{" + role + "}{" + location + "}")
        else:
            lines.append(r"\cvRoleNoLocation{" + company + "}{" + dates + "}{" + role + "}")
        lines.append(r"\vspace{-0.27em}")
        lines.append(r"\begin{cvWorkBullets}")
        for bullet in entry["bullets"]:
            lines.append(r"  \item {\cvBody " + tex_escape(bullet) + r"}")
        lines.append(r"\end{cvWorkBullets}")
        lines.append(r"\vspace{" + entry.get("gap_after", "0.20em") + "}")
        if idx != len(entries) - 1:
            lines.append("")
    return lines


def render_projects(data: dict[str, Any]) -> list[str]:
    lines = [r"\cvSection{" + tex_escape(data["title"]) + "}", ""]
    entries = data["entries"]
    for idx, entry in enumerate(entries):
        lines.append(r"{\cvBody\textbf{" + tex_escape(entry["name"]) + r"}}\par")
        lines.append(r"\vspace{0.08em}")
        lines.append(r"\begin{cvBullets}")
        for bullet in entry["bullets"]:
            lines.append(r"  \item {\cvBody " + tex_escape(bullet) + r"}")
        lines.append(r"\end{cvBullets}")
        lines.append(r"\vspace{" + entry.get("gap_after", "0.15em") + "}")
        if idx != len(entries) - 1:
            lines.append("")
    return lines


def render_education(data: dict[str, Any]) -> list[str]:
    lines = [r"\cvSection{" + tex_escape(data["title"]) + "}", ""]
    entries = data["entries"]
    for idx, entry in enumerate(entries):
        lines.append(r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}r@{}}")
        lines.append(
            r"{\cvBody\textbf{"
            + tex_escape(entry["institution"])
            + r"}} & {\cvBody\textbf{"
            + tex_escape(entry["dates"])
            + r"}} \\"
        )
        lines.append(r"\end{tabular*}")
        lines.append(r"{\cvBody " + tex_escape(entry["degree"]) + r"}\par")
        if idx != len(entries) - 1:
            lines.append(r"\vspace{0.40em}")
    return lines


def render_certifications(data: dict[str, Any]) -> list[str]:
    lines = [
        r"\cvSection{" + tex_escape(data["title"]) + "}",
        "",
        r"\vspace{" + data.get("intro_gap", "0.81em") + "}",
    ]
    items = data["items"]
    for idx, item in enumerate(items):
        escaped = tex_escape(item)
        if idx < len(items) - 1:
            lines.append(r"\cvCertItem{" + escaped + "}")
        else:
            lines.append(r"{\cvBody\textbf{" + escaped + r"}}\par")
    return lines


def render_languages(data: dict[str, Any]) -> list[str]:
    width = data.get("width", "1.18in")
    font_size = data.get("font_size", "10.500")
    line_height = data.get("line_height", "11.178")
    line_gap_adjust = data.get("line_gap_adjust", "-0.03em")
    items = data["items"]

    lines = [r"\vfill", r"\hfill", r"\begin{minipage}{" + width + "}", r"\raggedright"]
    for idx, item in enumerate(items):
        line = r"{\fontsize{" + font_size + "}{" + line_height + r"}\selectfont\textbf{" + tex_escape(item) + "}}"
        if idx < len(items) - 1:
            line += r"\\[" + line_gap_adjust + "]"
        lines.append(line)
    lines.append(r"\end{minipage}")
    return lines


def render_from_yaml(data_file: Path, sections_dir: Path = SECTIONS_DIR) -> None:
    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    try:
        source_label = str(data_file.relative_to(ROOT))
    except ValueError:
        source_label = str(data_file)

    write_section("header.tex", render_header(data["header"]), sections_dir, source_label)
    write_section("summary.tex", render_summary(data["summary"]), sections_dir, source_label)
    write_section("skills.tex", render_skills(data["skills"]), sections_dir, source_label)
    write_section("work_experience.tex", render_work(data["work_experience"]), sections_dir, source_label)
    write_section("projects.tex", render_projects(data["projects"]), sections_dir, source_label)
    write_section("education.tex", render_education(data["education"]), sections_dir, source_label)
    write_section("certifications.tex", render_certifications(data["certifications"]), sections_dir, source_label)
    write_section("languages.tex", render_languages(data["languages"]), sections_dir, source_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LaTeX sections from CV YAML data.")
    parser.add_argument("--data-file", type=Path, default=None, help="Path to CV YAML file.")
    parser.add_argument(
        "--sections-dir",
        type=Path,
        default=SECTIONS_DIR,
        help="Output directory for rendered section files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = resolve_data_file(args.data_file)
    render_from_yaml(data_file, args.sections_dir)


if __name__ == "__main__":
    main()
