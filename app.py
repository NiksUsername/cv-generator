from __future__ import annotations

import base64
import json
import os
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
import yaml
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tools.build_cv import ROOT, build_cv


app = FastAPI(title="CV Tailoring API", version="1.0.0")
_generation_lock = threading.Lock()

DATA_DIR = ROOT / "data"
TEMPLATE_FILE = DATA_DIR / "template.yaml"
TEMPLATE_EXAMPLE_FILE = DATA_DIR / "template.yaml.example"
CANDIDATE_FILE = DATA_DIR / "candidate.yaml"
CANDIDATE_EXAMPLE_FILE = DATA_DIR / "candidate.yaml.example"
CV_FILE = DATA_DIR / "cv.yaml"
TMP_DIR = Path("/tmp/cv_template")
DECODED_REQUEST_LOG_FILE = TMP_DIR / "decoded_job_description.log"
PDF_FILE = ROOT / "main_modular.pdf"

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"


class GenerateCVRequest(BaseModel):
    job_description_base64: str = Field(..., min_length=8)
    job_link: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Missing required file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail=f"Invalid YAML structure in {path}")
    return data


def _resolve_preferred_data_file(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise HTTPException(
        status_code=500,
        detail=f"Missing required files: {primary} or {fallback}",
    )


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _decode_job_description(encoded: str) -> str:
    try:
        decoded_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="job_description_base64 is not valid base64.") from exc

    try:
        decoded_text = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Decoded job description must be valid UTF-8 text.") from exc

    normalized = _normalize_whitespace(decoded_text)
    if len(normalized) < 20:
        raise HTTPException(status_code=422, detail="Decoded job description is too short.")
    return normalized


def _request_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _append_decoded_request_log(job_description: str, job_link: str | None, request_timestamp: str) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with DECODED_REQUEST_LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{request_timestamp}] job_link={job_link or ''}\n")
        log_file.write(job_description + "\n")
        log_file.write("-" * 80 + "\n")


def _write_debug_yaml_copy(cv_data: dict[str, Any], request_timestamp: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    debug_file = TMP_DIR / f"request_{request_timestamp}.yml"
    debug_file.write_text(yaml.safe_dump(cv_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return debug_file


def _extract_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    raise HTTPException(status_code=502, detail="Gemini did not return valid JSON.")


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d[\d.,+/%-]*\+?\b", text))


def _max_similarity(candidate: str, sources: list[str]) -> float:
    if not sources:
        return 0.0
    return max(SequenceMatcher(None, candidate.lower(), source.lower()).ratio() for source in sources)


def _sanitize_bullets(generated: list[Any], source_bullets: list[str], limit: int) -> list[str]:
    if not source_bullets:
        return []

    source_numbers = set().union(*(_extract_numbers(item) for item in source_bullets))
    output: list[str] = []

    for value in generated:
        if not isinstance(value, str):
            continue
        text = _normalize_whitespace(value)
        if not text:
            continue
        if _max_similarity(text, source_bullets) < 0.2:
            continue
        if not _extract_numbers(text).issubset(source_numbers):
            continue
        if text not in output:
            output.append(text)
        if len(output) >= limit:
            break

    if output:
        return output
    return source_bullets[:limit]


def _sanitize_skills(generated: list[Any], allowed: list[str], minimum: int = 8, maximum: int = 14) -> list[str]:
    canonical = {re.sub(r"[^a-z0-9]", "", item.lower()): item for item in allowed}
    result: list[str] = []

    for value in generated:
        if not isinstance(value, str):
            continue
        key = re.sub(r"[^a-z0-9]", "", value.lower())
        chosen = canonical.get(key)
        if chosen and chosen not in result:
            result.append(chosen)
        if len(result) >= maximum:
            break

    if len(result) >= minimum:
        return result
    return allowed[:maximum]


def _sanitize_summary(generated: list[Any], fallback: list[str]) -> list[str]:
    paragraphs: list[str] = []
    for value in generated:
        if not isinstance(value, str):
            continue
        cleaned = _normalize_whitespace(value)
        if cleaned:
            paragraphs.append(cleaned)
        if len(paragraphs) >= 2:
            break
    return paragraphs or fallback[:2]


def _sanitize_entries(
    generated: list[Any],
    source_entries: list[dict[str, Any]],
    *,
    max_entries: int,
    max_bullets: int,
    min_entries: int,
    is_project: bool,
) -> list[dict[str, Any]]:
    source_map = {entry["id"]: entry for entry in source_entries}
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for item in generated:
        if not isinstance(item, dict):
            continue
        entry_id = str(item.get("id", "")).strip()
        if not entry_id or entry_id in seen_ids or entry_id not in source_map:
            continue
        source_entry = source_map[entry_id]
        bullets = _sanitize_bullets(item.get("bullets", []), source_entry.get("bullet_bank", []), max_bullets)
        built: dict[str, Any] = {
            "bullets": bullets,
            "gap_after": source_entry.get("gap_after", "0.20em"),
        }
        if is_project:
            built["name"] = source_entry["name"]
        else:
            built["company"] = source_entry["company"]
            built["dates"] = source_entry["dates"]
            built["role"] = source_entry["role"]
            if source_entry.get("location"):
                built["location"] = source_entry["location"]
        selected.append(built)
        seen_ids.add(entry_id)
        if len(selected) >= max_entries:
            break

    if len(selected) < min_entries:
        for source_entry in source_entries:
            if source_entry["id"] in seen_ids:
                continue
            built = {
                "bullets": source_entry.get("bullet_bank", [])[:max_bullets],
                "gap_after": source_entry.get("gap_after", "0.20em"),
            }
            if is_project:
                built["name"] = source_entry["name"]
            else:
                built["company"] = source_entry["company"]
                built["dates"] = source_entry["dates"]
                built["role"] = source_entry["role"]
                if source_entry.get("location"):
                    built["location"] = source_entry["location"]
            selected.append(built)
            if len(selected) >= min_entries:
                break

    return selected[:max_entries]


def _fetch_job_listing_text(job_link: str | None) -> str:
    if not job_link:
        return ""

    normalized_link = job_link.strip()
    if not normalized_link:
        return ""
    if not normalized_link.startswith(("http://", "https://")):
        normalized_link = "https://" + normalized_link

    try:
        response = httpx.get(normalized_link, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError:
        return ""

    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    parts: list[str] = []
    if soup.title and soup.title.string:
        parts.append(soup.title.string)

    for tag in soup.select("h1, h2, p, li"):
        text = _normalize_whitespace(tag.get_text(" ", strip=True))
        if len(text) >= 30:
            parts.append(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            deduped.append(part)

    return "\n".join(deduped)[:6000]


def _build_prompt(job_description: str, job_link: str | None, job_link_text: str, source: dict[str, Any]) -> str:
    source_payload = {
        "skills_pool": source["skills_pool"],
        "work_experience": source["work_experience"]["entries"],
        "projects": source["projects"]["entries"],
    }

    return f"""
You tailor a one-page CV for job applications.

Hard constraints:
1. Use only facts from CANDIDATE_SOURCE below.
2. Do not invent technologies, dates, companies, roles, metrics, certifications, education, locations, or project details.
3. Rephrase source facts for clarity and ATS keyword alignment.
4. Keep output concise for one page.
5. Prefer to spend most space on work expirience, then projects, and only then summory

Return JSON only with this schema:
{{
  "summary_paragraphs": ["string"],
  "skills": ["string", "string"],
  "work_experience": [
    {{"id": "work_entry_id", "bullets": ["string"]}}
  ],
  "projects": [
    {{"id": "project_id", "bullets": ["string"]}}
  ]
}}

Formatting constraints:
- 1 summary_paragraph.
- skills must be a subset of skills_pool.
- work_experience: up to 3 entries, each up to 3 bullets.
- projects: up to 2 entries, each up to 2 bullets.
- Keep bullet text compact.

JOB_DESCRIPTION:
{job_description}

JOB_LINK:
{job_link or ""}

JOB_LINK_EXTRACTED_TEXT:
{job_link_text}

CANDIDATE_SOURCE:
{json.dumps(source_payload, ensure_ascii=False, indent=2)}
""".strip()


def _call_gemini(prompt: str) -> dict[str, Any]:
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise HTTPException(status_code=500, detail=f"Missing {GEMINI_API_KEY_ENV} environment variable.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = httpx.post(url, params={"key": api_key}, json=payload, timeout=60.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Gemini request failed due to network error.") from exc

    response_json = response.json()
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise HTTPException(status_code=502, detail="Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise HTTPException(status_code=502, detail="Gemini returned an empty response.")

    return _extract_json(text)


def _assemble_cv_data(model_output: dict[str, Any], source: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    cv_data = deepcopy(example)

    cv_data["header"] = deepcopy(source["header"])

    cv_data["summary"]["paragraphs"] = _sanitize_summary(
        model_output.get("summary_paragraphs", []),
        example.get("summary", {}).get("paragraphs", []),
    )

    cv_data["skills"]["items"] = _sanitize_skills(
        model_output.get("skills", []),
        source.get("skills_pool", []),
    )

    cv_data["work_experience"] = {
        "title": source["work_experience"].get("title", cv_data["work_experience"]["title"]),
        "entries": _sanitize_entries(
            model_output.get("work_experience", []),
            source["work_experience"]["entries"],
            max_entries=3,
            max_bullets=3,
            min_entries=2,
            is_project=False,
        ),
    }

    cv_data["projects"] = {
        "title": source["projects"].get("title", cv_data["projects"]["title"]),
        "entries": _sanitize_entries(
            model_output.get("projects", []),
            source["projects"]["entries"],
            max_entries=2,
            max_bullets=2,
            min_entries=1,
            is_project=True,
        ),
    }

    cv_data["education"] = deepcopy(source["education"])
    cv_data["certifications"] = deepcopy(source["certifications"])
    cv_data["languages"] = deepcopy(source["languages"])

    return cv_data


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-cv")
def generate_cv(request: GenerateCVRequest) -> FileResponse:
    example_data = _load_yaml(_resolve_preferred_data_file(TEMPLATE_FILE, TEMPLATE_EXAMPLE_FILE))
    source_data = _load_yaml(_resolve_preferred_data_file(CANDIDATE_FILE, CANDIDATE_EXAMPLE_FILE))

    job_description = _decode_job_description(request.job_description_base64)
    request_timestamp = _request_timestamp()
    _append_decoded_request_log(job_description, request.job_link, request_timestamp)

    link_text = _fetch_job_listing_text(request.job_link)
    prompt = _build_prompt(job_description, request.job_link, link_text, source_data)
    model_output = _call_gemini(prompt)
    cv_data = _assemble_cv_data(model_output, source_data, example_data)

    with _generation_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CV_FILE.write_text(yaml.safe_dump(cv_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        _write_debug_yaml_copy(cv_data, request_timestamp)
        try:
            build_cv(data_file=CV_FILE, output_dir=ROOT)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to compile CV PDF: {exc}") from exc

    if not PDF_FILE.exists():
        raise HTTPException(status_code=500, detail="PDF was not produced.")

    return FileResponse(
        path=str(PDF_FILE),
        filename="tailored_cv.pdf",
        media_type="application/pdf",
    )
