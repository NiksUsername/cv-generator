# CV Tailoring + PDF API

This project renders a one-page LaTeX CV from YAML and includes a FastAPI service that:

1. Accepts a job description and optional job listing URL.
2. Uses Gemini (`gemini-2.5-flash`) to tailor CV text to the role.
3. Writes generated content into `data/cv.yaml`.
4. Compiles and returns `main_modular.pdf`.

## Data Files

Tracked templates:
- `data/template.yaml.example`: starter CV template schema/content.
- `data/candidate.yaml.example`: starter factual source used for Gemini selection.

Local runtime files (ignored by git):
- `data/template.yaml`: active CV template source (copy from `.example` and edit locally).
- `data/candidate.yaml`: active candidate fact source (copy from `.example` and edit locally).
- `data/cv.yaml`: generated per request.

`sections/*.tex` are generated files and should not be edited manually.

## First-Time Setup

Create local runtime YAML files from examples:

```bash
cp data/template.yaml.example data/template.yaml
cp data/candidate.yaml.example data/candidate.yaml
```

Then edit `data/template.yaml` and `data/candidate.yaml` with your own content.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Set Gemini API key:

```bash
export GEMINI_API_KEY="your_key_here"
```

## Run With Docker

Build image:

```bash
docker build -t cv-tailoring-api .
```

Run container:

```bash
docker run --rm -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" cv-tailoring-api
```

## Run API

```bash
uvicorn app:app --reload
```

## Generate CV (API)

```bash
curl -X POST "http://127.0.0.1:8000/generate-cv" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description_base64": "RGV2T3BzIEVuZ2luZWVyIHJvbGUgZm9jdXNlZCBvbiBBV1MsIFRlcnJhZm9ybSwgQ0kvQ0QgYW5kIG9ic2VydmFiaWxpdHku",
    "job_link": "https://example.com/jobs/devops"
  }' \
  --output tailored_cv.pdf
```

`job_description_base64` must be base64-encoded UTF-8 text.

## Manual Build

If `data/cv.yaml` exists, build uses it.
If it does not exist, build falls back to `data/template.yaml`, then `data/template.yaml.example`.

```bash
python tools/build_cv.py
```

Manual render/compile:

```bash
python tools/render_sections.py
tectonic main_modular.tex -o .
```
