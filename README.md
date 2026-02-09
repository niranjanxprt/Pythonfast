# Pythonfast

Energy-focused FastAPI learning repository with a React tutorial frontend and FastAPI backend examples.

## Live Links

- GitHub Pages (tutorial app): https://niranjanxprt.github.io/Pythonfast/
- Repository: https://github.com/niranjanxprt/Pythonfast

## Repository Layout

```text
.
├── fastapi-tutorial/      # React + Vite interactive FastAPI tutorial (deployed to GitHub Pages)
├── my-fastapi-app/        # FastAPI example backend and Pydantic models
└── .github/workflows/     # CI/CD workflows (Pages deployment)
```

## What This Repo Includes

- FastAPI basics: routing, request validation, query/path params
- Pydantic models and validation patterns
- Energy-domain examples (panels, readings, monitoring)
- Pandas and NumPy examples for energy data analysis
- React tutorial UI deployed to GitHub Pages

## Quick Start

### 1. Frontend tutorial app

```bash
cd fastapi-tutorial
npm ci
npm run dev
```

Open: http://localhost:5173

### 2. FastAPI backend examples

```bash
cd my-fastapi-app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install fastapi "uvicorn[standard]"
uvicorn main:app --reload
```

Open:

- API root: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs

## Deployment

The frontend is deployed automatically with GitHub Actions:

- Workflow: `.github/workflows/deploy-fastapi-tutorial-pages.yml`
- Trigger: push to `main` for files under `fastapi-tutorial/**`
- Publish target: `gh-pages` branch
- Vite base path: `/Pythonfast/`

## Best Practices Followed

- Keep frontend and backend concerns separated by directory.
- Deploy only static frontend assets to Pages; keep API runtime separate.
- Use lockfiles (`package-lock.json`) for reproducible frontend builds.
- Use CI/CD workflow for repeatable deployment on every push.
- Keep secrets out of source code and use environment variables in real deployments.

## Notes

- This repository is tutorial-oriented and optimized for learning.
- For production APIs, add authentication, persistent storage, tests, and observability.
