# FastAPI Tutorial Frontend

Interactive React + Vite frontend for learning FastAPI with energy-data examples.

## Live URL

- https://niranjanxprt.github.io/Pythonfast/

## Local Development

```bash
npm ci
npm run dev
```

Open: http://localhost:5173

## Production Build

```bash
npm run build
npm run preview
```

## Deployment Notes

- This app is deployed via repository workflow:
  - `.github/workflows/deploy-fastapi-tutorial-pages.yml`
- GitHub Pages project URL requires Vite base path:
  - `base: "/Pythonfast/"`

## Best-Practice Notes

- Keep static docs/tutorial UI on Pages and backend API runtime separate.
- Use `npm ci` in CI for deterministic installs from `package-lock.json`.
- Use path-based workflow triggers to deploy only when frontend-related files change.
