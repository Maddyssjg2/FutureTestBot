# FutureTestBot

FutureTestBot is a mixed JavaScript, React, and Python workspace for trading automation experiments, dashboard UI work, and CLI demos.

## What this repo contains

- `src/` — Node.js CLI entrypoint for star pattern demos
- `frontend/` — React dashboard scaffold for trading status and controls
- `backend/` — Python trading and strategy utilities
- project docs and helper scripts for setup and experimentation

## Getting Started

### Node.js CLI

```bash
npm start
```

### Frontend

```bash
cd frontend
npm install
npm start
```

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

## Notes

- The repository is organized as a mixed Node.js + frontend + Python workspace.
- The frontend uses React tooling and can be extended as needed.
- The CLI is dependency-free and runs with built-in Node.js modules.

## License

MIT
