# Contributing

Thanks for helping improve Deferral Twin.

## Ground rules

1. **Honesty first.** Never weaken the deferral gate to make the twin answer more. If a change affects which tier a question lands in, include before/after numbers.
2. **Offline by default.** Nothing may leave the machine unless the person turned on Online mode.
3. **Consent-gated likeness.** Voice and face features only ever use the person who consented in the app.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ci.txt          # enough for the tests
PYTHONHASHSEED=0 python scripts/dev_server.py --port 8000   # UI without models
```

The full app (models, voice, video) needs an Apple Silicon Mac; see [`DESKTOP.md`](DESKTOP.md).

## Tests

Each file in [`tests/`](tests/) runs on its own and needs `ffmpeg` on your `PATH`:

```bash
PYTHONHASHSEED=0 python tests/test_gate_and_media.py
```

CI runs every suite on each push and pull request.

## Commits and pull requests

- Keep each commit to one change, with a message saying what it does ("Add …", "Fix …").
- Fill in the pull request checklist.
