# Development

## Environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Checks

```powershell
python -m pytest -q --cov=rsm_design
black --check .
ruff check .
```

The suite includes a regression test that refits the DLLME `RSM_3` sheet and
compares coefficients with the original 2024 results.

## Style

Conventions are in [CODING_STYLE.md](../CODING_STYLE.md) (written in Spanish):
thin scripts over a library, `params:` / `return:` docstrings with domain
reasoning, commented uppercase sections, keyword arguments, Black formatting,
lab-log style messages through `logging`, figures saved then closed.

## Contributing

- Keep commits small, with short messages in plain English.
- Add or update tests with each change; run the checks above first.
- Private data (`data/hazeladd/`) is git-ignored and must never be committed.
- Do not push to `main` without review.

Back to [index](index.md).
