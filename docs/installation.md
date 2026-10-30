# Installation

Requires **Python 3.10 or newer**.

## Create a virtual environment (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

On Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`.

## Install

From a clone (editable, with test and lint tools):

```powershell
pip install -e ".[dev]"
```

From GitHub (placeholder URL, replace `<user>`):

```powershell
pip install git+https://github.com/<user>/rsm_design
```

## Verify

```powershell
python -c "import rsm_design; print(rsm_design.__version__)"
rsm-design --help
```

Next: [Quickstart](quickstart.md). Back to [index](index.md).
