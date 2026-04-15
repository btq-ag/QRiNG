# Contributing to QRiNG

## Dev Install

```bash
git clone https://github.com/btq-ag/QRiNG.git
cd QRiNG
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=qring --cov-report=term-missing
```

## Lint and Type Check

```bash
ruff check qring/ tests/
ruff format --check qring/ tests/
mypy qring/ --ignore-missing-imports
```

## Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Style

- camelCase for all Python identifiers (overrides PEP 8)
- Type annotations on all public function signatures
- No bare `except` clauses
- No em dashes in comments, docstrings, or output
