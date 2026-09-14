.PHONY: docs-check

docs-check:
	uv run python scripts/check_docs.py
