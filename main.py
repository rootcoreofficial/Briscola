"""
Entrypoint a livello root per `fastapi run` / FastAPI Cloud.

Il codice applicativo vive in `src/briscola_ai/` (layout "src"), dove l'autodetection del comando
`fastapi` non guarderebbe (cerca `main.py`/`app.py` nella root). Questo shim ri-espone l'app ASGI
così che il build cloud (e `fastapi run`) la trovino come `main:app`. Il package `briscola_ai` è
installato dal `pyproject.toml`, quindi l'import funziona a runtime.
"""

from briscola_ai.main import app

__all__ = ["app"]
