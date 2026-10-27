"""Production entry point: gunicorn librarian.wsgi:app

Loads the index and models once at import so the first user request is not
the one that pays the ~10 s model load.
"""
from .server import app, lib

lib()
