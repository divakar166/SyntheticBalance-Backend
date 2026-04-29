from __future__ import annotations

from persistence import SupabaseS3Backend

_storage_backend = None


def get_storage_backend():
    global _storage_backend
    if _storage_backend is not None:
        return _storage_backend

    _storage_backend = SupabaseS3Backend()
    return _storage_backend
