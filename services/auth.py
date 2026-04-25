from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from settings import get_settings

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str | None = None


@lru_cache(maxsize=1)
def _supabase_auth_client():
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for API auth.")

    from supabase import create_client

    return create_client(settings.supabase_url, settings.supabase_service_role_key)


async def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    try:
        response = _supabase_auth_client().auth.get_user(credentials.credentials)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired bearer token.") from exc

    user = getattr(response, "user", None)
    user_id = getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired bearer token.")

    return AuthenticatedUser(id=str(user_id), email=getattr(user, "email", None))


def record_user_id(record: dict | None) -> str | None:
    if not record:
        return None
    metadata = record.get("metadata") or {}
    return record.get("user_id") or metadata.get("user_id")


def ensure_user_owns_record(record: dict | None, user: AuthenticatedUser, resource: str):
    if not record:
        raise HTTPException(status_code=404, detail=f"{resource} was not found.")
    if record_user_id(record) != user.id:
        raise HTTPException(status_code=404, detail=f"{resource} was not found.")
