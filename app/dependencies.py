import jwt
from fastapi import HTTPException, Request, status
from jwt import PyJWKClient, PyJWTError

from app.config import settings

_jwks_client = PyJWKClient(settings.CLERK_JWKS_URL)


async def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    try:
        unverified_header = jwt.get_unverified_header(token)
    except PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token header",
        )

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
    except PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signing key",
        )

    try:
        return jwt.decode(
            token,
            signing_key,
            algorithms=[unverified_header.get("alg", "RS256")],
            audience=settings.CLERK_AUDIENCE or None,
            issuer=settings.CLERK_ISSUER or None,
        )
    except PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
