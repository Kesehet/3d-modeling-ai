from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from urllib.parse import urlsplit

from fastapi import Form, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .config import API_TOKEN

SESSION_COOKIE = "model_maker_session"
SESSION_SECONDS = 12 * 60 * 60
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}


def _valid_token(value: str) -> bool:
    return bool(API_TOKEN) and hmac.compare_digest(value.encode(), API_TOKEN.encode())


def _signature(value: str) -> str:
    return hmac.new(API_TOKEN.encode(), ("dashboard-session:" + value).encode(), hashlib.sha256).hexdigest()


def _new_session() -> str:
    value = f"{int(time.time()) + SESSION_SECONDS}.{secrets.token_hex(16)}"
    return f"{value}.{_signature(value)}"


def _valid_session(value: str) -> bool:
    if not API_TOKEN or not value or len(value) > 180:
        return False
    try:
        expires, nonce, signature = value.split(".")
        deadline = int(expires)
    except (ValueError, TypeError):
        return False
    now = time.time()
    return (
        now < deadline <= now + SESSION_SECONDS
        and len(nonce) == 32
        and hmac.compare_digest(signature.encode(), _signature(f"{expires}.{nonce}").encode())
    )


def _same_origin(request: Request) -> bool:
    if request.headers.get("sec-fetch-site") == "cross-site":
        return False
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and parsed.netloc == request.headers.get("host")
    return True


def _login_page(error: str = "", *, code: int = 200) -> HTMLResponse:
    # Error strings are fixed server messages; never reflect the submitted key.
    return HTMLResponse("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Open your workspace · 3D Modeling AI</title>
<style>body{margin:0;background:#0e1421;color:#eaf0fa;font:16px system-ui;display:grid;min-height:100vh;place-items:center}
main{width:min(380px,85vw);padding:32px;border:1px solid #34415a;border-radius:18px;background:#172033}
h1{font-size:26px}p{line-height:1.6;color:#b8c7dc}label{display:block;margin:22px 0 8px}
input,button{box-sizing:border-box;width:100%;padding:14px;border-radius:9px;font:inherit}
input{background:#0e1421;border:1px solid #53627b;color:white}button{margin-top:16px;background:#86b9ff;border:0;color:#10213d;font-weight:650;cursor:pointer}
.error{color:#ffb4b4}small{color:#b8c7dc}</style><main><small>3D MODELING AI</small>
<h1>Open your workspace</h1><p>Create models, review their progress, and download your work.</p>
""" + (f'<p class="error" role="alert">{error}</p>' if error else "") + """
<form method="post" action="/auth/login"><label for="key">Workspace access key</label>
<input id="key" name="key" type="password" autocomplete="current-password" required autofocus maxlength="4096">
<button type="submit">Open workspace</button></form></main></html>""",
        status_code=code, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


async def dashboard_access(request: Request, call_next):
    path = request.url.path
    if API_TOKEN and (path == "/" or path.startswith("/dashboard/")):
        authorization = request.headers.get("authorization", "")
        bearer = authorization.startswith("Bearer ") and _valid_token(authorization[7:])
        session = _valid_session(request.cookies.get(SESSION_COOKIE, ""))
        if not bearer and not session:
            if path == "/" and request.method == "GET":
                return _login_page()
            return JSONResponse({"detail": "Sign in to your workspace."}, status_code=401,
                                headers={"Cache-Control": "no-store"})
        if not bearer and request.method not in {"GET", "HEAD", "OPTIONS"} and not _same_origin(request):
            return JSONResponse({"detail": "Cross-site action refused."}, status_code=403)
    response = await call_next(request)
    if path == "/" or path.startswith(("/dashboard/", "/auth/")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
    return response


async def dashboard_login(request: Request, key: str = Form(max_length=4096)):
    if not _same_origin(request):
        raise HTTPException(status_code=403, detail="Cross-site sign-in refused.")
    if not API_TOKEN:
        return RedirectResponse("/", status_code=303)
    now = time.monotonic()
    # Bound the in-memory window, including when many distinct clients submit keys.
    for address in list(_LOGIN_ATTEMPTS):
        _LOGIN_ATTEMPTS[address] = [t for t in _LOGIN_ATTEMPTS[address] if now - t < 60]
        if not _LOGIN_ATTEMPTS[address]:
            del _LOGIN_ATTEMPTS[address]
    address = request.client.host if request.client else "unknown"
    attempts = _LOGIN_ATTEMPTS.get(address, [])
    if len(attempts) >= 5 or len(_LOGIN_ATTEMPTS) >= 1000:
        return _login_page("Too many attempts. Try again in a minute.", code=429)
    if not _valid_token(key):
        _LOGIN_ATTEMPTS[address] = [*attempts, now]
        return _login_page("That access key is not valid.", code=401)
    _LOGIN_ATTEMPTS.pop(address, None)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(SESSION_COOKIE, _new_session(), max_age=SESSION_SECONDS,
                        httponly=True, secure=request.url.hostname not in {"localhost", "127.0.0.1", "::1"},
                        samesite="strict", path="/")
    return response


async def dashboard_logout(request: Request):
    if not _same_origin(request):
        raise HTTPException(status_code=403, detail="Cross-site sign-out refused.")
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return response


async def require_api_token(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer ") or not _valid_token(authorization[7:]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")
