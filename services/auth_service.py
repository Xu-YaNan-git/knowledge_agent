"""Local account authentication with PBKDF2 passwords and signed short-lived tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from pymongo import ASCENDING, ReturnDocument

from services.metadata_service import get_metadata_service, serialize, utc_now


ROLE_VIEWER = "viewer"
ROLE_EDITOR = "editor"
ROLE_ADMIN = "admin"
ROLES = {ROLE_VIEWER, ROLE_EDITOR, ROLE_ADMIN}
ROLE_RANK = {ROLE_VIEWER: 1, ROLE_EDITOR: 2, ROLE_ADMIN: 3}
PASSWORD_ITERATIONS = 600_000


class AuthenticationError(ValueError):
    """Safe authentication failure suitable for a generic client response."""


@dataclass(frozen=True)
class AuthConfig:
    enabled: bool
    secret: str
    token_ttl_minutes: int
    cookie_secure: bool
    bootstrap_username: str
    bootstrap_password: str


def load_auth_config() -> AuthConfig:
    return AuthConfig(
        enabled=(os.getenv("AUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes"}),
        secret=(os.getenv("AUTH_SECRET") or "").strip(),
        token_ttl_minutes=max(5, int(os.getenv("AUTH_TOKEN_TTL_MINUTES", "480"))),
        cookie_secure=(os.getenv("AUTH_COOKIE_SECURE", "false").strip().lower() in {"1", "true", "yes"}),
        bootstrap_username=(os.getenv("BOOTSTRAP_ADMIN_USERNAME") or "admin").strip(),
        bootstrap_password=(os.getenv("BOOTSTRAP_ADMIN_PASSWORD") or "").strip(),
    )


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("密码至少需要 10 个字符")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_base64url_encode(salt)}${_base64url_encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), _base64url_decode(salt), int(iterations)
        )
        return hmac.compare_digest(actual, _base64url_decode(expected))
    except (TypeError, ValueError):
        return False


_DUMMY_PASSWORD_HASH = hash_password("not-a-real-password")


class AuthService:
    def __init__(self) -> None:
        self.config = load_auth_config()
        self.users = get_metadata_service().db["users"]
        self.users.create_index([("username", ASCENDING)], unique=True)

    def validate_configuration(self) -> list[str]:
        if not self.config.enabled:
            return []
        missing = []
        if len(self.config.secret) < 32:
            missing.append("AUTH_SECRET")
        if self.users.count_documents({}) == 0 and not self.config.bootstrap_password:
            missing.append("BOOTSTRAP_ADMIN_PASSWORD")
        return missing

    def bootstrap_admin(self) -> None:
        if not self.config.enabled or self.users.count_documents({}) > 0:
            return
        missing = self.validate_configuration()
        if missing:
            raise RuntimeError(f"认证配置不完整: {', '.join(missing)}")
        self.create_user(
            self.config.bootstrap_username,
            self.config.bootstrap_password,
            ROLE_ADMIN,
        )

    def create_user(self, username: str, password: str, role: str) -> dict[str, Any]:
        normalized_username = username.strip().lower()
        if not normalized_username or len(normalized_username) > 64:
            raise ValueError("用户名长度必须为 1-64 个字符")
        if role not in ROLES:
            raise ValueError("用户角色不合法")
        now = utc_now()
        record = {
            "user_id": uuid.uuid4().hex,
            "username": normalized_username,
            "password_hash": hash_password(password),
            "role": role,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        }
        self.users.insert_one(record)
        return self._public_user(record)

    def list_users(self) -> list[dict[str, Any]]:
        return [self._public_user(item) for item in self.users.find({}).sort("created_at", ASCENDING)]

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        record = self.users.find_one({"user_id": user_id})
        return self._public_user(record) if record else None

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        record = self.users.find_one({"username": username.strip().lower()})
        password_hash = record.get("password_hash", "") if record else _DUMMY_PASSWORD_HASH
        password_matches = verify_password(password, password_hash)
        if not record or not record.get("is_active") or not password_matches:
            raise AuthenticationError("用户名或密码错误")
        return self._public_user(record)

    def update_user(
        self,
        user_id: str,
        *,
        role: str | None = None,
        is_active: bool | None = None,
        password: str | None = None,
    ) -> dict[str, Any] | None:
        current = self.users.find_one({"user_id": user_id})
        if not current:
            return None
        if role is not None and role not in ROLES:
            raise ValueError("用户角色不合法")
        removes_active_admin = current.get("role") == ROLE_ADMIN and current.get("is_active") and (
            role not in {None, ROLE_ADMIN} or is_active is False
        )
        if removes_active_admin and self.users.count_documents({"role": ROLE_ADMIN, "is_active": True}) <= 1:
            raise ValueError("不能停用或降级最后一个管理员")
        changes: dict[str, Any] = {"updated_at": utc_now()}
        if role is not None:
            changes["role"] = role
        if is_active is not None:
            changes["is_active"] = is_active
        if password is not None:
            changes["password_hash"] = hash_password(password)
        record = self.users.find_one_and_update(
            {"user_id": user_id}, {"$set": changes}, return_document=ReturnDocument.AFTER
        )
        return self._public_user(record) if record else None

    def create_token(self, user: dict[str, Any]) -> str:
        if len(self.config.secret) < 32:
            raise RuntimeError("AUTH_SECRET 至少需要 32 个字符")
        now = int(time.time())
        header = _base64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
        payload = _base64url_encode(json.dumps({
            "sub": user["user_id"], "username": user["username"], "role": user["role"],
            "iat": now, "exp": now + self.config.token_ttl_minutes * 60,
        }, separators=(",", ":")).encode())
        signature = hmac.new(self.config.secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
        return f"{header}.{payload}.{_base64url_encode(signature)}"

    def verify_token(self, token: str) -> dict[str, Any]:
        if len(self.config.secret) < 32:
            raise RuntimeError("认证签名密钥配置不安全")
        try:
            header, payload, signature = token.split(".", 2)
            expected = hmac.new(
                self.config.secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256
            ).digest()
            if not hmac.compare_digest(expected, _base64url_decode(signature)):
                raise AuthenticationError("登录凭证无效")
            claims = json.loads(_base64url_decode(payload))
            if int(claims.get("exp", 0)) <= int(time.time()):
                raise AuthenticationError("登录已过期")
            user = self.get_user(str(claims.get("sub") or ""))
            if not user or not user.get("is_active"):
                raise AuthenticationError("用户已停用")
            return user
        except AuthenticationError:
            raise
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise AuthenticationError("登录凭证无效") from exc

    @staticmethod
    def _public_user(record: dict[str, Any]) -> dict[str, Any]:
        return serialize({key: value for key, value in record.items() if key not in {"_id", "password_hash"}})


_auth_service: AuthService | None = None
_auth_lock = threading.Lock()


def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        with _auth_lock:
            if _auth_service is None:
                _auth_service = AuthService()
    return _auth_service
