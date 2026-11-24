from services.auth_service import (
    ROLE_ADMIN,
    ROLE_EDITOR,
    ROLE_VIEWER,
    AuthConfig,
    AuthService,
    hash_password,
    verify_password,
)
from app import _audit_action_and_target, _required_role
from fastapi import HTTPException
from fastapi.testclient import TestClient
from types import SimpleNamespace
import app as app_module


def test_password_hash_is_salted_and_verifiable() -> None:
    first = hash_password("correct-horse-battery")
    second = hash_password("correct-horse-battery")

    assert first != second
    assert "correct-horse-battery" not in first
    assert verify_password("correct-horse-battery", first) is True
    assert verify_password("wrong-password", first) is False


def test_signed_token_round_trip_uses_current_user_state() -> None:
    service = AuthService.__new__(AuthService)
    service.config = AuthConfig(
        enabled=True,
        secret="a-secure-test-secret-that-is-longer-than-32-characters",
        token_ttl_minutes=30,
        cookie_secure=False,
        bootstrap_username="admin",
        bootstrap_password="",
    )
    user = {"user_id": "user-1", "username": "alice", "role": ROLE_EDITOR, "is_active": True}
    service.get_user = lambda user_id: user if user_id == "user-1" else None

    token = service.create_token(user)
    verified = service.verify_token(token)

    assert verified["user_id"] == "user-1"
    assert verified["role"] == ROLE_EDITOR


def test_route_permissions_protect_mutations_and_administration() -> None:
    assert _required_role("POST", "/api/query") == ROLE_VIEWER
    assert _required_role("POST", "/api/knowledge-bases") == ROLE_EDITOR
    assert _required_role("DELETE", "/api/knowledge-bases/kb-1") == ROLE_ADMIN
    assert _required_role("GET", "/api/users") == ROLE_ADMIN
    assert _required_role("GET", "/api/audit-logs") == ROLE_ADMIN
    assert _required_role("GET", "/api/operations/summary") == ROLE_EDITOR
    assert _required_role("GET", "/api/evaluations") == ROLE_EDITOR
    assert _required_role("PATCH", "/api/evaluations/eval-1/baseline") == ROLE_EDITOR
    assert _required_role("GET", "/api/exports/task-1/download") == ROLE_EDITOR


def test_audit_actions_use_stable_names_and_resource_targets() -> None:
    assert _audit_action_and_target("POST", "/api/knowledge-bases") == (
        "knowledge_base.create", "knowledge_base", None,
    )
    assert _audit_action_and_target("DELETE", "/api/documents/doc-1") == (
        "document.delete", "document", "doc-1",
    )
    assert _audit_action_and_target("POST", "/api/tasks/task-1/cancel") == (
        "task.cancel", "task", "task-1",
    )
    assert _audit_action_and_target("PATCH", "/api/evaluations/eval-1/baseline") == (
        "evaluation.baseline", "evaluation", "eval-1",
    )
    assert _audit_action_and_target("POST", "/api/knowledge-bases/kb-1/exports") == (
        "knowledge_base.export", "knowledge_base", "kb-1",
    )
    assert _audit_action_and_target("GET", "/api/tasks/task-1") == (None, None, None)


def test_auth_middleware_rejects_anonymous_and_viewer_admin_access() -> None:
    class FakeAuthService:
        @staticmethod
        def verify_token(_token):
            return {"user_id": "viewer-1", "username": "viewer", "role": ROLE_VIEWER, "is_active": True}

    config = AuthConfig(
        enabled=True,
        secret="a-secure-test-secret-that-is-longer-than-32-characters",
        token_ttl_minutes=30,
        cookie_secure=False,
        bootstrap_username="admin",
        bootstrap_password="",
    )
    original_load_config = app_module.load_auth_config
    original_get_service = app_module.get_auth_service
    app_module.load_auth_config = lambda: config
    app_module.get_auth_service = lambda: FakeAuthService()
    try:
        client = TestClient(app_module.app)
        assert client.get("/api/health/live").status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        response = client.get("/api/operations/summary", cookies={"kb_access_token": "valid-token"})
        assert response.status_code == 403
    finally:
        app_module.load_auth_config = original_load_config
        app_module.get_auth_service = original_get_service


def test_login_rate_limit_blocks_repeated_failures() -> None:
    client_key = "test-client"
    app_module._clear_login_failures(client_key)
    try:
        for _ in range(app_module.LOGIN_MAX_ATTEMPTS):
            app_module._record_login_failure(client_key)
        assert app_module._check_login_rate_limit(client_key) > 0
    finally:
        app_module._clear_login_failures(client_key)


def test_task_ownership_hides_other_users_resources_but_allows_admin() -> None:
    config = AuthConfig(
        enabled=True,
        secret="a-secure-test-secret-that-is-longer-than-32-characters",
        token_ttl_minutes=30,
        cookie_secure=False,
        bootstrap_username="admin",
        bootstrap_password="",
    )
    original_load_config = app_module.load_auth_config
    app_module.load_auth_config = lambda: config
    try:
        viewer_request = SimpleNamespace(
            state=SimpleNamespace(user={"user_id": "viewer-1", "role": ROLE_VIEWER})
        )
        try:
            app_module._ensure_task_access({"owner_user_id": "viewer-2"}, viewer_request)
        except HTTPException as exc:
            assert exc.status_code == 404
        else:
            raise AssertionError("跨用户任务访问应返回 404")

        admin_request = SimpleNamespace(
            state=SimpleNamespace(user={"user_id": "admin-1", "role": ROLE_ADMIN})
        )
        app_module._ensure_task_access({"owner_user_id": "viewer-2"}, admin_request)
    finally:
        app_module.load_auth_config = original_load_config


def test_resource_quota_returns_retry_headers_when_exhausted() -> None:
    class ExhaustedQuotaService:
        @staticmethod
        def consume_rate_limit(**_kwargs):
            return {
                "allowed": False,
                "limit": 30,
                "remaining": 0,
                "reset_at": "2026-11-22T12:01:00+00:00",
                "retry_after_seconds": 20,
            }

    request = SimpleNamespace(
        state=SimpleNamespace(user={"user_id": "user-1", "role": ROLE_VIEWER}, request_id="request-1"),
        client=SimpleNamespace(host="127.0.0.1"),
    )
    original_get_service = app_module.get_metadata_service
    app_module.get_metadata_service = lambda: ExhaustedQuotaService()
    try:
        try:
            app_module._enforce_rate_limit(request, "query")
        except HTTPException as exc:
            assert exc.status_code == 429
            assert exc.headers["Retry-After"] == "20"
            assert exc.headers["X-RateLimit-Remaining"] == "0"
        else:
            raise AssertionError("配额耗尽时应拒绝请求")
    finally:
        app_module.get_metadata_service = original_get_service
