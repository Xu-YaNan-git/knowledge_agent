"""Evaluation history must return a real total and a stable page window."""

from types import SimpleNamespace
from unittest.mock import Mock

import app as app_module
from services.metadata_service import MetadataService


def test_evaluation_history_paginates_with_owner_scope() -> None:
    service = MetadataService.__new__(MetadataService)
    service.evaluation_runs = Mock()
    cursor = Mock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = [{"evaluation_id": "run-21"}]
    service.evaluation_runs.find.return_value = cursor
    service.evaluation_runs.count_documents.return_value = 31

    page = service.list_evaluation_runs(limit=10, owner_user_id="owner-1", skip=20)

    assert page == {
        "items": [{"evaluation_id": "run-21"}],
        "total": 31,
        "skip": 20,
        "limit": 10,
    }
    service.evaluation_runs.find.assert_called_once_with({"owner_user_id": "owner-1"})
    cursor.skip.assert_called_once_with(20)
    cursor.limit.assert_called_once_with(10)
    service.evaluation_runs.count_documents.assert_called_once_with({"owner_user_id": "owner-1"})


def test_evaluation_endpoint_passes_paging_to_service(monkeypatch) -> None:
    service = Mock()
    service.list_evaluation_runs.return_value = {"items": [], "total": 31, "skip": 20, "limit": 10}
    monkeypatch.setattr(app_module, "get_metadata_service", lambda: service)
    monkeypatch.setattr(app_module, "_owner_scope", lambda request: "owner-1")

    response = app_module.list_evaluations(SimpleNamespace(), skip=20, limit=10)

    assert response["total"] == 31
    service.list_evaluation_runs.assert_called_once_with(10, owner_user_id="owner-1", skip=20)
