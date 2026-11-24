from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from pymongo.errors import DuplicateKeyError

from app import (
    _document_source_object_name,
    _document_media_type,
    _document_export_path,
    _present_document,
    _present_task,
    _present_task_summary,
    _safe_remove_task_directory,
)
from processor.import_processor.node.c_node_md_img import NodeMDImg
from processor.query_processor.nodes.a_node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.f_node_rerank import NodeRerank
from services.metadata_service import MetadataService, rate_limit_window_start


def test_present_task_hides_private_payload() -> None:
    task = {
        "task_id": "task-1",
        "type": "query",
        "status": "pending",
        "payload": {"question": "private"},
        "started_at": None,
        "finished_at": None,
    }

    presented = _present_task(task)

    assert "payload" not in presented
    assert presented["duration_ms"] == 0
    assert presented["pipeline"]


def test_local_citation_keeps_document_metadata() -> None:
    node = NodeRerank()
    state = {
        "rrf_chunks": [{
            "chunk_id": 7,
            "content": "内存为 16GB",
            "file_title": "产品手册",
            "parent_title": "硬件规格",
            "document_id": "doc-1",
            "knowledge_base_id": "kb-1",
        }],
        "web_search_docs": [],
    }

    documents = node._step_1_merge_multi_source_docs(state)

    assert documents[0]["title"] == "硬件规格"
    assert documents[0]["document_id"] == "doc-1"
    assert documents[0]["knowledge_base_id"] == "kb-1"


def test_task_summary_does_not_expose_answer_or_sources() -> None:
    task = {
        "task_id": "task-1", "type": "query", "session_id": "session-1", "status": "completed",
        "answer": "private answer", "sources": [{"content": "private source"}], "payload": {"question": "private"},
        "started_at": None, "finished_at": None,
    }

    summary = _present_task_summary(task)

    assert "answer" not in summary
    assert "sources" not in summary
    assert "payload" not in summary


def test_safe_temp_cleanup_only_removes_direct_task_directory() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory) / "uploads"
        task_directory = root / "task-1"
        task_directory.mkdir(parents=True)
        (task_directory / "file.txt").write_text("temporary", encoding="utf-8")
        outside = Path(temporary_directory) / "outside"
        outside.mkdir()

        assert _safe_remove_task_directory(root, "task-1") is True
        assert task_directory.exists() is False
        assert _safe_remove_task_directory(root, "../outside") is False
        assert outside.exists() is True


def test_task_start_records_worker_lease() -> None:
    class CapturingTasks:
        def __init__(self) -> None:
            self.query = None
            self.update = None

        def find_one_and_update(self, query, update, **_kwargs):
            self.query = query
            self.update = update
            return {"task_id": "task-1"}

    tasks = CapturingTasks()
    service = MetadataService.__new__(MetadataService)
    service.tasks = tasks

    assert service.start_task("task-1", "worker-1", 180) is True
    assert tasks.query["status"]["$in"] == ["pending", "retrying"]
    assert tasks.update["$set"]["worker_id"] == "worker-1"
    assert tasks.update["$set"]["lease_expires_at"] > tasks.update["$set"]["started_at"]


def test_audit_event_is_append_only_and_contains_trace_context() -> None:
    class CapturingAuditLogs:
        def __init__(self) -> None:
            self.record = None

        def insert_one(self, record):
            self.record = record.copy()

    audit_logs = CapturingAuditLogs()
    service = MetadataService.__new__(MetadataService)
    service.audit_logs = audit_logs

    result = service.record_audit_event(
        action="document.delete",
        outcome="success",
        method="DELETE",
        path="/api/documents/doc-1",
        status_code=200,
        request_id="request-1",
        actor_user_id="user-1",
        actor_username="alice",
        actor_role="editor",
        client_ip="127.0.0.1",
        target_type="document",
        target_id="doc-1",
    )

    assert audit_logs.record["audit_id"]
    assert result["request_id"] == "request-1"
    assert result["actor_username"] == "alice"
    assert result["target_id"] == "doc-1"
    assert "password" not in result


def test_setting_evaluation_baseline_only_replaces_same_owner_and_dataset() -> None:
    class InMemoryEvaluationRuns:
        def __init__(self) -> None:
            self.records = [
                {"evaluation_id": "old", "owner_user_id": "user-1", "dataset_checksum": "same", "is_baseline": True},
                {"evaluation_id": "new", "owner_user_id": "user-1", "dataset_checksum": "same", "is_baseline": False},
                {"evaluation_id": "other", "owner_user_id": "user-2", "dataset_checksum": "same", "is_baseline": True},
            ]

        @staticmethod
        def _matches(record, query):
            return all(record.get(key) == value for key, value in query.items())

        def find_one(self, query, **_kwargs):
            return next((record for record in self.records if self._matches(record, query)), None)

        def update_many(self, query, update):
            for record in self.records:
                if self._matches(record, query):
                    record.update(update["$set"])

        def find_one_and_update(self, query, update, **_kwargs):
            record = self.find_one(query)
            if record:
                record.update(update["$set"])
            return record

    runs = InMemoryEvaluationRuns()
    service = MetadataService.__new__(MetadataService)
    service.evaluation_runs = runs

    result = service.set_evaluation_baseline("new", owner_user_id="user-1")

    assert result["is_baseline"] is True
    assert runs.find_one({"evaluation_id": "old"})["is_baseline"] is False
    assert runs.find_one({"evaluation_id": "other"})["is_baseline"] is True


def test_rate_limit_window_is_shared_and_counter_rejects_overflow() -> None:
    now = datetime(2026, 11, 22, 12, 0, 59, tzinfo=timezone.utc)
    assert rate_limit_window_start(now, 60) == datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)

    class InMemoryCounters:
        def __init__(self) -> None:
            self.record = None

        def find_one_and_update(self, query, update, *, upsert, **_kwargs):
            limit = query["$or"][0]["count"]["$lt"]
            if self.record is None:
                if not upsert:
                    return None
                self.record = {**update["$setOnInsert"], "count": 0}
            elif self.record["count"] >= limit:
                if upsert:
                    raise DuplicateKeyError("counter already exists")
                return None
            self.record["count"] += update["$inc"]["count"]
            return self.record.copy()

    service = MetadataService.__new__(MetadataService)
    service.rate_limit_counters = InMemoryCounters()
    results = [service.consume_rate_limit(
        scope_key="user:1", action="query", limit=2, window_seconds=60, now=now,
    ) for _ in range(3)]

    assert [item["allowed"] for item in results] == [True, True, False]
    assert results[-1]["remaining"] == 0


def test_document_archive_key_uses_internal_ids_and_api_hides_storage_paths() -> None:
    document = {
        "document_id": "doc-1",
        "logical_document_id": "logical-1",
        "knowledge_base_id": "kb-1",
        "version": 2,
        "filename": "../../产品手册.pdf",
        "source_path": "D:/private/uploads/file.pdf",
        "source_bucket": "private-documents",
        "source_object_name": "private/object.pdf",
    }

    object_name = _document_source_object_name(document)
    presented = _present_document(document)

    assert object_name == "sources/kb-1/logical-1/v2/doc-1.pdf"
    assert _document_media_type("产品手册.pdf") == "application/pdf"
    assert _document_media_type("说明.md") == "text/markdown"
    assert presented["source_archived"] is True
    assert "source_path" not in presented
    assert "source_bucket" not in presented
    assert "source_object_name" not in presented


def test_export_archive_path_cannot_escape_zip_root() -> None:
    document = {
        "document_id": "../doc-1",
        "logical_document_id": "../../logical-1",
        "version": 3,
        "filename": "../../dangerous.md",
    }

    archive_path = _document_export_path(document)

    assert archive_path.startswith("documents/")
    assert ".." not in archive_path
    assert archive_path.endswith("dangerous.md")


def test_markdown_without_images_directory_skips_image_processing() -> None:
    with TemporaryDirectory() as temporary_directory:
        missing_images = Path(temporary_directory) / "images"

        assert NodeMDImg()._step_2_scan_images("# 纯文本说明", missing_images) == []


def test_markdown_empty_default_content_reads_uploaded_file() -> None:
    with TemporaryDirectory() as temporary_directory:
        markdown_path = Path(temporary_directory) / "manual.md"
        markdown_path.write_text("# 产品手册\n\n内存容量：32GB", encoding="utf-8")

        content, resolved_path, images_dir = NodeMDImg()._step_1_get_content({
            "md_path": str(markdown_path),
            "md_content": "",
        })

        assert content == "# 产品手册\n\n内存容量：32GB"
        assert resolved_path == markdown_path
        assert images_dir == markdown_path.parent / "images"


def test_scoped_query_continues_when_product_name_is_not_recognized() -> None:
    state = {
        "knowledge_base_ids": ["kb-1"],
        "item_names": ["unrecognized-model"],
    }

    result = NodeItemNameConfirm()._step_7_check_confirmation(
        state,
        {"confirmed_item_names": [], "options": ["similar-model"]},
        history=[],
    )

    assert result["answer"] == ""
    assert result["item_names"] == []
