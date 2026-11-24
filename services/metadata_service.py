"""MongoDB-backed metadata and task persistence services."""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def rate_limit_window_start(now: datetime, window_seconds: int) -> datetime:
    """Align a timestamp to a deterministic fixed window shared by every app instance."""
    timestamp = int(now.timestamp())
    return datetime.fromtimestamp(timestamp - timestamp % window_seconds, timezone.utc)


def serialize(value: Any) -> Any:
    """Convert Mongo-specific values into JSON-safe values."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value


class MetadataService:
    """Own metadata collections and their consistency rules."""

    def __init__(self) -> None:
        mongo_url = (os.getenv("MONGO_URL") or "").strip()
        database_name = (os.getenv("MONGO_DB_NAME") or "").strip()
        if not mongo_url or not database_name:
            raise RuntimeError("MONGO_URL and MONGO_DB_NAME must be configured")
        self.client = MongoClient(mongo_url, serverSelectionTimeoutMS=5_000)
        self.db = self.client[database_name]
        self.knowledge_bases = self.db["knowledge_bases"]
        self.documents = self.db["knowledge_documents"]
        self.tasks = self.db["background_tasks"]
        self.chat_sessions = self.db["chat_sessions"]
        self.evaluation_runs = self.db["evaluation_runs"]
        self.audit_logs = self.db["audit_logs"]
        self.rate_limit_counters = self.db["rate_limit_counters"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.knowledge_bases.create_index("name", unique=True)
        self.documents.create_index(
            [("knowledge_base_id", ASCENDING), ("logical_document_id", ASCENDING), ("version", DESCENDING)]
        )
        self.documents.create_index(
            [("knowledge_base_id", ASCENDING), ("filename", ASCENDING), ("version", ASCENDING)], unique=True
        )
        self.documents.create_index([("knowledge_base_id", ASCENDING), ("status", ASCENDING)])
        self.tasks.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
        self.tasks.create_index("task_id", unique=True)
        self.chat_sessions.create_index("session_id", unique=True)
        self.chat_sessions.create_index([("updated_at", DESCENDING)])
        self.evaluation_runs.create_index([("created_at", DESCENDING)])
        self.audit_logs.create_index([("created_at", DESCENDING)])
        self.audit_logs.create_index([("actor_user_id", ASCENDING), ("created_at", DESCENDING)])
        self.audit_logs.create_index([("action", ASCENDING), ("created_at", DESCENDING)])
        self.rate_limit_counters.create_index(
            [("scope_key", ASCENDING), ("action", ASCENDING), ("window_start", ASCENDING)],
            unique=True,
        )
        self.rate_limit_counters.create_index("expires_at", expireAfterSeconds=0)

    def ping(self) -> None:
        self.client.admin.command("ping")

    # Audit logs are append-only: this service intentionally exposes no update or delete method.
    def record_audit_event(
        self,
        *,
        action: str,
        outcome: str,
        method: str,
        path: str,
        status_code: int,
        request_id: str,
        actor_user_id: str | None = None,
        actor_username: str | None = None,
        actor_role: str | None = None,
        client_ip: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "audit_id": uuid.uuid4().hex,
            "action": action,
            "outcome": outcome,
            "method": method,
            "path": path,
            "status_code": status_code,
            "request_id": request_id,
            "actor_user_id": actor_user_id,
            "actor_username": actor_username,
            "actor_role": actor_role,
            "client_ip": client_ip,
            "target_type": target_type,
            "target_id": target_id,
            "details": details or {},
            "created_at": utc_now(),
        }
        self.audit_logs.insert_one(record)
        return serialize(record)

    def list_audit_logs(
        self,
        *,
        action: str | None = None,
        outcome: str | None = None,
        actor_user_id: str | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if action:
            query["action"] = action
        if outcome:
            query["outcome"] = outcome
        if actor_user_id:
            query["actor_user_id"] = actor_user_id
        if start_at or end_at:
            created_at: dict[str, datetime] = {}
            if start_at:
                created_at["$gte"] = start_at
            if end_at:
                created_at["$lte"] = end_at
            query["created_at"] = created_at
        records = self.audit_logs.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)
        return {
            "items": [serialize(item) for item in records],
            "total": self.audit_logs.count_documents(query),
            "skip": skip,
            "limit": limit,
        }

    def consume_rate_limit(
        self,
        *,
        scope_key: str,
        action: str,
        limit: int,
        window_seconds: int,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically consume one fixed-window quota unit across all service instances."""
        current_time = now or utc_now()
        window_start = rate_limit_window_start(current_time, window_seconds)
        reset_at = window_start + timedelta(seconds=window_seconds)
        identity = {"scope_key": scope_key, "action": action, "window_start": window_start}
        available = {**identity, "$or": [{"count": {"$lt": limit}}, {"count": {"$exists": False}}]}
        update = {
            "$inc": {"count": 1},
            "$set": {"updated_at": current_time},
            "$setOnInsert": {"created_at": current_time, "expires_at": reset_at},
        }
        try:
            record = self.rate_limit_counters.find_one_and_update(
                available, update, upsert=True, return_document=ReturnDocument.AFTER
            )
        except DuplicateKeyError:
            # A concurrent first request may win the unique-key insert; retry only against that record.
            record = self.rate_limit_counters.find_one_and_update(
                available, update, upsert=False, return_document=ReturnDocument.AFTER
            )
        count = int(record.get("count", 0)) if record else limit
        return {
            "action": action,
            "allowed": record is not None,
            "limit": limit,
            "used": count,
            "remaining": max(0, limit - count),
            "reset_at": serialize(reset_at),
            "retry_after_seconds": max(1, int((reset_at - current_time).total_seconds())),
        }

    def get_rate_limit_usage(
        self,
        *,
        scope_key: str,
        action: str,
        limit: int,
        window_seconds: int,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current_time = now or utc_now()
        window_start = rate_limit_window_start(current_time, window_seconds)
        reset_at = window_start + timedelta(seconds=window_seconds)
        record = self.rate_limit_counters.find_one({
            "scope_key": scope_key, "action": action, "window_start": window_start,
        })
        used = int(record.get("count", 0)) if record else 0
        return {
            "action": action,
            "limit": limit,
            "used": used,
            "remaining": max(0, limit - used),
            "reset_at": serialize(reset_at),
        }

    # Knowledge bases
    def create_knowledge_base(self, name: str, description: str = "") -> dict[str, Any]:
        now = utc_now()
        record = {
            "knowledge_base_id": uuid.uuid4().hex,
            "name": name.strip(),
            "description": description.strip(),
            "created_at": now,
            "updated_at": now,
        }
        self.knowledge_bases.insert_one(record)
        return serialize(record)

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        result = []
        for record in self.knowledge_bases.find({}).sort("created_at", DESCENDING):
            item = serialize(record)
            item["document_count"] = self.documents.count_documents(
                {"knowledge_base_id": item["knowledge_base_id"], "is_current": True, "status": "ready"}
            )
            result.append(item)
        return result

    def get_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any] | None:
        record = self.knowledge_bases.find_one({"knowledge_base_id": knowledge_base_id})
        return serialize(record) if record else None

    def update_knowledge_base(self, knowledge_base_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        changes = {key: value.strip() if isinstance(value, str) else value for key, value in changes.items()}
        changes["updated_at"] = utc_now()
        record = self.knowledge_bases.find_one_and_update(
            {"knowledge_base_id": knowledge_base_id}, {"$set": changes}, return_document=ReturnDocument.AFTER
        )
        return serialize(record) if record else None

    def delete_knowledge_base(self, knowledge_base_id: str) -> bool:
        if self.documents.count_documents({"knowledge_base_id": knowledge_base_id, "is_current": True}):
            raise ValueError("知识库中仍有文档，请先删除文档")
        return self.knowledge_bases.delete_one({"knowledge_base_id": knowledge_base_id}).deleted_count == 1

    # Documents and versions
    def begin_document_version(
        self,
        knowledge_base_id: str,
        filename: str,
        checksum_sha256: str,
        source_path: str,
    ) -> tuple[dict[str, Any], bool]:
        checksum = checksum_sha256
        current = self.documents.find_one(
            {"knowledge_base_id": knowledge_base_id, "filename": filename, "is_current": True}
        )
        if current and current.get("checksum_sha256") == checksum and current.get("status") == "ready":
            return serialize(current), True

        logical_document_id = current.get("logical_document_id") if current else uuid.uuid4().hex
        version = int(current.get("version", 0)) + 1 if current else 1
        now = utc_now()
        record = {
            "document_id": uuid.uuid4().hex,
            "logical_document_id": logical_document_id,
            "knowledge_base_id": knowledge_base_id,
            "filename": filename,
            "checksum_sha256": checksum,
            "version": version,
            "status": "processing",
            "is_current": False,
            "source_path": source_path,
            "previous_document_id": current.get("document_id") if current else None,
            "chunk_count": 0,
            "item_name": "",
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        self.documents.insert_one(record)
        return serialize(record), False

    def complete_document_version(self, document_id: str, *, chunk_count: int, item_name: str) -> dict[str, Any]:
        record = self.documents.find_one({"document_id": document_id})
        if not record:
            raise KeyError(document_id)
        now = utc_now()
        # The new version is only made current after the vector write succeeded.
        self.documents.update_many(
            {"logical_document_id": record["logical_document_id"], "document_id": {"$ne": document_id}},
            {"$set": {"is_current": False, "updated_at": now}},
        )
        updated = self.documents.find_one_and_update(
            {"document_id": document_id},
            {"$set": {"status": "ready", "is_current": True, "chunk_count": chunk_count,
                      "item_name": item_name, "error": None, "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        return serialize(updated)

    def fail_document_version(self, document_id: str, error: str) -> None:
        self.documents.update_one(
            {"document_id": document_id},
            {"$set": {"status": "failed", "is_current": False, "error": error, "updated_at": utc_now()}},
        )

    def list_documents(self, knowledge_base_id: str, skip: int = 0, limit: int = 50) -> dict[str, Any]:
        query = {
            "knowledge_base_id": knowledge_base_id,
            "$or": [{"is_current": True}, {"status": {"$in": ["processing", "failed"]}}],
        }
        records = self.documents.find(query).sort("updated_at", DESCENDING).skip(skip).limit(limit)
        return {"items": [serialize(item) for item in records], "total": self.documents.count_documents(query)}

    def attach_document_task(self, document_id: str, task_id: str) -> None:
        self.documents.update_one(
            {"document_id": document_id}, {"$set": {"task_id": task_id, "updated_at": utc_now()}}
        )

    def attach_document_source(self, document_id: str, bucket_name: str, object_name: str, size: int) -> None:
        self.documents.update_one(
            {"document_id": document_id, "status": "processing"},
            {"$set": {
                "source_bucket": bucket_name,
                "source_object_name": object_name,
                "source_size": size,
                "source_archived_at": utc_now(),
                "updated_at": utc_now(),
            }},
        )

    def discard_processing_document(self, document_id: str) -> bool:
        """Compensate a failed archive/task setup without touching an existing current version."""
        result = self.documents.delete_one({"document_id": document_id, "status": "processing", "is_current": False})
        return result.deleted_count == 1

    def restart_document_version(self, document_id: str) -> bool:
        result = self.documents.update_one(
            {"document_id": document_id, "status": "failed"},
            {"$set": {"status": "processing", "error": None, "updated_at": utc_now()}},
        )
        return result.modified_count == 1

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        record = self.documents.find_one({"document_id": document_id})
        return serialize(record) if record else None

    def get_document_versions(self, logical_document_id: str) -> list[dict[str, Any]]:
        records = self.documents.find({"logical_document_id": logical_document_id}).sort("version", DESCENDING)
        return [serialize(item) for item in records]

    def list_document_versions_for_knowledge_base(self, knowledge_base_id: str) -> list[dict[str, Any]]:
        records = self.documents.find({"knowledge_base_id": knowledge_base_id}).sort([
            ("logical_document_id", ASCENDING), ("version", ASCENDING),
        ])
        return [serialize(item) for item in records]

    def mark_document_deleted(self, document_id: str) -> dict[str, Any] | None:
        record = self.documents.find_one_and_update(
            {"document_id": document_id, "is_current": True},
            {"$set": {"status": "deleted", "is_current": False, "updated_at": utc_now()}},
            return_document=ReturnDocument.BEFORE,
        )
        return serialize(record) if record else None

    # Chat sessions
    def touch_session(
        self,
        session_id: str,
        first_question: str,
        knowledge_base_ids: list[str] | None = None,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a session on its first question and refresh its activity timestamp."""
        existing = self.chat_sessions.find_one({"session_id": session_id})
        if owner_user_id and existing and existing.get("owner_user_id") not in {None, owner_user_id}:
            raise PermissionError("会话不属于当前用户")
        now = utc_now()
        title = first_question.strip().replace("\n", " ")[:40] or "新会话"
        record = self.chat_sessions.find_one_and_update(
            {"session_id": session_id},
            {
                "$set": {"updated_at": now, "knowledge_base_ids": knowledge_base_ids or []},
                "$setOnInsert": {"title": title, "created_at": now, "owner_user_id": owner_user_id},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return serialize(record)

    def list_sessions(self, limit: int = 30, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        query = {"owner_user_id": owner_user_id} if owner_user_id else {}
        records = self.chat_sessions.find(query).sort("updated_at", DESCENDING).limit(limit)
        return [serialize(item) for item in records]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        record = self.chat_sessions.find_one({"session_id": session_id})
        return serialize(record) if record else None

    def rename_session(self, session_id: str, title: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        query: dict[str, Any] = {"session_id": session_id}
        if owner_user_id:
            query["owner_user_id"] = owner_user_id
        record = self.chat_sessions.find_one_and_update(
            query,
            {"$set": {"title": title.strip(), "updated_at": utc_now()}},
            return_document=ReturnDocument.AFTER,
        )
        return serialize(record) if record else None

    def delete_session(self, session_id: str, owner_user_id: str | None = None) -> bool:
        if self.tasks.count_documents({
            "session_id": session_id,
            "type": "query",
            "status": {"$in": ["pending", "retrying", "processing"]},
        }):
            raise ValueError("会话仍有运行中的任务，请先停止任务")
        query: dict[str, Any] = {"session_id": session_id}
        if owner_user_id:
            query["owner_user_id"] = owner_user_id
        deleted = self.chat_sessions.delete_one(query).deleted_count == 1
        # Session messages share the same Mongo database; deleting the session is a cascade operation.
        if deleted:
            self.db["chat_message"].delete_many({"session_id": session_id})
        return deleted

    # Persistent tasks
    def create_task(
        self,
        task_id: str,
        task_type: str,
        session_id: str,
        payload: dict[str, Any],
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        task = {
            "task_id": task_id, "type": task_type, "session_id": session_id, "payload": payload,
            "owner_user_id": owner_user_id,
            "status": "pending", "current_node": None, "completed_nodes": [], "attempt": 0,
            "answer": "", "sources": [], "image_urls": [], "result": None, "error": None,
            "created_at": now, "updated_at": now, "started_at": None, "finished_at": None,
        }
        self.tasks.insert_one(task)
        return serialize(task)

    def update_task(self, task_id: str, **changes: Any) -> None:
        changes["updated_at"] = utc_now()
        self.tasks.update_one({"task_id": task_id}, {"$set": changes})

    def start_task(self, task_id: str, worker_id: str, lease_seconds: int) -> bool:
        now = utc_now()
        task = self.tasks.find_one_and_update(
            {"task_id": task_id, "status": {"$in": ["pending", "retrying"]}},
            {"$set": {"status": "processing", "started_at": now, "updated_at": now,
                      "error": None, "cancel_requested": False, "worker_id": worker_id,
                      "lease_expires_at": now + timedelta(seconds=lease_seconds)},
             "$inc": {"attempt": 1}},
            return_document=ReturnDocument.AFTER,
        )
        return task is not None

    def renew_task_lease(self, task_id: str, worker_id: str, lease_seconds: int) -> bool:
        now = utc_now()
        result = self.tasks.update_one(
            {"task_id": task_id, "status": "processing", "worker_id": worker_id},
            {"$set": {"lease_expires_at": now + timedelta(seconds=lease_seconds), "updated_at": now}},
        )
        return result.modified_count == 1

    def complete_task_node(self, task_id: str, node_name: str) -> None:
        self.tasks.update_one(
            {"task_id": task_id},
            {"$addToSet": {"completed_nodes": node_name}, "$set": {"current_node": node_name, "updated_at": utc_now()}},
        )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = self.tasks.find_one({"task_id": task_id})
        return serialize(task) if task else None

    def recoverable_tasks(self) -> Iterable[dict[str, Any]]:
        now = utc_now()
        self.tasks.update_many(
            {
                "status": "processing",
                "$or": [
                    {"lease_expires_at": {"$lt": now}},
                    {"lease_expires_at": {"$exists": False}},
                ],
            },
            {"$set": {"status": "pending", "current_node": None, "updated_at": now,
                      "error": "工作租约已过期，任务自动恢复"},
             "$unset": {"worker_id": "", "lease_expires_at": ""}},
        )
        return [
            serialize(task)
            for task in self.tasks.find({"status": {"$in": ["pending", "retrying"]}}).sort("created_at", ASCENDING)
        ]

    def retry_task(self, task_id: str) -> dict[str, Any] | None:
        task = self.tasks.find_one_and_update(
            {"task_id": task_id, "status": {"$in": ["failed", "cancelled"]}},
            {"$set": {"status": "retrying", "current_node": None, "completed_nodes": [], "error": None,
                      "finished_at": None, "cancel_requested": False, "updated_at": utc_now()}},
            return_document=ReturnDocument.AFTER,
        )
        return serialize(task) if task else None

    def request_task_cancel(self, task_id: str) -> dict[str, Any] | None:
        """Cancel queued tasks immediately or request cooperative cancellation for running tasks."""
        task = self.tasks.find_one({"task_id": task_id})
        if not task or task.get("status") not in {"pending", "retrying", "processing"}:
            return None
        if task.get("type") == "import" and task.get("status") == "processing":
            raise ValueError("文档正在写入，不能在处理中强制取消")
        if task.get("status") in {"pending", "retrying"}:
            changes = {"status": "cancelled", "finished_at": utc_now(), "current_node": None}
        else:
            changes = {"cancel_requested": True}
        changes["updated_at"] = utc_now()
        updated = self.tasks.find_one_and_update(
            {"task_id": task_id}, {"$set": changes}, return_document=ReturnDocument.AFTER
        )
        return serialize(updated) if updated else None

    def is_task_cancel_requested(self, task_id: str) -> bool:
        task = self.tasks.find_one({"task_id": task_id}, {"cancel_requested": 1, "status": 1})
        return bool(task and (task.get("cancel_requested") or task.get("status") == "cancelled"))

    def list_tasks(self, status: str | None = None, task_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        if task_type:
            query["type"] = task_type
        records = self.tasks.find(query).sort("created_at", DESCENDING).limit(limit)
        return [serialize(item) for item in records]

    def operations_summary(self) -> dict[str, Any]:
        def count_by(collection, field: str) -> dict[str, int]:
            return {
                str(item["_id"]): item["count"]
                for item in collection.aggregate([{"$group": {"_id": f"${field}", "count": {"$sum": 1}}}])
                if item.get("_id") is not None
            }

        return {
            "knowledge_base_count": self.knowledge_bases.count_documents({}),
            "session_count": self.chat_sessions.count_documents({}),
            "task_status_counts": count_by(self.tasks, "status"),
            "document_status_counts": count_by(self.documents, "status"),
            "evaluation_count": self.evaluation_runs.count_documents({}),
            "generated_at": serialize(utc_now()),
        }

    def cleanup_candidates(self, older_than: datetime) -> list[dict[str, Any]]:
        records = self.tasks.find({
            "status": {"$in": ["completed", "cancelled"]},
            "finished_at": {"$lt": older_than},
            "files_cleaned_at": {"$exists": False},
        })
        return [serialize(item) for item in records]

    def mark_task_files_cleaned(self, task_id: str) -> None:
        self.tasks.update_one({"task_id": task_id}, {"$set": {"files_cleaned_at": utc_now()}})

    # Evaluation reports
    def save_evaluation_run(self, run: dict[str, Any]) -> dict[str, Any]:
        record = {**run, "evaluation_id": run.get("evaluation_id") or uuid.uuid4().hex,
                  "is_baseline": bool(run.get("is_baseline", False)),
                  "created_at": run.get("created_at") or utc_now()}
        self.evaluation_runs.insert_one(record)
        return serialize(record)

    def list_evaluation_runs(self, limit: int = 20, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        query = {"owner_user_id": owner_user_id} if owner_user_id else {}
        records = self.evaluation_runs.find(query).sort("created_at", DESCENDING).limit(limit)
        return [serialize(item) for item in records]

    def get_evaluation_run(self, evaluation_id: str, owner_user_id: str | None = None) -> dict[str, Any] | None:
        query: dict[str, Any] = {"evaluation_id": evaluation_id}
        if owner_user_id:
            query["owner_user_id"] = owner_user_id
        record = self.evaluation_runs.find_one(query)
        return serialize(record) if record else None

    def set_evaluation_baseline(
        self, evaluation_id: str, owner_user_id: str | None = None
    ) -> dict[str, Any] | None:
        query: dict[str, Any] = {"evaluation_id": evaluation_id}
        if owner_user_id:
            query["owner_user_id"] = owner_user_id
        record = self.evaluation_runs.find_one(query)
        if not record:
            return None
        scope: dict[str, Any] = {"owner_user_id": record.get("owner_user_id")}
        if record.get("dataset_checksum"):
            scope["dataset_checksum"] = record["dataset_checksum"]
        else:
            scope["dataset_name"] = record.get("dataset_name")
        now = utc_now()
        self.evaluation_runs.update_many(scope, {"$set": {"is_baseline": False, "updated_at": now}})
        updated = self.evaluation_runs.find_one_and_update(
            {"evaluation_id": evaluation_id},
            {"$set": {"is_baseline": True, "baseline_at": now, "updated_at": now}},
            return_document=ReturnDocument.AFTER,
        )
        return serialize(updated) if updated else None

    def find_evaluation_baseline(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        query: dict[str, Any] = {
            "is_baseline": True,
            "owner_user_id": candidate.get("owner_user_id"),
        }
        if candidate.get("dataset_checksum"):
            query["dataset_checksum"] = candidate["dataset_checksum"]
        else:
            query["dataset_name"] = candidate.get("dataset_name")
        record = self.evaluation_runs.find_one(query, sort=[("baseline_at", DESCENDING)])
        return serialize(record) if record else None


_service: MetadataService | None = None
_service_lock = threading.Lock()


def get_metadata_service() -> MetadataService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = MetadataService()
    return _service
