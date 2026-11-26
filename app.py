"""Knowledge Base console API with durable metadata, tasks and evaluations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from evaluation.rag_evaluator import (
    compare_evaluation_summaries,
    evaluate_answer,
    load_dataset,
    summarize_results,
)
from services.auth_service import (
    ROLE_ADMIN,
    ROLE_EDITOR,
    ROLE_RANK,
    ROLE_VIEWER,
    AuthenticationError,
    get_auth_service,
    load_auth_config,
)
from services.metadata_service import get_metadata_service, utc_now
from tool.logger import logger
from utils.minio_utils import close_minio_client, initialize_minio_client, peek_minio_client
from utils.mongo_history_utils import (
    close_history_mongo_tool,
    initialize_history_mongo_tool,
    peek_history_mongo_tool,
)

PROJECT_ROOT = Path(__file__).resolve().parent
# Local development reads the project .env, while deployment-provided variables keep precedence.
load_dotenv(PROJECT_ROOT / ".env", override=False)
FRONT_DIR = PROJECT_ROOT / "front"
UPLOAD_DIR = PROJECT_ROOT / "temp_data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "temp_data" / "output"
EVALUATION_DIR = PROJECT_ROOT / "temp_data" / "evaluations"
EXPORT_DIR = PROJECT_ROOT / "temp_data" / "exports"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_EVALUATION_BYTES = 5 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".md"}
WORKER_ID = uuid.uuid4().hex
TASK_LEASE_SECONDS = max(60, int(os.getenv("TASK_LEASE_SECONDS", "180")))
TASK_HEARTBEAT_SECONDS = max(10, min(int(os.getenv("TASK_HEARTBEAT_SECONDS", "30")), TASK_LEASE_SECONDS // 2))
TEMP_FILE_RETENTION_DAYS = max(1, int(os.getenv("TEMP_FILE_RETENTION_DAYS", "7")))
SLOW_REQUEST_THRESHOLD_MS = max(100, int(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "2000")))
LOGIN_MAX_ATTEMPTS = max(3, int(os.getenv("LOGIN_MAX_ATTEMPTS", "5")))
LOGIN_WINDOW_SECONDS = max(60, int(os.getenv("LOGIN_WINDOW_SECONDS", "300")))
QUERY_RATE_LIMIT_PER_MINUTE = max(0, int(os.getenv("QUERY_RATE_LIMIT_PER_MINUTE", "30")))
IMPORT_RATE_LIMIT_PER_HOUR = max(0, int(os.getenv("IMPORT_RATE_LIMIT_PER_HOUR", "20")))
EVALUATION_RATE_LIMIT_PER_HOUR = max(0, int(os.getenv("EVALUATION_RATE_LIMIT_PER_HOUR", "10")))
EXPORT_RATE_LIMIT_PER_HOUR = max(0, int(os.getenv("EXPORT_RATE_LIMIT_PER_HOUR", "10")))
RATE_LIMIT_SPECS = {
    "query": (QUERY_RATE_LIMIT_PER_MINUTE, 60),
    "import": (IMPORT_RATE_LIMIT_PER_HOUR, 60 * 60),
    "evaluation": (EVALUATION_RATE_LIMIT_PER_HOUR, 60 * 60),
    "export": (EXPORT_RATE_LIMIT_PER_HOUR, 60 * 60),
}

QUERY_PIPELINE = [
    ("node_item_name_confirm", "产品识别"), ("node_multi_search", "多路检索"),
    ("node_search_embedding", "向量检索"), ("node_search_embedding_hyde", "HyDE 检索"),
    ("node_web_search_mcp", "网络搜索"), ("node_join", "检索合并"),
    ("node_rrf", "RRF 融合"), ("node_rerank", "GPT 精排"),
    ("node_answer_output", "答案生成"),
]
IMPORT_PIPELINE = [
    ("node_entry", "文件检查"), ("node_pdf_to_md", "PDF 解析"),
    ("node_md_img", "图片处理"), ("node_document_split", "文档切分"),
    ("node_item_name_recognition", "主体识别"), ("node_bge_embedding", "向量生成"),
    ("node_import_milvus", "写入 Milvus"),
]
EXPORT_PIPELINE = [("collect_metadata", "整理元数据"), ("archive_documents", "打包原文"), ("write_manifest", "生成清单")]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    knowledge_base_ids: list[str] = Field(default_factory=list, max_length=20)


class TaskCreatedResponse(BaseModel):
    task_id: str
    session_id: str
    duplicate: bool = False


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)


class KnowledgeBaseUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)


class SessionRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=10, max_length=256)
    role: str = Field(default=ROLE_VIEWER)


class UserUpdateRequest(BaseModel):
    password: str | None = Field(default=None, min_length=10, max_length=256)
    role: str | None = None
    is_active: bool | None = None


class TaskCancelledError(RuntimeError):
    """Raised at a safe workflow boundary after a cancellation request."""


executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb-task")
_submit_lock = threading.Lock()
_submitted_tasks: set[str] = set()
_login_attempts: dict[str, list[float]] = {}
_login_attempts_lock = threading.Lock()


def _check_login_rate_limit(client_key: str) -> int:
    """Return retry-after seconds, or zero when another attempt is allowed."""
    now = time.monotonic()
    with _login_attempts_lock:
        recent = [value for value in _login_attempts.get(client_key, []) if now - value < LOGIN_WINDOW_SECONDS]
        _login_attempts[client_key] = recent
        if len(recent) < LOGIN_MAX_ATTEMPTS:
            return 0
        return max(1, round(LOGIN_WINDOW_SECONDS - (now - recent[0])))


def _record_login_failure(client_key: str) -> None:
    with _login_attempts_lock:
        _login_attempts.setdefault(client_key, []).append(time.monotonic())


def _clear_login_failures(client_key: str) -> None:
    with _login_attempts_lock:
        _login_attempts.pop(client_key, None)


def _serialize_mongo_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_mongo_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_mongo_value(item) for item in value]
    return value


def _extract_task_output(state: dict[str, Any], session_id: str) -> tuple[str, list, list]:
    from utils.mongo_history_utils import get_recent_messages

    answer = state.get("answer") or ""
    sources = state.get("reranked_docs") or []
    image_urls: list[str] = []
    for message in reversed(get_recent_messages(session_id, limit=20)):
        if message.get("role") == "assistant":
            image_urls = message.get("image_urls") or []
            answer = answer or message.get("text") or ""
            break
    return answer, sources, image_urls


def _run_workflow(question: str, session_id: str, knowledge_base_ids: list[str], task_id: str | None = None):
    from processor.query_processor.main_graph import KBQueryWorkflow

    state: dict[str, Any] = {
        "original_query": question, "session_id": session_id, "is_stream": False,
        "knowledge_base_ids": knowledge_base_ids,
    }
    for event in KBQueryWorkflow().compile().stream(state, stream_mode="updates"):
        if task_id and get_metadata_service().is_task_cancel_requested(task_id):
            raise TaskCancelledError(task_id)
        for node_name, update in event.items():
            if task_id:
                get_metadata_service().complete_task_node(task_id, node_name)
            if isinstance(update, dict):
                state.update(update)
    return state, _extract_task_output(state, session_id)


def _run_query_task(task_id: str, payload: dict[str, Any]) -> None:
    service = get_metadata_service()
    if not service.start_task(task_id, WORKER_ID, TASK_LEASE_SECONDS):
        return
    try:
        _, (answer, sources, image_urls) = _run_workflow(
            payload["question"], payload["session_id"], payload.get("knowledge_base_ids") or [], task_id
        )
        if not answer:
            raise RuntimeError("查询流程完成但没有生成答案")
        service.update_task(task_id, status="completed", current_node=None, answer=answer,
                            sources=sources, image_urls=image_urls, finished_at=utc_now())
    except TaskCancelledError:
        service.update_task(task_id, status="cancelled", current_node=None,
                            error="用户已取消任务", finished_at=utc_now())
    except Exception:
        logger.exception("查询任务失败，task_id=%s", task_id)
        service.update_task(task_id, status="failed", current_node=None,
                            error="查询处理失败，请检查服务日志后重试。", finished_at=utc_now())


def _delete_document_vectors(document_id: str) -> None:
    from config.milvus_config import milvus_config
    from processor.import_processor.node.e_node_item_reconition import escape_milvus_string
    from utils.milvus_utils import get_milvus_client

    if document_id:
        get_milvus_client().delete(
            collection_name=milvus_config.chunks_collection,
            filter=f'document_id == "{escape_milvus_string(document_id)}"',
        )


def _delete_document_assets(document_id: str) -> None:
    from config.minio_config import minio_config
    from utils.minio_utils import get_minio_client

    prefix = f"{minio_config.img_dir}/{document_id}".strip("/")
    client = get_minio_client()
    objects = list(client.list_objects(minio_config.bucket_name, prefix=prefix, recursive=True))
    for item in objects:
        client.remove_object(minio_config.bucket_name, item.object_name)


def _document_source_object_name(document: dict[str, Any]) -> str:
    suffix = Path(document.get("filename") or "").suffix.lower()
    return (
        f"sources/{document['knowledge_base_id']}/{document['logical_document_id']}"
        f"/v{document['version']}/{document['document_id']}{suffix}"
    )


def _document_media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".md":
        return "text/markdown"
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _archive_document_source(document: dict[str, Any], source_path: Path) -> tuple[str, str, int]:
    from config.minio_config import minio_config
    from utils.minio_utils import get_minio_client

    client = get_minio_client()
    if client is None:
        raise RuntimeError("MinIO client is unavailable")
    object_name = _document_source_object_name(document)
    content_type = _document_media_type(document.get("filename") or "")
    client.fput_object(
        minio_config.document_bucket_name,
        object_name,
        str(source_path),
        content_type=content_type,
    )
    return minio_config.document_bucket_name, object_name, source_path.stat().st_size


def _remove_document_source(document: dict[str, Any]) -> None:
    from utils.minio_utils import get_minio_client

    bucket_name = document.get("source_bucket")
    object_name = document.get("source_object_name")
    if not bucket_name or not object_name:
        return
    client = get_minio_client()
    if client is None:
        raise RuntimeError("MinIO client is unavailable")
    client.remove_object(bucket_name, object_name)


def _restore_document_source(document: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Restore a retry source from private object storage when temporary files were cleaned."""
    from utils.minio_utils import get_minio_client

    payload = dict(task.get("payload") or {})
    current_path = Path(payload.get("file_path") or "")
    if current_path.is_file():
        return task
    bucket_name = document.get("source_bucket")
    object_name = document.get("source_object_name")
    if not bucket_name or not object_name:
        raise FileNotFoundError("原始文件未归档，无法恢复重试")
    task_id = task["task_id"]
    upload_dir = UPLOAD_DIR / task_id
    output_dir = OUTPUT_DIR / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / Path(document["filename"]).name
    client = get_minio_client()
    if client is None:
        raise RuntimeError("MinIO client is unavailable")
    client.fget_object(bucket_name, object_name, str(target_path))
    digest = hashlib.sha256()
    with target_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != document.get("checksum_sha256"):
        target_path.unlink(missing_ok=True)
        raise ValueError("归档原文件校验失败")
    payload.update({"file_path": str(target_path), "output_dir": str(output_dir)})
    get_metadata_service().update_task(task_id, payload=payload)
    return {**task, "payload": payload}


def _present_document(document: dict[str, Any]) -> dict[str, Any]:
    presented = dict(document)
    presented["source_archived"] = bool(document.get("source_bucket") and document.get("source_object_name"))
    for private_field in ("source_path", "source_bucket", "source_object_name"):
        presented.pop(private_field, None)
    return presented


def _safe_zip_segment(value: Any, fallback: str) -> str:
    cleaned = "".join(character if str(character).isalnum() or character in "._-" else "_" for character in str(value))
    cleaned = cleaned.strip("._")[:120]
    return cleaned or fallback


def _document_export_path(document: dict[str, Any]) -> str:
    logical_id = _safe_zip_segment(document.get("logical_document_id"), "document")
    document_id = _safe_zip_segment(document.get("document_id"), "version")
    version = max(1, int(document.get("version") or 1))
    filename = _safe_zip_segment(Path(document.get("filename") or "document").name, "document")
    return f"documents/{logical_id}/v{version}/{document_id}-{filename}"


def _run_export_task(task_id: str, payload: dict[str, Any]) -> None:
    from minio.error import S3Error
    from utils.minio_utils import get_minio_client

    service = get_metadata_service()
    if not service.start_task(task_id, WORKER_ID, TASK_LEASE_SECONDS):
        return
    export_path = EXPORT_DIR / f"{task_id}.zip"
    try:
        knowledge_base = service.get_knowledge_base(payload["knowledge_base_id"])
        if not knowledge_base:
            raise RuntimeError("知识库不存在")
        documents = service.list_document_versions_for_knowledge_base(payload["knowledge_base_id"])
        service.complete_task_node(task_id, "collect_metadata")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        minio_client = get_minio_client()
        if minio_client is None:
            raise RuntimeError("MinIO client is unavailable")
        manifest_documents: list[dict[str, Any]] = []
        missing_sources: list[dict[str, str]] = []
        archived_file_count = 0
        with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for document in documents:
                if service.is_task_cancel_requested(task_id):
                    raise TaskCancelledError(task_id)
                manifest_document = _present_document(document)
                manifest_document.pop("_id", None)
                bucket_name = document.get("source_bucket")
                object_name = document.get("source_object_name")
                if not bucket_name or not object_name:
                    missing_sources.append({"document_id": document["document_id"], "reason": "not_archived"})
                    manifest_documents.append(manifest_document)
                    continue
                source = None
                try:
                    source = minio_client.get_object(bucket_name, object_name)
                    archive_path = _document_export_path(document)
                    with archive.open(archive_path, "w", force_zip64=True) as destination:
                        chunk_index = 0
                        while data := source.read(1024 * 1024):
                            destination.write(data)
                            chunk_index += 1
                            if chunk_index % 8 == 0 and service.is_task_cancel_requested(task_id):
                                raise TaskCancelledError(task_id)
                    manifest_document["archive_path"] = archive_path
                    archived_file_count += 1
                except S3Error as exc:
                    if exc.code not in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                        raise
                    missing_sources.append({"document_id": document["document_id"], "reason": "object_missing"})
                finally:
                    if source is not None:
                        source.close()
                        source.release_conn()
                manifest_documents.append(manifest_document)
            service.complete_task_node(task_id, "archive_documents")
            manifest = {
                "format_version": 1,
                "exported_at": utc_now().isoformat(),
                "knowledge_base": {key: value for key, value in knowledge_base.items() if key != "_id"},
                "documents": manifest_documents,
                "missing_sources": missing_sources,
            }
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        service.complete_task_node(task_id, "write_manifest")
        service.update_task(
            task_id,
            status="completed",
            current_node=None,
            result={
                "download_url": f"/api/exports/{task_id}/download",
                "filename": f"knowledge-base-{payload['knowledge_base_id']}.zip",
                "file_size": export_path.stat().st_size,
                "document_version_count": len(documents),
                "archived_file_count": archived_file_count,
                "missing_source_count": len(missing_sources),
            },
            finished_at=utc_now(),
        )
    except TaskCancelledError:
        export_path.unlink(missing_ok=True)
        service.update_task(task_id, status="cancelled", current_node=None,
                            error="用户已取消任务", finished_at=utc_now())
    except Exception:
        export_path.unlink(missing_ok=True)
        logger.exception("知识库导出失败，task_id=%s knowledge_base_id=%s", task_id, payload.get("knowledge_base_id"))
        service.update_task(task_id, status="failed", current_node=None,
                            error="知识库导出失败，请检查存储服务和应用日志。", finished_at=utc_now())


def _run_import_task(task_id: str, payload: dict[str, Any]) -> None:
    service = get_metadata_service()
    if not service.start_task(task_id, WORKER_ID, TASK_LEASE_SECONDS):
        return
    document_id = payload["document_id"]
    try:
        from processor.import_processor.main_graph import KBImportWorkflow
        from processor.import_processor.state import create_default_state

        state = create_default_state(
            task_id=task_id, import_file_path=payload["file_path"], file_dir=payload["output_dir"],
            knowledge_base_id=payload["knowledge_base_id"], document_id=document_id,
            previous_document_id=payload.get("previous_document_id"), version=payload["version"],
        )
        final_state: dict[str, Any] = dict(state)
        for event in KBImportWorkflow().graph.stream(state, stream_mode="updates"):
            for node_name, update in event.items():
                service.complete_task_node(task_id, node_name)
                if isinstance(update, dict):
                    final_state.update(update)
        chunk_count = len(final_state.get("chunks") or [])
        item_name = final_state.get("item_name") or ""
        chunks_data = final_state.get("chunks") or []
        if chunks_data:
            service.insert_chunks(
                chunks_data,
                knowledge_base_id=payload["knowledge_base_id"],
                document_id=document_id,
                version=payload["version"],
            )
        service.complete_document_version(document_id, chunk_count=chunk_count, item_name=item_name)

        if payload.get("previous_document_id"):
            try:
                _delete_document_assets(payload["previous_document_id"])
            except Exception:
                logger.warning("旧文档图片清理失败，document_id=%s", payload["previous_document_id"], exc_info=True)
        service.update_task(task_id, status="completed", current_node=None,
                            result={"document_id": document_id, "file_title": final_state.get("file_title"),
                                    "item_name": item_name, "chunk_count": chunk_count}, finished_at=utc_now())
    except Exception:
        logger.exception("导入任务失败，task_id=%s document_id=%s", task_id, document_id)
        service.fail_document_version(document_id, "资料导入失败，请检查服务日志后重试。")
        service.update_task(task_id, status="failed", current_node=None,
                            error="资料导入失败，请检查服务日志后重试。", finished_at=utc_now())


def _run_evaluation_task(task_id: str, payload: dict[str, Any]) -> None:
    service = get_metadata_service()
    if not service.start_task(task_id, WORKER_ID, TASK_LEASE_SECONDS):
        return
    results = []
    try:
        cases = load_dataset(payload["dataset_path"])
        for index, case in enumerate(cases, start=1):
            if service.is_task_cancel_requested(task_id):
                raise TaskCancelledError(task_id)
            started = time.monotonic()
            _, (answer, sources, _) = _run_workflow(
                case.question, f"eval-{task_id}-{index}",
                case.knowledge_base_ids or payload.get("knowledge_base_ids", []),
            )
            result = evaluate_answer(case, answer, sources)
            result.update({"answer": answer, "latency_ms": round((time.monotonic() - started) * 1000)})
            results.append(result)
            service.update_task(task_id, current_node=f"case_{index}", result={"completed": index, "total": len(cases)})
        evaluation_run = {
            "task_id": task_id, "dataset_name": payload["dataset_name"],
            "dataset_checksum": payload.get("dataset_checksum"),
            "knowledge_base_ids": payload.get("knowledge_base_ids") or [],
            "owner_user_id": payload.get("owner_user_id"),
            "summary": summarize_results(results), "results": results,
        }
        baseline = service.find_evaluation_baseline(evaluation_run)
        if baseline:
            evaluation_run["baseline_evaluation_id"] = baseline["evaluation_id"]
            evaluation_run["comparison"] = compare_evaluation_summaries(
                evaluation_run["summary"], baseline.get("summary") or {}
            )
        report = service.save_evaluation_run(evaluation_run)
        service.update_task(task_id, status="completed", current_node=None,
                            result={"evaluation_id": report["evaluation_id"], **report["summary"],
                                    "baseline_evaluation_id": report.get("baseline_evaluation_id"),
                                    "comparison": report.get("comparison")},
                            finished_at=utc_now())
    except TaskCancelledError:
        service.update_task(task_id, status="cancelled", current_node=None,
                            error="用户已取消任务", finished_at=utc_now())
    except Exception:
        logger.exception("评测任务失败，task_id=%s", task_id)
        service.update_task(task_id, status="failed", current_node=None,
                            error="评测执行失败，请检查数据集和服务日志。", finished_at=utc_now())


def _submit_task(task: dict[str, Any]) -> None:
    task_id = task["task_id"]
    with _submit_lock:
        if task_id in _submitted_tasks:
            return
        _submitted_tasks.add(task_id)

    def run() -> None:
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(TASK_HEARTBEAT_SECONDS):
                try:
                    get_metadata_service().renew_task_lease(task_id, WORKER_ID, TASK_LEASE_SECONDS)
                except Exception:
                    logger.exception("任务租约续期失败，task_id=%s worker_id=%s", task_id, WORKER_ID)

        heartbeat_thread = threading.Thread(
            target=heartbeat, name=f"kb-heartbeat-{task_id[:8]}", daemon=True
        )
        heartbeat_thread.start()
        try:
            handlers = {
                "query": _run_query_task,
                "import": _run_import_task,
                "evaluation": _run_evaluation_task,
                "export": _run_export_task,
            }
            handlers[task["type"]](task_id, task["payload"])
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
            with _submit_lock:
                _submitted_tasks.discard(task_id)
    executor.submit(run)


def _safe_remove_task_directory(root: Path, task_id: str) -> bool:
    """Delete only a direct task directory below a known temporary root."""
    target = (root / task_id).resolve()
    resolved_root = root.resolve()
    if target.parent != resolved_root or not target.exists():
        return False
    shutil.rmtree(target)
    return True


def _cleanup_expired_task_files() -> int:
    service = get_metadata_service()
    cutoff = datetime.now(timezone.utc) - timedelta(days=TEMP_FILE_RETENTION_DAYS)
    cleaned_count = 0
    for task in service.cleanup_candidates(cutoff):
        task_id = task["task_id"]
        try:
            removed = _safe_remove_task_directory(UPLOAD_DIR, task_id)
            removed = _safe_remove_task_directory(OUTPUT_DIR, task_id) or removed
            evaluation_path = EVALUATION_DIR / f"{task_id}.jsonl"
            if evaluation_path.is_file():
                evaluation_path.unlink()
                removed = True
            export_path = EXPORT_DIR / f"{task_id}.zip"
            if export_path.is_file():
                export_path.unlink()
                removed = True
            service.mark_task_files_cleaned(task_id)
            cleaned_count += int(removed)
        except OSError:
            logger.exception("临时文件清理失败，task_id=%s", task_id)
    if cleaned_count:
        logger.info("临时任务文件清理完成，count=%d retention_days=%d", cleaned_count, TEMP_FILE_RETENTION_DAYS)
    return cleaned_count


async def _maintenance_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(_cleanup_expired_task_files)
        except Exception:
            logger.exception("后台维护任务执行失败")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=6 * 60 * 60)
        except TimeoutError:
            continue


async def _initialize_external_dependencies(application: FastAPI) -> dict[str, str]:
    """Initialize independent external clients without coupling their failures."""
    errors: dict[str, str] = {}
    initializers = (
        ("minio", initialize_minio_client),
        ("history_mongodb", initialize_history_mongo_tool),
    )
    async def initialize_one(name: str, initializer) -> None:
        try:
            await asyncio.to_thread(initializer)
        except Exception:
            errors[name] = "unavailable"
            logger.exception("外部依赖初始化失败，dependency=%s", name)

    await asyncio.gather(*(initialize_one(name, initializer) for name, initializer in initializers))
    application.state.dependency_errors = errors
    return errors


async def _shutdown_external_dependencies() -> None:
    """Drain business work before closing the shared external clients."""
    await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)
    for name, closer in (
        ("history_mongodb", close_history_mongo_tool),
        ("minio", close_minio_client),
    ):
        try:
            await asyncio.to_thread(closer)
        except Exception:
            logger.exception("外部依赖关闭失败，dependency=%s", name)


@asynccontextmanager
async def lifespan(application: FastAPI):
    maintenance_stop = asyncio.Event()
    maintenance_task: asyncio.Task | None = None
    await _initialize_external_dependencies(application)
    try:
        for task in get_metadata_service().recoverable_tasks():
            _submit_task(task)
        get_auth_service().bootstrap_admin()
    except Exception:
        # 保持存活检查可用，让就绪检查明确报告依赖异常。
        logger.exception("持久化或认证初始化失败，相关业务接口暂不可用")
    maintenance_task = asyncio.create_task(_maintenance_loop(maintenance_stop))
    try:
        yield
    finally:
        maintenance_stop.set()
        if maintenance_task:
            await maintenance_task
        await _shutdown_external_dependencies()


app = FastAPI(title="Knowledge Base Console", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")


def _is_public_path(path: str) -> bool:
    return (
        path == "/"
        or path.startswith("/static/")
        or path.startswith("/api/health")
        or path in {"/api/auth/config", "/api/auth/login", "/docs", "/openapi.json", "/redoc"}
    )


def _required_role(method: str, path: str) -> str:
    if path.startswith("/api/users") or path.startswith("/api/audit-logs"):
        return ROLE_ADMIN
    if method == "DELETE" and path.startswith("/api/knowledge-bases/"):
        return ROLE_ADMIN
    if path.startswith("/api/evaluations"):
        return ROLE_EDITOR
    if path.startswith("/api/exports"):
        return ROLE_EDITOR
    if path.startswith("/api/operations/") or path == "/api/tasks":
        return ROLE_EDITOR
    if method in {"POST", "PATCH", "DELETE"} and (
        path.startswith("/api/knowledge-bases")
        or path.startswith("/api/documents")
        or path.startswith("/api/evaluations")
        or path == "/api/import"
        or (path.startswith("/api/tasks/") and path.endswith("/retry"))
    ):
        return ROLE_EDITOR
    return ROLE_VIEWER


def _authorize_request(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    config = load_auth_config()
    if not config.enabled or _is_public_path(request.url.path):
        return None, None
    token = request.cookies.get("kb_access_token")
    authorization = request.headers.get("Authorization", "")
    if not token and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        return None, JSONResponse(status_code=401, content={"detail": "请先登录"})
    try:
        user = get_auth_service().verify_token(token)
    except AuthenticationError as exc:
        return None, JSONResponse(status_code=401, content={"detail": str(exc)})
    except Exception:
        logger.exception("认证服务不可用")
        return None, JSONResponse(status_code=503, content={"detail": "认证服务当前不可用"})
    required_role = _required_role(request.method, request.url.path)
    if ROLE_RANK.get(user.get("role"), 0) < ROLE_RANK[required_role]:
        return user, JSONResponse(status_code=403, content={"detail": "当前账号没有执行此操作的权限"})
    return user, None


def _owner_scope(request: Request) -> str | None:
    """Return the current user ID for ownership filters; admins and auth-disabled mode are unscoped."""
    if not load_auth_config().enabled:
        return None
    user = getattr(request.state, "user", None) or {}
    return None if user.get("role") == ROLE_ADMIN else user.get("user_id")


def _request_user_id(request: Request) -> str | None:
    user = getattr(request.state, "user", None) or {}
    return user.get("user_id")


def _rate_limit_scope(request: Request) -> str:
    user_id = _request_user_id(request)
    if user_id:
        return f"user:{user_id}"
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


def _enforce_rate_limit(request: Request, action: str) -> dict[str, Any] | None:
    limit, window_seconds = RATE_LIMIT_SPECS[action]
    if limit <= 0:
        return None
    try:
        result = get_metadata_service().consume_rate_limit(
            scope_key=_rate_limit_scope(request),
            action=action,
            limit=limit,
            window_seconds=window_seconds,
        )
    except Exception as exc:
        logger.exception(
            "资源配额检查失败，action=%s request_id=%s",
            action, getattr(request.state, "request_id", ""),
        )
        raise HTTPException(status_code=503, detail="资源配额服务当前不可用") from exc
    if not result["allowed"]:
        retry_after = str(result["retry_after_seconds"])
        logger.warning(
            "资源配额已用尽，action=%s scope=%s retry_after=%s request_id=%s",
            action, _rate_limit_scope(request), retry_after, getattr(request.state, "request_id", ""),
        )
        raise HTTPException(
            status_code=429,
            detail="当前操作请求过于频繁，请稍后重试",
            headers={
                "Retry-After": retry_after,
                "X-RateLimit-Limit": str(result["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": result["reset_at"],
            },
        )
    return result


def _ensure_task_access(task: dict[str, Any], request: Request) -> None:
    owner_user_id = _owner_scope(request)
    if owner_user_id and task.get("owner_user_id") != owner_user_id:
        raise HTTPException(status_code=404, detail="任务不存在")


def _ensure_session_access(session_id: str, request: Request) -> None:
    owner_user_id = _owner_scope(request)
    if not owner_user_id:
        return
    session = get_metadata_service().get_session(session_id)
    if not session or session.get("owner_user_id") != owner_user_id:
        raise HTTPException(status_code=404, detail="会话不存在")


def _audit_action_and_target(method: str, path: str) -> tuple[str | None, str | None, str | None]:
    """Map state-changing API routes to stable audit actions without inspecting sensitive bodies."""
    if method not in {"POST", "PATCH", "DELETE"}:
        return None, None, None
    segments = [segment for segment in path.split("/") if segment]
    if path == "/api/auth/login":
        return "auth.login", "account", None
    if path == "/api/auth/logout":
        return "auth.logout", "account", None
    if path.startswith("/api/users"):
        return ("user.create" if method == "POST" else "user.update", "user", segments[2] if len(segments) > 2 else None)
    if path == "/api/query":
        return "query.create", "query", None
    if path.startswith("/api/tasks/"):
        suffix = segments[-1] if segments else ""
        return f"task.{suffix}", "task", segments[2] if len(segments) > 2 else None
    if path.startswith("/api/documents/"):
        action = "document.delete" if method == "DELETE" else "document.retry"
        return action, "document", segments[2] if len(segments) > 2 else None
    if path.startswith("/api/knowledge-bases/") and path.endswith("/documents"):
        return "document.import", "knowledge_base", segments[2] if len(segments) > 2 else None
    if path.startswith("/api/knowledge-bases/") and path.endswith("/exports"):
        return "knowledge_base.export", "knowledge_base", segments[2] if len(segments) > 2 else None
    if path.startswith("/api/knowledge-bases"):
        verb = {"POST": "create", "PATCH": "update", "DELETE": "delete"}[method]
        return f"knowledge_base.{verb}", "knowledge_base", segments[2] if len(segments) > 2 else None
    if path == "/api/import":
        return "document.import", "knowledge_base", None
    if path.startswith("/api/evaluations"):
        action = "evaluation.baseline" if method == "PATCH" else "evaluation.create"
        return action, "evaluation", segments[2] if len(segments) > 2 else None
    if path.startswith("/api/sessions/"):
        verb = "rename" if method == "PATCH" else "delete"
        return f"session.{verb}", "session", segments[2] if len(segments) > 2 else None
    if path.startswith("/api/history/"):
        return "history.delete", "session", segments[2] if len(segments) > 2 else None
    return "api.mutation", "api", path


def _record_request_audit(request: Request, status_code: int, *, details: dict[str, Any] | None = None) -> None:
    action, target_type, target_id = _audit_action_and_target(request.method, request.url.path)
    if not action:
        return
    user = getattr(request.state, "user", None) or {}
    outcome = "success" if status_code < 400 else "denied" if status_code in {401, 403} else "failure"
    try:
        get_metadata_service().record_audit_event(
            action=action,
            outcome=outcome,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            request_id=getattr(request.state, "request_id", ""),
            actor_user_id=user.get("user_id"),
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            client_ip=request.client.host if request.client else None,
            target_type=target_type,
            target_id=target_id,
            details=details,
        )
    except Exception:
        # Audit is a non-critical side effect; the primary request result remains authoritative.
        logger.exception(
            "审计日志写入失败，request_id=%s action=%s status=%d",
            getattr(request.state, "request_id", ""), action, status_code,
        )


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id_header = (request.headers.get("X-Request-ID") or "").strip()
    request_id = request_id_header if 0 < len(request_id_header) <= 128 else uuid.uuid4().hex
    started = time.monotonic()
    status_code = 500
    should_auto_audit = request.url.path != "/api/auth/login"
    request.state.request_id = request_id
    try:
        user, auth_response = _authorize_request(request)
        request.state.user = user
        response = auth_response or await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception(
            "请求处理失败，request_id=%s method=%s path=%s",
            request_id, request.method, request.url.path,
        )
        raise
    finally:
        if should_auto_audit:
            # PyMongo is synchronous; keep audit persistence off the event loop.
            await asyncio.to_thread(_record_request_audit, request, status_code)
        duration_ms = round((time.monotonic() - started) * 1000)
        if not request.url.path.startswith("/static/") and request.url.path not in {"/api/health", "/api/health/live", "/api/health/ready"}:
            log_method = logger.warning if duration_ms >= SLOW_REQUEST_THRESHOLD_MS else logger.info
            log_method(
                "请求完成，request_id=%s method=%s path=%s status=%d duration_ms=%d client_ip=%s",
                request_id, request.method, request.url.path, status_code, duration_ms,
                request.client.host if request.client else "unknown",
            )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONT_DIR / "index.html")


@app.get("/api/health/live")
def liveness() -> dict[str, str]:
    return {"status": "ok", "service": "knowledge-base"}


@app.get("/api/health")
@app.get("/api/health/ready")
def readiness() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    try:
        get_metadata_service().ping()
        dependencies["mongodb"] = "ok"
    except Exception:
        dependencies["mongodb"] = "unavailable"
    try:
        from utils.milvus_utils import get_milvus_client
        get_milvus_client().list_collections()
        dependencies["milvus"] = "ok"
    except Exception:
        dependencies["milvus"] = "unavailable"
    try:
        from config.minio_config import minio_config
        minio_client = peek_minio_client()
        buckets_ready = (
            minio_client is not None
            and minio_client.bucket_exists(minio_config.bucket_name)
            and minio_client.bucket_exists(minio_config.document_bucket_name)
        )
        dependencies["minio"] = "ok" if buckets_ready else (
            "unavailable" if minio_client is None else "bucket_missing"
        )
    except Exception:
        dependencies["minio"] = "unavailable"
    history_tool = peek_history_mongo_tool()
    if history_tool is None:
        dependencies["history_mongodb"] = "unavailable"
    else:
        try:
            history_tool.client.admin.command("ping")
            dependencies["history_mongodb"] = "ok"
        except Exception:
            dependencies["history_mongodb"] = "unavailable"
    required_config = [
        "OPENAI_API_KEY", "OPENAI_API_BASE", "LLM_DEFAULT_MODEL",
        "MILVUS_URL", "MONGO_URL", "MONGO_DB_NAME",
        "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_NAME",
    ]
    auth_config = load_auth_config()
    if auth_config.enabled:
        required_config.append("AUTH_SECRET")
    missing_config = [key for key in required_config if not (os.getenv(key) or "").strip()]
    dependencies["configuration"] = "ok" if not missing_config else "missing"
    if auth_config.enabled:
        try:
            dependencies["authentication"] = "ok" if not get_auth_service().validate_configuration() else "misconfigured"
        except Exception:
            dependencies["authentication"] = "unavailable"
    else:
        dependencies["authentication"] = "ok"
    status = "ok" if all(value == "ok" for value in dependencies.values()) else "degraded"
    return {
        "status": status,
        "service": "knowledge-base",
        "dependencies": dependencies,
        "missing_config": missing_config,
        "worker_id": WORKER_ID,
        "auth_enabled": auth_config.enabled,
    }


@app.get("/api/auth/config")
def auth_config() -> dict[str, Any]:
    config = load_auth_config()
    return {"enabled": config.enabled, "token_ttl_minutes": config.token_ttl_minutes}


@app.post("/api/auth/login")
def login(request: LoginRequest, http_request: Request) -> JSONResponse:
    config = load_auth_config()
    if not config.enabled:
        return JSONResponse({"enabled": False, "user": None})
    client_key = http_request.client.host if http_request.client else "unknown"
    retry_after = _check_login_rate_limit(client_key)
    if retry_after:
        _record_request_audit(http_request, 429, details={"attempted_username": request.username})
        raise HTTPException(
            status_code=429, detail="登录尝试过多，请稍后重试", headers={"Retry-After": str(retry_after)}
        )
    try:
        service = get_auth_service()
        user = service.authenticate(request.username, request.password)
        token = service.create_token(user)
    except AuthenticationError as exc:
        _record_login_failure(client_key)
        _record_request_audit(http_request, 401, details={"attempted_username": request.username})
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    _clear_login_failures(client_key)
    http_request.state.user = user
    _record_request_audit(http_request, 200)
    response = JSONResponse({"enabled": True, "user": user})
    response.set_cookie(
        "kb_access_token", token,
        max_age=config.token_ttl_minutes * 60,
        httponly=True, secure=config.cookie_secure, samesite="strict", path="/",
    )
    return response


@app.post("/api/auth/logout")
def logout() -> JSONResponse:
    response = JSONResponse({"logged_out": True})
    response.delete_cookie("kb_access_token", path="/", samesite="strict")
    return response


@app.get("/api/auth/me")
def current_user(request: Request) -> dict[str, Any]:
    if not load_auth_config().enabled:
        return {"enabled": False, "user": None}
    return {"enabled": True, "user": request.state.user}


@app.get("/api/usage")
def usage_limits(request: Request) -> dict[str, Any]:
    scope_key = _rate_limit_scope(request)
    items = []
    for action, (limit, window_seconds) in RATE_LIMIT_SPECS.items():
        if limit <= 0:
            items.append({"action": action, "enabled": False, "limit": 0})
            continue
        item = get_metadata_service().get_rate_limit_usage(
            scope_key=scope_key,
            action=action,
            limit=limit,
            window_seconds=window_seconds,
        )
        item["enabled"] = True
        item["window_seconds"] = window_seconds
        items.append(item)
    return {"items": items}


@app.get("/api/users")
def list_users() -> dict[str, Any]:
    items = get_auth_service().list_users()
    return {"items": items, "total": len(items)}


@app.post("/api/users", status_code=201)
def create_user(request: UserCreateRequest) -> dict[str, Any]:
    try:
        return get_auth_service().create_user(request.username, request.password, request.role)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="用户名已存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/api/users/{user_id}")
def update_user(user_id: str, request: UserUpdateRequest) -> dict[str, Any]:
    changes = request.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=422, detail="没有可更新的字段")
    try:
        user = get_auth_service().update_user(user_id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


@app.get("/api/audit-logs")
def list_audit_logs(
    action: str | None = Query(default=None, max_length=80),
    outcome: str | None = Query(default=None, pattern="^(success|failure|denied)$"),
    actor_user_id: str | None = Query(default=None, max_length=64),
    start_at: datetime | None = Query(default=None),
    end_at: datetime | None = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    if start_at and start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=timezone.utc)
    if end_at and end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=timezone.utc)
    if start_at and end_at and start_at > end_at:
        raise HTTPException(status_code=422, detail="开始时间不能晚于结束时间")
    return get_metadata_service().list_audit_logs(
        action=action,
        outcome=outcome,
        actor_user_id=actor_user_id,
        start_at=start_at,
        end_at=end_at,
        skip=skip,
        limit=limit,
    )


@app.post("/api/knowledge-bases", status_code=201)
def create_knowledge_base(request: KnowledgeBaseCreateRequest) -> dict[str, Any]:
    try:
        return get_metadata_service().create_knowledge_base(request.name, request.description)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="知识库名称已存在") from exc


@app.get("/api/knowledge-bases")
def list_knowledge_bases() -> dict[str, Any]:
    items = get_metadata_service().list_knowledge_bases()
    return {"items": items, "total": len(items)}


@app.get("/api/knowledge-bases/{knowledge_base_id}")
def get_knowledge_base(knowledge_base_id: str) -> dict[str, Any]:
    record = get_metadata_service().get_knowledge_base(knowledge_base_id)
    if not record:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return record


@app.get("/api/knowledge-bases/{knowledge_base_id}/chunks")
def list_knowledge_base_chunks(
    knowledge_base_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: str | None = Query(None, max_length=200),
    document_id: str | None = Query(None),
) -> dict[str, Any]:
    """List knowledge chunks (paginated, searchable) for a knowledge base."""
    service = get_metadata_service()
    if not service.get_knowledge_base(knowledge_base_id):
        raise HTTPException(status_code=404, detail="知识库不存在")

    skip = (page - 1) * page_size
    result = service.list_chunks(
        knowledge_base_id,
        skip=skip,
        limit=page_size,
        keyword=keyword,
        document_id=document_id,
    )
    return {
        "items": result["items"],
        "total": result["total"],
        "page": page,
        "page_size": page_size,
    }


@app.get("/api/knowledge-bases/{knowledge_base_id}/chunks/{chunk_id}")
def get_knowledge_base_chunk(knowledge_base_id: str, chunk_id: str) -> dict[str, Any]:
    """Get single chunk detail."""
    service = get_metadata_service()
    if not service.get_knowledge_base(knowledge_base_id):
        raise HTTPException(status_code=404, detail="知识库不存在")

    chunk = service.get_chunk(knowledge_base_id, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="知识条目不存在")
    return chunk


@app.get("/api/knowledge-bases/{knowledge_base_id}/stats")
def get_knowledge_base_stats(knowledge_base_id: str) -> dict[str, Any]:
    """Knowledge base health stats (total chunks, docs, last update)."""
    service = get_metadata_service()
    if not service.get_knowledge_base(knowledge_base_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    return service.get_kb_stats(knowledge_base_id)


@app.post("/api/knowledge-bases/{knowledge_base_id}/exports", response_model=TaskCreatedResponse, status_code=202)
def create_knowledge_base_export(knowledge_base_id: str, request: Request) -> TaskCreatedResponse:
    service = get_metadata_service()
    if not service.get_knowledge_base(knowledge_base_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    _enforce_rate_limit(request, "export")
    task_id = uuid.uuid4().hex
    task = service.create_task(
        task_id,
        "export",
        knowledge_base_id,
        {"knowledge_base_id": knowledge_base_id},
        owner_user_id=_request_user_id(request),
    )
    _submit_task(task)
    return TaskCreatedResponse(task_id=task_id, session_id=knowledge_base_id)


@app.patch("/api/knowledge-bases/{knowledge_base_id}")
def update_knowledge_base(knowledge_base_id: str, request: KnowledgeBaseUpdateRequest) -> dict[str, Any]:
    changes = request.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=422, detail="没有可更新的字段")
    try:
        record = get_metadata_service().update_knowledge_base(knowledge_base_id, changes)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="知识库名称已存在") from exc
    if not record:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return record


@app.delete("/api/knowledge-bases/{knowledge_base_id}")
def delete_knowledge_base(knowledge_base_id: str) -> dict[str, bool]:
    try:
        deleted = get_metadata_service().delete_knowledge_base(knowledge_base_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return {"deleted": True}


@app.get("/api/knowledge-bases/{knowledge_base_id}/documents")
def list_documents(knowledge_base_id: str, skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    if not get_metadata_service().get_knowledge_base(knowledge_base_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    result = get_metadata_service().list_documents(knowledge_base_id, skip, limit)
    result["items"] = [_present_document(item) for item in result["items"]]
    return result


@app.get("/api/documents/{document_id}")
def get_document(document_id: str) -> dict[str, Any]:
    record = get_metadata_service().get_document(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="文档不存在")
    record["versions"] = [
        _present_document(version)
        for version in get_metadata_service().get_document_versions(record["logical_document_id"])
    ]
    return _present_document(record)


@app.get("/api/documents/{document_id}/download")
def download_document(document_id: str) -> StreamingResponse:
    from minio.error import S3Error
    from utils.minio_utils import get_minio_client

    record = get_metadata_service().get_document(document_id)
    if not record or not record.get("source_bucket") or not record.get("source_object_name"):
        raise HTTPException(status_code=404, detail="该文档版本没有可下载的归档原文件")
    client = get_minio_client()
    if client is None:
        raise HTTPException(status_code=503, detail="对象存储当前不可用")
    try:
        source = client.get_object(record["source_bucket"], record["source_object_name"])
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            raise HTTPException(status_code=404, detail="归档原文件不存在") from exc
        logger.exception("读取归档原文件失败，document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="归档原文件读取失败") from exc

    def stream_source():
        try:
            while data := source.read(1024 * 1024):
                yield data
        finally:
            source.close()
            source.release_conn()

    filename = Path(record.get("filename") or "document").name
    media_type = _document_media_type(filename)
    return StreamingResponse(
        stream_source(),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, bool]:
    record = get_metadata_service().get_document(document_id)
    if not record or not record.get("is_current"):
        raise HTTPException(status_code=404, detail="文档不存在或已删除")
    try:
        _delete_document_vectors(document_id)
    except Exception as exc:
        logger.exception("删除文档向量失败，document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="向量清理失败，文档未删除") from exc
    versions = get_metadata_service().get_document_versions(record["logical_document_id"])
    for version in versions:
        try:
            _delete_document_assets(version["document_id"])
            _remove_document_source(version)
        except Exception:
            # 向量已经移除，资产清理失败不能恢复检索状态，保留告警供运维补偿。
            logger.warning("删除文档归档资产失败，document_id=%s", version["document_id"], exc_info=True)
    get_metadata_service().mark_document_deleted(document_id)
    return {"deleted": True}


@app.post("/api/documents/{document_id}/retry", response_model=TaskCreatedResponse, status_code=202)
def retry_document(document_id: str) -> TaskCreatedResponse:
    service = get_metadata_service()
    document = service.get_document(document_id)
    if not document or document.get("status") != "failed" or not document.get("task_id"):
        raise HTTPException(status_code=409, detail="仅导入失败且保留任务信息的文档可以重试")
    existing_task = service.get_task(document["task_id"])
    if not existing_task:
        raise HTTPException(status_code=409, detail="文档关联任务不存在")
    try:
        _restore_document_source(document, existing_task)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        logger.exception("恢复归档原文件失败，document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="归档原文件恢复失败") from exc
    except Exception as exc:
        logger.exception("对象存储恢复失败，document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="对象存储当前不可用") from exc
    if not service.restart_document_version(document_id):
        raise HTTPException(status_code=409, detail="文档任务当前不可重试")
    task = service.retry_task(document["task_id"])
    if not task:
        service.fail_document_version(document_id, "关联任务当前不可重试")
        raise HTTPException(status_code=409, detail="文档任务当前不可重试")
    _submit_task(task)
    return TaskCreatedResponse(task_id=task["task_id"], session_id=document_id)


@app.post("/api/documents/{document_id}/discard-processing")
def discard_processing_document(document_id: str) -> dict[str, bool]:
    """清除卡住的 processing 状态记录，允许重新上传相同文件。"""
    service = get_metadata_service()
    document = service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")

    if document.get("status") != "processing":
        raise HTTPException(status_code=409, detail="只有 processing 状态的文档才能清除")

    if document.get("is_current"):
        raise HTTPException(status_code=409, detail="当前版本不能直接清除")

    deleted = service.discard_processing_document(document_id)
    if not deleted:
        raise HTTPException(status_code=409, detail="未能清除 processing 记录")

    return {"discarded": True}


@app.post("/api/query", response_model=TaskCreatedResponse, status_code=202)
def create_query(query_request: QueryRequest, request: Request) -> TaskCreatedResponse:
    question = query_request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    service = get_metadata_service()
    for knowledge_base_id in query_request.knowledge_base_ids:
        if not service.get_knowledge_base(knowledge_base_id):
            raise HTTPException(status_code=404, detail=f"知识库不存在: {knowledge_base_id}")
    _enforce_rate_limit(request, "query")
    session_id = (query_request.session_id or uuid.uuid4().hex).strip()
    owner_user_id = _request_user_id(request)
    try:
        service.touch_session(session_id, question, query_request.knowledge_base_ids, owner_user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    task_id = uuid.uuid4().hex
    task = service.create_task(task_id, "query", session_id, {
        "question": question, "session_id": session_id, "knowledge_base_ids": query_request.knowledge_base_ids,
    }, owner_user_id=owner_user_id)
    _submit_task(task)
    return TaskCreatedResponse(task_id=task_id, session_id=session_id)


def _present_task(task: dict[str, Any]) -> dict[str, Any]:
    """Remove private payload fields and add display-oriented duration data."""
    pipelines = {"query": QUERY_PIPELINE, "import": IMPORT_PIPELINE, "export": EXPORT_PIPELINE}
    task["pipeline"] = pipelines.get(task["type"], [])
    started, finished = task.get("started_at"), task.get("finished_at")
    if started:
        from datetime import datetime
        start_time = datetime.fromisoformat(started).timestamp()
        end_time = datetime.fromisoformat(finished).timestamp() if finished else time.time()
        task["duration_ms"] = max(0, int((end_time - start_time) * 1000))
    else:
        task["duration_ms"] = 0
    task.pop("payload", None)
    return task


def _present_task_summary(task: dict[str, Any]) -> dict[str, Any]:
    presented = _present_task(task)
    allowed_fields = {
        "task_id", "type", "session_id", "status", "current_node", "completed_nodes",
        "attempt", "error", "result", "created_at", "updated_at", "started_at", "finished_at", "duration_ms",
    }
    return {key: value for key, value in presented.items() if key in allowed_fields}


@app.get("/api/tasks")
def list_tasks(
    status: str | None = Query(default=None),
    task_type: str | None = Query(default=None, alias="type"),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    allowed_statuses = {"pending", "retrying", "processing", "completed", "failed", "cancelled"}
    allowed_types = {"query", "import", "evaluation", "export"}
    if status and status not in allowed_statuses:
        raise HTTPException(status_code=422, detail="不支持的任务状态")
    if task_type and task_type not in allowed_types:
        raise HTTPException(status_code=422, detail="不支持的任务类型")
    items = [
        _present_task_summary(task)
        for task in get_metadata_service().list_tasks(status=status, task_type=task_type, limit=limit)
    ]
    return {"items": items, "total": len(items)}


@app.get("/api/operations/summary")
def operations_summary() -> dict[str, Any]:
    return get_metadata_service().operations_summary()


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, request: Request) -> dict[str, Any]:
    task = get_metadata_service().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    _ensure_task_access(task, request)
    return _present_task(task)


@app.get("/api/exports/{task_id}/download")
def download_knowledge_base_export(task_id: str, request: Request) -> FileResponse:
    task = get_metadata_service().get_task(task_id)
    if not task or task.get("type") != "export":
        raise HTTPException(status_code=404, detail="导出任务不存在")
    _ensure_task_access(task, request)
    if task.get("status") != "completed":
        raise HTTPException(status_code=409, detail="导出任务尚未完成")
    export_path = (EXPORT_DIR / f"{task['task_id']}.zip").resolve()
    if export_path.parent != EXPORT_DIR.resolve() or not export_path.is_file():
        raise HTTPException(status_code=410, detail="导出文件已过期，请重新生成")
    filename = Path((task.get("result") or {}).get("filename") or f"knowledge-base-{task_id}.zip").name
    return FileResponse(
        export_path,
        media_type="application/zip",
        filename=filename,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@app.get("/api/tasks/{task_id}/events")
async def stream_task_events(task_id: str, request: Request) -> StreamingResponse:
    initial_task = get_metadata_service().get_task(task_id)
    if not initial_task:
        raise HTTPException(status_code=404, detail="任务不存在")
    _ensure_task_access(initial_task, request)

    async def event_stream():
        previous_snapshot = ""
        terminal_statuses = {"completed", "failed", "cancelled"}
        while not await request.is_disconnected():
            task = get_metadata_service().get_task(task_id)
            if not task:
                yield 'event: error\ndata: {"detail":"任务不存在"}\n\n'
                return
            presented = _present_task(task)
            snapshot = json.dumps(presented, ensure_ascii=False, sort_keys=True)
            if snapshot != previous_snapshot:
                event_name = "final" if presented["status"] in terminal_statuses else "progress"
                yield f"event: {event_name}\ndata: {snapshot}\n\n"
                previous_snapshot = snapshot
            if presented["status"] in terminal_statuses:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/tasks/{task_id}/retry", response_model=TaskCreatedResponse, status_code=202)
def retry_task(task_id: str) -> TaskCreatedResponse:
    task = get_metadata_service().retry_task(task_id)
    if not task:
        raise HTTPException(status_code=409, detail="仅失败或已取消任务可以重试")
    _submit_task(task)
    return TaskCreatedResponse(task_id=task_id, session_id=task["session_id"])


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str, request: Request) -> dict[str, Any]:
    existing = get_metadata_service().get_task(task_id)
    if not existing:
        raise HTTPException(status_code=404, detail="任务不存在")
    _ensure_task_access(existing, request)
    try:
        task = get_metadata_service().request_task_cancel(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not task:
        raise HTTPException(status_code=409, detail="任务不存在或当前状态不可取消")
    if task.get("type") == "import" and task.get("status") == "cancelled":
        document_id = (task.get("payload") or {}).get("document_id")
        if document_id:
            get_metadata_service().fail_document_version(document_id, "用户在任务开始前取消导入")
    return _present_task(task)


@app.get("/api/sessions")
def list_sessions(request: Request, limit: int = Query(30, ge=1, le=100)) -> dict[str, Any]:
    items = get_metadata_service().list_sessions(limit, owner_user_id=_owner_scope(request))
    return {"items": items, "total": len(items)}


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, payload: SessionRenameRequest, request: Request) -> dict[str, Any]:
    session = get_metadata_service().rename_session(session_id, payload.title, owner_user_id=_owner_scope(request))
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, request: Request) -> dict[str, bool]:
    try:
        deleted = get_metadata_service().delete_session(session_id, owner_user_id=_owner_scope(request))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": True}


@app.get("/api/history/{session_id}")
def history(session_id: str, request: Request) -> dict[str, Any]:
    _ensure_session_access(session_id, request)
    try:
        from utils.mongo_history_utils import get_recent_messages
        messages = get_recent_messages(session_id, limit=50)
    except Exception:
        logger.warning("历史记录服务当前不可用，session_id=%s", session_id)
        messages = []
    return {"session_id": session_id, "messages": _serialize_mongo_value(messages)}


@app.delete("/api/history/{session_id}")
def delete_history(session_id: str, request: Request) -> dict[str, Any]:
    _ensure_session_access(session_id, request)
    try:
        from utils.mongo_history_utils import clear_history
        deleted_count = clear_history(session_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="历史记录服务当前不可用") from exc
    return {"session_id": session_id, "deleted_count": deleted_count}


async def _save_upload(file: UploadFile, target_path: Path, max_bytes: int) -> str:
    digest, total_bytes = hashlib.sha256(), 0
    try:
        with target_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="文件大小超过限制")
                digest.update(chunk)
                output.write(chunk)
    finally:
        await file.close()
    return digest.hexdigest()


@app.post("/api/knowledge-bases/{knowledge_base_id}/documents", response_model=TaskCreatedResponse, status_code=202)
async def import_document(knowledge_base_id: str, request: Request, file: UploadFile = File(...)) -> TaskCreatedResponse:
    service = get_metadata_service()
    if not service.get_knowledge_base(knowledge_base_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    original_name = Path(file.filename or "").name
    if Path(original_name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=415, detail="仅支持 PDF 和 Markdown 文件")
    _enforce_rate_limit(request, "import")
    task_id = uuid.uuid4().hex
    upload_dir, output_dir = UPLOAD_DIR / task_id, OUTPUT_DIR / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / original_name
    checksum = await _save_upload(file, target_path, MAX_UPLOAD_BYTES)
    try:
        document, duplicate = service.begin_document_version(
            knowledge_base_id, original_name, checksum, str(target_path)
        )
    except DuplicateKeyError as exc:
        target_path.unlink(missing_ok=True)
        # 尝试找到正在 processing 的文档，返回 document_id 给前端
        existing = service.documents.find_one({
            "knowledge_base_id": knowledge_base_id,
            "filename": original_name,
            "status": "processing",
            "is_current": False
        })
        detail = {
            "message": "同一文档已有版本正在处理中，请稍后重试",
            "document_id": str(existing["_id"]) if existing else None
        }
        raise HTTPException(status_code=409, detail=detail) from exc
    if duplicate:
        target_path.unlink(missing_ok=True)
        _safe_remove_task_directory(UPLOAD_DIR, task_id)
        _safe_remove_task_directory(OUTPUT_DIR, task_id)
        return TaskCreatedResponse(task_id="", session_id=document["document_id"], duplicate=True)
    try:
        source_bucket, source_object_name, source_size = _archive_document_source(document, target_path)
        document.update({
            "source_bucket": source_bucket,
            "source_object_name": source_object_name,
            "source_size": source_size,
        })
        service.attach_document_source(
            document["document_id"], source_bucket, source_object_name, source_size
        )
    except Exception as exc:
        try:
            _remove_document_source(document)
        except Exception:
            logger.warning("回滚未登记的归档原文件失败，document_id=%s", document["document_id"], exc_info=True)
        service.discard_processing_document(document["document_id"])
        target_path.unlink(missing_ok=True)
        _safe_remove_task_directory(UPLOAD_DIR, task_id)
        _safe_remove_task_directory(OUTPUT_DIR, task_id)
        logger.exception("归档上传原文件失败，document_id=%s", document["document_id"])
        raise HTTPException(status_code=503, detail="原文件归档失败，未创建导入任务") from exc
    payload = {
        "file_path": str(target_path), "output_dir": str(output_dir), "knowledge_base_id": knowledge_base_id,
        "document_id": document["document_id"], "previous_document_id": document.get("previous_document_id"),
        "version": document["version"],
    }
    try:
        task = service.create_task(
            task_id, "import", document["document_id"], payload, owner_user_id=_request_user_id(request)
        )
        service.attach_document_task(document["document_id"], task_id)
    except Exception:
        try:
            _remove_document_source(document)
        except Exception:
            logger.warning("回滚归档原文件失败，document_id=%s", document["document_id"], exc_info=True)
        service.discard_processing_document(document["document_id"])
        target_path.unlink(missing_ok=True)
        _safe_remove_task_directory(UPLOAD_DIR, task_id)
        _safe_remove_task_directory(OUTPUT_DIR, task_id)
        raise
    _submit_task(task)
    return TaskCreatedResponse(task_id=task_id, session_id=document["document_id"])


@app.post("/api/import", response_model=TaskCreatedResponse, status_code=202)
async def import_document_legacy(request: Request, file: UploadFile = File(...)) -> TaskCreatedResponse:
    service = get_metadata_service()
    knowledge_bases = service.list_knowledge_bases()
    knowledge_base_id = (knowledge_bases[0]["knowledge_base_id"] if knowledge_bases else
                         service.create_knowledge_base("默认知识库", "由兼容上传接口自动创建")["knowledge_base_id"])
    return await import_document(knowledge_base_id, request, file)


@app.post("/api/evaluations", response_model=TaskCreatedResponse, status_code=202)
async def create_evaluation(
    request: Request,
    file: UploadFile = File(...),
    knowledge_base_ids: str = Form(default=""),
) -> TaskCreatedResponse:
    filename = Path(file.filename or "dataset.jsonl").name
    if Path(filename).suffix.lower() != ".jsonl":
        raise HTTPException(status_code=415, detail="评测数据集必须是 JSONL 文件")
    _enforce_rate_limit(request, "evaluation")
    task_id = uuid.uuid4().hex
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = EVALUATION_DIR / f"{task_id}.jsonl"
    dataset_checksum = await _save_upload(file, dataset_path, MAX_EVALUATION_BYTES)
    try:
        load_dataset(dataset_path)
    except (ValueError, FileNotFoundError) as exc:
        dataset_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    kb_ids = [item.strip() for item in knowledge_base_ids.split(",") if item.strip()]
    task = get_metadata_service().create_task(
        task_id,
        "evaluation",
        task_id,
        {"dataset_path": str(dataset_path), "dataset_name": filename, "dataset_checksum": dataset_checksum,
         "knowledge_base_ids": kb_ids, "owner_user_id": _request_user_id(request)},
        owner_user_id=_request_user_id(request),
    )
    _submit_task(task)
    return TaskCreatedResponse(task_id=task_id, session_id=task_id)


@app.get("/api/evaluations")
def list_evaluations(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    return get_metadata_service().list_evaluation_runs(
        limit, owner_user_id=_owner_scope(request), skip=skip
    )


@app.get("/api/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str, request: Request) -> dict[str, Any]:
    record = get_metadata_service().get_evaluation_run(evaluation_id, owner_user_id=_owner_scope(request))
    if not record:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return record


@app.patch("/api/evaluations/{evaluation_id}/baseline")
def set_evaluation_baseline(evaluation_id: str, request: Request) -> dict[str, Any]:
    record = get_metadata_service().set_evaluation_baseline(
        evaluation_id, owner_user_id=_owner_scope(request)
    )
    if not record:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return record


@app.get("/api/evaluations/{evaluation_id}/comparison")
def compare_evaluation(
    evaluation_id: str,
    request: Request,
    baseline_id: str | None = Query(default=None, max_length=64),
    pass_rate_drop: float = Query(default=0.05, ge=0, le=1),
    keyword_recall_drop: float = Query(default=0.05, ge=0, le=1),
    citation_validity_drop: float = Query(default=0.05, ge=0, le=1),
    latency_increase_ratio: float = Query(default=0.25, ge=0, le=10),
) -> dict[str, Any]:
    service = get_metadata_service()
    owner_scope = _owner_scope(request)
    candidate = service.get_evaluation_run(evaluation_id, owner_user_id=owner_scope)
    if not candidate:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    baseline = (
        service.get_evaluation_run(baseline_id, owner_user_id=owner_scope)
        if baseline_id else service.find_evaluation_baseline(candidate)
    )
    if not baseline:
        raise HTTPException(status_code=404, detail="未找到同一数据集的评测基线")
    if candidate.get("dataset_checksum") and baseline.get("dataset_checksum") != candidate["dataset_checksum"]:
        raise HTTPException(status_code=409, detail="只能比较使用相同数据集的评测")
    if not candidate.get("dataset_checksum") and baseline.get("dataset_name") != candidate.get("dataset_name"):
        raise HTTPException(status_code=409, detail="历史评测只能与同名数据集比较")
    comparison = compare_evaluation_summaries(
        candidate.get("summary") or {},
        baseline.get("summary") or {},
        {
            "pass_rate_drop": pass_rate_drop,
            "keyword_recall_drop": keyword_recall_drop,
            "citation_validity_drop": citation_validity_drop,
            "latency_increase_ratio": latency_increase_ratio,
        },
    )
    return {
        "candidate_evaluation_id": candidate["evaluation_id"],
        "baseline_evaluation_id": baseline["evaluation_id"],
        **comparison,
    }
