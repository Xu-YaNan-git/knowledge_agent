"""Run the production-like knowledge-base flow against a running API and clean test data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pymongo import MongoClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)


def log(stage: str, value: Any) -> None:
    print(json.dumps({"stage": stage, "value": value}, ensure_ascii=False), flush=True)


def wait_until_ready(client: httpx.Client, timeout_seconds: int = 120) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_health: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            response = client.get("/api/health/ready")
            if response.status_code == 200:
                last_health = response.json()
                if last_health.get("status") == "ok":
                    return last_health
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"service readiness timeout: {last_health}")


def wait_task(client: httpx.Client, task_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_snapshot: tuple[Any, ...] | None = None
    while time.time() < deadline:
        response = client.get(f"/api/tasks/{task_id}")
        response.raise_for_status()
        task = response.json()
        snapshot = (task.get("status"), task.get("current_node"), tuple(task.get("completed_nodes") or []))
        if snapshot != last_snapshot:
            log("task_progress", {
                "task_id": task_id,
                "type": task.get("type"),
                "status": task.get("status"),
                "node": task.get("current_node"),
                "completed": task.get("completed_nodes"),
            })
            last_snapshot = snapshot
        if task.get("status") in {"completed", "failed", "cancelled"}:
            return task
        time.sleep(2)
    raise TimeoutError(f"task timeout: {task_id}")


def cleanup_test_data(
    client: httpx.Client,
    knowledge_base_id: str | None,
    session_id: str,
    task_ids: list[str],
) -> None:
    mongo_url = os.getenv("MONGO_URL")
    database_name = os.getenv("MONGO_DB_NAME")
    if not mongo_url or not database_name:
        log("cleanup_warning", "MongoDB configuration is unavailable")
        return
    mongo = MongoClient(mongo_url, serverSelectionTimeoutMS=5_000)
    database = mongo[database_name]
    source_records = list(database["knowledge_documents"].find(
        {"knowledge_base_id": knowledge_base_id}
    )) if knowledge_base_id else []
    try:
        from app import _delete_document_assets, _delete_document_vectors
        from config.minio_config import minio_config
        from utils.minio_utils import get_minio_client

        minio_client = get_minio_client()
        if minio_client and knowledge_base_id:
            for item in minio_client.list_objects(
                minio_config.document_bucket_name,
                prefix=f"sources/{knowledge_base_id}/",
                recursive=True,
            ):
                minio_client.remove_object(minio_config.document_bucket_name, item.object_name)
        for record in source_records:
            document_id = record.get("document_id")
            if document_id:
                try:
                    _delete_document_vectors(document_id)
                    _delete_document_assets(document_id)
                except Exception as exc:
                    log("vector_asset_cleanup_warning", {"document_id": document_id, "error": str(exc)})
            if minio_client and record.get("source_bucket") and record.get("source_object_name"):
                minio_client.remove_object(record["source_bucket"], record["source_object_name"])
    except Exception as exc:
        log("storage_cleanup_warning", str(exc))

    if knowledge_base_id:
        database["knowledge_documents"].delete_many({"knowledge_base_id": knowledge_base_id})
        database["knowledge_bases"].delete_many({"knowledge_base_id": knowledge_base_id})
    if task_ids:
        database["background_tasks"].delete_many({"task_id": {"$in": task_ids}})
        database["evaluation_runs"].delete_many({"task_id": {"$in": task_ids}})
    database["chat_sessions"].delete_many({"session_id": session_id})
    database["chat_message"].delete_many({"session_id": session_id})
    mongo.close()

    for task_id in task_ids:
        if len(task_id) != 32 or any(character not in "0123456789abcdef" for character in task_id):
            continue
        shutil.rmtree(PROJECT_ROOT / "temp_data" / "uploads" / task_id, ignore_errors=True)
        shutil.rmtree(PROJECT_ROOT / "temp_data" / "output" / task_id, ignore_errors=True)
        (PROJECT_ROOT / "temp_data" / "evaluations" / f"{task_id}.jsonl").unlink(missing_ok=True)
        (PROJECT_ROOT / "temp_data" / "exports" / f"{task_id}.zip").unlink(missing_ok=True)
    log("direct_cleanup", {"task_count": len(task_ids), "knowledge_base_id": knowledge_base_id})


def run(base_url: str) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:10]
    knowledge_base_name = f"E2E-整体流程-{run_id}"
    session_id = f"e2e-{uuid.uuid4().hex}"
    knowledge_base_id: str | None = None
    document_id: str | None = None
    task_ids: list[str] = []
    results: dict[str, Any] = {}
    # Local smoke tests must not be routed through workstation HTTP proxy settings.
    client = httpx.Client(base_url=base_url, timeout=60.0, trust_env=False)
    try:
        health = wait_until_ready(client)
        results["health"] = health["dependencies"]
        log("health", results["health"])

        response = client.post("/api/knowledge-bases", json={
            "name": knowledge_base_name,
            "description": "自动化整体流程临时知识库",
        })
        response.raise_for_status()
        knowledge_base_id = response.json()["knowledge_base_id"]
        log("knowledge_base_created", knowledge_base_id)

        document_content = f"""# FLOW-TEST-{run_id} 产品手册

## 硬件规格

产品型号：FLOW-TEST-{run_id}

- 内存容量：32GB
- 存储容量：1TB SSD
- 质保期限：3年

## 使用说明

该设备用于知识库端到端检索验证。
""".encode("utf-8")
        response = client.post(
            f"/api/knowledge-bases/{knowledge_base_id}/documents",
            files={"file": (f"flow-test-{run_id}.md", document_content, "text/markdown")},
        )
        response.raise_for_status()
        import_response = response.json()
        document_id = import_response["session_id"]
        task_ids.append(import_response["task_id"])
        import_task = wait_task(client, import_response["task_id"], 900)
        results["import"] = {"status": import_task["status"], "result": import_task.get("result"), "error": import_task.get("error")}
        if import_task["status"] != "completed":
            raise RuntimeError(f"import failed: {import_task.get('error')}")

        documents = client.get(f"/api/knowledge-bases/{knowledge_base_id}/documents").json()
        download = client.get(f"/api/documents/{document_id}/download")
        download.raise_for_status()
        results["document"] = {
            "listed": documents.get("total") == 1,
            "archived": documents["items"][0].get("source_archived") is True,
            "download_matches": hashlib.sha256(download.content).digest() == hashlib.sha256(document_content).digest(),
        }
        log("document_verified", results["document"])

        response = client.post("/api/query", json={
            "question": f"FLOW-TEST-{run_id} 的内存容量和存储容量分别是多少？",
            "session_id": session_id,
            "knowledge_base_ids": [knowledge_base_id],
        })
        response.raise_for_status()
        task_ids.append(response.json()["task_id"])
        query_task = wait_task(client, response.json()["task_id"], 900)
        normalized_answer = (query_task.get("answer") or "").upper().replace(" ", "")
        results["query"] = {
            "status": query_task["status"],
            "has_32gb": "32GB" in normalized_answer,
            "has_1tb": "1TB" in normalized_answer,
            "source_count": len(query_task.get("sources") or []),
            "error": query_task.get("error"),
        }
        log("query_verified", results["query"])
        if query_task["status"] != "completed":
            raise RuntimeError(f"query failed: {query_task.get('error')}")
        if not results["query"]["has_32gb"] or not results["query"]["has_1tb"]:
            raise AssertionError("query answer is missing required hardware specifications")
        if results["query"]["source_count"] < 1:
            raise AssertionError("query answer does not contain a knowledge-base citation")

        dataset = (json.dumps({
            "case_id": f"flow-{run_id}",
            "question": f"FLOW-TEST-{run_id} 的内存容量和存储容量分别是多少？",
            "expected_keywords": ["32GB", "1TB"],
            "knowledge_base_ids": [knowledge_base_id],
        }, ensure_ascii=False) + "\n").encode("utf-8")
        response = client.post(
            "/api/evaluations",
            data={"knowledge_base_ids": knowledge_base_id},
            files={"file": (f"flow-eval-{run_id}.jsonl", dataset, "application/jsonl")},
        )
        response.raise_for_status()
        task_ids.append(response.json()["task_id"])
        evaluation_task = wait_task(client, response.json()["task_id"], 1_200)
        results["evaluation"] = {"status": evaluation_task["status"], "result": evaluation_task.get("result"), "error": evaluation_task.get("error")}
        log("evaluation_verified", results["evaluation"])
        if evaluation_task["status"] != "completed":
            raise RuntimeError(f"evaluation failed: {evaluation_task.get('error')}")
        if evaluation_task["result"].get("pass_rate") != 1.0:
            raise AssertionError(f"evaluation did not fully pass: {evaluation_task['result']}")
        if evaluation_task["result"].get("citation_validity") != 1.0:
            raise AssertionError(f"evaluation citations are invalid: {evaluation_task['result']}")
        evaluation_id = evaluation_task["result"]["evaluation_id"]
        baseline = client.patch(f"/api/evaluations/{evaluation_id}/baseline")
        baseline.raise_for_status()
        results["baseline"] = baseline.json().get("is_baseline") is True

        response = client.post(f"/api/knowledge-bases/{knowledge_base_id}/exports")
        response.raise_for_status()
        task_ids.append(response.json()["task_id"])
        export_task = wait_task(client, response.json()["task_id"], 300)
        if export_task["status"] != "completed":
            raise RuntimeError(f"export failed: {export_task.get('error')}")
        export_response = client.get(export_task["result"]["download_url"])
        export_response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(export_response.content)) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read("manifest.json"))
        results["export"] = {
            "status": export_task["status"],
            "manifest": "manifest.json" in names,
            "archived_files": len([name for name in names if name.startswith("documents/")]),
            "manifest_documents": len(manifest["documents"]),
        }
        log("export_verified", results["export"])

        audit = client.get("/api/audit-logs", params={"action": "knowledge_base.export", "limit": 10})
        audit.raise_for_status()
        results["audit"] = {"export_events": audit.json()["total"]}

        if document_id:
            client.delete(f"/api/documents/{document_id}").raise_for_status()
        client.delete(f"/api/sessions/{session_id}").raise_for_status()
        client.delete(f"/api/knowledge-bases/{knowledge_base_id}").raise_for_status()
        results["api_cleanup"] = True
        log("result", results)
        return results
    except Exception as exc:
        log("failure", {"type": type(exc).__name__, "message": str(exc)})
        traceback.print_exc()
        raise
    finally:
        try:
            cleanup_test_data(client, knowledge_base_id, session_id, task_ids)
        finally:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    arguments = parser.parse_args()
    run(arguments.base_url)
