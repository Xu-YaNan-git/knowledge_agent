"""Knowledge Base 可视化控制台的 FastAPI 入口。"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from bson import ObjectId
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tool.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parent
FRONT_DIR = PROJECT_ROOT / "front"
UPLOAD_DIR = PROJECT_ROOT / "temp_data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "temp_data" / "output"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".md"}

QUERY_PIPELINE = [
    ("node_item_name_confirm", "产品识别"),
    ("node_multi_search", "多路检索"),
    ("node_search_embedding", "向量检索"),
    ("node_search_embedding_hyde", "HyDE 检索"),
    ("node_web_search_mcp", "网络搜索"),
    ("node_join", "检索合并"),
    ("node_rrf", "RRF 融合"),
    ("node_rerank", "GPT 精排"),
    ("node_answer_output", "答案生成"),
]

IMPORT_PIPELINE = [
    ("node_entry", "文件检查"),
    ("node_pdf_to_md", "PDF 解析"),
    ("node_md_img", "图片处理"),
    ("node_document_split", "文档切分"),
    ("node_item_name_recognition", "主体识别"),
    ("node_bge_embedding", "向量生成"),
    ("node_import_milvus", "写入 Milvus"),
]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)


class TaskCreatedResponse(BaseModel):
    task_id: str
    session_id: str


class TaskRegistry:
    """保存单进程演示环境中的后台任务状态。"""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(self, task_id: str, task_type: Literal["query", "import"], session_id: str) -> None:
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "type": task_type,
                "session_id": session_id,
                "status": "pending",
                "current_node": None,
                "completed_nodes": [],
                "answer": "",
                "sources": [],
                "image_urls": [],
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
            }

    def update(self, task_id: str, **changes: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.update(changes)

    def complete_node(self, task_id: str, node_name: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if node_name not in task["completed_nodes"]:
                task["completed_nodes"].append(node_name)
            task["current_node"] = node_name

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task else None


task_registry = TaskRegistry()
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb-task")

app = FastAPI(title="Knowledge Base Console", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")


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
    messages = get_recent_messages(session_id, limit=20)
    for message in reversed(messages):
        if message.get("role") == "assistant":
            image_urls = message.get("image_urls") or []
            if not answer:
                answer = message.get("text") or ""
            break
    return answer, sources, image_urls


def _run_query_task(task_id: str, session_id: str, question: str) -> None:
    task_registry.update(task_id, status="processing")
    accumulated_state: dict[str, Any] = {
        "original_query": question,
        "session_id": session_id,
        "is_stream": False,
    }
    try:
        # 工作流依赖较重，延迟到后台任务中加载，避免阻塞 Web 服务启动。
        from processor.query_processor.main_graph import KBQueryWorkflow

        workflow = KBQueryWorkflow()
        for event in workflow.compile().stream(accumulated_state, stream_mode="updates"):
            for node_name, update in event.items():
                task_registry.complete_node(task_id, node_name)
                if isinstance(update, dict):
                    accumulated_state.update(update)

        answer, sources, image_urls = _extract_task_output(accumulated_state, session_id)
        if not answer:
            raise RuntimeError("查询流程完成但没有生成答案")
        task_registry.update(
            task_id,
            status="completed",
            current_node=None,
            answer=answer,
            sources=sources,
            image_urls=image_urls,
            finished_at=time.time(),
        )
    except Exception:
        logger.exception("可视化查询任务失败，task_id=%s", task_id)
        task_registry.update(
            task_id,
            status="failed",
            current_node=None,
            error="查询处理失败，请检查服务日志后重试。",
            finished_at=time.time(),
        )


def _run_import_task(task_id: str, file_path: Path, output_dir: Path) -> None:
    task_registry.update(task_id, status="processing")
    # MinerU 与向量依赖较重，延迟加载以保持首页和查询 API 快速可用。
    from processor.import_processor.main_graph import KBImportWorkflow
    from processor.import_processor.state import create_default_state

    state = create_default_state(
        task_id=task_id,
        import_file_path=str(file_path),
        file_dir=str(output_dir),
    )
    try:
        workflow = KBImportWorkflow()
        final_state: dict[str, Any] = dict(state)
        for event in workflow.graph.stream(state, stream_mode="updates"):
            for node_name, update in event.items():
                task_registry.complete_node(task_id, node_name)
                if isinstance(update, dict):
                    final_state.update(update)
        task_registry.update(
            task_id,
            status="completed",
            current_node=None,
            result={
                "file_title": final_state.get("file_title"),
                "item_name": final_state.get("item_name"),
                "chunk_count": len(final_state.get("chunks") or []),
            },
            finished_at=time.time(),
        )
    except Exception:
        logger.exception("可视化导入任务失败，task_id=%s", task_id)
        task_registry.update(
            task_id,
            status="failed",
            current_node=None,
            error="资料导入失败，请检查服务日志后重试。",
            finished_at=time.time(),
        )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONT_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "knowledge-base"}


@app.post("/api/query", response_model=TaskCreatedResponse, status_code=202)
def create_query(request: QueryRequest) -> TaskCreatedResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    session_id = (request.session_id or uuid.uuid4().hex).strip()
    task_id = uuid.uuid4().hex
    task_registry.create(task_id, "query", session_id)
    executor.submit(_run_query_task, task_id, session_id, question)
    return TaskCreatedResponse(task_id=task_id, session_id=session_id)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    task = task_registry.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    task["pipeline"] = QUERY_PIPELINE if task["type"] == "query" else IMPORT_PIPELINE
    task["duration_ms"] = int(((task.get("finished_at") or time.time()) - task["started_at"]) * 1000)
    return task


@app.get("/api/history/{session_id}")
def history(session_id: str) -> dict[str, Any]:
    try:
        from utils.mongo_history_utils import get_recent_messages

        messages = get_recent_messages(session_id, limit=50)
    except Exception:
        logger.warning("历史记录服务当前不可用，session_id=%s", session_id)
        messages = []
    return {"session_id": session_id, "messages": _serialize_mongo_value(messages)}


@app.delete("/api/history/{session_id}")
def delete_history(session_id: str) -> dict[str, Any]:
    try:
        from utils.mongo_history_utils import clear_history

        deleted_count = clear_history(session_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="历史记录服务当前不可用") from exc
    return {"session_id": session_id, "deleted_count": deleted_count}


@app.post("/api/import", response_model=TaskCreatedResponse, status_code=202)
async def import_document(file: UploadFile = File(...)) -> TaskCreatedResponse:
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=415, detail="仅支持 PDF 和 Markdown 文件")

    task_id = uuid.uuid4().hex
    upload_dir = UPLOAD_DIR / task_id
    output_dir = OUTPUT_DIR / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / original_name

    total_bytes = 0
    try:
        with target_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="文件不能超过 50 MB")
                output.write(chunk)
    finally:
        await file.close()

    task_registry.create(task_id, "import", task_id)
    executor.submit(_run_import_task, task_id, target_path, output_dir)
    return TaskCreatedResponse(task_id=task_id, session_id=task_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
