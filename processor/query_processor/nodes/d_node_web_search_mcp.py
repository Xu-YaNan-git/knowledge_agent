import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from agents import Runner, RunConfig
from agents.exceptions import ModelTimeoutError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils import web_search_utils
from utils.web_search_utils import WebSearchDocument, WebSearchOutput


MAX_WEB_SEARCH_DOCUMENTS = 5


class NodeWebSearchMcp(NodeBase):
    """通过 OpenAI Responses API 的 WebSearchTool 执行联网搜索。"""

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query") or state.get("original_query")
        if not isinstance(query, str) or not query.strip():
            logger.warning("GPT 联网搜索跳过：查询内容为空")
            return {"web_search_docs": []}

        started_at = time.perf_counter()
        try:
            documents = self._run_async(self._web_search, query.strip())
            docs = [document.model_dump() for document in self._normalize_documents(documents)]
            logger.info(
                "GPT联网搜索完成，document_count=%d，duration_ms=%d",
                len(docs),
                int((time.perf_counter() - started_at) * 1000),
            )
            return {"web_search_docs": docs}
        except ModelTimeoutError:
            logger.warning(
                "GPT联网搜索超时，已降级为空结果，duration_ms=%d",
                int((time.perf_counter() - started_at) * 1000),
            )
            return {"web_search_docs": []}
        except Exception:
            # 联网搜索是并行增强分支，失败时返回空结果，让本地知识库链路继续执行。
            logger.exception(
                "GPT联网搜索失败，已降级为空结果，duration_ms=%d",
                int((time.perf_counter() - started_at) * 1000),
            )
            return {"web_search_docs": []}

    @staticmethod
    def _run_async(async_callable, *args):
        """兼容普通同步调用以及已有事件循环的运行环境。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(async_callable(*args))

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(async_callable(*args))).result()

    @staticmethod
    def _normalize_documents(documents: list[WebSearchDocument]) -> list[WebSearchDocument]:
        """过滤无效 URL、空摘要和重复网页，限制下游上下文规模。"""
        normalized: list[WebSearchDocument] = []
        seen_urls: set[str] = set()
        for document in documents:
            url = document.url.strip()
            snippet = document.snippet.strip()
            parsed_url = urlparse(url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc or not snippet:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            normalized.append(
                WebSearchDocument(
                    title=document.title.strip() or parsed_url.netloc,
                    url=url,
                    snippet=snippet,
                )
            )
            if len(normalized) >= MAX_WEB_SEARCH_DOCUMENTS:
                break
        return normalized

    async def _web_search(self, query: str) -> list[WebSearchDocument]:

        agent = web_search_utils.get_web_search_client()
        result = await Runner.run(
            agent,
            query,
            max_turns=3,
            run_config=RunConfig(
                model_provider=web_search_utils.get_model_provider(),
                tracing_disabled=True,
                trace_include_sensitive_data=False,
            ),
        )

        output = result.final_output
        if isinstance(output, WebSearchOutput):
            return output.documents
        if isinstance(output, str):
            return WebSearchOutput.model_validate(json.loads(output)).documents
        return WebSearchOutput.model_validate(output).documents


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    node = NodeWebSearchMcp()
    result = node(
        {
            "rewritten_query": "搜索华为擎云 B530 台式计算机的官方产品信息"
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
