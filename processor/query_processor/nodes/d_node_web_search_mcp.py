import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents import Agent, ModelSettings, RunConfig, Runner, WebSearchTool
from agents.models.openai_provider import OpenAIProvider
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class WebSearchDocument(BaseModel):
    """GPT 联网搜索返回的单条网页文档。"""

    title: str = ""
    url: str = ""
    snippet: str = ""


class WebSearchOutput(BaseModel):
    """约束 GPT 将联网搜索结果输出为稳定结构。"""

    documents: list[WebSearchDocument] = Field(default_factory=list)


class NodeWebSearchMcp(NodeBase):
    """通过 OpenAI Responses API 的 WebSearchTool 执行联网搜索。"""

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        query = state.get("rewritten_query") or state.get("original_query")
        if not isinstance(query, str) or not query.strip():
            logger.warning("GPT 联网搜索跳过：查询内容为空")
            return {"web_search_docs": []}

        try:
            documents = self._run_async(self._web_search(query.strip()))
            docs = [document.model_dump() for document in documents]
            logger.info(f"GPT 联网搜索完成，共返回 {len(docs)} 条文档")
            return {"web_search_docs": docs}
        except Exception as exc:
            logger.exception(f"GPT 联网搜索失败: {exc}")
            return {"web_search_docs": []}

    @staticmethod
    def _run_async(coroutine):
        """兼容普通同步调用以及已有事件循环的运行环境。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coroutine).result()

    async def _web_search(self, query: str) -> list[WebSearchDocument]:
        model = (
            os.getenv("WEB_SEARCH_MODEL")
            or os.getenv("LLM_DEFAULT_MODEL")
            or "gpt-5.4-mini"
        ).strip()

        openai_api_key = (
            os.getenv("WEB_SEARCH_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        if not openai_api_key:
            raise ValueError("缺少 GPT Key：请配置 OPENAI_API_KEY")

        openai_base_url = (
            os.getenv("WEB_SEARCH_OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).strip()

        agent = Agent(
            name="GPT Web Search Agent",
            model=model,
            instructions=(
                "必须先调用联网搜索工具检索用户问题。"
                "只根据搜索结果生成 documents；每条包含 title、url、snippet，"
                "不要编造网址，没有有效结果时返回空列表。"
            ),
            tools=[
                WebSearchTool(
                    search_context_size="medium",
                    external_web_access=True,
                )
            ],
            model_settings=ModelSettings(tool_choice="required"),
            output_type=WebSearchOutput,
        )

        provider = OpenAIProvider(
            api_key=openai_api_key,
            base_url=openai_base_url,
            use_responses=True,
        )
        result = await Runner.run(
            agent,
            query,
            run_config=RunConfig(
                model_provider=provider,
                tracing_disabled=True,
            ),
        )

        output = result.final_output
        if isinstance(output, WebSearchOutput):
            return output.documents
        return WebSearchOutput.model_validate(output).documents


if __name__ == "__main__":
    node = NodeWebSearchMcp()
    result = node(
        {
            "rewritten_query": "搜索华为擎云 B530 台式计算机的官方产品信息"
        }
    )
    print(result)
