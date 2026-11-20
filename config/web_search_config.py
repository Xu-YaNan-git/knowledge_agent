import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass
class WebSearchConfig:
    web_search_model: str
    openai_api_key: str
    openai_base_url: str
    timeout_seconds: float

web_search_config = WebSearchConfig(
    web_search_model=os.getenv("LLM_DEFAULT_MODEL",""),
    openai_api_key=os.getenv("OPENAI_API_KEY",""),
    openai_base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL",""),
    timeout_seconds=float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "90")),
)
