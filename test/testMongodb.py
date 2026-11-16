"""MongoDB 连接验证脚本。"""

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, PyMongoError, ServerSelectionTimeoutError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
SERVER_SELECTION_TIMEOUT_MS = 5_000
TEST_COLLECTION_NAME = "mongodb_connection_test"


def load_mongodb_config() -> tuple[str, str]:
    """从项目根目录的 .env 加载并校验 MongoDB 配置。"""
    load_dotenv(dotenv_path=ENV_FILE)
    mongo_url = os.getenv("MONGO_URL", "").strip()
    database_name = os.getenv("MONGO_DB_NAME", "").strip()

    missing_keys = [
        key
        for key, value in (
            ("MONGO_URL", mongo_url),
            ("MONGO_DB_NAME", database_name),
        )
        if not value
    ]
    if missing_keys:
        raise ValueError(f"MongoDB 配置缺失: {', '.join(missing_keys)}")

    return mongo_url, database_name


def test_mongodb_connection() -> None:
    """连接 MongoDB，插入一条测试数据，再按主键查询并校验。"""
    mongo_url, database_name = load_mongodb_config()
    client: MongoClient | None = None

    try:
        client = MongoClient(
            mongo_url,
            serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        )
        client.admin.command("ping")
        database = client[database_name]
        test_collection = database[TEST_COLLECTION_NAME]

        print(f"MongoDB 连接成功，database={database_name}")

        test_document = {
            "test_id": str(uuid4()),
            "message": "MongoDB 增查测试数据",
            "created_at": datetime.now(timezone.utc),
        }
        insert_result = test_collection.insert_one(test_document)
        print(f"数据新增成功，collection={TEST_COLLECTION_NAME}, id={insert_result.inserted_id}")

        queried_document = test_collection.find_one({"_id": insert_result.inserted_id})
        if queried_document is None:
            raise RuntimeError("测试数据已写入，但未能按主键查询到")
        if queried_document.get("test_id") != test_document["test_id"]:
            raise RuntimeError("查询结果与新增数据不一致")

        print(f"数据查询成功: {queried_document}")
    except (ConfigurationError, ServerSelectionTimeoutError) as exc:
        raise RuntimeError("MongoDB 配置无效或服务器不可达") from exc
    except PyMongoError as exc:
        raise RuntimeError("MongoDB 操作失败") from exc
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    test_mongodb_connection()
