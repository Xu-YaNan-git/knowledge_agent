import importlib
import asyncio
import sys
import threading

import minio
import pymongo
import pytest


def _fresh_import(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_minio_module_import_does_not_create_client(monkeypatch) -> None:
    calls = []

    def fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("MinIO client must not be created while importing the module")

    monkeypatch.setattr(minio, "Minio", fail_if_called)

    module = _fresh_import("utils.minio_utils")

    assert calls == []
    assert module.peek_minio_client() is None


def test_mongo_history_module_import_does_not_create_client(monkeypatch) -> None:
    calls = []

    def fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("MongoDB client must not be created while importing the module")

    monkeypatch.setattr(pymongo, "MongoClient", fail_if_called)

    module = _fresh_import("utils.mongo_history_utils")

    assert calls == []
    assert module.peek_history_mongo_tool() is None


def test_minio_initialization_can_retry_after_failure(monkeypatch) -> None:
    module = _fresh_import("utils.minio_utils")
    working_client = object()
    attempts = []

    def client_factory(**_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary MinIO failure")
        return working_client

    module.reset_minio_client()
    with pytest.raises(RuntimeError, match="temporary MinIO failure"):
        module.initialize_minio_client(client_factory=client_factory)

    assert module.peek_minio_client() is None
    assert module.initialize_minio_client(client_factory=client_factory) is working_client
    assert len(attempts) == 2


def test_history_mongo_initialization_can_retry_after_failure(monkeypatch) -> None:
    module = _fresh_import("utils.mongo_history_utils")
    working_tool = object()
    attempts = []

    def tool_factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary MongoDB failure")
        return working_tool

    module.reset_history_mongo_tool()
    with pytest.raises(RuntimeError, match="temporary MongoDB failure"):
        module.initialize_history_mongo_tool(tool_factory=tool_factory)

    assert module.peek_history_mongo_tool() is None
    assert module.initialize_history_mongo_tool(tool_factory=tool_factory) is working_tool
    assert len(attempts) == 2


def test_external_clients_close_and_can_be_initialized_again() -> None:
    minio_module = _fresh_import("utils.minio_utils")
    mongo_module = _fresh_import("utils.mongo_history_utils")

    class ClosableMinio:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    class ClosableMongoClient:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    class MongoTool:
        def __init__(self) -> None:
            self.client = ClosableMongoClient()

    first_minio = ClosableMinio()
    first_mongo = MongoTool()
    minio_module.initialize_minio_client(client_factory=lambda **_kwargs: first_minio)
    mongo_module.initialize_history_mongo_tool(tool_factory=lambda: first_mongo)

    minio_module.close_minio_client()
    mongo_module.close_history_mongo_tool()
    minio_module.close_minio_client()
    mongo_module.close_history_mongo_tool()

    assert first_minio.closed == 1
    assert first_mongo.client.closed == 1
    assert minio_module.peek_minio_client() is None
    assert mongo_module.peek_history_mongo_tool() is None

    second_minio = ClosableMinio()
    second_mongo = MongoTool()
    assert minio_module.initialize_minio_client(client_factory=lambda **_kwargs: second_minio) is second_minio
    assert mongo_module.initialize_history_mongo_tool(tool_factory=lambda: second_mongo) is second_mongo


def test_app_external_dependency_startup_isolates_failures(monkeypatch) -> None:
    app_module = _fresh_import("app")
    calls = []

    def failing_minio() -> None:
        calls.append("minio")
        raise RuntimeError("minio unavailable")

    def working_history() -> None:
        calls.append("history_mongodb")

    monkeypatch.setattr(app_module, "initialize_minio_client", failing_minio)
    monkeypatch.setattr(app_module, "initialize_history_mongo_tool", working_history)

    errors = asyncio.run(app_module._initialize_external_dependencies(app_module.app))

    assert calls == ["minio", "history_mongodb"]
    assert errors == {"minio": "unavailable"}
    assert app_module.app.state.dependency_errors == errors


def test_readiness_does_not_initialize_minio_or_history_mongodb(monkeypatch) -> None:
    app_module = _fresh_import("app")

    class ReadyMetadata:
        @staticmethod
        def ping() -> None:
            return None

    class ReadyMilvus:
        @staticmethod
        def list_collections() -> list:
            return []

    monkeypatch.setattr(app_module, "get_metadata_service", lambda: ReadyMetadata())
    monkeypatch.setattr("utils.milvus_utils.get_milvus_client", lambda: ReadyMilvus())
    monkeypatch.setattr(app_module, "peek_minio_client", lambda: None)
    monkeypatch.setattr(app_module, "peek_history_mongo_tool", lambda: None)
    monkeypatch.setattr(
        "utils.minio_utils.get_minio_client",
        lambda: (_ for _ in ()).throw(AssertionError("readiness initialized MinIO")),
    )

    result = app_module.readiness()

    assert result["dependencies"]["minio"] == "unavailable"
    assert result["dependencies"]["history_mongodb"] == "unavailable"


def test_minio_partial_initialization_failure_closes_client() -> None:
    module = _fresh_import("utils.minio_utils")

    class FailingClient:
        def __init__(self) -> None:
            self.closed = 0

        def bucket_exists(self, _bucket: str) -> bool:
            raise RuntimeError("bucket check failed")

        def make_bucket(self, _bucket: str) -> None:
            return None

        def set_bucket_policy(self, _bucket: str, _policy: str) -> None:
            return None

        def close(self) -> None:
            self.closed += 1

    client = FailingClient()
    module.reset_minio_client()

    with pytest.raises(RuntimeError, match="bucket check failed"):
        module.initialize_minio_client(client_factory=lambda **_kwargs: client)

    assert client.closed == 1
    assert module.peek_minio_client() is None


def test_history_mongo_constructor_failure_closes_client(monkeypatch) -> None:
    module = _fresh_import("utils.mongo_history_utils")

    class FailingCollection:
        @staticmethod
        def create_index(_index) -> None:
            raise RuntimeError("index creation failed")

    class FakeDatabase:
        def __getitem__(self, _name: str):
            return FailingCollection()

    class FakeClient:
        def __init__(self) -> None:
            self.closed = 0

        def __getitem__(self, _name: str):
            return FakeDatabase()

        def close(self) -> None:
            self.closed += 1

    client = FakeClient()
    monkeypatch.setattr(module, "MongoClient", lambda *_args, **_kwargs: client)

    with pytest.raises(RuntimeError, match="index creation failed"):
        module.HistoryMongoTool()

    assert client.closed == 1


def test_external_dependencies_initialize_concurrently(monkeypatch) -> None:
    app_module = _fresh_import("app")
    both_started = threading.Event()
    release = threading.Event()
    started = []
    lock = threading.Lock()

    def initializer(name: str) -> None:
        with lock:
            started.append(name)
            if len(started) == 2:
                both_started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(app_module, "initialize_minio_client", lambda: initializer("minio"))
    monkeypatch.setattr(app_module, "initialize_history_mongo_tool", lambda: initializer("history"))

    async def run_initialization():
        task = asyncio.create_task(app_module._initialize_external_dependencies(app_module.app))
        started_together = await asyncio.to_thread(both_started.wait, 1)
        release.set()
        return started_together, await task

    started_together, errors = asyncio.run(run_initialization())

    assert started_together is True
    assert errors == {}


def test_shutdown_waits_for_executor_before_closing_clients(monkeypatch) -> None:
    app_module = _fresh_import("app")
    calls = []

    class FakeExecutor:
        @staticmethod
        def shutdown(*, wait: bool, cancel_futures: bool) -> None:
            calls.append(("executor", wait, cancel_futures))

    monkeypatch.setattr(app_module, "executor", FakeExecutor())
    monkeypatch.setattr(app_module, "close_history_mongo_tool", lambda: calls.append("history"))
    monkeypatch.setattr(app_module, "close_minio_client", lambda: calls.append("minio"))

    asyncio.run(app_module._shutdown_external_dependencies())

    assert calls == [("executor", True, False), "history", "minio"]
