import json
import os
import threading
from typing import Callable, Optional, TypeVar

from minio import Minio
from urllib3 import PoolManager, Retry, Timeout

from config.minio_config import minio_config


_Client = TypeVar("_Client")
_minio_client: Optional[object] = None
# Backward-compatible module attribute. New code should use the lifecycle helpers.
minio_client: Optional[object] = None
_minio_client_lock = threading.RLock()


def _default_minio_client_factory(**kwargs):
    timeout_seconds = max(1.0, float(os.getenv("MINIO_TIMEOUT_SECONDS", "10")))
    retries = max(0, int(os.getenv("MINIO_MAX_RETRIES", "2")))
    kwargs["http_client"] = PoolManager(
        timeout=Timeout(connect=timeout_seconds, read=timeout_seconds),
        retries=Retry(total=retries, connect=retries, read=retries, status=retries),
    )
    return Minio(**kwargs)


def _configure_buckets(client: object) -> None:
    """Create required buckets and apply the public image policy."""
    bucket_exists = getattr(client, "bucket_exists", None)
    make_bucket = getattr(client, "make_bucket", None)
    set_bucket_policy = getattr(client, "set_bucket_policy", None)

    # Factories used for lifecycle tests may return an opaque sentinel. Real
    # MinIO clients (and functional fakes) expose all three operations.
    if not all(callable(method) for method in (bucket_exists, make_bucket, set_bucket_policy)):
        return

    if not bucket_exists(minio_config.bucket_name):
        make_bucket(minio_config.bucket_name)
    if not bucket_exists(minio_config.document_bucket_name):
        make_bucket(minio_config.document_bucket_name)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"],
            }
        ],
    }
    set_bucket_policy(minio_config.bucket_name, json.dumps(policy))


def _close_client(client: object) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    http = getattr(client, "_http", None)
    clear = getattr(http, "clear", None)
    if callable(clear):
        clear()


def peek_minio_client() -> Optional[object]:
    """Return the cached client without initializing MinIO."""
    with _minio_client_lock:
        return _minio_client


def initialize_minio_client(client_factory: Callable[..., _Client] = _default_minio_client_factory) -> _Client:
    """Initialize and cache the MinIO client once.

    Construction and bucket setup happen while holding the lock, so concurrent
    callers cannot create duplicate clients. A failed attempt is never cached.
    """
    global _minio_client, minio_client

    with _minio_client_lock:
        if _minio_client is not None:
            return _minio_client  # type: ignore[return-value]

        client = client_factory(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=False,
        )
        try:
            _configure_buckets(client)
        except Exception:
            _close_client(client)
            raise
        _minio_client = client
        minio_client = client
        return client


def get_minio_client() -> Optional[object]:
    """Return the shared MinIO client, lazily initializing it if necessary."""
    try:
        return initialize_minio_client()
    except Exception as exc:
        print(f"Minio init failed:{exc}")
        return None


def reset_minio_client() -> None:
    """Clear the cached client so a later initialization can retry."""
    global _minio_client, minio_client

    with _minio_client_lock:
        _minio_client = None
        minio_client = None


def close_minio_client() -> None:
    """Release the cached MinIO client without initializing a new one."""
    global _minio_client, minio_client

    with _minio_client_lock:
        client = _minio_client
        _minio_client = None
        minio_client = None

    if client is None:
        return
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    http = getattr(client, "_http", None)
    clear = getattr(http, "clear", None)
    if callable(clear):
        clear()
