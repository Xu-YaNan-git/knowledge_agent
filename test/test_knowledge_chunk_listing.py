"""Checks that the knowledge view only exposes current, ready document chunks."""

from unittest.mock import Mock

from services.metadata_service import MetadataService


def test_chunk_search_is_literal_and_scoped_to_current_documents() -> None:
    service = MetadataService.__new__(MetadataService)
    service.documents = Mock()
    service.documents.distinct.return_value = ["doc-current"]
    service.chunks = Mock()
    cursor = Mock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = []
    service.chunks.find.return_value = cursor
    service.chunks.count_documents.return_value = 0

    result = service.list_chunks("kb-1", keyword="a.*", document_id="doc-old")

    query = service.chunks.find.call_args.args[0]
    assert query["knowledge_base_id"] == "kb-1"
    assert query["document_id"] == {"$in": []}
    assert query["$or"][0]["content"]["$regex"] == r"a\.\*"
    assert query["$or"][1]["title"]["$regex"] == r"a\.\*"
    assert result == {"items": [], "total": 0}


def test_chunk_stats_count_only_current_ready_documents() -> None:
    service = MetadataService.__new__(MetadataService)
    service.documents = Mock()
    service.documents.distinct.return_value = ["doc-1", "doc-2"]
    service.chunks = Mock()
    service.chunks.count_documents.return_value = 3
    cursor = Mock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = []
    service.chunks.find.return_value = cursor

    stats = service.get_kb_stats("kb-1")

    assert stats["total_documents"] == 2
    assert stats["total_chunks"] == 3
    assert service.chunks.count_documents.call_args.args[0] == {
        "knowledge_base_id": "kb-1", "document_id": {"$in": ["doc-1", "doc-2"]}
    }
