from processor.import_processor.node.g_node_import_milvus import NodeImportMilvus


def test_new_vectors_are_inserted_before_previous_version_is_deleted() -> None:
    node = NodeImportMilvus()
    call_order: list[str] = []
    client = object()
    node._step_2_prepare_collection = lambda _dimension: client

    def insert_data(_client, chunks):
        call_order.append("insert")
        chunks[0]["chunk_id"] = 1
        return chunks

    def clean_previous(_client, previous_document_id):
        call_order.append(f"delete:{previous_document_id}")

    node._step_4_insert_data = insert_data
    node._step_5_clean_previous_version = clean_previous
    state = {
        "knowledge_base_id": "kb-1",
        "document_id": "doc-v2",
        "previous_document_id": "doc-v1",
        "version": 2,
        "chunks": [{"dense_vector": [0.1], "sparse_vector": {1: 0.5}}],
    }

    result = node.process(state)

    assert call_order == ["insert", "delete:doc-v1"]
    assert result["chunks"][0]["document_id"] == "doc-v2"
    assert result["chunks"][0]["knowledge_base_id"] == "kb-1"
