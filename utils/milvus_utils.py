import json
import os
import sys
import time
from urllib.parse import urlsplit

from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest, connections

from config.milvus_config import milvus_config
from tool.logger import logger

_milvus_client = None


def ensure_milvus_no_proxy(uri: str) -> None:
    """让直连的 Milvus 地址绕过工作站代理，保留已有代理例外。"""
    hostname = urlsplit(uri).hostname
    if not hostname:
        return
    bypass_hosts = [value.strip() for value in os.environ.get("NO_PROXY", "").split(",") if value.strip()]
    if hostname not in bypass_hosts:
        os.environ["NO_PROXY"] = ",".join([*bypass_hosts, hostname])


def build_metadata_filter(item_names=None, knowledge_base_ids=None):
    """Build an escaped Milvus filter for optional product and knowledge-base scopes."""
    conditions = []
    valid_item_names = [value.strip() for value in (item_names or []) if isinstance(value, str) and value.strip()]
    valid_kb_ids = [value.strip() for value in (knowledge_base_ids or []) if isinstance(value, str) and value.strip()]
    if valid_item_names:
        conditions.append(f"item_name in {json.dumps(valid_item_names, ensure_ascii=False)}")
    if valid_kb_ids:
        conditions.append(f"knowledge_base_id in {json.dumps(valid_kb_ids, ensure_ascii=False)}")
    return " and ".join(conditions) or None
def get_milvus_client():
    """
    获取全局单例的MilvusClient对象
    :return: MilvusClient实例
    """
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    ensure_milvus_no_proxy(milvus_config.milvus_url)
    _milvus_client = MilvusClient(uri=milvus_config.milvus_url)

    return _milvus_client


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算，避免一个得分特别大、另一个特别小导致权重失效。
        # 版本：V2.4
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None


def test_milvus_connection(timeout: int = 10) -> bool:
    """
    测试 Milvus 连接是否正常（用于诊断网络/配置问题）

    Args:
        timeout: 连接超时时间（秒）

    Returns:
        bool: 连接是否成功
    """
    print("=" * 60)
    print("开始测试 Milvus 连接")
    print(f"MILVUS_URL = {milvus_config.milvus_url}")
    print(f"超时设置   = {timeout} 秒")
    print("=" * 60)

    start_time = time.time()

    try:
        # 使用项目统一的 MilvusClient
        print("\n[1/3] 正在创建 MilvusClient ...")
        ensure_milvus_no_proxy(milvus_config.milvus_url)
        client = MilvusClient(uri=milvus_config.milvus_url)

        print("[2/3] 正在测试连接（list_collections）...")
        collections = client.list_collections()
        print(f"✓ 连接成功！当前集合数量: {len(collections)}")

        if collections:
            print(f"  部分集合: {collections[:5]}")

        # 底层健康检查
        print("\n[3/3] 执行底层连接健康检查 ...")
        connections.connect(
            alias="test_conn",
            uri=milvus_config.milvus_url,
            timeout=timeout
        )
        connections.disconnect("test_conn")
        print("✓ 底层连接测试通过")

        elapsed = time.time() - start_time
        print(f"\n✅ Milvus 连接测试成功！耗时: {elapsed:.2f} 秒")
        logger.info(f"Milvus 连接测试成功，耗时 {elapsed:.2f} 秒")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Milvus 连接失败！")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print(f"耗时: {elapsed:.2f} 秒")
        logger.error(f"Milvus 连接测试失败: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    # 支持命令行传入超时时间
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    success = test_milvus_connection(timeout=timeout)
    sys.exit(0 if success else 1)
