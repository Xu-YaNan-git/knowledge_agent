
"""
    工程环境
"""
conda create -n py311 python=3.11
conda activate py311
python -m venv kb311
.\Scripts\activate

"""
    pytorch
"""
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 (5090)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 (5090)
cuda开发工具：https://developer.nvidia.com/cuda-12-4-0-download-archive
检查pytorch状态：python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

测试：
import torch
print(torch.cuda.is_available())  # 如果输出 True，说明成功认出显卡！
print(torch.version.cuda)         # 这会显示你安装的 PyTorch 自带的 CUDA 版本（比如 12.1）

"""
    mineru
"""
官方：https://opendatalab.github.io/MinerU/zh/
官方：https://mineru.net/apiManage/docs
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
pip install uv -i https://mirrors.aliyun.com/pypi/simple
uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple
测试： mineru --version

"""
    mineru模型
"""
模型缓存：$env:MODELSCOPE_CACHE="D:\ai_models\modelscope_cache"
下模型：mineru-models-download
桌面客户端：https://mineru.net/
测试：mineru -p xxx.pdf -o output --backend pipeline

"""
    milvus
"""
pip uninstall -y pymilvus milvus-model FlagEmbedding
uv add pymilvus[model] transformers FlagEmbedding
pip install "pymilvus[model]" transformers FlagEmbedding

"""
    安装BGE-M3
"""
uv add pymilvus[model] transformers FlagEmbedding
pip install "pymilvus[model]" transformers FlagEmbedding
pip install modelscope
执行脚本：python download_bgem3.py
"""
    from modelscope.hub.snapshot_download import snapshot_download
    # 下载模型到当前目录下的 models/bge-m3 文件夹
    model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:/ai_models/modelscope_cache/models')
    print(f"模型已下载到: {model_dir}")
"""
测试：python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
万一测试不可用：pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 (仅限于5090)



"""
    fastapi
"""
pip install fastapi
pip install "uvicorn[standard]"



"""
    清华镜像
"""
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

"""
    若安装minio会报错，升级版本
"""
python -m pip install --upgrade pip


knowledge/
├── api/                              # API 路由层
│   ├── query_router.py              # 查询服务路由 (port 8001)
│   │   ├── POST /query              # 发起查询
│   │   ├── GET /stream/{session_id} # SSE 流式获取
│   │   ├── GET /history/{session_id}# 获取历史
│   │   └── DELETE /history/...      # 清除历史
│   └── import_router.py             # 导入服务路由 (port 8000)
│       ├── POST /upload             # 上传文件
│       └── GET /status/{task_id}    # 查询任务状态
│
├── core/                             # 核心配置
│   ├── deps.py                      # 依赖注入（单例管理）
│   └── paths.py                     # 路径常量配置
│
├── processor/                        # 业务处理流程（LangGraph）
│   ├── import_processor/               # 导入流程
│   │   ├── base.py           		 # 导入节点基类
│   │   ├── config.py           	  # 导入流程配置管理
│   │   ├── exceptions.py             # 导入流程自定义异常
│   │   ├── main_graph.py           # 导入流程图定义
│   │   ├── state.py                # 状态类型定义
│   │   └── nodes/                  # 处理节点
│   │       ├── node_entry.py            # 入口节点
│   │       ├── node_pdf_to_md.py        # PDF 转 MD
│   │       ├── node_md_img.py           # 图片处理
│   │       ├── node_document_split.py   # 文档切分
│   │       ├── node_item_name_recognition.py  # 商品识别
│   │       ├── node_bge_embedding.py    # 向量嵌入
│   │       └── node_import_milvus.py    # Milvus 存储
│   │
│   └── query_processor/              	# 查询流程
│       ├── base.py           		    # 查询节点基类
│       ├── config.py           	    # 查询流程配置管理
│       ├── exceptions.py               # 查询流程自定义异常
│       ├── main_graph.py           	# 查询流程图定义
│       ├── state.py                	# 状态类型定义
│       ├── prompt.py               	# 提示词模板
│       └── nodes/                  	# 处理节点
│           ├── node_item_name_confirm.py    # 商品确认
│           ├── node_vector_search.py        # 向量检索
│           ├── node_hyde_search.py          # HyDE 检索
│           ├── node_web_search_mcp.py       # Web 搜索
│           ├── node_rrf.py                  # RRF 融合
│           ├── node_rerank.py               # 重排序
│           └── node_answer_output.py        # 答案生成
│
├── schema/                          # 数据模型定义
│   ├── query_schema.py              # 查询请求/响应模型
│   ├── upload_schema.py             # 上传响应模型
│   └── task_schema.py               # 任务状态模型
│
├── services/                        # 业务服务层
│   ├── file_import_service.py       # 文件导入服务
│   └── task_service.py              # 任务管理服务
│
├── utils/                           # 工具函数库
│   ├── milvus_utils.py              # Milvus 向量库操作
│   ├── embedding_utils.py           # 向量嵌入工具
│   ├── llm_utils.py                 # LLM 客户端封装
│   ├── reranker_utils.py            # Reranker 模型
│   ├── mongo_history_utils.py       # MongoDB 历史记录
│   ├── sse_utils.py                 # SSE 流式推送
│   ├── task_utils.py                # 任务状态管理
│   └── minio_utils.py               # MinIO 对象存储
│
├── front/                           # 前端页面
│   ├── chat.html                    # 聊天界面
│   └── import.html                  # 导入界面
│
├── test/                             # 测试代码
├── docs/                             # 文档目录
├── temp_data/                        # 临时数据目录
├── .env                              # 环境配置文件
├── .env.example                      # 环境配置文件模板
├── pyproject.toml                    # Python 依赖声明
└── uv.locl							# uv版本锁定文件

"""
    笔记
"""
扫描后的图片：List[Tuple[str,str,Tuple[str,str]]]
扫描后的图片：list[(图片名，图片路径,(图片上文，图片下文))]


"""
    可视化控制台
"""
# 在项目根目录启动，浏览器访问 http://127.0.0.1:8000
\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000

# 页面功能：
# 1. 智能问答与会话历史
# 2. LangGraph 节点执行轨迹
# 3. 精排资料、引用和图片展示
# 4. PDF / Markdown 知识入库

## 2.0 管理与评测能力

本版本在原有 RAG 查询和导入流程上新增：

应用入口会自动加载项目根目录 `.env`，但不会覆盖操作系统、容器或部署平台已经注入的同名环境变量。

- 知识库 CRUD：`/api/knowledge-bases`
- 文档上传、列表、版本查询和安全删除：`/api/knowledge-bases/{id}/documents`、`/api/documents/{id}`
- 安全更新：相同文件名形成版本，新向量写入成功后才清理旧版本；相同哈希直接判重
- 持久化任务：查询、导入和评测任务保存在 MongoDB，服务重启后恢复未完成任务
- 失败重试：`POST /api/tasks/{task_id}/retry`
- RAG 评测：上传 JSONL 到 `POST /api/evaluations`，结果通过 `/api/evaluations` 查询
- 就绪检查：`/api/health/ready` 会检查 MongoDB 和 Milvus
- 会话管理：`/api/sessions` 支持最近会话、重命名和级联删除消息
- 实时进度：`/api/tasks/{task_id}/events` 通过 SSE 推送任务节点和最终结果，断线可回退轮询
- 任务控制：`POST /api/tasks/{task_id}/cancel` 支持安全取消查询/评测及尚未启动的导入任务
- 文档失败重试：`POST /api/documents/{document_id}/retry` 复用原持久化任务和上传文件
- 引用详情：本地引用保留知识库、文档、章节和切片信息，前端可展开查看完整原文
- 任务租约：后台任务定期续租，只有租约过期的处理中任务才会被其他实例恢复
- 可观测性：所有动态请求返回 `X-Request-ID`，日志记录状态码、耗时和慢请求
- 运行面板：`/api/operations/summary` 与 `/api/tasks` 提供任务、文档、会话和评测概况
- 临时文件治理：完成或取消任务的临时文件按 `TEMP_FILE_RETENTION_DAYS` 自动清理
- 本地账号认证：PBKDF2-SHA256 密码哈希、HttpOnly 登录 Cookie 和 HMAC-SHA256 签名令牌
- 三级权限：`admin` 管理账号和高风险删除，`editor` 管理知识与评测，`viewer` 只读并可发起问答
- 安全审计：知识、账号、任务和会话等写操作追加保存到 `audit_logs`，管理员可通过 `/api/audit-logs` 按操作、结果、人员和时间检索
- 评测基线：`PATCH /api/evaluations/{id}/baseline` 将同一数据集的一次运行设为当前用户基线
- 回归门禁：`GET /api/evaluations/{id}/comparison` 对比通过率、关键词召回、引用有效率和平均延迟，默认允许质量指标下降 5%、延迟增加 25%
- 资源配额：问答、文档入库和评测使用 MongoDB 原子计数进行跨实例限流，超限返回 `429`、`Retry-After` 和 `X-RateLimit-*` 响应头；`GET /api/usage` 可查询当前用量
- 原文归档：每个文档版本的原始 PDF/Markdown 保存到独立私有 MinIO Bucket，可通过 `/api/documents/{id}/download` 下载；重试时本地临时文件缺失会自动从归档恢复并校验 SHA-256
- 知识库导出：`POST /api/knowledge-bases/{id}/exports` 通过持久化任务生成包含 `manifest.json` 和各版本归档原文的 ZIP，完成后通过 `/api/exports/{task_id}/download` 下载

审计记录只保存操作标识、资源标识、结果、请求 ID、客户端地址和操作者，不记录密码、令牌、问题正文或上传内容。应用不提供审计记录修改和删除接口。

生产环境启用认证前，至少配置：

```env
AUTH_ENABLED=true
AUTH_SECRET=长度至少32位的随机密钥
AUTH_COOKIE_SECURE=true
BOOTSTRAP_ADMIN_USERNAME=admin
BOOTSTRAP_ADMIN_PASSWORD=长度至少10位的强密码
```

首次启用且用户表为空时会创建引导管理员；创建成功后可以删除环境中的明文引导密码。

资源配额默认值可通过以下配置调整，设置为 `0` 可关闭对应限制：

```env
QUERY_RATE_LIMIT_PER_MINUTE=30
IMPORT_RATE_LIMIT_PER_HOUR=20
EVALUATION_RATE_LIMIT_PER_HOUR=10
EXPORT_RATE_LIMIT_PER_HOUR=10
```

评测数据集每行一个 JSON 对象：

```json
{"case_id":"product-001","question":"产品有哪些配置？","expected_keywords":["内存","硬盘"],"knowledge_base_ids":["可选知识库ID"]}
```

当前基线指标包括关键词召回率、引用编号有效率、通过率和平均耗时。样例见
`evaluation/sample_dataset.jsonl`。生产评测建议维护 30～100 条经过人工确认的黄金用例。

开发测试依赖安装：

```powershell
uv sync --group dev
uv run pytest -q test/test_evaluation_metrics.py
```
