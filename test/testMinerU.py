"""使用本地 MinerU 模型解析 PDF，并输出解析日志。"""

import os
import time
from pathlib import Path

from mineru.kit.common import save_parse_result
from mineru.parser import parse


def test_mineru(import_file_path: str, file_dir_path: str) -> None:
    """将 PDF 解析到指定目录；失败时抛出异常。"""
    input_path = Path(import_file_path).expanduser().resolve()
    output_dir = Path(file_dir_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件不是 PDF: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_source = os.environ.get("MINERU_MODEL_SOURCE")
    os.environ["MINERU_MODEL_SOURCE"] = "local"
    process_start_time = time.monotonic()
    try:
        result = parse(input_path, tier="basic", page_range="all")
        zip_path = output_dir / f"{input_path.stem}.zip"
        save_parse_result(result, zip_path, "zip")
    finally:
        if previous_source is None:
            os.environ.pop("MINERU_MODEL_SOURCE", None)
        else:
            os.environ["MINERU_MODEL_SOURCE"] = previous_source

    elapsed = time.monotonic() - process_start_time
    print(f"MinerU 成功解析 PDF，耗时: {elapsed:.2f}s；输出文件: {zip_path}")


if __name__ == "__main__":
    test_mineru(r"E:\华为擎云B530.pdf", r"E:\output")
