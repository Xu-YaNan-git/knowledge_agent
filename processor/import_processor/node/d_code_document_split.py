import re
from typing import Tuple, List, Dict

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from langchain_text_splitters import RecursiveCharacterTextSplitter


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        """
                节点：文档切分（node_document_split）
                整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
                核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
                后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

                必要参数：md_content、file_title
                更新参数：chunks

                :param state: 工作流状态对象
                :return: 更新后的状态对象
                """

        # 1.参数处理
        content, file_title = self._step_1_get_inputs(state)

        # 2.标题切（初切）
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)
        for section in sections:
            print(f"{section['title']}")
            print(f"{section['content']}")
            print("============================")
        print(f"标题数量：{title_count}")
        print(f"行数：{lines_count}")

        # 3.无标题兜底（默认标题）
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # 4.块精细化处理（长切结合）
        sections = self._step_4_refine_chunks(sections)

        # 5.打印日志
        self._step_5_print_stats(lines_count, sections)

        # 6.备份
        self._step_6_backup(state, sections)

        state["chunks"] = None
        return state

    # 1.参数处理
    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题
        """

        print("node_document_split:步骤1:参数处理")
        # 1、非空校验3/
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        content = state.get("md_content")
        if not content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        # 2、基础标准化：统一换行符
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        return content, file_title


        return state

    # 2.标题切（初切）
    def _step_2_split_by_titles(self, content, file_title):
        print("node_document_split:步骤2:标题切（初切）")

        #参数说明
        sections: List[Dict[str,str]]=[]
        title_count = 0
        lines=content.split("\n")
        current_lines=[]
        current_title=""
        in_code_block = False

        #切换逻辑（标题切）
        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        title_pattern = r'^\s*#{1,6}\s+.+'

        #独立封装刷新块的独立函数
        def _flush_section():
            if not current_lines:
                return
                # 封装section块的案例代码
            sections.append(
                {
                    "title": current_lines,
                    "content": "\n".join(current_lines),
                    "parent_title": "",
                    "file_title": file_title
                }
            )


        for line in lines:
            striped_line=line.strip()

            #判断是否在代码块中
            if striped_line.startswith("```") or striped_line.startswith("~~~"):
                in_code_block = True
                current_lines.append(line)
                continue

            if(not in_code_block) and re.match(title_pattern, line):
                #标题处理，将快内容封装成一个section块
                _flush_section()
                current_title=striped_line  #换标题
                current_lines=[current_title]
                title_count += 1

            else:
                current_lines.append(line)  #普通行或代码块

        _flush_section()
        print(f"标题初切：section :{sections}")
        return sections, title_count, len(lines)

    # 3.无标题兜底（默认标题）
    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        print("node_document_split:步骤3:无标题兜底（默认标题）")
        if title_count==0:
            return [{f"title":"无标题","content":content,"file_title":file_title}]
        return sections

    # 4.块精细化处理（长切结合）
    def _step_4_refine_chunks(self, sections):
        print("node_document_split:步骤4:块精细化处理（长切结合）")

        #长切结合
        refined_split=[]
        for sec in sections:
            refined_split.extend(self._split_long_section(sec))  #长切操作


        #短合列表
        final_sections = self._merge_short_sections(refined_split)
        self.logger.info(f"步骤4-2：过短章节合并完成，最终得到{len(final_sections)}个Chunk")

        # 阶段3：父标题兜底 → 适配Milvus向量库schema（parent_title为必填字段）
        # 兜底规则：无parent_title则用自身title，title也无则填空字符串
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""
        self.logger.debug(f"步骤4-3：父标题兜底完成，所有Chunk均包含parent_title字段")

        return sections

    # 5.打印日志
    def _step_5_print_stats(self, lines_count, sections):
        print("node_document_split:步骤5:打印日志")
        pass

    # 6.备份
    def _step_6_backup(self, state, sections):
        print("node_document_split:步骤6:备份")
        pass

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
        """
        # 内容空值兜底：无内容直接返回原章节
        content = section.get("content", "")
        # 长度未超限，无需切分，直接返回原章节（列表格式保持统一）
        if len(content) <= self.config.max_content_length:
            return [section]

        # 提取章节标题，用于组装子Chunk前缀（保留标题上下文）
        title = section.get("title", "")
        # 标题前缀：带空行分隔，与正文区分开
        prefix = f"{title}\n\n" if title else ""
        # 计算正文可用长度：总长度 - 标题前缀长度（避免标题占满Chunk额度）
        available_len = self.config.max_content_length - len(prefix)
        # 极端情况：标题长度超过阈值，无法切分，返回原章节
        if available_len <= 0:
            self.logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
            return [section]

        # 清理正文重复标题：避免原章节中正文开头重复标题，导致子Chunk内容冗余
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()

        # 初始化LangChain递归分割器（核心工具：按优先级分隔符切分，保留语义）
        # separators：分割符优先级（从粗到细），优先按大语义单元切分，最后才硬拆
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,  # 正文部分最大长度（已扣除标题）
            chunk_overlap=0,           # 无重叠：按标题切分后语义完整，无需重叠
            # 分割符优先级：空行(段落)→换行→中文标点→英文标点→空格，最后硬拆（在 chunk_size 位置强制切断）
            # 先用第一个分隔符进行切分，切分后如果某个 Chunk 还是超过 chunk_size，则继续用下一个优先级的分隔符切分
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        # 切分正文并组装子章节（带完整元信息，便于溯源）
        sub_sections = []

        # 遍历切分后的每个文本块，idx 从 1 开始计数
        for idx, chunk in enumerate(splitter.split_text(body), start=1):

            # 清理空内容：跳过切分后的空字符串
            text = chunk.strip()
            if not text:
                continue

            # 组装子Chunk完整内容 = 标题前缀 + 切分后的正文
            full_text = (prefix + text).strip()

            # 子章节元信息：保留父级关联，添加序号，便于后续检索/溯源
            sub_sections.append({
                "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # 子Chunk标题（带序号）
                "content": full_text,                                     # 切分后的完整内容
                "parent_title": title,                                    # 父章节标题（用于后续合并）
                "part": idx,                                              # 子Chunk序号
                "file_title": section.get("file_title"),                  # 所属文件标题
            })

        self.self.logger.debug(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")
        return sub_sections


    #短合
    def _merge_short_sections(self, refined_split):
        return refined_split


if __name__ == "__main__":
    node = NodeDocumentSplit()

    with open("E:\output\H3C\H3C_new.md", "r", encoding="utf-8") as file:
        md_content = file.read()

    init_state={
        "md_path":"E:\output\H3C\H3C_new.md",
        "md_content":md_content,
        "file_title":"H3C_new",
    }
    process = node.process(init_state)
    print(f"切分即诶单执行流程:{process}")