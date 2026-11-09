import logging


from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.node.a_node_entry import NodeEntry
from processor.import_processor.node.b_node_pdf_to_md import NodePDFToMD
from processor.import_processor.node.c_node_md_img import NodeMDImg
from processor.import_processor.node.d_code_document_split import NodeDocumentSplit
from processor.import_processor.node.e_node_item_reconition import NodeItemNameRecognition
from processor.import_processor.node.f_node_bge_embdding import NodeBGEEmbedding
from processor.import_processor.node.g_node_import_milvus import NodeImportMilvus
from processor.import_processor.state import ImportGraphState



class KBImportWorkflow:

    def __init__(self):
        """
        初始化工作流
         """
        self._compiled_graph = None


    @property
    def graph(self):
        """
        返回图实例
        :return:
        """
        logging.info("获取图实例")

        if self._compiled_graph is None:
            self._compiled_graph=self.build_graph()  #船舰图
        return self._compiled_graph

    @staticmethod
    def route_after_entry(state: ImportGraphState) -> str:
        """
        入口节点后的条件路由函数
        :param state: 当前状态
        :return: 下一个节点名称
        """
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "node_md_img"
        else:
            logging.info("未指定导入文件类型")
            return END

    def build_graph(self):
        """
        创建主图
        :return:
        """
        graph=StateGraph(ImportGraphState)

        #1.注册节点
        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        # 3. 设置入口节点
        graph.set_entry_point("node_entry")
        #2.节点边
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                # key：路由函数的返回值，value:节点的名字
                "node_md_img": "node_md_img",
                "node_pdf_to_md": "node_pdf_to_md",
                END: END
            }
        )

        # 5. 注册顺序边
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        #3.编译图
        return graph.compile()


    def run(self,state: ImportGraphState,stream:bool=False):

        if stream:
            return self.graph.stream(state,stream_mode="values")
        else:
            return self.graph.invoke(state)



if __name__ == "__main__":
    # 启用日志
    setup_logging()

    workflow = KBImportWorkflow()
    init_state={
        "import_file_path": r"E:\H3C.pdf"
    }
    for event in workflow.run(init_state, stream=True):
        print(f"state: {event}")

    # final_state = workflow.run(init_state, stream=False)
    # print(json.dumps(final_state,ensure_ascii=False, indent=4))
