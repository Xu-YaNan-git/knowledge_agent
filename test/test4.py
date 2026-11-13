from utils.embedding_utils import get_bge_m3_ef

model=get_bge_m3_ef()
embedding = model.encode_documents(["你好,hello world"])
print(embedding)