from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter

#64
# splitter=CharacterTextSplitter(
#     separator="",
#     chunk_size=1000,
#     chunk_overlap=10
# )

#28
# splitter=TokenTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=10
# )

data=PyPDFLoader("text spliter/GRU.pdf")
docs=data.load()

chunk=splitter.split_documents(docs)
print(len(chunk))