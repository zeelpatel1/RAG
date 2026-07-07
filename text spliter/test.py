from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter

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

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    # chunk_overlap=10
)

data=TextLoader("text spliter/note1.txt")
docs=data.load()

chunk=splitter.split_documents(docs)
for i in chunk:
    print(i.page_content)
    print()