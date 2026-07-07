from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

data=TextLoader("document loaders/notes.txt")
docs=data.load()

print(docs)