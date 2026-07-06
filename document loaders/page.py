from langchain_community.document_loaders import WebBaseLoader

url="https://docs.langchain.com/oss/python/integrations/document_loaders"

data=WebBaseLoader(url)
docs=data.load()
print(docs)