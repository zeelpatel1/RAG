from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data=PyPDFLoader("DeepLearning.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks=splitter.split_documents(docs)

template=ChatPromptTemplate.from_messages(
    [("system","you are a AI that summerizes the text"),("human","{data}")]
)

prompt=template.invoke({
    "data": docs
})

model=ChatMistralAI(model="mistral-small-2506")

res=model.invoke(prompt)
print(res.content)