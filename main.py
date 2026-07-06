from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data=PyPDFLoader("document loaders/GRU.pdf")
docs=data.load()

template=ChatPromptTemplate.from_messages(
    [("system","you are a AI that summerizes the text"),("human","{data}")]
)

prompt=template.invoke({
    "data": docs
})

model=ChatMistralAI(model="mistral-small-2506")

res=model.invoke(prompt)
print(res.content)