from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

model=ChatMistralAI(model="mistral-small-2506")
res=model.invoke("hello")
print(res.content)