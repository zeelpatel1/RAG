from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model=MistralAIEmbeddings()

vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriver=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatMistralAI(model="mistal-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant. Use ONLY the provided context to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Context:
{context}
""",
        ),
        ("human", "{question}"),
    ]
)

print("Rag system created.")
print("press 0 to exit")

while True:
    query=input("You:")
    if query==0:
        break

    docs=retriver.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })

    res=llm.invoke(final_prompt)
    print(f"\n AI:{res.content}")