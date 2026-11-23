from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()

#Creating Vector embeddings for user query

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
                                                                    # from_documents() -> Documents se embeddings bana kar collection mein store karna
vector_store = QdrantVectorStore.from_existing_collection(          # from_existing_collection() ->	Pehle se bani collection se connect karna           
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

#taking user query and doing similarity search in the vector database for collecting revelent chunks of data from vector_db and can send it to llm.

user_query = input("Ask something: ")

search_results = vector_store.similarity_search(query=user_query)


context = "\n\n\n".join([
    f"""Page Content: {result.page_content}
Page Number: {result.metadata['page_label']}
File Location: {result.metadata['source']}"""
    for result in search_results
])

SYSTEM_PROMPT = f"""
 You are a helpfull AI assistant who answeres user query based on the available conetext 
 retrieved from a PDF file along with page_contents and page number.

 You should only answer the user based on the following context and navigate the user to open the right page number to know more.

 Context:
    {context}
"""
                                                                   
response = openai_client.chat.completions.create(                   
    model="gpt-5",
    messages=[
        {"role": "system", "content":SYSTEM_PROMPT },
        {"role": "user", "content":user_query }
    ]
)

print(f"🤖: {response.choices[0].message.content}")




# ------------------------------------------------------------



