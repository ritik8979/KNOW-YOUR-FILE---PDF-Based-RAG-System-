# this file is resonsible for indexing phase of the data.
from dotenv import load_dotenv

from pathlib import Path                                                #pdf ke path ko handle karne ke liye.
from langchain_community.document_loaders import PyPDFLoader            #PDF file ko load karna aur uske andar ka text extract karna.Kiyoki Python ko PDF ke andar ka text automatically samajh nahi aata.
from langchain_text_splitters import RecursiveCharacterTextSplitter     #Chunkings- Iska kaam hai bade text ko chhote-chhote pieces mein todna.In chhote pieces ko hum chunks kehte hain.
from langchain_openai import OpenAIEmbeddings                           #Ye line OpenAI ke embedding model ko use karne ke liye hai.LangChain → OpenAI model ko call karne ka interface (OpenAIEmbeddings) provide karta hai.
from langchain_qdrant import QdrantVectorStore                          #Ye Qdrant database ke saath interact karne ke liye hai.
                                                                        #Qdrant-> Qdrant ek Vector Database hai.Iska kaam tumhare vectors aur unse related original text ko store karna aur similarity search karna hai.
load_dotenv()

pdf_path = Path(__file__).parent / "history.pdf"                        #__file__-->Ye current Python file ka path batata hai.
                                                                        #.parent ka matlab hai current file jis folder ke andar hai, uska parent folder.
                                                                        #Yahan result hoga:    /Users/ritik/Desktop/Udemy Ai/rag
                                                                        #Path(__file__).parent / "history.pdf"=====/Users/ritik/Desktop/Udemy Ai/rag/history.pdf
                                                                  
# loading this file pdf in the current python program
loader = PyPDFLoader(file_path=pdf_path)                            #Yahan tum PyPDFLoader class ka object bana rahe ho.
docs = loader.load()                                                # With the help of object of Pypdfloader created above- gives page by page pdf docs/pdf broked into pages and every page is given one by one to docs variable.
                                                                    #Maan lo tumhari PDF mein 3 pages hain.
                                                                    # docs = [
                                                                    #             Document(page_content="Page 1 ka text...", metadata={...}),
                                                                    #             Document(page_content="Page 2 ka text...", metadata={...}),
                                                                    #             Document(page_content="Page 3 ka text...", metadata={...})
                                                                    #         ]

# print(docs[10])

#CHUNCKING DATA OF PDF - Split the docs into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(                         #text_splitter = tumhara splitter object.
    chunk_size=1000,                                                    #chunk sixe defining.
    chunk_overlap=400                                                   #Ek chunk ka kuch text next chunk mein repeat hoga.Maan lo ek sentence ka pehla part Chunk 1 mein hai aur doosra part Chunk 2 mein.Agar overlap nahi hoga toh sentence ka context toot sakta hai.Isse context maintain karne mein help milti hai.
    )
chunks = text_splitter.split_documents(documents=docs)

#Creating Vector embeddings for these chunks

embedding_model = OpenAIEmbeddings(                                     #Ab tumne embedding model ka object banaya.OpenAIEmbeddings class ka object.
    model="text-embedding-3-large"
)
                                                                        #QdrantVectorStore object hai .. aur vector_store -> Ye ek variable hai jisme QdrantVectorStore ka object store hota hai.
vector_store = QdrantVectorStore.from_documents(                        #QdrantVectorStore.from_documents -> Ye LangChain ka method hai.Is method ko tum documents dete ho aur ye unhe embeddings ke saath Qdrant Vector Database mein store karne ka kaam karta hai.
    documents=chunks,                                                   #jo pdf ke chote chote chunks(parts) banaye the unhe store kro.
    embedding=embedding_model,                                          #QdrantVectorStore ko batana ki chunks ki embeddings banane ke liye kaunsa model use karna hai. Is stage pr embeddings create hui hai.
    url="http://localhost:6333",                                        #Tumhare Python program ko batata hai ki Qdrant se kahan connect karna hai jo docker ke through run ho rhi hai.
    collection_name="learning_rag"                                      #Qdrant mein data collections ke andar organize hota hai.Collection ko tum ek folder ya table ki tarah imagine kar sakte ho.
)

print("Indexing of documents done....")





# -----------------------------------------------------------------------------------

