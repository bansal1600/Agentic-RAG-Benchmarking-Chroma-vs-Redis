from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

# Use the environment variable if set, otherwise default to localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")
index_name = "rag-redis-index"

# Connect to existing Redis vector store
vectorstore = RedisVectorStore(
    redis_url=REDIS_URL,
    index_name=index_name,
    embeddings=OpenAIEmbeddings(),
)

# Create retriever with similarity search
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)