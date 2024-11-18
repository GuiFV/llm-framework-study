import os
from sqlalchemy import create_engine
from llama_index.core import download_loader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.database import DatabaseReader
from llama_index.core import StorageContext, load_index_from_storage

"""regular DB"""

# reader = DatabaseReader(
#     scheme=os.getenv("DB_SCHEME"),
#     host=os.getenv("DB_HOST"),
#     port=os.getenv("DB_PORT"),
#     user=os.getenv("DB_USER"),
#     password=os.getenv("DB_PASS"),
#     dbname=os.getenv("DB_NAME"),
# )

"""SQLite3"""

# Specify the SQLite database file inside the 'dbs' folder
db_path = os.path.join("..", "dbs", "db.sqlite3")

# Construct the SQLite connection URI
uri = f'sqlite:///{db_path}'

# Create SQLAlchemy engine
engine = create_engine(uri)

reader = DatabaseReader(
    engine=engine,
)

query = "SELECT * FROM core_project"

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

PERSIST_DIR = "../storage"

def add_db_data_to_documents():
    documents = reader.load_data(query=query)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    for doc in documents:
        index.insert(doc)

if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = reader.load_data(query=query)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # ONLY UNCOMMENT BELOW IF YOU WANT TO ADD (INDEX) DB TO DOCUMENTS
    # add_db_data_to_documents()
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


# ADD TO INDEX DOCS ADDED LATER ON FOLDER/API ENDPOINT/ETC
# index = VectorStoreIndex([])
# for doc in documents:
#     index.insert(doc)

query_engine = index.as_query_engine()
response = query_engine.query(
    "What projects are related to twitch.tv?"
)
print(response)
