import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

#Confifure the inference model
Settings.llm = Groq(model = 'llama-3.3-70b-versatile', api_key = os.getenv('GROQ_API_KEY'))

#Configure the embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name= 'BAAI/bge-small-en-v1.5')
Settings.chunk_size = 500
Settings.chunk_overlap = 50

#Load the docs
docs = SimpleDirectoryReader('data').load_data()
print(f"Documents loaded {len(docs)} pages")

#Chunking + embedding + storing
index = VectorStoreIndex.from_documents(docs)

#create query engine

query_engine = index.as_query_engine(similarity_top_k=5)

qa = ({
    "q": "Who are the two parties in this agreement and what are their roles?",
    "a": "Whitesmoke Inc. is the Distributor, Google Inc. is the provider of Distribution Products"
},
{
    "q": "When did this agreement come into effect and when does it end?",
    "a": "Effective 1 August 2011, ends 31 July 2013 (2 year term) or when Maximum Distribution Commitment is reached"
},
{
    "q": "What law governs this agreement?",
    "a": "English law, with exclusive jurisdiction of English courts"
},
{
    "q": "Under what conditions can Google terminate this agreement immediately?",
    "a": "If Distributor breaches License Grants, EULA, Accurate Reproduction, or Confidentiality clauses; violates Anti-Bribery Laws; or is in material breach more than a specified number of times"
},
{
    "q": "What are the Distributor's obligations when an end user wants to uninstall the application?",
    "a": "Must permit uninstall in the standard OS location (e.g. Add/Remove Programs), provide clear instructions on how to uninstall, and leave no functionality or setting changes behind after uninstall"
}
)
questions =[]

for item in qa:
    questions.append(item['q'])





while True:
    query = input("Your question: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        print("Exiting. Good luck on your AI engineering journey!")
        break
    response = query_engine.query(query)

    print("─" * 55)
    print(f"answer :{response}")
    print("─" * 55 + "\n")











