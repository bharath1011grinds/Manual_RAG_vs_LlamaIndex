from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import os
import uuid


from main import (
    load_index,
    retrieve,
    generate_answer,
    EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME,
)
from groq import Groq

#Defining the models - schemas

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="The question to ask about your contracts",
        examples=["Which contract has a sponsorship fee of $750,000?"]
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of passages to retrieve (1-10, default 3)"
    )
    session_id: str = Field(default='',description=(
            "Session ID for conversation memory. "
            "Leave blank to start a new session — one will be created and returned. "
            "Pass the same session_id in follow-up questions to maintain history."
        ))

class RetrievedPassage(BaseModel):
    source:       str   # full file path of the source contract
    text:         str   # the retrieved chunk text
    rerank_score: float # cross-encoder relevance score
    rrf_score:    float # reciprocal rank fusion score
    in_faiss:     bool  # was this chunk retrieved by FAISS?
    in_bm25:      bool # was this chunk retrieved by bm25? 

class QueryResponse(BaseModel):
    question:   str                    # echoed back for clarity
    answer:     str                    # LLM generated answer
    passages:   list[RetrievedPassage] # retrieved chunks with scores
    model_used: str                    # LLM model name
    session_id : str #session id for chat context


#PIPELINE STATE:

class PipelineState:
    def __init__(self):
        self.embed_model:   SentenceTransformer | None = None #Field can take either a SentenceTransformer object or(union operator) a None, default value is None.
        self.reranker:      CrossEncoder | None        = None
        self.index:         object | None              = None
        self.bm25:          object | None              = None
        self.passages:      list[str] | None           = None
        self.chunk_sources: list[str] | None           = None
        self.groq_client:   Groq | None                = None
        # Conversation memory — maps session_id → list of messages
        # Each message is {"role": "user"|"assistant", "content": "..."}
        # This is in-memory only — clears on server restart.
        # For production you'd persist this in Redis or a database.
        self.conversations : dict[str, list[dict]] = {}

MAX_HISTORY_TURNS = 5 #keeps last 10 conversations as history, prevents context window overflow

state = PipelineState()

@asynccontextmanager
async def lifespan(app:FastAPI):

    #Startup
    print("[startup] Loading pipeline...")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "  Mac/Linux : export GROQ_API_KEY='your_key_here'\n"
            "  Windows   : set GROQ_API_KEY=your_key_here"
        )
    
    index, bm25, passages, chunk_sources = load_index()
    if index is None:
        raise RuntimeError(
            "No cache found. Run rag_pipeline.py first to build and save the index.\n"
            "  python rag_pipeline.py --pdf_dir ./contracts"
        )
    

    state.index         = index
    state.bm25          = bm25
    state.passages      = passages
    state.chunk_sources = chunk_sources
    state.groq_client   = Groq(api_key=api_key)

    print("[startup] Loading embedding model...")
    state.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[startup] Loading reranker...")
    state.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    print("[startup] Pipeline ready. Visit http://localhost:8000/docs")

    yield #Control passed on to FastAPI

    #Shutdown
    print("[shutdown] Cleaning up...")
    state.embed_model   = None
    state.reranker      = None
    state.index         = None
    state.bm25          = None
    state.passages      = None
    state.chunk_sources = None
    state.groq_client   = None


app = FastAPI(title="Legal Contract RAG API",
    description=(
        "Hybrid RAG pipeline for legal contract Q&A.\n\n"
        "**Stack:** BAAI/bge-small-en-v1.5 · FAISS · BM25 · RRF · "
        "cross-encoder/ms-marco-MiniLM-L-6-v2 · Llama 3.3 70b via Groq\n\n"
        "Run `python rag_pipeline.py --pdf_dir ./contracts` first to build the index."
    ),
    version="1.0.0",
    lifespan=lifespan)



#ENDPOINTS
@app.delete('/session/{session_id}', tags=['Memory'])
async def clear_session(session_id):
    #Clear the conversation history of a particular session, call this when the user wants to start a fresh conversation.
    if session_id in state.conversations:
        del state.conversations['session_id']
        return {"status" :"cleared", "session_id" : session_id}
    return {"status":"session not found", "session_id" : session_id}



@app.get('/health', tags=['health'])
async def health():

    #Check if pipeline variables have the values in them
    ready = all([
        state.embed_model is not None,
        state.reranker    is not None,
        state.index       is not None,
        state.bm25        is not None,
    ])

    return {
        "status":  "ready" if ready else "not ready",
        "passages_loaded": len(state.passages) if state.passages else 0
    }


@app.post('/query', response_model=QueryResponse, tags = ['RAG'], summary='Ask a question about your contract.')
async def query(request: QueryRequest):

    if state.index is None:
        raise HTTPException(status_code=503, detail = "Pipeline not ready!")
    try:
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        if session_id not in state.conversations:
            state.conversations[session_id] = []
        history = state.conversations[session_id]    

        if history:
            history_text = '\n'.join([f"{m['role']} : {m['content']}" for m in history[-6:]])
        else: 
            history_text = "None"

        retrieval_query = request.question
        rewrite_response = state.groq_client.chat.completions.create(model= 'llama-3.3-70b-versatile', 
                                                                     messages=[{'role': 'system',
                                                                                 'content':
                                                                                    "You are a query rewriter for a RAG system. "
                                                                                    "Given a conversation history and a follow-up question, "
                                                                                    "rewrite the follow-up into a single fully self-contained search query "
                                                                                    "that includes all necessary context from the history. "
                                                                                    "If the question is already self-contained, return it unchanged. "
                                                                                    "Return ONLY the rewritten query — no explanation, no preamble."
                                                                                    "DO NOT change anything and retun the same query if the Conversation history is None."
                                                            
                                                                                }, {'role':'user', 'content': f"Conversation history:\n{history_text}\n\nFollow-up question: {retrieval_query}\n\nRewritten query:"}])

        retrieval_query = rewrite_response.choices[0].message.content.strip()
        retrieved = retrieve(retrieval_query,state.embed_model, state.reranker, state.index, state.bm25, state.passages,
                             state.chunk_sources,top_k=request.top_k)
        
        #Generation with Memory
        context_block = "\n\n---\n\n".join(
                    [f"[Source: {r['source']}]\n{r['text']}" for r in retrieved]
                )

        system_with_context = (
            "You are a precise legal contract analyst. "
            "Answer the user's question using ONLY the contract passages provided. "
            "Ignore any retrieved passage that is not relevant to the question. "
            "Structure your answer as follows:\n"
            "1. A direct one-sentence answer to the question\n"
            "2. Supporting details as a numbered list if multiple values or contracts are involved\n"
            "Keep your answer concise — no more than 150 words unless the question requires more. "
            "Never infer, assume, or hallucinate information not present in the passages. "
            "If the answer is not found say: "
            "'I could not find this information in the provided contract passages.'"
            f"\n\nRetrieved contract passages for the current question:\n{context_block}"
        )
        messages = (
            [{"role": "system", "content": system_with_context}]
            + history[-MAX_HISTORY_TURNS * 2:]   # keep llast 20 items, 10 user query and 10 assistant responses.
            + [{"role": "user", "content": retrieval_query}]
        )

        response = state.groq_client.chat.completions.create(model= "llama-3.3-70b-versatile", messages=messages,
                                                             temperature=0.0, max_tokens=512)
        answer = response.choices[0].message.content

        #add the new request and response to history...
        history.append({
            "role": "user", 
            "content": f"{retrieval_query} (sources: {', '.join(set(r['source'] for r in retrieved))})"
        })        
        history.append({'role': 'assistant', 'content':answer})

        #BUild the response as per the model definition    
        passages = [RetrievedPassage(
                source=r["source"],
                text=r["text"],
                rerank_score=r["reranked_score"],
                rrf_score=r["rrf_score"],
                in_faiss=r["in_faiss"],
                in_bm25=r["in_bm25"]
        ) for r in retrieved]

        return QueryResponse(question = retrieval_query, answer = answer, passages = passages, model_used="llama-3.3-70b-versatile", session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))