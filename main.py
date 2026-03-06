import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
from pathlib import Path
import glob
import pdfplumber
import json
import pickle
import re
import argparse
from sentence_transformers.cross_encoder import CrossEncoder
from nltk import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet = True)


EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
RETRIEVAL_K = 20
RRF_K = 60 #constant used in the denominator of the Reciprocal Rank Fusion(RRF)
MAX_FILE_SIZE = 20 
CACHE_DIR = ".rag_cache"

parent = Path(__file__).resolve().parent.resolve().parent
file_path = parent/"legal contract analyzer/data/raw/CUAD_v1/CUAD_v1.json"

#Text extraciton from each file
def extract_text_from_pdf(pdf_path : str) -> str:

    fulltext = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages: 
            page_text = page.extract_text()
            if page_text:
                fulltext += page_text+"\n"
        
        return fulltext

#File Loading
def load_pdfs(pdf_dir : str, max_files : int = MAX_FILE_SIZE) ->  tuple[list[str], list[str]]:

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, '*.pdf')))[:max_files] #takes all the .pdf in the dir and joins it to the rest of the path, and sorts all the paths and fetches the first max_files entries 

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{pdf_dir}'.\n"
            f"Make sure your contract PDFs are in that folder."
        )
    print(f"Found {len(pdf_paths)} files, extracting text...")

    texts, sources = [], []

    for path in pdf_paths:

        filename = os.path.basename(path)
        print(f"Reading {filename}")
        text = extract_text_from_pdf(path)
        
        if text.strip():
            texts.append(text)
            sources.append(filename)
        else:
            print(f"        WARNING: Could not extract text from {filename} (may be scanned/image-based) — skipping.")

    print(f"Successfully extracted text from {len(texts)} files.")

    return texts, sources


#Cleaning the loaded text
def clean_text(text: str) -> str:

    text = re.sub(r'\n{3,}', '\n\n', text)   # collapse 3+ newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)   # collapse multiple spaces/tabs
    text = re.sub(r'\f', '\n', text)          # form feeds → newlines
    return text.strip()
        
def chunk_document(text : str, source : str, sentences_per_chunk : int = 14, sentence_overlap : int = 4) -> list[dict]:

    text = clean_text(text)
    sentences = sent_tokenize(text)
    start = 0
    chunks = []
    while True:

        if start >= len(sentences):
            break 
        #print(start, len(words))
        end = start + sentences_per_chunk
        chunk = sentences[start:end]
        chunk_text = " ".join(chunk)
        start = end - sentence_overlap

        enriched = f"Source: {source}/n/n {chunk_text}"

        chunks.append({'text':enriched, 'source': source})


    return chunks        
    

def build_chunks(texts: list[str], sources: list[str]) -> tuple[list[str], list[str]]:

    all_passages = []
    all_sources = []
    print("Entering chunk loop")
    for text, source in zip(texts, sources):

        chunks = chunk_document(text, source)

        all_passages.extend([c['text'] for c in chunks])
        all_sources.extend([c['source'] for c in chunks])
        print(f"{source}: {len(chunks)} chunks")

    print(f"\nTotal passages to index: {len(all_passages)}")

    return all_passages, all_sources


#embedding the passages[chunks]
def embed_passages(passages : list[str], model : SentenceTransformer) -> np.ndarray:

    print("Embedding the passages, might take a moment...")
    embeddings = model.encode(passages, 
                              batch_size=64, 
                              show_progress_bar=True, 
                              normalize_embeddings=True, 
                              convert_to_numpy=True
                            )
    
    print(f"Embedding shape : {embeddings.shape}")

    return embeddings


#BUILD FAISS[Facebook AI Similarity Search] INDICES
def build_index(embeddings: np.ndarray ) -> faiss.IndexFlatIP:

    #Inner Product(IP) is same as consine similarity when the vectors are L2 normalised which they are in this case, by the BGE model.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Indices created - {index.ntotal} indices of dimension {dim} each.")

    return index

#BUILD BM25 INDICES
def build_bm25_index(passages : list[str]) -> BM25Okapi:

    tokenized = [word_tokenize(p.lower()) for p in passages]  #one chunk at a time
    bm25 = BM25Okapi(tokenized)#both vectorization and indexing takes place here 
    print(f" BM25 index ready — {len(passages)} passages.")
    
    return bm25



#Retrieve the matching chunks
def retrieve(query: str, model: SentenceTransformer, reranker : CrossEncoder ,index: faiss.IndexFlatIP, bm25 : BM25Okapi,
             passages: list[str], sources: list[str], retrieval_k = RETRIEVAL_K, top_k: int = TOP_K, rrf_k : int = RRF_K ) -> list[dict]:
    
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}" #prepended the text specific to BGE to improve retrieval recall.

    query_embedded = model.encode(prefixed_query, normalize_embeddings=True, convert_to_numpy=True)

    faiss_scores, faiss_indices = index.search(x=query_embedded.reshape(1, -1), k=top_k)

    faiss_scores = faiss_scores[0]#because FAISS returns a 2d array[this is useful in batch processing, here we are passing only 1 query at a time, so we have only 1 top_k list.]
    #NOTE: indices returned by the faiss are sorted by default, need not sort again.
    faiss_hits = list(faiss_indices[0])#because FAISS returns a 2d array[this is useful in batch processing, here we are passing only 1 query at a time, so we have only 1 top_k list.]


    query_words = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(query_words)
    bm25_hits = np.argsort(bm25_scores)[::-1][:retrieval_k].tolist()

    #Reciprocal Rank fusion
    rrf_scores :dict[int,float] = {} #maps passage index -> cummulative rrf score

    for rank, idx in enumerate(faiss_hits,): #relative difference between scores is preserved even if enumerate starts from 0. We can change it to 1 if needed.
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0/(rrf_k+rank) #using the formula for rrf, summation_over_i(1/k+rank_i)

    for rank, idx in enumerate(bm25_hits): #relative difference between scores is preserved even if enumerate starts from 0. We can change it to 1 if needed.
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0/(rrf_k+rank) #using the formula for rrf, summation_over_i(1/k+rank_i)

    #returns a list not a dict
    merged_indices = sorted(rrf_scores, key= rrf_scores.get, reverse=True)[:retrieval_k] #sort by the combined scores not the indices[keys]

    candidates = [{
        'text' : passages[idx],
        'source' : sources[idx],
        'rrf_score' : rrf_scores[idx],
        'in_faiss' : idx in faiss_hits,
        'in_bm25' : idx in bm25_hits
    }
    for idx in merged_indices
    ]


    #CrossEncoder - Reranker

    pairs = [[query, c['text']] for c in candidates]
    reranked_scores = reranker.predict(pairs)
    
    for candidate, reranked_score in zip(candidates, reranked_scores):
        candidate['reranked_score'] = float(reranked_score)
        
    reranked = sorted(candidates, key= lambda x:x['reranked_score'], reverse=True)[:top_k]
    print(f"\n  Top {top_k} passages after reranking (from {retrieval_k} candidates):")
    for rank, r in enumerate(reranked):
        print(f"  [{rank+1}] Rerank: {r['reranked_score']:.4f} | RRF SCORE: {r['rrf_score']:.4f} | Source: {os.path.basename(r['source'])} | Preview: {r['text'][:80]}...")

    return reranked 

#Generate answer with GROQ:

def generate_answer(query: str, retrieved: list[dict], client: Groq) -> str:
    
    context_block = "\n\n---\n\n".join(
        [f"[Source: {r['source']}]\n{r['text']}" for r in retrieved]
    )

    system_prompt = (
        "You are a precise legal contract analyst. "
        "Answer the user's question using ONLY the contract passages provided. "
        "Ignore any retrieved passage that is not relevant to the question. "
        "Structure your answer as follows:"
        "1. A direct one-sentence answer to the question"
        "2. Supporting details as a numbered list if multiple values or contracts are involved"
        "Keep your answer concise — no more than 150 words unless the question explicitly requires more detail. "
        
        "NOTE: Never infer, assume, or hallucinate information not present in the passages. "
        "If the answer is not found say exactly: "
        "'I could not find this information in the provided contract passages.'"
    )

    

    user_prompt = f"""Contract Passages:
{context_block}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(model = GROQ_MODEL, messages=[{"role": "system", "content": system_prompt},
                                               {"role": "user", "content": user_prompt}], temperature=0.0, max_tokens=512)
    
    return response.choices[0].message.content

#Save the indices after first run:
def save_index( index, passages, source, bm_25, cache_dir : str = CACHE_DIR ):

    #Save the faiss indices, bm25 pkl file and passages, sources JSON file.
    os.makedirs(cache_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(cache_dir, "faiss.index"))
    
    #open and write to the bm25 pkl file, in binary format
    with open(os.path.join(cache_dir,'bm25.pkl'), 'wb') as f:
        pickle.dump(bm_25, f)
    with open(os.path.join(cache_dir, 'passages.json'), 'w') as f:
        json.dump({"passages" : passages, "sources" :source}, f)
    print(f"Indices saved to {cache_dir}/")

def load_index(cache_dir : str = CACHE_DIR):
    
    faiss_path = os.path.join(cache_dir, 'faiss.index')
    bm25_path  = os.path.join(cache_dir, 'bm25.pkl')
    passages_path = os.path.join(cache_dir, 'passages.json')

    if not all(os.path.exists(p) for p in [faiss_path, bm25_path, passages_path]):
        return None, None, None, None
    print(f"cache found in {cache_dir}/ - loading saved indices...")
    index = faiss.read_index(faiss_path)

    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    with open(passages_path, 'r') as f:
        passages = json.load(f)

    print(f"Loaded the vectors and passages successfully...")

    return index, bm25, passages['passages'], passages['sources']


#main function
def main():
    parser = argparse.ArgumentParser(description="RAG pipeline for local legal contract PDFs")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="./data",
        help="Path to folder containing your PDF contracts (default: ./contracts)"
    )

    #store_true makes just adding the --rebuild argument to change args.rebuild to TRUE, we would have to do --rebuild True or --rebuild 1 without the store_true as action.
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index even if cache exists")
    args = parser.parse_args()

    api_key = os.environ.get('GROQ_API_KEY')

    if not api_key:
        raise EnvironmentError("GROQ API Key not set in .env, please set it.")
    
    groq_client = Groq(api_key=api_key)

    #Trigger for --rebuild
    index, index_bm25, passages, chunk_sources = load_index() if not args.rebuild else (None, None, None, None)

    if index is None:
        print("No cache found, starting process from beginning...")

        texts, sources = load_pdfs(args.pdf_dir, 20)
        passages, chunk_sources = build_chunks(texts, sources)

        print("\n[*] Loading embedding model...")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME, device='cuda')

        embeddings = embed_passages(passages, embed_model)
        index = build_index(embeddings)
        index_bm25 = build_bm25_index(passages=passages)

        print("\n[*] Saving index to cache...")
        save_index(index, passages, chunk_sources, index_bm25)
    
    else: 

        print("Cached data present, loading just the embedding model...")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME, device='cuda')

    print("Loading the reranker model...")
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device='cuda')


    print("\n" + "=" * 55)
    print("  Pipeline ready! Ask questions about your contracts.")
    print("  Type 'quit' to exit.")
    print("=" * 55)
    print("\nExample questions:")
    print("  - What are the termination conditions in these contracts?")
    print("  - Which contracts mention a non-compete clause?")
    print("  - What is the governing law in these agreements?")
    print("  - Are there any indemnification clauses?\n")


    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Exiting. Good luck on your AI engineering journey!")
            break
        if not query:
            continue

        retrieved = retrieve(query, embed_model, reranker, index, index_bm25 ,passages, chunk_sources)

        print("\n[Generating answer with Groq...]\n")
        answer = generate_answer(query, retrieved, groq_client)

        print("─" * 55)
        print(f"ANSWER:\n{answer}")
        print("─" * 55 + "\n")

if __name__ == "__main__":
    main()


    

    














