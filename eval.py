import os
import re
import json
import time
import argparse
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from main import (
    load_index, build_chunks, load_pdfs,
    embed_passages, build_index, build_bm25_index,
    retrieve, EMBED_MODEL_NAME, RERANKER_MODEL_NAME,
    CACHE_DIR, RETRIEVAL_K, TOP_K
)

#NOTE: the eval code is ready to be used but the eval set creation left pending because its tedious and boring. We were able to test manually on a few samples tho. 
EVAL_SET = [
    {
        "question":    "Which contract has a sponsorship fee of $750,000.00?",
        "expected":    "GAINSCOINC",
        "source_hint": "GAINSCOINC",          # fill in partial filename e.g. "STAMPS"
    },
    {
        "question":    "What does the Force Majure clause of Canopetroleum agreement say?.",
        "expected":    "",          
        "source_hint": "",
    },
    {
        "question":    "What are the termination conditions in the Stamps contract?",
        "expected":    "",
        "source_hint": ["stamps", "intuit"],
    },
    {
        "question":    "Which contracts mention a non-compete clause?",
        "expected":    "",
        "source_hint": "",
    },
    {
        "question":    "What is the sponsorship fee per contract year in the Violin Memory agreement?",
        "expected":    "4,000,000",
        "source_hint": "VIOLIN",
    },
]

''' NOTE: hit_rate and mrr are re-generation(retrieval) metrics that shows the quality of the retrieval in place,
 while exact_match and llm_judge are post-generation metrics that tells the quality of the whole pipeline
 
 '''

#Below function checks if the correct source was identified in the top_k chunks, returns 1 if it was, 0 if not, -1 if there was not source hint in the eval set for that sample
#retrieved - top_k chunks, source_hint - partial filename of the source.
def hit_rate(retrieved : list[dict], source_hint : str, expected : str) -> int:

    if not source_hint or not expected:
        return -1
    
    for r in retrieved:
        if (source_hint.lower() in r['source'].lower()) and (expected.lower() in r['text'].lower()):
            return 1
    return 0

#Mean Recriprocal Rate - how high is the right chunk ranked? Also checks if the chunk is from the right source.
def mrr(retrieved : list[dict], source_hint : str, expected : str) -> float:

    if not source_hint or not expected:
        return -1.0
    
    for rank,r in enumerate(retrieved, start=1):
        if (source_hint.lower() in r['source'].lower()) and (expected.lower() in r['text'].lower()):
            return (1.0/rank)
    return 0.0

#Post Generation metric that checks if the answer contains the expected text.
#returns 1 if match is present in the answer, 0 otherwise
def exact_match(answer : str, expected: str) -> int:
    
    if not expected:
        return -1
    return 1 if expected.lower() in answer.lower() else 0

def llm_judge(question : str, answer : str, context : str, client : Groq) -> int:
    #Use groq llm to score our answer from 1-5

    prompt = f"""You are evaluating the quality of an answer produced by a RAG system.

Question: {question}

Retrieved Context:
{context}

Answer:
{answer}

Score the answer from 1 to 5 using these criteria:
1 - Completely wrong or not grounded in the context
2 - Mostly wrong but contains some relevant info
3 - Partially correct, missing key details
4 - Mostly correct with minor gaps
5 - Accurate, fully grounded in context, well cited

Respond with ONLY a single integer between 1 and 5. No explanation.

    """
    response = client.chat.completions.create(model = "llama-3.3-70b-versatile", messages={'role': 'user', 'content':prompt},
                                              temperature=0.0, max_tokens=5)
    
    raw = response.choices[0].message.content.strip()

    #Extract the first digit in the response
    score = re.findall(r"1-5", raw)
    return int(score[0]) if score else 0
        
#has the same config as llamaindex rag, could have just used that code itself
def build_llamaindex_pipeline(pdf_dir : str, groq_api_key: str):

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.groq import Groq as LlamaGroq

    print("Loading LlamaIndex model...")
    documents = SimpleDirectoryReader(pdf_dir).load_data()

    Settings.embed_model =HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = LlamaGroq(model = "llama-3.3-70b-versatile", api_key=groq_api_key)
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50

    index = VectorStoreIndex.from_documents(documents) #Creating the indices
    query_engine = index.as_query_engine(similarity_top_k = TOP_K)  #Initializing the query engine with the indices

    print("LlamaIndex ready...")

    return query_engine


#Eval for manual Pipeline:
def run_manual_eval(eval_set, embed_model, reranker, index, bm25, passages, chunk_sources, groq_client):

    results = []
    for item in eval_set:
        q = eval_set['question']
        expected = eval_set['expected']
        source_hint = eval_set['source_hint']

        start = time.time()
        retrieved = retrieve(q, embed_model, reranker, index, bm25, passages, chunk_sources)
        latency = time.time() - start()

        #Context block for LLM judge
        context = "\n\n".join([r['text'] for r in retrieved])

        from main import generate_answer
        answer = generate_answer(q, retrieved, groq_client)

        #compute the metrics
        hr = hit_rate(retrieved, source_hint, expected)
        mean_rr = mrr(retrieved ,source_hint, expected)
        em =exact_match(answer, expected)
        judge = llm_judge(q, answer, context, groq_client)

        results.append({
            "question":    q,
            "answer":      answer,
            "hit_rate":    hr,
            "mrr":         mean_rr,
            "exact_match": em,
            "llm_judge":   judge,
            "latency_s":   round(latency, 2),
        })

    return results


def llamaindex_eval(eval_set, query_engine, groq_client):
    results = []
    for item in eval_set:
        q           = item["question"]
        expected    = item["expected"]
        source_hint = item["source_hint"]

        start    = time.time()
        response = query_engine.query(q)
        latency  = time.time() - start
        #this is done because, llamaindex returns a response object that contains the text, metadata and the source_nodes(chunks from which answer was created i.e. retrieved chunks)
        answer = str(response) #wrapping it with str() returns just the text part.
        context = "\n\n".join([n.node.text for n in response.source_nodes])

        #Build a fake list of source and text for the metric functions to use. Its called fake because, 
        # response.source_nodes is originally not a list of dict but a NodewithScore object.
        #NOTE: Its called fake, cos we are synthetically creating this list, nothing else is fake about it. It retrieves all the chunks that were passed as context to the LLM.
        retrieved_fake = [{'source' : n.node.metadata.get('file_path',''), 'text': n.text} for n in response.source_nodes]

        hr    = hit_rate(retrieved_fake, source_hint, expected)
        mrr_  = mrr(retrieved_fake, source_hint, expected)
        em    = exact_match(answer, expected)
        judge = llm_judge(q, answer, context, groq_client)

        results.append({
            "question":    q,
            "answer":      answer,
            "hit_rate":    hr,
            "mrr":         mrr_,
            "exact_match": em,
            "llm_judge":   judge,
            "latency_s":   round(latency, 2),
        })

    return results
    

#Print Summary function

def print_summary(manual_results, llamaindex_results):

    #Defining an average function to use for each metric in the upcoming lines...
    def avg(results, key):
        vals = [r['key'] for r in results if r['key'] != -1]
        return round(sum(vals)/len(vals), 3) if vals else "N/A"
    
    print("\n" + "=" * 65)
    print(f"  {'METRIC':<25} {'MANUAL RAG':>15} {'LLAMAINDEX':>15}")
    print("=" * 65)
    metrics = [
        ("Hit Rate",    "hit_rate"),
        ("MRR",         "mrr"),
        ("Exact Match", "exact_match"),
        ("LLM Judge /5","llm_judge"),
        ("Avg Latency s","latency_s"),
    ]

    for label, key in metrics:
        m = avg(manual_results, key)
        l = avg(llamaindex_results, key)
        print(f"{label:<25} {str(m):>15} {str(l):>15}") #alignment syntax, refer google if needed.
    print("=" * 65)

    #save full results in a json for furthur exploration
    outputs ={'manual_results': manual_results, 'llamaindex_results' : llamaindex_results}

    with open('eval_results.json', 'w') as f:
        json.dump(outputs, f, indent=2)
    print("\n  Full results saved to eval_results.json")

def main():
    parser = argparse.ArgumentParser(description="Evaluate manual RAG vs LlamaIndex")
    parser.add_argument('--pdf_dir',type=str, default='./data')
    args = parser.parse_args()

    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
   
    groq_client = Groq(api_key=api_key)

    #Load the manual RAG from cache folder
    print("=" * 65)
    print("  RAG Evaluation Framework")
    print("=" * 65)

    index, bm25, passages, chunk_sources = load_index(CACHE_DIR)
    if index is None:
        raise RuntimeError(
            "No cache found. Run main.py first to build and save the index."
        )

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    #Build LlamaIndex Pipeline
    llamaindex_qe = build_llamaindex_pipeline(args.pdf_dir, api_key)

    #Run the Evals:
    print("Evaluating manual pipeline...")
    manual_results = run_manual_eval(EVAL_SET, embed_model, reranker, index, bm25, passages, chunk_sources, groq_client)

    print("Evaluating LlamaIndex pipeline...")
    llamaindex_results = llamaindex_eval(EVAL_SET, llamaindex_qe, groq_client)


    #Print the summary:
    print_summary(manual_results, llamaindex_results)


if __name__ == '__main__':
    main()