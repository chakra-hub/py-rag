from repository.vector_database import VectorRepository
from repository.bm25_repository import BM25Repository
from repository.semantic_cache import SemanticCacheRepository
from services.ask_llm import AskLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict

vector_db = VectorRepository()
ask_llm = AskLLM()
bm25_db = BM25Repository()
cache_repo = SemanticCacheRepository()


class AgentState(TypedDict):
    question: str
    query: str
    chunks: list[str]
    is_relevant: bool
    rewrite_attempts: int
    answer: str
    session_id:str


def hybrid_retrieve(state: AgentState) -> AgentState:
    vector_results = vector_db.query_text_raw(state["query"])
    bm25_results = bm25_db.query_raw(state["query"])
    seen = set()
    combined = []
    for chunk in vector_results + bm25_results:
        if chunk not in seen:
            seen.add(chunk)
            combined.append(chunk)
    state["chunks"] = combined[:3]
    return state


def grade_retrieval(state: AgentState) -> AgentState:
    if not state["chunks"]:
        state["grade_retrieval"] = False
        return state

    context = "\n\n".join(state["chunks"])
    prompt = f"""Question: {state['question']}
    Retrieved context:
{context}

Is this context relevant enough to answer the question?
Reply with only YES or NO."""
    
    response = ask_llm.model.invoke(prompt)
    answer = response.content.strip().upper()
    is_relevant = "YES" in answer
    state["is_relevant"] = is_relevant
    return state


def agent_call(state: AgentState) -> AgentState:
    context = "\n\n---\n\n".join(state["chunks"])
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I cannot find this information."

Context:
{context}

Question: {state['question']}"""
    response = ask_llm.model.invoke(prompt)
    state["answer"] = response.content
    return state


def rewrite_query(state: AgentState) -> AgentState:
    prompt = f"""The search query "{state['query']}" did not find relevant information.

Rewrite it to be more specific and likely to find the answer to: "{state['question']}"
Return only the rewritten query, nothing else."""
    response = ask_llm.model.invoke(prompt)
    new_query = response.content.strip()
    state["query"]=new_query
    state['rewrite_attempts']=state["rewrite_attempts"]+1
    return state

def no_info(state: AgentState) -> AgentState:
    state["answer"]= "I could not find relevant information in the document to answer this question."
    return state

def decide_after_grading(state:AgentState) -> str:
    if state["is_relevant"]:
        return 'generate'
    elif state["rewrite_attempts"] >= 2:
        return 'no_info'
    else:
        return 'rewrite'
def build_rag_graph():   
    graph = StateGraph(AgentState)

    graph.add_node('retrieval', hybrid_retrieve)
    graph.add_node('grading', grade_retrieval)
    graph.add_node('rewrite', rewrite_query)
    graph.add_node('agent', agent_call)
    graph.add_node('no_info', no_info)
    graph.set_entry_point('retrieval')


    graph.add_edge('retrieval', "grading")

    graph.add_conditional_edges('grading',decide_after_grading,{
        "generate": "agent",
        "rewrite": "rewrite",
        "no_info": "no_info"
    })

    graph.add_edge("rewrite","retrieval")
    graph.add_edge("agent", END)
    graph.add_edge("no_info", END)

    return graph.compile()

rag_graph = build_rag_graph()


