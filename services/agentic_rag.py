from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from repository.bm25_repository import BM25Repository
from repository.semantic_cache import SemanticCacheRepository
from repository.vector_database import VectorRepository
from services.ask_llm import AskLLM
from services.retrieval.reranker import CrossEncoderReranker

vector_db = VectorRepository()
bm25_db = BM25Repository()
cache_repo = SemanticCacheRepository()
ask_llm = AskLLM()
reranker = CrossEncoderReranker()


class AgentState(TypedDict):
    question: str
    query: str
    chunks: list[Document]
    is_relevant: bool
    rewrite_attempts: int
    answer: str
    session_id: str


def hybrid_retrieve(state: AgentState) -> AgentState:
    query = state["query"]

    vector_results = vector_db.query_text_raw(
        query,
        n_results=50,
    )

    bm25_results = bm25_db.query_raw(
        query,
        n_results=50,
    )

    seen = set()
    combined = []

    for chunk in vector_results + bm25_results:
        key = (
            chunk.metadata.get("doc_id"),
            chunk.metadata.get("chunk_index"),
        )

        if key not in seen:
            seen.add(key)
            combined.append(chunk)

    combined = reranker.rerank(
        query=query,
        documents=combined,
        top_k=5,
    )

    state["chunks"] = combined

    return state


def grade_retrieval(state: AgentState) -> AgentState:

    if not state["chunks"]:
        state["is_relevant"] = False
        return state

    context = "\n\n".join(
        doc.page_content
        for doc in state["chunks"]
    )

    prompt = f"""
Question:
{state["question"]}

Retrieved context:
{context}

Is this context sufficient to answer the question?

Reply ONLY with YES or NO.
"""

    response = ask_llm.invoke(prompt)

    if not response.success:
        state["answer"] = response.error
        state["is_relevant"] = False
        return state

    state["is_relevant"] = (
        "YES" in response.content.upper()
    )

    return state


def agent_call(state: AgentState) -> AgentState:

    context = "\n\n---\n\n".join(
        doc.page_content
        for doc in state["chunks"]
    )

    prompt = f"""
Answer the question using ONLY the context below.

If the answer is not in the context, say
"I cannot find this information."

Context:
{context}

Question:
{state["question"]}
"""

    response = ask_llm.invoke(prompt)

    if not response.success:
        state["answer"] = response.error
        return state

    state["answer"] = response.content

    return state


def rewrite_query(state: AgentState) -> AgentState:

    prompt = f"""
The search query "{state['query']}" did not find relevant information.

Rewrite it to be more specific and likely to find the answer to:

"{state['question']}"

Return ONLY the rewritten query.
"""

    response = ask_llm.invoke(prompt)

    if not response.success:
        state["answer"] = response.error
        state["is_relevant"] = False
        return state

    state["query"] = response.content.strip()
    state["rewrite_attempts"] += 1

    return state


def no_info(state: AgentState) -> AgentState:
    if not state.get("answer"):
        state["answer"] = (
            "I could not find relevant information in the document to answer this question."
        )
    return state


def decide_after_grading(state: AgentState) -> str:

    if state.get("answer"):
        return "no_info"

    if state["is_relevant"]:
        return "generate"

    if state["rewrite_attempts"] >= 2:
        return "no_info"

    return "rewrite"


def build_rag_graph():

    graph = StateGraph(AgentState)

    graph.add_node("retrieval", hybrid_retrieve)
    graph.add_node("grading", grade_retrieval)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("agent", agent_call)
    graph.add_node("no_info", no_info)

    graph.set_entry_point("retrieval")

    graph.add_edge("retrieval", "grading")

    graph.add_conditional_edges(
        "grading",
        decide_after_grading,
        {
            "generate": "agent",
            "rewrite": "rewrite",
            "no_info": "no_info",
        },
    )

    graph.add_edge("rewrite", "retrieval")
    graph.add_edge("agent", END)
    graph.add_edge("no_info", END)

    return graph.compile()


rag_graph = build_rag_graph()