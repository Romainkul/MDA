from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

from rag import get_rag_chain, RAGRequest, RAGResponse

app = FastAPI(title="Enhanced Async RAG Assistant")


def rag_chain_depender():
    """
    Dependency injector for the RAG chain stored in app.state.
    Raises HTTPException if not initialized.
    """
    from fastapi import Request
    # Access the global FastAPI app state
    from fastapi import current_app
    chain = current_app.state.rag_chain
    if chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    return chain

@app.post("/rag", response_model=RAGResponse)
async def ask_rag(
    req: RAGRequest,
    rag_chain = Depends(rag_chain_depender)
):
    """
    Handle a RAG query. Uses session memory and the provided RAG chain.
    """
    # Invoke the chain with the named input
    result = await rag_chain.ainvoke({"question": req.query})
    return RAGResponse(answer=result["answer"])