from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rag import search_labels


app = FastAPI(title="Drug Encyclopedia API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # allow all methods, including OPTIONS
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class ContextItem(BaseModel):
    title: str
    text: str


class AskResponse(BaseModel):
    question: str
    contexts: list[ContextItem]


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    hits = search_labels(req.question, req.top_k)
    return AskResponse(
        question=req.question,
        contexts=[ContextItem(**h) for h in hits],
    )
