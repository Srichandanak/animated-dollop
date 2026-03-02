# # main.py

# from fastapi import FastAPI
# from pydantic import BaseModel
# from rag import query_drug

# app = FastAPI(title="Drug Information API")

# class QueryRequest(BaseModel):
#     query: str

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def root():
#     return {"message": "Drug Information API running"}


# @app.post("/ask")
# def ask_drug_info(request: QueryRequest):
#     response = query_drug(request.query)
#     return response


# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import prescription, homeopathic

app = FastAPI(title="Drug Information API")

# CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prescription.router)
app.include_router(homeopathic.router)

from rag import index_data

@app.on_event("startup")
def startup_event():
    index_data()

@app.get("/")
def root():
    return {"message": "Drug Information API running"}
