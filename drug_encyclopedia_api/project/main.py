# # main.py

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from routers import prescription, homeopathic

# app = FastAPI(title="Drug Information API")

# # CORS
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

# # Include routers
# app.include_router(prescription.router)
# app.include_router(homeopathic.router)

# from rag import index_data

# @app.on_event("startup")
# def startup_event():
#     index_data()

# @app.get("/")
# def root():
#     return {"message": "Drug Information API running"}


# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import prescription, homeopathic
from routers import drug_resolver  # NEW: import the drug resolver router

app = FastAPI(title="Drug Information API")

# CORS (unchanged)
origins = ["*"
    # "http://localhost:5173",
    # "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prescription.router)
app.include_router(homeopathic.router)
app.include_router(drug_resolver.router)  # NEW: register drug resolver endpoints

from rag import index_data

@app.on_event("startup")
def startup_event():
    index_data()

@app.get("/")
def root():
    return {"message": "Drug Information API running"}