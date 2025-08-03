from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rag_agent import initialize_rag

# Initialize app and QA chain
app = FastAPI()
qa_chain = initialize_rag()

# CORS (important for allowing browser requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://127.0.0.1:5500"] if using Live Server in VS Code
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "College Admission Agent is running!"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("query")  # ✅ match the frontend
    if not query:
        return {"answer": "Please provide a valid query."}
    
    result = qa_chain.run(query)
    return {"answer": result}

