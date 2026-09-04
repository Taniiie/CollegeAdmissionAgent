# College Admission Agent

A Retrieval-Augmented Generation (RAG) chatbot that answers college admission questions — eligibility, fees, courses, and deadlines — by grounding a Gemini LLM in a college prospectus PDF. It exposes a simple `/ask` API and a minimal HTML chat UI.

## Problem statement

College admission information is often scattered across lengthy, hard-to-search prospectus documents, leaving students overwhelmed and unsure where to find accurate answers. This project centralizes that information into a single AI assistant that answers admission-related questions instantly, in plain language.

## How it works

1. The prospectus PDF (`data/Prospectus2025-26.pdf`) is loaded and split into text chunks.
2. Each chunk is embedded with Google's `embedding-001` model and stored in a FAISS vector index.
3. A user question is sent to the `/ask` API endpoint.
4. LangChain's `RetrievalQA` chain retrieves the most relevant chunks and passes them to a Gemini chat model (`gemini-pro`), which generates a grounded answer.
5. The answer is returned as JSON and rendered in the chat UI.

## Tech stack

- **Backend:** Python, [FastAPI](https://fastapi.tiangolo.com/)
- **RAG / orchestration:** [LangChain](https://www.langchain.com/)
- **Vector store:** [FAISS](https://github.com/facebookresearch/faiss)
- **LLM & embeddings:** Google Generative AI (Gemini `gemini-pro`, `embedding-001`) via `langchain-google-genai`
- **PDF loading:** `PyPDFLoader` (`langchain_community`)
- **Frontend:** Plain HTML/CSS/JavaScript (`index.html`), calling the backend with `fetch`

## Project structure

```
CollegeAdmissionAgent/
├── main (1).py     # FastAPI app — exposes GET / and POST /ask
├── rag_agent.py     # Builds the RAG pipeline (PDF load, embed, FAISS, QA chain)
├── index.html       # Minimal chat UI that calls the /ask endpoint
├── env              # Environment variable file (should define GOOGLE_API_KEY)
└── data/            # Expected location for the prospectus PDF (not included)
    └── Prospectus2025-26.pdf
```

## Setup & running locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/Taniiie/CollegeAdmissionAgent.git
   cd CollegeAdmissionAgent
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn python-dotenv langchain langchain-community langchain-google-genai faiss-cpu pypdf
   ```

3. **Add your Google API key**
   Create a `.env` file in the project root with:
   ```
   GOOGLE_API_KEY=your_google_generative_ai_api_key
   ```

4. **Add the source document**
   Place your college prospectus PDF at `data/Prospectus2025-26.pdf` (create the `data/` folder if needed), or update the path in `rag_agent.py` to point to your own document.

5. **Rename and start the backend**
   ```bash
   # main (1).py needs to be run as a module — rename it first, e.g.:
   mv "main (1).py" main.py
   uvicorn main:app --reload --port 8000
   ```

6. **Open the frontend**
   Open `index.html` directly in your browser (it calls `http://127.0.0.1:8000/ask`), or serve it locally:
   ```bash
   python3 -m http.server 5500
   ```

## API

**`GET /`**
Health check — returns `{"message": "College Admission Agent is running!"}`

**`POST /ask`**
Request body:
```json
{ "query": "What is the eligibility for B.Tech?" }
```
Response:
```json
{ "answer": "..." }
```

## Notes / known gaps

- The filename `main (1).py` (with a space and parentheses) isn't a valid Python module name — rename it to `main.py` before running with `uvicorn main:app`.
- The `env` file should likely be named `.env` (a leading dot) for `python-dotenv`'s default `load_dotenv()` to pick it up automatically.
- The `data/Prospectus2025-26.pdf` source document is not included in the repo and must be supplied separately.
- CORS is currently open to all origins (`allow_origins=["*"]`) — restrict this before deploying publicly.
