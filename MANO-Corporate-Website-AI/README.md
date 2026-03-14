# MANO Corporate Website AI

This folder contains the isolated AI logic (RAG Retrieval + LLM Execution) and knowledge processing scripts for the **MANO Projects Corporate Website**.

## Architecture & Integration

This module is **not** a standalone server. It is designed to be imported and executed by the master `Centralized-AI-Server` router. 

The master server handles all Express routing, CORS, and Port configuration (running on Port `5555`), and passes `/chat-corporate/*` requests down to `rag.js` in this folder.

### Core Files
- `rag.js` ← Core RAG logic: Embeds the question, searches the local `chroma_data/` DB, and queries Groq.
- `index_knowledge.py` ← Reads scraped `.txt` files, generates embeddings, and saves them to local Chroma DB.
- `../scripts/scrape_website.py` ← Dynamic Web Crawler that scrapes the live MANO website for updated knowledge.
- `requirements.txt` ← Python dependencies (`chromadb`, `playwright`, etc.) required to run the Scraper and Indexer.

---

## Updating the AI Knowledge Base

When the live website (e.g. Services, Projects, Careers) is updated, you must re-run the Python scripts to update the AI's brain. 

### 1) Install Python Dependencies (One-time)
```bash
# You must have python3 installed
pip install -r requirements.txt
playwright install
```

### 2) Run the Web Scraper
*(Make sure the `BASE_URL` inside `scripts/scrape_website.py` points to your live domain)*
```bash
# Run from the root of MANO-Corporate-Website-AI
python scripts/scrape_website.py
```
*This will crawl your site and output new text files to `knowledge_base/pages/`.*

### 3) Re-Index the Vector Database
```bash
# Optional: Chunk the data if needed, then index it
python backend/chunk_data.py
python backend/index_knowledge.py
```
*This converts the new text files into embeddings and saves them inside `chroma_data/`.*

### 4) Restart the Central Server
```bash
# Run from the root Centralized-AI-Server folder
pm2 restart all
# Or if running locally:
npm start
```
The AI will immediately begin using the newly scraped information!
