# VectorDB — Build a Vector Database from Scratch in Python

A fully working **Vector Database** built from scratch in Python (Flask) with a web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

You need **3 things** installed on your system:

1. **Python 3.8+** (with pip)
2. **Git**
3. **Ollama** (runs the local AI models)

---

## Step-by-Step Setup

### Step 1 — Install Python
1. Download and install Python from the official site (https://www.python.org/downloads/) or your package manager.
2. Make sure `python` and `pip` are added to your system's PATH.
3. Verify in your terminal/PowerShell:
   ```bash
   python --version
   pip --version
   ```

### Step 2 — Install Git
1. Install Git from https://git-scm.com/downloads.
2. Verify in your terminal:
   ```bash
   git --version
   ```

### Step 3 — Install Ollama (Local AI Models)
1. Go to https://ollama.com and download/install the version for your OS.
2. Make sure Ollama is running (it usually runs in the background / system tray).
3. Open a terminal and pull the two required models:
   ```bash
   ollama pull nomic-embed-text
   ```
   *(~274 MB — this is the embedding model)*
   ```bash
   ollama pull llama3.2
   ```
   *(~2 GB — this is the language model)*
4. Verify they are available:
   ```bash
   ollama list
   ```

### Step 4 — Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/shahanasheikh12/MY-OWN-AI-CHATBOT.git
cd MY-OWN-AI-CHATBOT
```

### Step 5 — Install Python Dependencies
Install the required packages using pip:
```bash
pip install flask numpy requests
```

### Step 6 — Run the Application
1. Make sure Ollama is running.
2. Start the Python database and Flask server:
   ```bash
   python main.py
   ```
3. You should see:
   ```
   === VectorDB Engine (Python Flask Port) ===
   http://localhost:8080
   20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
   Ollama: ONLINE
     embed model: nomic-embed-text  gen model: llama3.2
   ```
4. Open your web browser and navigate to:
   `http://localhost:8080`

---

## Using the Application

### Tab 1: Search (Demo Vectors)
- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

**The scatter plot** shows all 20 vectors projected to 2D using PCA. Notice how the 4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)
This uses Ollama to generate **real 768-dimensional embeddings** from any text.
1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)
1. Make sure you have inserted some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:
1. Your question → embedded with `nomic-embed-text` (768D vector)
2. HNSW search → finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to `llama3.2`
4. `llama3.2` → generates an answer based only on your documents

---

## REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

---

## Project Structure

```
MY-OWN-AI-CHATBOT/
├── main.py        ← Python backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── index.html     ← Frontend (PCA scatter plot, chat UI, benchmark)
└── README.md      ← This file
```

### Architecture (main.py)

```
BruteForce          O(N·d)      Exact, baseline
KDTree              O(log N)    Exact, axis-aligned partitioning
HNSW                O(log N)    Approximate, multilayer small-world graph

VectorDB            Unified interface over all 3 (16D demo vectors)
DocumentDB          HNSW-only index for real Ollama embeddings (768D)
OllamaClient        HTTP client → /api/embeddings + /api/generate
```

---

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)
Nodes are inserted into a multilayer graph. Each node randomly gets assigned a maximum layer. Layer 0 has all nodes with many connections; higher layers have fewer nodes (exponentially fewer) with longer-range connections.

- **Insert:** Start at the top layer, greedily find the nearest node, drop a layer, repeat. At each layer from your assigned max down to 0, run a beam search (`ef_construction=200`) and connect to the M nearest neighbors bidirectionally.
- **Search:** Same greedy descent from top layer. At layer 0, expand to ef nearest candidates using a priority queue.

*Why it's fast:* The upper layers act like a highway — you quickly get to the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)
Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree can't beat the current best — the "ball within hyperslab" check.

*Weakness:* Degrades with high dimensions (curse of dimensionality). Works well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions
KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all the space is near the boundary of the hypersphere — no subtrees get pruned. HNSW's graph-based approach doesn't have this problem.

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` in a terminal or start the Ollama desktop app |
| Embedding takes forever | Ollama is downloading the model on first use, wait a few minutes |
| `python: command not found` | Install Python and add it to your system's environment PATH variable |
| `ModuleNotFoundError` | Install the required dependencies: `pip install flask numpy requests` |
| Port 8080 already in use | Kill the process running on port 8080 or specify another port in `main.py` |
| LLM answer is slow | Normal on CPUs. To speed it up, switch to a smaller model like `llama3.2:1b` in `main.py` |

---

## License
MIT — use this however you want.
