import math
import random
import threading
import time
from flask import Flask, request, jsonify, make_response
import numpy as np
import requests

app = Flask(__name__)

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.sqrt(np.sum((a - b) ** 2)))

def cosine(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na = np.sum(a * a)
    nb = np.sum(b * b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    dot = np.sum(a * b)
    # Cosine distance is 1.0 - cosine similarity
    return float(1.0 - dot / (np.sqrt(na) * np.sqrt(nb)))

def manhattan(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.sum(np.abs(a - b)))

def get_dist_fn(metric):
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items = []

    def insert(self, item):
        self.items.append(item)

    def knn(self, q, k, dist_fn):
        res = []
        for item in self.items:
            d = dist_fn(q, item["emb"])
            res.append((d, item["id"]))
        res.sort(key=lambda x: x[0])
        return res[:k]

    def remove(self, id):
        self.items = [item for item in self.items if item["id"] != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, dims):
        self.dims = dims
        self.root = None

    def insert(self, item):
        def _ins(node, item, d):
            if not node:
                return KDNode(item)
            ax = d % self.dims
            if item["emb"][ax] < node.item["emb"][ax]:
                node.left = _ins(node.left, item, d + 1)
            else:
                node.right = _ins(node.right, item, d + 1)
            return node
        self.root = _ins(self.root, item, 0)

    def knn(self, q, k, dist_fn):
        import heapq
        heap = []  # max-heap storing (-dist, id)

        def _knn(node, d):
            if not node:
                return
            dn = dist_fn(q, node.item["emb"])
            if len(heap) < k:
                heapq.heappush(heap, (-dn, node.item["id"]))
            elif dn < -heap[0][0]:
                heapq.heappushpop(heap, (-dn, node.item["id"]))

            ax = d % self.dims
            diff = q[ax] - node.item["emb"][ax]
            closer = node.left if diff < 0 else node.right
            farther = node.right if diff < 0 else node.left

            _knn(closer, d + 1)
            if len(heap) < k or abs(diff) < -heap[0][0]:
                _knn(farther, d + 1)

        _knn(self.root, 0)
        res = [(-dist, id) for dist, id in heap]
        res.sort(key=lambda x: x[0])
        return res

    def rebuild(self, items):
        self.root = None
        for item in items:
            self.insert(item)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    def __init__(self, item, max_lyr):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs = [[] for _ in range(max_lyr + 1)]

class HNSW:
    def __init__(self, m=16, ef_build=200):
        self.G = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self.random = random.Random(42)

    def rand_level(self):
        u = self.random.random()
        if u == 0:
            u = 1e-9
        return int(math.floor(-math.log(u) * self.mL))

    def search_layer(self, q, ep, ef, lyr, dist_fn):
        import heapq
        vis = {ep}
        d0 = dist_fn(q, self.G[ep].item["emb"])
        
        cands = [(d0, ep)] # min-heap of (dist, id)
        found = [(-d0, ep)] # max-heap of (-dist, id)

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.G.get(cid)
            if not node or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist_fn(q, self.G[nid].item["emb"])
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)
                        
        res = [(-d, id) for d, id in found]
        res.sort(key=lambda x: x[0])
        return res

    def select_nbrs(self, cands, max_m):
        return [cid for dist, cid in cands[:max_m]]

    def insert(self, item, dist_fn):
        id_ = item["id"]
        lvl = self.rand_level()
        self.G[id_] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(item["emb"], ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self.search_layer(item["emb"], ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self.select_nbrs(W, max_m)
            self.G[id_].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                node_n = self.G[nid]
                if len(node_n.nbrs) <= lc:
                    while len(node_n.nbrs) <= lc:
                        node_n.nbrs.append([])
                conn = node_n.nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ds = []
                    for c in conn:
                        if c in self.G:
                            ds.append((dist_fn(self.G[nid].item["emb"], self.G[c].item["emb"]), c))
                    ds.sort(key=lambda x: x[0])
                    node_n.nbrs[lc] = [c for d, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q, k, ef, dist_fn):
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self.search_layer(q, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]
        W = self.search_layer(q, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, id_):
        if id_ not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_pt == id_:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id_:
                    self.entry_pt = nid
                    break
        del self.G[id_]

    def get_info(self):
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for id_, nd in self.G.items():
            nodes.append({
                "id": id_,
                "metadata": nd.item["metadata"],
                "category": nd.item["category"],
                "maxLyr": nd.max_lyr
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges
        }

    def __len__(self):
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE
# =====================================================================

class VectorDB:
    def __init__(self, dims):
        self.dims = dims
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.next_id = 1

    def insert(self, meta, cat, emb, dist_fn):
        with self.lock:
            id_ = self.next_id
            self.next_id += 1
            item = {"id": id_, "metadata": meta, "category": cat, "emb": emb}
            self.store[id_] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            return id_

    def remove(self, id_):
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q, k, metric, algo):
        with self.lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn(q, k, 50, dist_fn)

            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                if id_ in self.store:
                    item = self.store[id_]
                    hits.append({
                        "id": id_,
                        "metadata": item["metadata"],
                        "category": item["category"],
                        "distance": d,
                        "embedding": item["emb"]
                    })
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q, k, metric):
        with self.lock:
            dist_fn = get_dist_fn(metric)

            def time_fn(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            bf_us = time_fn(lambda: self.bf.knn(q, k, dist_fn))
            kd_us = time_fn(lambda: self.kdt.knn(q, k, dist_fn))
            hnsw_us = time_fn(lambda: self.hnsw.knn(q, k, 50, dist_fn))

            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": len(self.store)
            }

    def all(self):
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.get_info()

    def size(self):
        with self.lock:
            return len(self.store)

# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================

class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title, text, emb):
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            id_ = self.next_id
            self.next_id += 1
            item = {"id": id_, "title": title, "text": text, "emb": emb}
            self.store[id_] = item
            vi = {"id": id_, "metadata": title, "category": "doc", "emb": emb}
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return id_

    def search(self, q, k, max_dist=0.7):
        with self.lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            
            out = []
            for d, id_ in raw:
                if id_ in self.store and d <= max_dist:
                    out.append((d, self.store[id_]))
            return out

    def remove(self, id_):
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self):
        with self.lock:
            return list(self.store.values())

    def size(self):
        with self.lock:
            return len(self.store)

# =====================================================================
#  OLLAMA CLIENT & TEXT CHUNKER
# =====================================================================

class OllamaClient:
    def __init__(self, host="127.0.0.1", port=11434):
        self.host = host
        self.port = port
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self):
        try:
            res = requests.get(f"http://{self.host}:{self.port}/api/tags", timeout=2)
            return res.status_code == 200
        except Exception:
            return False

    def embed(self, text):
        try:
            url = f"http://{self.host}:{self.port}/api/embeddings"
            payload = {"model": self.embed_model, "prompt": text}
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                return res.json().get("embedding", [])
        except Exception as e:
            print("Embedding failed:", e)
        return []

    def generate(self, prompt):
        try:
            url = f"http://{self.host}:{self.port}/api/generate"
            payload = {"model": self.gen_model, "prompt": prompt, "stream": False}
            res = requests.post(url, json=payload, timeout=180)
            if res.status_code == 200:
                return res.json().get("response", "")
        except Exception as e:
            print("Generate failed:", e)
        return "ERROR: Ollama unavailable. Run: ollama serve"

def chunk_text(text, chunk_words=250, overlap_words=30):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        chunk_words_list = words[i : i + chunk_words]
        chunks.append(" ".join(chunk_words_list))
        if i + chunk_words >= len(words):
            break
    return chunks

# =====================================================================
#  LOAD DEMO DATA
# =====================================================================

def load_demo(db):
    dist_fn = get_dist_fn("cosine")
    demos = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80])
    ]
    for meta, cat, emb in demos:
        db.insert(meta, cat, emb, dist_fn)

# =====================================================================
#  FLASK SERVER SETUP
# =====================================================================

DIMS = 16
db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

# Load demo vectors
load_demo(db)

def cors_response(data, status=200):
    res = make_response(jsonify(data), status)
    res.headers["Access-Control-Allow-Origin"] = "*"
    res.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    res.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return res

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        res = make_response("", 204)
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return res

@app.route("/", methods=["GET"])
def index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    except Exception:
        return "index.html not found", 404

# --- DEMO VECTOR ENDPOINTS ---

@app.route("/search", methods=["GET"])
def search():
    v_str = request.args.get("v", "")
    try:
        q = [float(x) for x in v_str.split(",") if x.strip()]
    except Exception:
        return cors_response({"error": "invalid vector format"}, 400)
        
    if len(q) != DIMS:
        return cors_response({"error": f"need {DIMS}D vector"}, 400)
        
    try:
        k = int(request.args.get("k", 5))
    except Exception:
        k = 5
        
    metric = request.args.get("metric", "cosine")
    algo = request.args.get("algo", "hnsw")
    
    out = db.search(q, k, metric, algo)
    return cors_response(out)

@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding", [])
    
    if not meta or not emb or len(emb) != DIMS:
        return cors_response({"error": "invalid body"}, 400)
        
    dist_fn = get_dist_fn("cosine")
    id_ = db.insert(meta, cat, emb, dist_fn)
    return cors_response({"id": id_})

@app.route("/delete/<int:id_>", methods=["DELETE"])
def delete(id_):
    ok = db.remove(id_)
    return cors_response({"ok": ok})

@app.route("/items", methods=["GET"])
def items():
    items_list = db.all()
    out = []
    for item in items_list:
        out.append({
            "id": item["id"],
            "metadata": item["metadata"],
            "category": item["category"],
            "embedding": item["emb"]
        })
    return cors_response(out)

@app.route("/benchmark", methods=["GET"])
def benchmark():
    v_str = request.args.get("v", "")
    try:
        q = [float(x) for x in v_str.split(",") if x.strip()]
    except Exception:
        return cors_response({"error": "invalid vector format"}, 400)
        
    if len(q) != DIMS:
        return cors_response({"error": f"need {DIMS}D vector"}, 400)
        
    try:
        k = int(request.args.get("k", 5))
    except Exception:
        k = 5
        
    metric = request.args.get("metric", "cosine")
    b = db.benchmark(q, k, metric)
    return cors_response(b)

@app.route("/hnsw-info", methods=["GET"])
def hnsw_info():
    info = db.hnsw_info()
    return cors_response(info)

# --- DOCUMENT + RAG ENDPOINTS ---

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    
    if not title or not text:
        return cors_response({"error": "need title and text"}, 400)
        
    chunks = chunk_text(text, 250, 30)
    ids = []
    
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return cors_response({
                "error": "Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2"
            }, 400)
        chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))
        
    return cors_response({
        "ids": ids,
        "chunks": len(chunks),
        "dims": doc_db.dims
    })

@app.route("/doc/delete/<int:id_>", methods=["DELETE"])
def doc_delete(id_):
    ok = doc_db.remove(id_)
    return cors_response({"ok": ok})

@app.route("/doc/list", methods=["GET"])
def doc_list():
    docs = doc_db.all()
    out = []
    for doc in docs:
        preview = doc["text"][:120]
        if len(doc["text"]) > 120:
            preview += "…"
        out.append({
            "id": doc["id"],
            "title": doc["title"],
            "preview": preview,
            "words": len(doc["text"].split())
        })
    return cors_response(out)

@app.route("/doc/search", methods=["POST"])
def doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = body.get("k", 3)
    
    if not question:
        return cors_response({"error": "need question"}, 400)
        
    q_emb = ollama.embed(question)
    if not q_emb:
        return cors_response({"error": "Ollama unavailable"}, 400)
        
    hits = doc_db.search(q_emb, k)
    contexts = []
    for d, item in hits:
        contexts.append({
            "id": item["id"],
            "title": item["title"],
            "distance": d
        })
    return cors_response({"contexts": contexts})

@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = body.get("k", 3)
    
    if not question:
        return cors_response({"error": "need question"}, 400)
        
    q_emb = ollama.embed(question)
    if not q_emb:
        return cors_response({"error": "Ollama unavailable"}, 400)
        
    hits = doc_db.search(q_emb, k)
    
    ctx_parts = []
    for i, (d, item) in enumerate(hits):
        ctx_parts.append(f"[{i+1}] {item['title']}:\n{item['text']}\n\n")
    ctx_str = "".join(ctx_parts)
    
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{ctx_str}"
        f"Question: {question}\n\n"
        "Answer:"
    )
    
    answer = ollama.generate(prompt)
    
    contexts = []
    for d, item in hits:
        contexts.append({
            "id": item["id"],
            "title": item["title"],
            "text": item["text"],
            "distance": d
        })
        
    return cors_response({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": contexts,
        "docCount": doc_db.size()
    })

@app.route("/status", methods=["GET"])
def status_endpoint():
    up = ollama.is_available()
    return cors_response({
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.dims,
        "demoDims": DIMS,
        "demoCount": db.size()
    })

@app.route("/stats", methods=["GET"])
def stats_endpoint():
    return cors_response({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"]
    })

if __name__ == "__main__":
    print("=== VectorDB Engine (Python Flask Port) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    ollama_up = ollama.is_available()
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
        
    # Run the server on port 8080
    app.run(host="0.0.0.0", port=8080, debug=False)
