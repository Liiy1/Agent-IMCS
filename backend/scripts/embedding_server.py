"""
Embedding API Server for Coze Studio
Wraps local bge-small-zh model as an HTTP API service.

API format (Coze Studio backend expects):
  GET  /support_status  ->  plain text "3" (SupportDenseAndSparse)
  POST /embedding       ->  {"texts":[...], "need_sparse":false}
                        ->  {"dense":[[...],...], "sparse":[...]}
"""
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# 加载本地 bge-small-zh 模型
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bge-small-zh")
print("Loading model...")
model = SentenceTransformer(MODEL_PATH)
dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dimension: {dim}")

@app.route("/support_status", methods=["GET"])
def support_status():
    """告知 Coze Studio 支持 Dense（纯向量）模式"""
    return "1", 200  # 1 = SupportDense（纯向量，不返回稀疏向量）

@app.route("/embedding", methods=["POST"])
def embedding():
    """Coze Studio HTTP Embedding API 端点"""
    data = request.json
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400

    texts = data["texts"]
    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(texts).tolist()

    return jsonify({
        "dense": embeddings,
        "sparse": []
    })

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Coze Studio Embedding Server",
        "model": "bge-small-zh",
        "dimension": dim,
        "status": "running"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
