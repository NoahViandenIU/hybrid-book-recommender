from pathlib import Path
from flask import Flask, jsonify, request
from src.recommender import create_engine

BASE_DIR = Path(__file__).resolve().parent
engine = create_engine(BASE_DIR / "data")
app = Flask(__name__, static_folder="frontend", static_url_path="/")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/books")
def books():
    return jsonify([book.__dict__ for book in engine.books.values()])

@app.route("/api/recommendations/<user_id>")
def recommendations(user_id: str):
    k = int(request.args.get("k", 5))
    recs = []
    for bid, score in engine.recommend(user_id, k=k):
        book = engine.books[bid]
        item = book.__dict__.copy()
        item["score"] = round(score, 4)
        recs.append(item)
    return jsonify({"user_id": user_id, "recommendations": recs})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
