from pathlib import Path

from flask import Flask, jsonify, request

from src.recommender import create_engine

BASE_DIR = Path(__file__).resolve().parent
MAX_RECOMMENDATIONS = 20


def create_app() -> Flask:
    app = Flask(__name__, static_folder="frontend", static_url_path="/")
    engine = create_engine(BASE_DIR / "data", alpha=0.7)

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/books")
    def books():
        return jsonify([book.__dict__ for book in engine.books.values()])

    @app.get("/api/recommendations/<user_id>")
    def recommendations(user_id: str):
        normalized_user_id = user_id.strip()
        if not normalized_user_id or len(normalized_user_id) > 64:
            return jsonify({"error": "user_id must contain 1 to 64 characters"}), 400

        raw_k = request.args.get("k", "5")
        try:
            k = int(raw_k)
        except ValueError:
            return jsonify({"error": "k must be an integer"}), 400
        if not 1 <= k <= MAX_RECOMMENDATIONS:
            return jsonify(
                {"error": f"k must be between 1 and {MAX_RECOMMENDATIONS}"}
            ), 400

        recs = []
        for bid, score in engine.recommend(normalized_user_id, k=k):
            book = engine.books[bid]
            item = book.__dict__.copy()
            item["score"] = round(score, 4)
            recs.append(item)
        return jsonify({"user_id": normalized_user_id, "recommendations": recs})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(port=5000)
