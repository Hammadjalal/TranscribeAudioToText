import os
import whisper
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Models
whisper_model = whisper.load_model("base")
sentiment_model = pipeline("sentiment-analysis")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["audio"]
    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Transcribe
    result = whisper_model.transcribe(filepath)
    transcription = result["text"]

    # Sentiment
    sentiment = sentiment_model(transcription)[0]

    return jsonify({
        "transcription": transcription,
        "sentiment": sentiment["label"],
        "confidence": round(sentiment["score"], 2)
    })


if __name__ == "__main__":
    app.run(debug=True)