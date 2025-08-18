from flask import Flask, render_template, request
import joblib

# Load trained model and accuracy info
model = joblib.load("news_classifier_model.pkl")
model_info = joblib.load("model_info.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form.get("news_text", "").strip()
        
        if text_input:  # Only predict if not empty
            probs = model.predict_proba([text_input])[0]
            predicted_class = model.classes_[probs.argmax()]
            prediction = predicted_class
            confidence = round(probs.max() * 100, 2)  # confidence percentage

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text_input=text_input,
        accuracy=round(model_info.get("accuracy", 0) * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
