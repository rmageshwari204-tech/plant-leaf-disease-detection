from flask import Flask, render_template, request
import os
import random

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        diseases = ["Healthy Leaf", "Brown Spot Disease", "Leaf Blast Disease"]
        result = random.choice(diseases)

        return render_template("index.html", filename=file.filename, prediction=result)

    return "No file uploaded"

if __name__ == "__main__":
    app.run(debug=True)