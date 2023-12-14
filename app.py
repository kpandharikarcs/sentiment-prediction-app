from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the trained model generated in notebook code
model = joblib.load('sentiment_model.joblib')

#rendering html code on application
@app.route('/')
def index():
    return render_template('index.html')

#code responsible for HTTP POST for review extraction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        prediction = model.predict([review])[0]
        return render_template('index.html', prediction=prediction, review=review)

#making the port dynamic
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
