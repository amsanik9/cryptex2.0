from flask import Flask, request, render_template, jsonify
from main import predict_algorithm

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ciphertext = data.get('ciphertext', '')
    result = predict_algorithm(ciphertext)
    return jsonify({'algorithm': result})

if __name__ == '__main__':
    app.run(debug=True)
