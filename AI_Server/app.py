from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'better_spam_model.pkl')
model = None
model_classes = None

try:
    model = joblib.load(MODEL_FILE)
    # Get the class labels from the trained model (e.g., ['ham', 'spam'])
    model_classes = model.classes_
    print(f"✅ AI model loaded successfully from '{MODEL_FILE}'.")
    print(f"🔬 Model classes are: {model_classes}")
except FileNotFoundError:
    print(f"❌ Error: Model file '{MODEL_FILE}' not found.")
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")

@app.route('/analyze', methods=['POST'])
def analyze_email():
    if model is None:
        return jsonify({'error': 'AI model is not available'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input: Missing "text" field in JSON body'}), 400

    email_text = data['text']

    # --- Safer Prediction ---
    # 1. Get probabilities for all classes
    prediction_prob = model.predict_proba([email_text])[0]
    
    # 2. Find the index corresponding to the 'spam' class
    # This makes the code robust, regardless of class order
    try:
        spam_index = np.where(model_classes == 'spam')[0][0]
        spam_probability = prediction_prob[spam_index]
    except (IndexError, ValueError):
        # Fallback if 'spam' class isn't found for some reason
        return jsonify({'error': "Model is not a spam classifier or is misconfigured."}), 500

    # 3. Get the direct prediction ('ham' or 'spam')
    prediction_label = model.predict([email_text])[0]

    # Send the result back in a clear JSON format
    return jsonify({
        'predicted_category': prediction_label,
        'spam_probability': float(spam_probability) # Ensure it's a standard float
    })

@app.route('/model', methods=['GET'])
def download_model():
    if not os.path.exists(MODEL_FILE):
        return jsonify({'error': 'Model file not found'}), 404
    return send_file(MODEL_FILE, as_attachment=True, download_name='better_spam_model.pkl', mimetype='application/octet-stream')

if __name__ == '__main__':
    print("🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
