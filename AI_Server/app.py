from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
from create_model import clean_text

import os

app = Flask(__name__)

# --- Model Loading ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, 'better_spam_model.pkl')
model = None
model_classes = None

try:
    model = joblib.load(MODEL_FILE)
    # Get the class labels from the trained model (e.g., ['ham', 'spam'])
    model_classes = model.classes_
    print(f"AI model loaded successfully from '{MODEL_FILE}'.")
    print(f"Model classes are: {model_classes}")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

@app.route('/', methods=['GET'])
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Email Spam & Phishing Analyzer</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
                --card-bg: rgba(30, 41, 59, 0.7);
                --card-border: rgba(255, 255, 255, 0.08);
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --accent-purple: #a855f7;
                --accent-blue: #3b82f6;
                --spam-color: #ef4444;
                --ham-color: #10b981;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: var(--bg-gradient);
                color: var(--text-primary);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2rem;
                overflow-x: hidden;
            }

            .container {
                max-width: 680px;
                width: 100%;
                background: var(--card-bg);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 2.5rem;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 
                            0 0 80px rgba(168, 85, 247, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            h1 {
                font-size: 2.2rem;
                font-weight: 700;
                background: linear-gradient(to right, #3b82f6, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
                text-align: center;
            }

            .subtitle {
                color: var(--text-secondary);
                font-size: 0.975rem;
                margin-bottom: 2rem;
                text-align: center;
            }

            .form-group {
                margin-bottom: 1.5rem;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            label {
                font-size: 0.875rem;
                font-weight: 600;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            textarea {
                width: 100%;
                height: 180px;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 1.2rem;
                color: var(--text-primary);
                font-family: inherit;
                font-size: 0.95rem;
                line-height: 1.6;
                resize: vertical;
                outline: none;
                transition: border-color 0.2s ease, box-shadow 0.2s ease;
            }

            textarea:focus {
                border-color: var(--accent-purple);
                box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.25);
            }

            .btn {
                width: 100%;
                background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
                color: white;
                border: none;
                border-radius: 16px;
                padding: 1rem 2rem;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s ease, filter 0.2s ease, box-shadow 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
                box-shadow: 0 4px 12px rgba(168, 85, 247, 0.2);
            }

            .btn:hover {
                filter: brightness(1.1);
                box-shadow: 0 6px 20px rgba(168, 85, 247, 0.35);
                transform: translateY(-1px);
            }

            .btn:active {
                transform: translateY(1px);
            }

            .btn:disabled {
                background: rgba(255, 255, 255, 0.1);
                color: var(--text-secondary);
                cursor: not-allowed;
                box-shadow: none;
                filter: none;
                transform: none;
            }

            .result-card {
                margin-top: 2rem;
                padding: 1.5rem;
                border-radius: 16px;
                background: rgba(15, 23, 42, 0.4);
                border: 1px solid var(--card-border);
                display: none;
                animation: fadeIn 0.4s ease-out forwards;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }

            .result-title {
                font-size: 0.875rem;
                font-weight: 600;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .badge {
                padding: 0.4rem 1rem;
                border-radius: 9999px;
                font-size: 0.9rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                display: inline-block;
            }

            .badge-spam {
                background: rgba(239, 68, 68, 0.15);
                color: var(--spam-color);
                border: 1px solid rgba(239, 68, 68, 0.3);
                box-shadow: 0 0 15px rgba(239, 68, 68, 0.1);
            }

            .badge-ham {
                background: rgba(16, 185, 129, 0.15);
                color: var(--ham-color);
                border: 1px solid rgba(16, 185, 129, 0.3);
                box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
            }

            .confidence-container {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }

            .confidence-text {
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
                font-weight: 500;
            }

            .progress-bar-bg {
                width: 100%;
                height: 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 9999px;
                overflow: hidden;
            }

            .progress-bar-fill {
                height: 100%;
                width: 0%;
                border-radius: 9999px;
                transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .fill-spam {
                background: linear-gradient(90deg, #ef4444, #f87171);
            }

            .fill-ham {
                background: linear-gradient(90deg, #10b981, #34d399);
            }

            .footer {
                margin-top: 2rem;
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.25);
                text-align: center;
            }

            /* Loader Animation */
            .spinner {
                width: 20px;
                height: 20px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 0.8s linear infinite;
                display: none;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Email Spam Analyzer</h1>
            <p class="subtitle">Enter email content below to identify if it is Spam/Phishing or Legitimate (Ham) using the custom-trained Naive Bayes model.</p>
            
            <div class="form-group">
                <label for="email-content">Email Content</label>
                <textarea id="email-content" placeholder="Paste email subject and body content here..."></textarea>
            </div>

            <button id="analyze-btn" class="btn" onclick="analyzeEmail()">
                <span class="spinner" id="btn-spinner"></span>
                <span id="btn-text">Analyze Email</span>
            </button>

            <div class="result-card" id="result-card">
                <div class="result-header">
                    <span class="result-title">Classification Result</span>
                    <span id="result-badge" class="badge">Spam</span>
                </div>
                <div class="confidence-container">
                    <div class="confidence-text">
                        <span style="color: var(--text-secondary);">Spam Probability</span>
                        <span id="confidence-val" style="font-weight: 700;">0.00%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div id="progress-fill" class="progress-bar-fill"></div>
                    </div>
                </div>
            </div>
        </div>
        <div class="footer">
            Powered by Custom-Trained Multinomial Naive Bayes Model with SMOTE
        </div>

        <script>
            async function analyzeEmail() {
                const textarea = document.getElementById('email-content');
                const btn = document.getElementById('analyze-btn');
                const spinner = document.getElementById('btn-spinner');
                const btnText = document.getElementById('btn-text');
                const resultCard = document.getElementById('result-card');
                const badge = document.getElementById('result-badge');
                const confVal = document.getElementById('confidence-val');
                const fill = document.getElementById('progress-fill');

                const text = textarea.value.trim();
                if (!text) {
                    alert('Please enter some text to analyze.');
                    return;
                }

                // Show loading state
                btn.disabled = true;
                spinner.style.display = 'inline-block';
                btnText.textContent = 'Analyzing...';
                resultCard.style.display = 'none';

                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: text })
                    });

                    if (!response.ok) {
                        throw new Error('API request failed');
                    }

                    const data = await response.json();
                    
                    // Display results
                    const isSpam = data.predicted_category === 'spam';
                    const prob = data.spam_probability * 100;

                    badge.textContent = data.predicted_category;
                    badge.className = 'badge ' + (isSpam ? 'badge-spam' : 'badge-ham');
                    
                    confVal.textContent = prob.toFixed(2) + '%';
                    
                    fill.className = 'progress-bar-fill ' + (isSpam ? 'fill-spam' : 'fill-ham');
                    
                    // Show card and trigger width animation
                    resultCard.style.display = 'block';
                    setTimeout(() => {
                        fill.style.width = prob.toFixed(1) + '%';
                    }, 50);

                } catch (error) {
                    alert('Error analyzing the email. Please verify if the server is running correctly.');
                } finally {
                    btn.disabled = false;
                    spinner.style.display = 'none';
                    btnText.textContent = 'Analyze Email';
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/analyze', methods=['POST'])
def analyze_email():
    if model is None:
        return jsonify({'error': 'AI model is not available'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input: Missing "text" field in JSON body'}), 400

    email_text = data['text']
    cleaned_email = clean_text(email_text)

    # --- Safer Prediction ---
    # 1. Get probabilities for all classes
    prediction_prob = model.predict_proba([cleaned_email])[0]
    
    # 2. Find the index corresponding to the 'spam' class
    # This makes the code robust, regardless of class order
    try:
        spam_index = np.where(model_classes == 'spam')[0][0]
        spam_probability = prediction_prob[spam_index]
    except (IndexError, ValueError):
        # Fallback if 'spam' class isn't found for some reason
        return jsonify({'error': "Model is not a spam classifier or is misconfigured."}), 500

    # 3. Get the direct prediction ('ham' or 'spam')
    prediction_label = model.predict([cleaned_email])[0]

    # Send the result back in a clear JSON format
    return jsonify({
        'predicted_category': prediction_label,
        'spam_probability': float(spam_probability) # Ensure it's a standard float
    })

if __name__ == '__main__':
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
