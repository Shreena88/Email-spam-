# AI Spam/Ham Email Classifier

A comprehensive email spam detection system consisting of a Chrome browser extension and an AI-powered backend server. The system automatically analyzes emails in Gmail and provides real-time spam classification using machine learning.

## 🚀 Features

- **Real-time Email Analysis**: Automatically scans emails as you read them in Gmail
- **AI-Powered Classification**: Uses a trained Naive Bayes classifier for accurate spam detection
- **Visual Alerts**: Modern, non-intrusive warning banners for suspicious emails
- **Local Processing**: All analysis happens locally on your machine for privacy
- **Chrome Extension**: Seamless integration with Gmail interface
- **REST API**: Flask-based server for email classification

## 📁 Project Structure

```
├── Spam_filter/              # Chrome Extension
│   ├── manifest.json         # Extension configuration
│   ├── content.js           # Gmail integration script
│   ├── background.js        # Extension background service
│   └── alert.css           # Styling for warning banners
├── AI_Server/               # Backend AI Server
│   ├── app.py              # Flask API server
│   ├── create_model.py     # Model training script
│   ├── better_spam_model.pkl # Trained ML model
│   ├── Merged1_data.csv    # Training dataset
│   └── requirements.txt    # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7+
- Google Chrome browser
- Gmail account

### Step 1: Set Up the AI Server

1. **Navigate to the AI Server directory:**
   ```bash
   cd AI_Server
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model (if needed):**
   ```bash
   python create_model.py
   ```

5. **Start the Flask server:**
   ```bash
   python app.py
   ```
   The server will run at `http://127.0.0.1:5000`

### Step 2: Install the Chrome Extension

1. **Open Chrome and navigate to:**
   ```
   chrome://extensions/
   ```

2. **Enable Developer mode** (toggle in the top right)

3. **Click "Load unpacked"** and select the `Spam_filter` directory

4. **The extension should now appear** in your extensions list

## 🎯 How It Works

### Chrome Extension Flow

1. **Content Script Injection**: The extension injects `content.js` into Gmail pages
2. **Email Detection**: Uses MutationObserver to detect when new emails are opened
3. **Data Extraction**: Scrapes sender, subject, and body content from Gmail's DOM
4. **API Communication**: Sends email data to the local Flask server via `background.js`
5. **Result Display**: Shows warning banners based on the AI analysis

### AI Classification Process

1. **Text Processing**: Combines email subject and body into a single text string
2. **Model Prediction**: Uses a trained Naive Bayes classifier to analyze the text
3. **Probability Calculation**: Returns spam probability and predicted category
4. **Verdict Assignment**: 
   - **Spam**: Model predicts spam category
   - **Not Spam**: Model predicts ham category
   - **Suspicious**: Server connection issues

## 🔧 Configuration

### Extension Permissions

The extension requires:
- **Storage**: For caching analysis results
- **Host Permissions**: Access to `http://127.0.0.1:5000/*` for API calls
- **Content Scripts**: Injection into `*://mail.google.com/*`

### API Endpoints

- **POST** `/analyze`: Analyzes email text
  ```json
  {
    "text": "Email subject and body combined"
  }
  ```
  
  **Response:**
  ```json
  {
    "predicted_category": "spam|ham",
    "spam_probability": 0.85
  }
  ```

## 🎨 User Interface

### Warning Banner Types

- **🔴 Spam**: Red banner for high-confidence spam detection
- **🟡 Suspicious**: Yellow banner for connection issues or uncertain results  
- **🟢 Not Spam**: No banner shown (safe emails)




