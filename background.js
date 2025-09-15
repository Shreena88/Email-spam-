// The URL where your Python Flask server is running
const API_ENDPOINT = 'http://127.0.0.1:5000/analyze';
const MODEL_DOWNLOAD_URL = 'http://127.0.0.1:5000/model';

// Thresholds to determine the verdict based on the AI's confidence
const SPAM_THRESHOLD = 0.75;      // 75%+ -> Spam (red)
const SUSPICIOUS_THRESHOLD = 0.50; // 50%-74% -> Suspicious (yellow)

// The new analysis function that calls the Python API
async function analyzeEmailWithPythonAPI(data) {
  const emailText = `${data.subject} ${data.body}`;

  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: emailText }),
    });

    if (!response.ok) {
      throw new Error(`Server returned an error: ${response.status}`);
    }

    const result = await response.json();
    const spamProbability = result.spam_probability;
    const predictedCategory = (result.predicted_category || '').toString().trim().toLowerCase();

    // Convert the probability score into our verdict system
    let verdict = 'Not Spam';
    let reasons = [];
    const riskScore = Math.round((spamProbability || 0) * 100);

    // Binary decision driven by model label (spam vs ham)
    if (predictedCategory === 'spam') {
      verdict = 'Spam';
      if (!Number.isNaN(riskScore)) {
        reasons.push(`AI classified as SPAM (${riskScore}% probability).`);
      } else {
        reasons.push('AI classified as SPAM.');
      }
    } else {
      verdict = 'Not Spam';
      if (!Number.isNaN(riskScore)) {
        reasons.push(`AI classified as NOT SPAM (${100 - riskScore}% confidence ham).`);
      } else {
        reasons.push('AI classified as NOT SPAM.');
      }
    }

    return { verdict, riskScore, reasons };

  } catch (error) {
    console.error("Email Spam Detector: Could not connect to the Python AI server.", error);
    // Show a visible warning in the UI instead of staying silent
    return { verdict: 'Suspicious', riskScore: 0, reasons: ['Could not reach the AI analysis server.'] };
  }
}

// Download the latest model (.pkl) and store in chrome storage as Base64
async function downloadAndStoreModel() {
  try {
    const response = await fetch(MODEL_DOWNLOAD_URL);
    if (!response.ok) throw new Error(`Failed to download model: ${response.status}`);
    const arrayBuffer = await response.arrayBuffer();
    const base64 = arrayBufferToBase64(arrayBuffer);
    await chrome.storage.local.set({ spamModelPkl: base64, spamModelUpdatedAt: Date.now() });
    console.log('Email Spam Detector: Model downloaded and stored.');
  } catch (err) {
    console.error('Email Spam Detector: Model download failed.', err);
  }
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

chrome.runtime.onInstalled.addListener(() => {
  downloadAndStoreModel();
});

// This listener part remains the same
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "analyzeEmail") {
    analyzeEmailWithPythonAPI(request.data).then(sendResponse);
  }
  // Return true to enable asynchronous response
  return true;
});