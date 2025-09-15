// The URL where your Python Flask server is running
const API_ENDPOINT = 'http://127.0.0.1:5000/analyze';

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

// This listener part remains the same
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "analyzeEmail") {
    analyzeEmailWithPythonAPI(request.data).then(sendResponse);
  }
  // Return true to enable asynchronous response
  return true;
});