// The URL where your Python Flask server is running
const API_ENDPOINT = 'http://127.0.0.1:5000/analyze';

// Thresholds to determine the verdict based on the AI's confidence
const SPAM_THRESHOLD = 85; // Score > 85 -> Spam, otherwise Not Spam (Ham)

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

    // Convert the probability score into our verdict system
    let verdict = 'Not Spam';
    let reasons = [];
    const riskScore = Math.round((spamProbability || 0) * 100);

    // Decision driven by the 85% threshold
    if (riskScore > SPAM_THRESHOLD) {
      verdict = 'Spam';
      reasons.push(`AI classified as SPAM (${riskScore}% probability).`);
    } else {
      verdict = 'Not Spam';
      reasons.push(`AI classified as NOT SPAM (${100 - riskScore}% confidence ham).`);
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
  return true;
});