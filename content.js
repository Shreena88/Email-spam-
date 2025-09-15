// This function creates and displays the modern warning banner
function displayWarningBanner(result, emailElement) {
  // Do nothing if the email is determined to be safe
  if (result.verdict === 'Safe') {
    console.log("Email Spam Scan: Email is safe.");
    return;
  }

  // Check if a banner already exists to avoid duplicates
  if (document.getElementById('spam-alert-banner-unique')) {
    return;
  }

  const banner = document.createElement('div');
  banner.id = 'spam-alert-banner-unique'; // Use an ID to prevent duplicates
  banner.className = `spam-alert-banner alert-${result.verdict.toLowerCase().replace(' ', '-')}`;

  // New: SVG Icon for a modern look
  const iconSVG = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="alert-icon">
      <path fill-rule="evenodd" d="M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003zM12 8.25a.75.75 0 01.75.75v3.75a.75.75 0 01-1.5 0V9a.75.75 0 01.75-.75zm0 8.25a.75.75 0 100-1.5.75.75 0 000 1.5z" clip-rule="evenodd" />
    </svg>
  `;
  
  let reasonsHtml = result.reasons.map(reason => `<li>${reason}</li>`).join('');

  if (result.verdict === 'Not Spam') {
    // Minimal content for Not Spam: only show these two lines
    banner.innerHTML = `
      <div class="alert-text-content">
        <strong>Not Spam</strong>
        <span>This email appears safe.</span>
      </div>
    `;
  } else {
    banner.innerHTML = `
      ${iconSVG}
      <div class="alert-text-content">
        <strong>${result.verdict}</strong>
        <span>Our scan identified the following potential risks (Score: ${result.riskScore}):</span>
        <ul>${reasonsHtml}</ul>
      </div>
    `;
  }
  
  // Inject banner directly into the main body so it can be fixed to the viewport
  document.body.appendChild(banner);

  // Optional: Remove the banner after a few seconds
  setTimeout(() => {
    if (banner) {
      banner.style.opacity = '0';
      // Remove from DOM after transition ends
      setTimeout(() => banner.remove(), 500);
    }
  }, 8000); // Banner stays for 8 seconds
}

// This function scrapes the key data from the email's HTML
function scanAndAnalyzeEmail() {
  // IMPORTANT: These selectors are for Gmail's current interface and may change.
  const emailBodyEl = document.querySelector('.adn.ads'); // The main email container

  // Check if an email body is visible and if it has NOT been scanned yet
  if (emailBodyEl && !emailBodyEl.hasAttribute('data-spam-scan')) {
    // Mark the email as scanned to prevent re-scanning
    emailBodyEl.setAttribute('data-spam-scan', 'processed');

    const senderEl = document.querySelector('.gD');
    const subjectEl = document.querySelector('.hP');

    // Make sure all elements exist before proceeding
    if (senderEl && subjectEl) {
      const emailData = {
        sender: senderEl.innerText,
        subject: subjectEl.innerText,
        body: emailBodyEl.innerText
      };
      
      console.log("Email Spam Scan: New email detected. Sending for analysis.", emailData);
      
      // Send the scraped data to our background.js for analysis
      chrome.runtime.sendMessage({ action: "analyzeEmail", data: emailData }, (response) => {
        if (chrome.runtime.lastError) {
          console.error("Error:", chrome.runtime.lastError.message);
          return;
        }
        console.log("Email Spam Scan: Analysis received.", response);
        // Display the result on the page
        displayWarningBanner(response, emailBodyEl);
      });
    }
  }
}

// Use a MutationObserver to detect when the user opens a new email
const observer = new MutationObserver((mutations) => {
  // We don't need to inspect the mutations themselves, just run our scan function
  // whenever the page's content changes. Our function will handle the rest.
  scanAndAnalyzeEmail();
});

// Start observing the entire document body for changes
observer.observe(document.body, {
  childList: true,
  subtree: true
});

console.log("Email Spam Detector content script is active.");