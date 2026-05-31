// Inject modern CSS styling directly into Gmail document head
function injectStyles() {
  if (document.getElementById('spam-alert-banner-styles')) return;

  const styleEl = document.createElement('style');
  styleEl.id = 'spam-alert-banner-styles';
  styleEl.textContent = `
    /* Main banner positioning and modern style */
    .spam-alert-banner {
      position: fixed !important;
      left: 50% !important;
      transform: translateX(-50%) !important;
      width: auto !important;
      max-width: 600px !important;
      min-width: 420px !important;
      z-index: 2147483647 !important; /* Max possible 32-bit z-index to overlay Gmail controls */
      
      /* Animation-controlled properties (Do NOT use !important here, otherwise animation is ignored) */
      top: -150px;
      opacity: 0;
      
      /* Modern aesthetics */
      background-color: #ffffff !important;
      border-radius: 12px !important;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.18) !important;
      border: 1px solid rgba(0, 0, 0, 0.08) !important;
      padding: 16px 20px !important;
      
      /* Flexbox for alignment */
      display: flex !important;
      align-items: flex-start !important;
      gap: 15px !important;

      /* Animation and transition */
      animation: spamDetectorSlideDown 0.6s cubic-bezier(0.25, 0.8, 0.25, 1) forwards !important;
      transition: opacity 0.5s ease-in-out, top 0.5s ease-in-out !important;
    }

    /* Animation keyframes to slide the banner down */
    @keyframes spamDetectorSlideDown {
      from {
        top: -150px;
        opacity: 0;
      }
      to {
        top: 30px;
        opacity: 1;
      }
    }

    /* Icon styling */
    .alert-icon {
      width: 24px !important;
      height: 24px !important;
      flex-shrink: 0 !important;
      margin-top: 3px !important;
    }

    /* Text content styling */
    .alert-text-content {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
      line-height: 1.5 !important;
      color: #333333 !important;
      flex-grow: 1 !important;
    }

    .alert-text-content strong {
      font-size: 16px !important;
      font-weight: 600 !important;
      display: block !important;
      margin-bottom: 2px !important;
    }

    .alert-text-content span {
      font-size: 14px !important;
      color: #555555 !important;
    }

    .alert-text-content ul {
      margin: 8px 0 0 0 !important;
      padding-left: 20px !important;
      font-size: 13px !important;
    }

    /* Style for 'Warning' level alerts (Amber/Yellow) */
    .alert-suspicious {
      background-color: #fffbeb !important;
    }
    .alert-suspicious .alert-icon {
      color: #f59e0b !important;
    }
    .alert-suspicious .alert-text-content strong {
      color: #b45309 !important;
    }

    /* Style for 'High Risk' level alerts (Red) */
    .alert-spam {
      background-color: #fef2f2 !important;
    }
    .alert-spam .alert-icon {
      color: #ef4444 !important;
    }
    .alert-spam .alert-text-content strong {
      color: #b91c1c !important;
    }

    /* Style for 'Not Spam' (Green) */
    .alert-not-spam {
      background-color: #ecfdf5 !important;
    }
    .alert-not-spam .alert-icon {
      color: #10b981 !important;
    }
    .alert-not-spam .alert-text-content strong {
      color: #047857 !important;
    }

    /* Close Button Styling */
    .alert-close-btn {
      background: none !important;
      border: none !important;
      font-size: 20px !important;
      font-weight: bold !important;
      line-height: 1 !important;
      color: #9ca3af !important;
      cursor: pointer !important;
      margin-left: 10px !important;
      padding: 0 !important;
      align-self: flex-start !important;
      transition: color 0.2s !important;
    }
    .alert-close-btn:hover {
      color: #4b5563 !important;
    }
  `;
  (document.head || document.documentElement).appendChild(styleEl);
  console.log("Email Spam Scan: Banner styles injected into head.");
}

// Call styles injection immediately
injectStyles();

// This function creates and displays the modern warning banner
function displayWarningBanner(result, emailElement) {
  console.log("Email Spam Scan: displayWarningBanner called with verdict:", result.verdict);

  // Do nothing if the email is determined to be safe
  if (result.verdict === 'Safe') {
    console.log("Email Spam Scan: Email is safe. Not showing banner.");
    return;
  }

  // Ensure styles are injected
  injectStyles();

  // Remove any existing banner to avoid overlapping and to show the latest result
  const existingBanner = document.getElementById('spam-alert-banner-unique');
  if (existingBanner) {
    console.log("Email Spam Scan: Removing existing banner.");
    existingBanner.remove();
  }

  const banner = document.createElement('div');
  banner.id = 'spam-alert-banner-unique'; // Use an ID to prevent duplicates
  banner.className = `spam-alert-banner alert-${result.verdict.toLowerCase().replace(' ', '-')}`;

  // SVG Icon for warning/spam
  const warningIconSVG = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="alert-icon">
      <path fill-rule="evenodd" d="M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003zM12 8.25a.75.75 0 01.75.75v3.75a.75.75 0 01-1.5 0V9a.75.75 0 01.75-.75zm0 8.25a.75.75 0 100-1.5.75.75 0 000 1.5z" clip-rule="evenodd" />
    </svg>
  `;

  // SVG Icon for Not Spam (checkmark)
  const checkIconSVG = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="alert-icon">
      <path fill-rule="evenodd" d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm13.36-1.814a.75.75 0 10-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 00-1.06 1.06l2.5 2.5a.75.75 0 001.14-.094l3.75-5.25z" clip-rule="evenodd" />
    </svg>
  `;
  
  let reasonsHtml = result.reasons.map(reason => `<li>${reason}</li>`).join('');
  
  const closeBtnHtml = `<button class="alert-close-btn" id="spam-alert-close-btn-unique" aria-label="Dismiss alert">&times;</button>`;

  if (result.verdict === 'Not Spam') {
    banner.innerHTML = `
      ${checkIconSVG}
      <div class="alert-text-content">
        <strong>Not Spam</strong>
        <span>This email appears safe.</span>
      </div>
      ${closeBtnHtml}
    `;
  } else {
    banner.innerHTML = `
      ${warningIconSVG}
      <div class="alert-text-content">
        <strong>${result.verdict}</strong>
        <span>Our scan identified the following potential risks (Score: ${result.riskScore}):</span>
        <ul>${reasonsHtml}</ul>
      </div>
      ${closeBtnHtml}
    `;
  }
  
  // Inject banner directly into the main body so it can float over the viewport
  document.body.appendChild(banner);
  console.log("Email Spam Scan: Banner appended to document.body:", banner);

  // Setup manual dismiss handler
  const closeBtn = banner.querySelector('#spam-alert-close-btn-unique');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      banner.style.opacity = '0';
      setTimeout(() => banner.remove(), 500);
    });
  }

  // Auto-dismiss the banner after a few seconds
  const autoDismissTimeout = setTimeout(() => {
    if (banner && document.body.contains(banner)) {
      banner.style.opacity = '0';
      setTimeout(() => banner.remove(), 500);
    }
  }, 10000); // Extended banner duration to 10 seconds to allow reading
}

// This function scrapes the key data from the email's HTML
function scanAndAnalyzeEmail() {
  // Select all email containers that have not been processed yet
  const emailElements = document.querySelectorAll('.adn:not([data-spam-scan])');

  emailElements.forEach((emailEl) => {
    // Find the sender inside this specific email container
    const senderEl = emailEl.querySelector('.gD');
    // Find the thread subject (subject is thread-wide, in the main header .hP)
    const subjectEl = document.querySelector('.hP');
    // Find the message body text container inside this email (.a3s is standard for email body in Gmail)
    const bodyEl = emailEl.querySelector('.a3s') || emailEl.querySelector('.ii.gt');

    // Only proceed if the email content has loaded (we have a sender and body text)
    if (senderEl && bodyEl && bodyEl.innerText.trim().length > 0) {
      // Mark this specific email as processed so we don't scan it again
      emailEl.setAttribute('data-spam-scan', 'processed');

      const emailData = {
        sender: senderEl.innerText.trim(),
        subject: subjectEl ? subjectEl.innerText.trim() : 'No Subject',
        body: bodyEl.innerText.trim()
      };

      console.log("Email Spam Scan: Scraped email data:", emailData);

      // Send the scraped data to our background.js for analysis
      chrome.runtime.sendMessage({ action: "analyzeEmail", data: emailData }, (response) => {
        if (chrome.runtime.lastError) {
          console.error("Email Spam Scan: Communication error:", chrome.runtime.lastError.message);
          return;
        }
        console.log("Email Spam Scan: Analysis received.", response);
        // Display the result on the page
        displayWarningBanner(response, emailEl);
      });
    }
  });
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