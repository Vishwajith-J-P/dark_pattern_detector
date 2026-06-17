// =====================================================================
// 1. ASYNCHRONOUS SCREENSHOT CAPTURE PROMISE WRAPPER
// =====================================================================
function captureTabAsync() {
    return new Promise((resolve) => {
        chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {
            if (chrome.runtime.lastError) {
                console.warn("Screenshot capture skipped or throttled:", chrome.runtime.lastError.message);
                resolve(null); // Fallback: allow pipeline to process text even if screen grab fails
            } else {
                resolve(dataUrl);
            }
        });
    });
}

// =====================================================================
// 2. BACKGROUND LIFECYCLE ROUTER
// =====================================================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "CAPTURE_AND_AUDIT") {
        
        // Execute the processing stack inside an isolated async self-invoking runner
        (async () => {
            try {
                // 1. Grab visual canvas view state safely using our promise hook
                const dataUrl = await captureTabAsync();

                const backendPayload = {
                    screenshot: dataUrl,
                    elements: request.elements
                };

                console.log(`🌐 Background Worker: Forwarding ${request.elements.length} components to Python processing stack...`);

                // 2. Transmit payload to the absolute localhost loop IP
                const response = await fetch("http://127.0.0.1:8000/api/v2/audit", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(backendPayload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP Error Status: ${response.status} - ${response.statusText}`);
                }

                const jsonResponse = await response.json();
                console.log("📥 Background Worker: Successfully parsed prediction array matrix.", jsonResponse);

                // 3. ENFORCE INTERFACE STRUCTURE UNIFICATION
                // If your FastAPI returns the object directly, wrap it inside the expected shape
                if (jsonResponse && jsonResponse.ok !== undefined) {
                    sendResponse(jsonResponse);
                } else {
                    sendResponse({ ok: true, data: jsonResponse });
                }

            } catch (error) {
                console.error("❌ Dark Pattern Detector API Connection Error:", error);
                sendResponse({ ok: false, error: `Backend Pipeline Down: ${error.message}` });
            }
        })();

        return true; // Strictly holds the message network pipe open across promises
    }
});