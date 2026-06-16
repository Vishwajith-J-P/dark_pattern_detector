// =====================================================================
// 1. CONFIGURATION & STATE DICTIONARIES
// =====================================================================
const PATTERN_COLORS = {
    urgency:                     '#ef4444',
    scarcity:                    '#f97316',
    confirmshaming:              '#a855f7',
    hidden_costs:                '#ec4899',
    forced_continuity:           '#8b5cf6',
    misdirection:                '#3b82f6',
    price_comparison_prevention: '#06b6d4',
    disguised_ads:               '#f59e0b',
    trick_questions:             '#22c55e',
    social_proof_manipulation:   '#0ea5e9',
};

// =====================================================================
// 2. CORE UI STATE ENGINE HELPERS
// =====================================================================
function showState(name) {
    const states = ['idle', 'loading', 'results', 'error'];
    states.forEach(s => {
        const el = document.getElementById('state-' + s);
        if (el) el.classList.remove('active');
    });
    
    const targetState = document.getElementById('state-' + name);
    if (targetState) targetState.classList.add('active');
    
    const actionsEl = document.getElementById('actions');
    if (actionsEl) {
        actionsEl.style.display = name === 'results' ? 'flex' : 'none';
    }
}

function setLoading(text) {
    showState('loading');
    const loadingTextEl = document.getElementById('loading-text');
    if (loadingTextEl) {
        loadingTextEl.textContent = text;
    }
}

// =====================================================================
// 3. METRIC & SCORE RENDER COMPONENT PIPELINES
// =====================================================================
function renderScore(score) {
    const el  = document.getElementById('score-number');
    const bdg = document.getElementById('risk-badge');
    if (!el || !bdg) return;
    
    el.textContent = score.toFixed(1);

    if (score < 40) {
        el.className  = 'score-number score-low';
        bdg.className = 'risk-badge risk-low';
        bdg.textContent = 'Low Risk';
    } else if (score < 70) {
        el.className  = 'score-number score-medium';
        bdg.className = 'risk-badge risk-medium';
        bdg.textContent = 'Moderate Risk';
    } else {
        el.className  = 'score-number score-high';
        bdg.className = 'risk-badge risk-high';
        bdg.textContent = 'High Risk';
    }
}

function renderPatterns(perPattern) {
    const list = document.getElementById('pattern-list');
    if (!list) return;
    list.innerHTML = '';

    const sorted = Object.entries(perPattern || {}).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([label, score]) => {
        const pct   = Math.round(score * 100);
        const color = PATTERN_COLORS[label] || '#38bdf8';
        const name  = label.replace(/_/g, ' ');

        list.innerHTML += `
            <div class="pattern-item">
                <div class="pattern-left">
                    <div class="pattern-dot" style="background:${color}"></div>
                    <div class="pattern-name">${name}</div>
                </div>
                <div class="pattern-bar-wrap">
                    <div class="pattern-bar" style="width:${pct}%;background:${color}"></div>
                </div>
                <div class="pattern-pct">${pct}%</div>
            </div>`;
    });
}

// =====================================================================
// 4. MAIN MULTI-MODAL AUDIT COORDINATOR
// =====================================================================
async function runScan() {
    console.log("🚀 DARK_LENS: Scan execution loop triggered.");
    const scanBtn = document.getElementById('scan-btn');
    if (scanBtn) scanBtn.disabled = true;
    
    setLoading('Connecting to page contents...');

    try {
        // 1. Fetch target tab context
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab || !tab.id || tab.url.startsWith('chrome://') || tab.url.startsWith('edge://') || tab.url.startsWith('brave://')) {
            throw new Error('DarkLens cannot run audits on internal browser configurations. Please try an active shop site.');
        }

        setLoading('Analysing DOM structure...');
        
        // 2. Handle Handshake with Fallback Programmatic Script Injection & Delay Retry Buffer
        let extracted;
        try {
            extracted = await new Promise((resolve, reject) => {
                chrome.tabs.sendMessage(tab.id, { action: "START_AUDIT" }, response => {
                    if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
                    else resolve(response);
                });
            });
        } catch (connectionError) {
            console.warn("Content environment unreachable. Injecting auditor components manually...", connectionError);
            
            // Force injection of content_script.js into target page
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ['content_script.js']
            });
            
            // Give the browser 750ms to mount hooks safely
            await new Promise(r => setTimeout(r, 750)); 
            
            try {
                extracted = await new Promise((resolve, reject) => {
                    chrome.tabs.sendMessage(tab.id, { action: "START_AUDIT" }, response => {
                        if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
                        else resolve(response);
                    });
                });
            } catch (retryError) {
                console.warn("First handshake retry failed. Attempting final recovery check...", retryError);
                await new Promise(r => setTimeout(r, 500));
                extracted = await new Promise((resolve, reject) => {
                    chrome.tabs.sendMessage(tab.id, { action: "START_AUDIT" }, response => {
                        if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
                        else resolve(response);
                    });
                });
            }
        }

        if (!extracted || !extracted.elements) {
            throw new Error('Failed to parse document elements inside the active viewport.');
        }

        // 3. Transmit payload to neural classification backend via localized background messaging
        setLoading('Running classifier models...');
        const response = await new Promise((resolve, reject) => {
            chrome.runtime.sendMessage({
                action: "CAPTURE_AND_AUDIT",
                elements: extracted.elements
            }, result => {
                if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
                else resolve(result);
            });
        });

        if (!response || !response.ok) {
            throw new Error(response?.error || 'Local API backend connection failure.');
        }
        
        const data = response.data;

        // 4. Dispatch prediction metrics to Content script for physical bounding box overlays
        await chrome.tabs.sendMessage(tab.id, {
            action: 'RENDER_HIGHLIGHTS',
            results: data
        });

        // 5. Build presentation matrices dynamically onto popup frame
        if (typeof renderScore === 'function') renderScore(data.score);
        
        const subText = document.getElementById('score-sub');
        if (subText) subText.textContent = `${data.total_detections || 0} elements flagged`;
        
        if (typeof renderPatterns === 'function') renderPatterns(data.per_pattern);
        if (typeof renderCompounds === 'function') renderCompounds(data.compounds);
        if (typeof renderCards === 'function') renderCards(data.cards);
        
        showState('results');

    } catch (err) {
        console.error("DarkLens Pipeline Execution Halt:", err);
        const errorText = document.getElementById('error-text');
        if (errorText) errorText.textContent = err.message;
        showState('error');
    } finally {
        if (scanBtn) scanBtn.disabled = false;
    }
}

// Bind function out to window scope globally
window.runScan = runScan;

// =====================================================================
// 5. ASYNCHRONOUS SECURE EVENT LIFECYCLE HANDLERS (ANTI-CSP)
// =====================================================================
window.onload = function() {
    console.log("🧩 DarkLens Popup DOM loaded. Attaching secure listeners...");
    
    const scanBtn = document.getElementById('scan-btn');
    if (scanBtn) {
        scanBtn.addEventListener('click', () => {
            console.log("🚀 Secure click captured! Starting pipeline...");
            runScan();
        });
    } else {
        console.error("❌ Fatal Error: Could not find 'scan-btn' in popup.html!");
    }

    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', async () => {
            try {
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                if (tab && tab.id) {
                    await chrome.tabs.sendMessage(tab.id, { action: 'CLEAR_HIGHLIGHTS' });
                }
                showState('idle');
            } catch (err) {
                console.error("Failed to transmit clean signal loop:", err);
            }
        });
    }

    const rescanBtn = document.getElementById('rescan-btn');
    if (rescanBtn) {
        rescanBtn.addEventListener('click', runScan);
    }
};