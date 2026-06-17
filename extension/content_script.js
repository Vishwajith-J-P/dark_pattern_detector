console.log("✅ Dark Pattern Detector Content Script Loaded");

// ======================================================
// 1. DOM FEATURE EXTRACTION ENGINE
// ======================================================

function extractDOMFeatures() {
    console.log("🔍 Dark Pattern Detector Engine: Scanning active viewport DOM...");

    const extractedElements = [];

    const targetTags = [
    'button',
    'a',
    'span',
    'p',
    'strong',
    'em',
    'b'
];

    const ignorePatterns = [
        'contact us',
        'about us',
        'privacy policy',
        'terms of use',
        'terms & conditions',
        'customer care',
        'help center',
        'help centre',
        'return policy',
        'refund policy',
        'shipping policy',
        'track order',
        'careers',
        'investor relations',
        'corporate office',
        'registered office',
        'all rights reserved',
        'follow us on',
        'facebook',
        'instagram',
        'twitter',
        'linkedin'
    ];

    targetTags.forEach(tagName => {

        const elements = document.querySelectorAll(tagName);

        elements.forEach((el, index) => {

            const textContent = el.innerText
                ? el.innerText.trim().replace(/\s+/g, ' ')
                : '';
            if (textContent.split(" ").length > 25) {
                return;
            }
            if (!textContent) return;
            if (textContent.length < 2) return;
            if (textContent.length > 250) return;

            if (el.offsetWidth === 0 || el.offsetHeight === 0) return;

            const lowerText = textContent.toLowerCase();

            const navigationWords = [
                "login",
                "sign in",
                "cart",
                "wishlist",
                "profile",
                "account",
                "home",
                "fashion",
                "beauty",
                "electronics",
                "mobiles",
                "travel",
                "more"
            ];

            if (
                navigationWords.includes(lowerText)
            ) {
                return;
            }

            const footerAncestor = el.closest(
                'footer,[class*="footer"],[id*="footer"]'
            );

            if (footerAncestor) {
                return;
            }
            const headerAncestor = el.closest(
                'header,[class*="header"],[id*="header"],nav'
            );

            if (headerAncestor) {
                return;
            }

            if (ignorePatterns.some(pattern => lowerText.includes(pattern))) {
                return;
            }

            const rect = el.getBoundingClientRect();

            if (rect.width < 20 || rect.height < 10) {
                return;
            }

            // Reject huge containers

            const viewportArea =
                window.innerWidth * window.innerHeight;

            const elementArea =
                rect.width * rect.height;

            if (elementArea > viewportArea * 0.20) {
                return;
            }

            if (rect.width > window.innerWidth * 0.70) {
                return;
            }

            if (rect.height > window.innerHeight * 0.50) {
                return;
            }

            let elementId = el.getAttribute('data-darklens-id');

            if (!elementId) {
                elementId = `dl-${tagName}-${index}-${Math.floor(Math.random() * 100000)}`;
                el.setAttribute('data-darklens-id', elementId);
            }

            let priority = 0;

            const shoppingSignals = [
                '₹',
                '% off',
                'discount',
                'offer',
                'sale',
                'limited',
                'left',
                'selling fast',
                'buy now',
                'add to bag',
                'add to cart'
            ];

            shoppingSignals.forEach(signal => {
                if (lowerText.includes(signal)) {
                    priority += 1;
                }
            });

            if (textContent.match(/₹\s?\d+/)) {
                priority += 2;
            }

            if (priority === 0) {
                return;
            }

            extractedElements.push({
                element_id: elementId,
                text: textContent,
                tag: tagName,
                priority: priority,

                bbox: {
                    top: Math.round(rect.top + window.scrollY),
                    left: Math.round(rect.left + window.scrollX),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                },

                styles: {
                    font_size: getComputedStyle(el).fontSize,
                    font_weight: getComputedStyle(el).fontWeight,
                    color: getComputedStyle(el).color
                },

                attributes: {
                    class: el.className || '',
                    id: el.id || ''
                }
            });
        });
    });

    console.log(`📊 Dark Pattern Detector: Captured ${extractedElements.length} filtered candidates`);
    console.log(extractedElements.slice(0, 50).map(e => ({ tag: e.tag, text: e.text })));
    extractedElements.sort((a, b) => b.priority - a.priority);

    return extractedElements;
}

// ======================================================
// 2. HIGHLIGHT OVERLAY ENGINE
// ======================================================

function clearHighlights() {
    document
        .querySelectorAll(".darklens-overlay")
        .forEach(el => el.remove());

    console.log("🧹 Dark Pattern Detector overlays cleared");
}

function renderHighlights(results) {

    clearHighlights();

    if (!results || !results.detections) {
        console.warn("No detections available");
        return;
    }

    console.log(`🎯 Rendering ${results.detections.length} detections`);

    console.log(results.detections);

    results.detections.forEach(det => {

        if ((det.max_confidence || 0) < 0.90) {
            return;
        }
        const elementId = det.element_id;

        if (!elementId) return;

        const target = document.querySelector(
            `[data-darklens-id="${elementId}"]`
        );

        console.log(
            "Trying to highlight:",
            elementId,
            det.text,
            det.max_confidence,
            target?.getBoundingClientRect()
        );

        if (!target) {
            console.warn(
                "Missing target for:",
                elementId,
                det.text
            );
            return;
        }

        const rect = target.getBoundingClientRect();

        if (rect.width > window.innerWidth * 0.80 || rect.height > window.innerHeight * 0.80) {
            console.warn(
                "Skipping giant element",
                elementId,
                rect.width,
                rect.height
            );

            return;
        }

        const overlay = document.createElement("div");

        overlay.className = "darklens-overlay";

        overlay.style.position = "absolute";
        overlay.style.left = `${rect.left + window.scrollX}px`;
        overlay.style.top = `${rect.top + window.scrollY}px`;
        overlay.style.width = `${rect.width}px`;
        overlay.style.height = `${rect.height}px`;
        overlay.style.border = "3px solid #ef4444";
        overlay.style.background = "rgba(239,68,68,0.15)";
        overlay.style.pointerEvents =
            "none";

        overlay.style.zIndex =
            "2147483647";

        overlay.style.boxSizing =
            "border-box";

        document.body.appendChild(overlay);
    });
}

// ======================================================
// 3. MESSAGE ROUTER
// ======================================================

chrome.runtime.onMessage.addListener(
    (request, sender, sendResponse) => {

        console.log(
            "📨 Content Script Message:",
            request.action
        );

        try {

            switch (request.action) {

                case "START_AUDIT": {

                    const elements =
                        extractDOMFeatures();

                    sendResponse({
                        ok: true,
                        elements
                    });

                    break;
                }

                case "RENDER_HIGHLIGHTS": {

                    renderHighlights(
                        request.results
                    );

                    sendResponse({
                        ok: true
                    });

                    break;
                }

                case "CLEAR_HIGHLIGHTS": {

                    clearHighlights();

                    sendResponse({
                        ok: true
                    });

                    break;
                }

                default:

                    sendResponse({
                        ok: false,
                        error:
                            "Unknown action: " +
                            request.action
                    });
            }

        } catch (err) {

            console.error(
                "❌ Content Script Error:",
                err
            );

            sendResponse({
                ok: false,
                error: err.message
            });
        }

        return true;
    }
);