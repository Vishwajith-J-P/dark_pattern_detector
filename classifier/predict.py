import torch
import numpy as np
import sys, os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi import Request
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = FastAPI(title="Dark Pattern Detection Engine")

# Enable CORS so your local browser extension can seamlessly call your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/distilbert_finetuned"
THRESHOLD = 0.75
MAX_LEN    = 64
BATCH_SIZE = 32

PATTERN_LABELS = [
    "urgency", "scarcity", "confirmshaming", "hidden_costs",
    "forced_continuity", "misdirection", "price_comparison_prevention",
    "disguised_ads", "trick_questions", "social_proof_manipulation"
]

COGNITIVE_BIAS = {
    "urgency": "FOMO + Loss Aversion",
    "scarcity": "Reactance + Scarcity Heuristic",
    "confirmshaming": "Guilt + Identity Threat",
    "hidden_costs": "Sunk Cost Fallacy",
    "forced_continuity": "Status Quo Bias + Inertia",
    "misdirection": "Attentional Bias",
    "price_comparison_prevention": "Anchoring Bias",
    "disguised_ads": "Familiarity + Trust Bias",
    "trick_questions": "Cognitive Load Exploitation",
    "social_proof_manipulation": "Bandwagon Effect + Conformity Bias",
}

PATTERN_DESCRIPTIONS = {
    "urgency": "Creates artificial time pressure to rush your decision.",
    "scarcity": "Manufactures false scarcity to make products seem more desirable.",
    "confirmshaming": "Uses guilt-inducing language to shame you into accepting an offer.",
    "hidden_costs": "Conceals additional charges until the final payment step.",
    "forced_continuity": "Enrolls you in recurring charges after a free trial without clear consent.",
    "misdirection": "Draws your attention away from important information or better options.",
    "price_comparison_prevention": "Makes it difficult to compare prices.",
    "disguised_ads": "Presents paid advertising as organic content.",
    "trick_questions": "Uses confusing double negatives or pre-ticked checkboxes.",
    "social_proof_manipulation": "Inflates or fabricates social signals to create false popularity.",
}

SEVERITY_WEIGHTS = {
    "urgency": 0.8, "scarcity": 0.8, "confirmshaming": 1.0, "hidden_costs": 1.0,
    "forced_continuity": 1.0, "misdirection": 0.7, "price_comparison_prevention": 0.7,
    "disguised_ads": 0.9, "trick_questions": 0.9, "social_proof_manipulation": 0.6,
}

BOILERPLATE_SIGNALS = [
    'cin',
    'telephone',
    'private limited',
    'copyright',
    'all rights reserved',
    'customer care',
    'registered office',
    'corporate office',
    'investor relations',
    'grievance',
    'shipping',
    'cancellation',
    'faq',
    'site map',
    'blog',
    'contact us'
]

CATEGORY_WORDS = {
    "handbags",
    "boxers",
    "smart watches",
    "eye makeup",
    "goggles",
    "wallets",
    "tops",
    "rings",
    "bags",
    "nike",
    "puma shoes",
    "sport shoes"
}

# --- Pydantic Schemas for Strict Input Parsing ---

class BBoxInput(BaseModel):
    """Bounding box of a DOM element relative to the full scrolled page."""
    top:    Optional[float] = None
    left:   Optional[float] = None
    width:  Optional[float] = None
    height: Optional[float] = None


class StylesInput(BaseModel):
    """Computed CSS properties captured by the content script."""
    color:             Optional[str] = None
    background_color:  Optional[str] = None  # camelCase variant also accepted below
    font_size:         Optional[str] = None
    font_weight:       Optional[str] = None
    text_decoration:   Optional[str] = None
    opacity:           Optional[str] = None
    z_index:           Optional[str] = None

    class Config:
        # Accept both snake_case and camelCase / hyphen-case keys sent by JS
        populate_by_name = True
        extra = "allow"   # forward-compat: ignore any extra style keys


class AttributesInput(BaseModel):
    """HTML attributes captured by the content script."""
    id:    Optional[str] = None
    cls:   Optional[str] = None   # 'class' is a reserved word; JS sends 'class'

    class Config:
        extra = "allow"


class TimerHistoryEntry(BaseModel):
    """A single snapshot of a timer element at a given moment."""
    timestamp: int           # Unix epoch milliseconds
    text:      str           # Raw text of the element at that moment


class TimerVerificationMetrics(BaseModel):
    """Persistent cross-visit timer state, built by timer_tracker.js."""
    session_key:         Optional[str]  = None
    reset_detected:      Optional[bool] = None   # Timer restarted on reload
    increase_detected:   Optional[bool] = None   # Timer value went *up*
    is_revisit:          Optional[bool] = None   # This domain was scanned before
    previous_start_time: Optional[int]  = None   # ms epoch of earlier scan


class TimerInput(BaseModel):
    """Timer element payload including observation history and verification data."""
    element_id:           str
    history:              List[TimerHistoryEntry] = []
    verification_metrics: Optional[TimerVerificationMetrics] = None


class ElementInput(BaseModel):
    """A single DOM element candidate sent from the content script."""
    element_id:   str
    text:         str
    tag:          str
    bbox:         Optional[BBoxInput]       = None   # Optional so dict {} still works
    styles:       Optional[StylesInput]     = None
    attributes:   Optional[AttributesInput] = None
    image_base64: Optional[str]             = None   # Reserved for Phase 3 PaddleOCR

    class Config:
        extra = "allow"   # gracefully absorb any extra fields from future versions


class AuditPayload(BaseModel):
    """Top-level payload for POST /api/v2/audit.

    Backward compatible: the old schema { screenshot, elements } is still
    accepted because all new top-level fields are Optional.
    """
    # ── Existing fields (kept exactly as before) ─────────────────────────
    elements:   List[ElementInput]
    screenshot: Optional[str] = None   # Base64 PNG; fallback OCR crop in Phase 3

    # ── New v2 fields (ignored by current processing; reserved for later) ─
    url:      Optional[str]          = None   # Full page URL for session tracking
    timers:   Optional[List[TimerInput]] = [] # Timer history; analysed in Phase 2
    metadata: Optional[Dict[str, Any]]   = {} # Extensible page metadata

    class Config:
        extra = "allow"

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        _model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
    return _model, _tokenizer

def is_boilerplate(text):
    text_lower = text.lower().strip()

    if any(signal in text_lower for signal in BOILERPLATE_SIGNALS):
        return True

    if re.search(r'cin[:\s]', text_lower):
        return True

    if re.search(r'\b\d{10,}\b', text_lower):
        return True

    if re.search(r'\+?\d[\d\-\s]{8,}', text_lower):
        return True

    if len(text_lower) < 3:
        return True

    return False

def predict_elements(elements):
    model, tokenizer = load_model()
    texts = [e.text for e in elements]
    if not texts:
        return []
        
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors='pt')
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.sigmoid(logits).numpy()
        all_probs.append(probs)

    all_probs = np.vstack(all_probs) if all_probs else np.empty((0, len(PATTERN_LABELS)))
    results = []

    for i, element in enumerate(elements):
        probs = all_probs[i]
        detected = []
        for j, label in enumerate(PATTERN_LABELS):
            if probs[j] >= THRESHOLD:
                detected.append({'label': label, 'confidence': round(float(probs[j]), 3)})

        if not detected or is_boilerplate(element.text):
            continue

        high_conf = [d for d in detected if d['confidence'] >= 0.80]
        if high_conf:
            if element.text.lower().strip() in CATEGORY_WORDS:
                continue
            results.append({
                "element_id": element.element_id,
                "text": element.text,
                "detected_patterns": high_conf,
                "max_confidence": max(d['confidence'] for d in high_conf),
            })
    return results

def calculate_manipulation_score(detections):
    if not detections:
        return 0.0, [], {}

    detected_labels = set()
    pattern_scores = {label: [] for label in PATTERN_LABELS}

    for element in detections:
        for pattern in element['detected_patterns']:
            label = pattern['label']
            detected_labels.add(label)
            pattern_scores[label].append(pattern['confidence'] * SEVERITY_WEIGHTS[label])

    per_pattern = {}
    for label, scores in pattern_scores.items():
        if scores:
            mean_conf = np.mean(scores)
            freq_boost = min(1 + 0.05 * np.log1p(len(scores)), 1.2)
            per_pattern[label] = min(mean_conf * freq_boost, 1.0)

    n_patterns = len(per_pattern)
    if n_patterns == 0:  # Safeguard against zero-division errors
        return 0.0, [], {}
        
    pattern_sum = sum(per_pattern.values())
    base_score = (pattern_sum / n_patterns) * 75

    # Simple placeholder calculation for compound detections
    active_compounds = []
    final_score = min(base_score, 95.0)
    return round(final_score, 1), active_compounds, per_pattern

@app.post("/api/v2/audit")
async def audit_endpoint(payload: AuditPayload):
    try:
        raw_detections = predict_elements(payload.elements)
        raw_detections = [
            d
            for d in raw_detections
            if d["max_confidence"] >= 0.80
        ]
        print("\n=== DETECTIONS ===")

        for d in raw_detections:
            print(
                d["text"],
                d["max_confidence"]
            )
        score, compounds, per_pattern = calculate_manipulation_score(raw_detections)
        
        # Build explainability elements structure
        cards = []
        seen_labels = set()
        for det in raw_detections:
            for pat in det['detected_patterns']:
                lbl = pat['label']
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    cards.append({
                        "label": lbl,
                        "display_name": lbl.replace('_', ' ').title(),
                        "description": PATTERN_DESCRIPTIONS[lbl],
                        "cognitive_bias": COGNITIVE_BIAS[lbl],
                        "confidence": pat['confidence']
                    })

        return {
            "ok": True,
            "data": {
                "score": score,
                "total_detections": len(raw_detections),
                "detections": raw_detections,
                "per_pattern": per_pattern,
                "compounds": compounds,
                "cards": cards
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))