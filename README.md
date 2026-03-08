# AI-PROOF — Content Verification System

> Multi-model AI content verification with blockchain proof, explainable AI, and real-time scanning.

## Architecture

```
futuristic-ai-proof/
├── index.html          # Landing page
├── analyze.html        # Submission form
├── result.html         # Verification result
├── timeline.html       # Blockchain history
├── lab.html            # Developer tools
├── vercel.json         # Deployment config
└── assets/
    ├── css/
    │   ├── theme.css       # CSS variables & palette
    │   ├── style.css       # Global base styles
    │   └── animations.css  # Keyframes & motion
    └── js/
        ├── config.js       # Configuration & thresholds
        ├── app.js          # Bootstrap
        ├── router.js       # Client navigation
        ├── ui.js           # Shared UI utilities
        ├── effects.js      # Visual effects
        ├── ai.js           # AI detection engine
        ├── consensus.js    # Multi-model aggregation
        ├── ai-explain.js   # Explainable AI highlights
        ├── realtime-scan.js# Live typing analysis
        ├── trust-score.js  # Score rendering
        ├── blockchain.js   # Proof-of-work ledger
        ├── merkle.js       # Merkle tree utilities
        ├── db.js           # LocalStorage persistence
        ├── hash.js         # Cryptographic hashing
        ├── blockchain-export.js
        ├── blockchain-import.js
        ├── qr.js           # QR code renderer
        ├── version-compare.js  # Text diff engine
        ├── analyze-page.js
        ├── result-page.js
        └── timeline-page.js
backend/
├── main.py             # FastAPI app
├── requirements.txt
└── services/
    ├── ai_text.py          # Text AI detection
    ├── fake_news.py        # Misinformation detection
    ├── ai_image.py         # Image AI detection
    ├── explainable_ai.py   # SHAP/LIME explainability
    ├── consensus.py        # Weighted vote aggregation
    ├── blockchain_verify.py# Chain verification
    └── version_compare.py  # Text diff
```

## Quick Start

### Frontend
Open `index.html` in a browser or serve with any static server:
```bash
npx serve .
```

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
API runs at `http://localhost:8000`. The frontend auto-detects localhost and routes requests there.

## Features

| Feature | Status |
|---------|--------|
| Text AI detection | ✅ (mock + API) |
| Multi-model consensus | ✅ |
| Explainable highlights | ✅ |
| Blockchain proof-of-work | ✅ |
| Merkle tree | ✅ |
| Real-time scan | ✅ |
| Image detection | ✅ (mock + API) |
| URL analysis | ✅ |
| Version diff | ✅ |
| Export / Import chain | ✅ |
| QR share | ✅ |

## Connecting Real Models

Replace mock implementations in `backend/services/` with real HuggingFace pipelines:

```python
from transformers import pipeline
detector = pipeline("text-classification", model="roberta-base-openai-detector")
result = detector(text[:512])
```

## License
MIT