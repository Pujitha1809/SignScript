# SignScript — Real-Time ASL to Text Translator

> Point your camera. Sign a letter. Watch it appear on screen.

SignScript is a full-stack, real-time **American Sign Language (A–Z) translator** built from scratch — custom dataset, custom-trained CNN, live webcam inference, and text-to-speech output. No paid APIs. No pre-built models. Everything trained and deployed locally.

---

## Demo

| Feature | Preview |
|--------|---------|
| 🖐 Hand detection | MediaPipe crops your hand in real time |
| 🔤 Letter prediction | CNN predicts with up to 99%+ confidence |
| 🧱 Word builder | Auto-builds words letter by letter |
| 🔊 Text-to-speech | Speaks your sentence aloud |

---

## How It Works

```
📷 Webcam (browser)
        │
        │  base64 JPEG frame every ~200ms
        ▼
🐍 Flask API  (/predict)
        ├── MediaPipe Hands  ──▶  crop hand ROI
        └── CNN (sign_model.keras)  ──▶  letter + confidence
        │
        │  JSON response
        ▼
🌐 Frontend
        ├── Displays letter + confidence bar
        ├── Shows top-3 predictions
        ├── Builds words automatically
        └── 🔊 Text-to-Speech (Web Speech API)
```

---

## Project Phases

| # | Phase | Tools | Status |
|---|-------|-------|--------|
| 1 | Data Collection — webcam images per ASL letter | OpenCV, MediaPipe | ✅ Done |
| 2 | Model Training — custom CNN, 20 epochs | TensorFlow, Keras | ✅ Done |
| 3 | Backend — REST API for real-time prediction | Flask, MediaPipe | ✅ Done |
| 4 | Frontend — live webcam UI, word builder, TTS | HTML, CSS, JS | ✅ Done |
| 5 | Polish — confidence display, sentence builder, README | — | ✅ Done |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Architecture | Custom CNN — 6 Conv layers + 3 Dense layers |
| Input size | 64 × 64 RGB |
| Classes | 29 (A–Z + `del`, `nothing`, `space`) |
| Total parameters | ~4.6M |
| Training accuracy | **99.76%** |
| Validation accuracy | **95.99%** |
| Dataset | ASL Alphabet (Kaggle) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model training | TensorFlow 2.x + Keras |
| Hand detection | MediaPipe Hands |
| Backend | Python + Flask |
| Frontend | Vanilla HTML / CSS / JS |
| Speech output | Web Speech API (browser-native) |

---

## Project Structure

```
SignLanguage/
├── model/
│   ├── sign_model.keras       # Trained CNN model
│   └── class_labels.json      # Label index map
├── static/
│   └── index.html             # Frontend (webcam + UI)
├── app.py                     # Flask backend
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Clone / download the project

```bash
cd SignLanguage
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
python app.py
```

Open **http://localhost:5000** in your browser, click **▶ Start Camera**, and start signing!

---

## API Reference

### `POST /predict`

Accepts a base64-encoded webcam frame and returns the predicted ASL letter.

**Request**
```json
{ "frame": "data:image/jpeg;base64,/9j/4AAQ..." }
```

**Response**
```json
{
  "hand_detected": true,
  "letter":        "A",
  "confidence":    0.994,
  "accepted":      true,
  "top3": [
    { "label": "A", "confidence": 0.994 },
    { "label": "S", "confidence": 0.004 },
    { "label": "E", "confidence": 0.001 }
  ],
  "bbox": [120, 80, 310, 280]
}
```

### `GET /health`

```json
{ "status": "ok", "classes": 29, "model": "model/sign_model.keras" }
```

---

## Frontend Features

- **Live mirrored webcam** with purple bounding box around detected hand
- **Animated letter display** — pops in with each new prediction
- **Confidence bar** — fills based on model certainty
- **Top-3 predictions** — live bar chart of the closest matches
- **Word builder** — auto-adds letters after N stable frames (configurable)
- **Space / Delete / Clear** controls for editing
- **🔊 Speak button** — reads the full sentence aloud
- **Settings panel** — tune hold frames, API interval, min confidence, auto-add

---

## Tips for Best Results

- ✅ Use a **plain, light background** — the model was trained on clean images
- ✅ Keep your hand **well-lit** and centred in frame
- ✅ Hold each sign **steady** for ~1 second before it registers
- ⚙️ Increase **Hold frames** in settings if letters trigger too fast
- 📦 The purple box shows exactly what MediaPipe detected

---

## Requirements

```
flask
flask-cors
tensorflow
keras
mediapipe
opencv-python-headless
numpy
```

---

## License

MIT — free to use, modify, and share.

---

*Built end-to-end from data collection to deployment — no shortcuts, no pre-trained models.*

---

## Author

**Pujitha Mamidishetty**