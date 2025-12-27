# JUSTICE AI - Technical Stack Documentation

> A comprehensive guide to the technologies powering the AI-assisted FIR Analysis and Investigation Support Dashboard

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Backend Technologies](#backend-technologies)
4. [AI/ML Technologies](#aiml-technologies)
5. [Image Processing & Computer Vision](#image-processing--computer-vision)
6. [Frontend Technologies](#frontend-technologies)
7. [Database Layer](#database-layer)
8. [External Services & APIs](#external-services--apis)
9. [Reporting & Communication](#reporting--communication)
10. [Development Tools](#development-tools)
11. [Feature-to-Tech Mapping](#feature-to-tech-mapping)

---

## Project Overview

**JUSTICE AI** is a hackathon project designed to assist law enforcement with FIR (First Information Report) analysis. It extracts text from FIR images, generates investigative questions, creates character profiles, builds investigation roadmaps, and provides an AI-powered chat advisor.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JUSTICE AI ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   Frontend   │    │                    Backend                        │   │
│  │              │    │  ┌─────────────────────────────────────────────┐  │   │
│  │  HTML/CSS    │◄──►│  │              Flask Application              │  │   │
│  │  JavaScript  │    │  │  ┌─────────┐  ┌──────────┐  ┌───────────┐  │  │   │
│  │  MapLibre    │    │  │  │ Routes  │  │ Services │  │  Models   │  │  │   │
│  │  Spline 3D   │    │  │  └────┬────┘  └────┬─────┘  └─────┬─────┘  │  │   │
│  └──────────────┘    │  │       │            │              │        │  │   │
│                      │  └───────┼────────────┼──────────────┼────────┘  │   │
│                      │          │            │              │           │   │
│                      │  ┌───────▼────────────▼──────────────▼────────┐  │   │
│                      │  │           Processing Layer                  │  │   │
│                      │  │  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │  │   │
│                      │  │  │  OCR    │ │   AI     │ │  Face        │ │  │   │
│                      │  │  │Tesseract│ │ Analysis │ │ Recognition  │ │  │   │
│                      │  │  │ OpenCV  │ │ OpenAI   │ │ DeepFace     │ │  │   │
│                      │  │  └─────────┘ │ Gemini   │ │ LBP/ORB      │ │  │   │
│                      │  │              └──────────┘ └──────────────┘ │  │   │
│                      │  └────────────────────────────────────────────┘  │   │
│                      └──────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Data & External Services                      │   │
│  │  ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌─────────────────┐  │   │
│  │  │ SQLite   │  │ Google Apps   │  │ Nominatim│  │  Gmail SMTP     │  │   │
│  │  │ Database │  │ Script Proxy  │  │ Geocoding│  │  Email Service  │  │   │
│  │  └──────────┘  └───────────────┘  └──────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Backend Technologies

### 1. Flask (Web Framework)

| Aspect | Details |
|--------|---------|
| **Package** | `flask`, `flask-cors` |
| **Version** | Latest stable |
| **Purpose** | Core web application framework |

**Why Flask?**
- **Lightweight & Minimal**: Perfect for hackathon projects where rapid development is crucial
- **Flexibility**: No rigid project structure, allowing custom architecture
- **Easy Integration**: Seamless integration with AI/ML libraries (OpenCV, TensorFlow, etc.)
- **Blueprint Support**: Modular route organization for clean code structure
- **Jinja2 Templating**: Built-in HTML templating for server-side rendering

**Used For:**
- Route handling (`/upload`, `/rescan`, `/chat`, `/admin`, `/match-image`)
- Request/Response management
- File upload processing
- Session management
- Template rendering

```python
# Example: Blueprint-based routing
main = Blueprint("main", __name__)

@main.route("/upload", methods=["POST"])
def upload():
    # Handle FIR image upload
    ...
```

---

### 2. python-dotenv (Environment Management)

| Aspect | Details |
|--------|---------|
| **Package** | `python-dotenv` |
| **Purpose** | Secure environment variable management |

**Why python-dotenv?**
- **Security**: Keeps API keys and secrets out of source code
- **Environment Flexibility**: Different configs for dev/prod environments
- **12-Factor App Compliance**: Follows modern application security practices

**Environment Variables Managed:**
- `OPENAI_API_KEY` - OpenAI API access
- `GEMINI_API_KEY` - Google Gemini API access
- `GEMINI_APPS_SCRIPT_URL` - Apps Script proxy URL
- `REPORT_EMAIL_USER` / `REPORT_EMAIL_PASSWORD` - SMTP credentials

---

### 3. Werkzeug (WSGI Utilities)

| Aspect | Details |
|--------|---------|
| **Package** | Built into Flask |
| **Purpose** | Secure file handling |

**Why Werkzeug?**
- **Security**: `secure_filename()` prevents directory traversal attacks
- **File Validation**: Safe handling of user-uploaded files

```python
from werkzeug.utils import secure_filename
filename = secure_filename(file.filename)
```

---

## AI/ML Technologies

### 1. OpenAI GPT-4o-mini (Primary AI Engine)

| Aspect | Details |
|--------|---------|
| **Package** | `openai` |
| **Model** | `gpt-4o-mini` |
| **Purpose** | Text analysis, question generation, insights extraction |

**Why GPT-4o-mini?**
- **Cost-Effective**: Lower cost than GPT-4 while maintaining quality
- **Speed**: Faster inference times suitable for real-time applications
- **JSON Mode**: Native structured output support for reliable parsing
- **Multimodal**: Supports both text and image inputs
- **Context Window**: Sufficient for FIR document analysis

**Features Powered:**

| Feature | Function | Temperature | Max Tokens |
|---------|----------|-------------|------------|
| OCR Text Cleanup | `format_text()` | 0.2 | 600 |
| Question Generation | `build_questions()` | 0.4 | 500 |
| Character Profile Extraction | `extract_character_profiles()` | 0.2 | 400 |
| Location Extraction | `extract_locations()` | 0.2 | 400 |
| Investigation Roadmap | `build_roadmap()` | 0.35 | 700 |
| Legal Sections Analysis | `build_judicial_sections()` | 0.25 | 500 |
| Audio Transcript Analysis | `_analyze_audio_insights()` | 0.35 | 450 |
| Facial Expression Analysis | `_analyze_facial_truth_signals()` | 0.25 | 400 |

```python
client = OpenAI(api_key=api_key)
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    temperature=0.4,
    max_tokens=500,
    response_format={"type": "json_object"},  # Structured output
)
```

---

### 2. Google Gemini (Secondary AI Engine)

| Aspect | Details |
|--------|---------|
| **Package** | `google-genai` |
| **Purpose** | Backup AI provider, chatbot functionality |

**Why Gemini?**
- **Redundancy**: Fallback when OpenAI is unavailable or rate-limited
- **Cost Distribution**: Spread API costs across providers
- **Feature Parity**: Comparable capabilities for text analysis

**Implementation Pattern:**
```python
def _invoke_chat_response(...):
    """Try Gemini first, then fall back to OpenAI"""
    reply = _invoke_gemini_text(...)
    if reply:
        return reply
    # Fallback to OpenAI
    return _invoke_openai_text(...)
```

---

### 3. OpenAI Whisper (Speech Recognition)

| Aspect | Details |
|--------|---------|
| **Model** | `whisper-1` |
| **Purpose** | Audio/video transcription for interrogation analysis |

**Why Whisper?**
- **Accuracy**: State-of-the-art speech recognition
- **Multilingual**: Supports multiple languages (important for Indian FIRs)
- **Noise Robustness**: Handles real-world audio quality
- **Easy Integration**: Native OpenAI API integration

```python
resp = client.audio.transcriptions.create(model="whisper-1", file=fh)
```

---

### 4. OpenAI TTS (Text-to-Speech)

| Aspect | Details |
|--------|---------|
| **Model** | `gpt-4o-mini-tts` |
| **Voice** | `alloy` |
| **Purpose** | Voice responses for chat advisor |

**Why OpenAI TTS?**
- **Natural Voice**: High-quality, natural-sounding speech
- **Low Latency**: Real-time audio generation
- **Multiple Voices**: Customizable voice options

---

### 5. DeepFace (Face Recognition)

| Aspect | Details |
|--------|---------|
| **Package** | `deepface`, `tf-keras` |
| **Model** | VGG-Face |
| **Detector Backend** | OpenCV |
| **Purpose** | Criminal face matching and verification |

**Why DeepFace?**
- **Pre-trained Models**: No training required, works out-of-the-box
- **High Accuracy**: VGG-Face achieves >97% accuracy on LFW benchmark
- **Multiple Backends**: Supports various face detection methods
- **Verification & Identification**: Both 1:1 and 1:N matching

**Features:**

| Function | Purpose |
|----------|---------|
| `_verify_faces_deepface()` | Compare two faces for identity match |
| `_get_face_embedding_deepface()` | Extract 2622-dimensional face embeddings |
| `_find_best_image_match()` | Search criminal database for matches |

```python
result = DeepFace.verify(
    img1_path=path1,
    img2_path=path2,
    model_name="VGG-Face",
    detector_backend="opencv",
    enforce_detection=False,
    align=True,
)
```

---

## Image Processing & Computer Vision

### 1. OpenCV (Computer Vision)

| Aspect | Details |
|--------|---------|
| **Package** | `opencv-contrib-python` |
| **Purpose** | Image preprocessing, face detection, feature extraction |

**Why OpenCV?**
- **Industry Standard**: Most widely used computer vision library
- **Performance**: Optimized C++ backend with Python bindings
- **Comprehensive**: 2500+ algorithms for image/video processing
- **Hardware Acceleration**: GPU support for heavy processing

**Features Implemented:**

| Feature | OpenCV Functions Used |
|---------|----------------------|
| Image Enhancement | `CLAHE`, `fastNlMeansDenoising` |
| Thresholding | `adaptiveThreshold`, `bitwise_not` |
| Face Detection | `CascadeClassifier` (Haar cascades) |
| Feature Extraction | `ORB_create`, `detectAndCompute` |
| Video Processing | `VideoCapture`, frame sampling |
| Image Encoding | `imencode`, `imdecode` |
| Color Conversion | `cvtColor` (BGR↔GRAY↔RGB) |
| Resizing | `resize` with various interpolations |
| DCT (Perceptual Hash) | `dct` for image similarity |

**OCR Preprocessing Pipeline:**
```python
def extract_text(image_path):
    # 1. Read and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Upscale small images
    gray = cv2.resize(gray, ..., interpolation=cv2.INTER_CUBIC)
    
    # 3. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 4. Denoise
    denoised = cv2.fastNlMeansDenoising(equalized, h=10)
    
    # 5. Adaptive threshold
    binary = cv2.adaptiveThreshold(denoised, 255, ...)
```

---

### 2. Tesseract OCR (Text Extraction)

| Aspect | Details |
|--------|---------|
| **Package** | `pytesseract` |
| **Engine** | Tesseract 4.x+ with LSTM |
| **Purpose** | Extract text from FIR images |

**Why Tesseract?**
- **Open Source**: Free, no API costs for OCR
- **LSTM Engine**: Modern deep learning-based recognition
- **Multi-language**: Supports 100+ languages including Hindi
- **Configurable**: PSM/OEM modes for different document types

**Configuration Used:**
```python
config = "--oem 3 --psm 6"
# --oem 3: LSTM only (most accurate)
# --psm 6: Assume uniform block of text
```

---

### 3. Pillow (Image I/O)

| Aspect | Details |
|--------|---------|
| **Package** | `Pillow` |
| **Purpose** | Image format handling and conversion |

**Why Pillow?**
- **Format Support**: JPEG, PNG, BMP, TIFF, WebP, etc.
- **Easy I/O**: Simple file reading/writing
- **Integration**: Works seamlessly with NumPy and OpenCV

---

### 4. NumPy (Numerical Computing)

| Aspect | Details |
|--------|---------|
| **Package** | `numpy` |
| **Purpose** | Array operations, mathematical computations |

**Why NumPy?**
- **Performance**: Vectorized operations 50x faster than Python loops
- **Memory Efficiency**: Contiguous memory arrays
- **Foundation**: Required by OpenCV, DeepFace, and ML libraries

**Use Cases:**
- Image array manipulation
- Face embedding computations
- Cosine similarity calculations
- Histogram operations for LBP

```python
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

---

### 5. Custom Face Recognition (Fallback)

When DeepFace is unavailable, the system uses custom algorithms:

| Algorithm | Purpose | Implementation |
|-----------|---------|----------------|
| **LBP (Local Binary Patterns)** | Face texture encoding | `_compute_lbp_histogram()` |
| **ORB (Oriented FAST and Rotated BRIEF)** | Feature point matching | `_face_embedding_cv()` |
| **pHash (Perceptual Hash)** | Image similarity | `_image_phash()` |
| **Haar Cascades** | Face detection | OpenCV's `haarcascade_frontalface_default.xml` |

---

## Frontend Technologies

### 1. HTML5 + Jinja2 Templates

| Aspect | Details |
|--------|---------|
| **Templates** | `landing.html`, `dashboard.html`, `admin.html`, `index.html` |
| **Engine** | Jinja2 (Flask built-in) |
| **Purpose** | Server-side rendered UI |

**Why Server-Side Rendering?**
- **SEO Friendly**: Content visible to search engines
- **Fast Initial Load**: No JavaScript framework overhead
- **Simplicity**: Perfect for hackathon timeline
- **Data Binding**: Direct Python → HTML data flow

---

### 2. CSS3 (Styling)

| Aspect | Details |
|--------|---------|
| **Approach** | Custom CSS with modern features |
| **Fonts** | Space Grotesk, Sora (Google Fonts) |
| **Theme** | Dark mode with glassmorphism |

**Design Features:**
- **CSS Variables**: Consistent theming
- **Animations**: `@keyframes` for blob animations, fade-ins
- **Glassmorphism**: `backdrop-filter: blur()` effects
- **Responsive**: `clamp()` for fluid typography
- **Grid/Flexbox**: Modern layout systems

```css
.card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    backdrop-filter: blur(14px);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5);
}
```

---

### 3. MapLibre GL JS (Mapping)

| Aspect | Details |
|--------|---------|
| **Package** | `maplibre-gl@3.6.1` (CDN) |
| **Purpose** | Interactive location mapping |

**Why MapLibre?**
- **Open Source**: Free alternative to Mapbox GL JS
- **Vector Tiles**: Smooth, performant map rendering
- **Customizable**: Full control over map styling
- **Mobile-Friendly**: Touch gestures and responsive

**Integration with Nominatim:**
```javascript
// Geocode addresses from FIR
const response = await fetch(
    `https://nominatim.openstreetmap.org/search?q=${address}&format=json`
);
// Plot markers on MapLibre map
```

---

### 4. Spline 3D (Visual Effects)

| Aspect | Details |
|--------|---------|
| **Package** | `@splinetool/viewer@1.12.27` (unpkg CDN) |
| **Purpose** | 3D background animations |

**Why Spline?**
- **Web-Native 3D**: No WebGL coding required
- **Visual Impact**: Professional 3D animations
- **Easy Integration**: Web component-based

```html
<script type="module" src="https://unpkg.com/@splinetool/viewer@1.12.27/build/spline-viewer.js"></script>
<spline-viewer class="spline-bg" url="..."></spline-viewer>
```

---

## Database Layer

### SQLite (Local Database)

| Aspect | Details |
|--------|---------|
| **Package** | `sqlite3` (Python standard library) |
| **File** | `database/data.db` |
| **Purpose** | Criminal records storage with image BLOBs |

**Why SQLite?**
- **Zero Configuration**: No server setup required
- **Single File**: Easy to deploy and backup
- **BLOB Support**: Stores images directly in database
- **Sufficient Performance**: Handles thousands of records easily
- **Perfect for Hackathons**: Instant setup, no dependencies

**Schema:**
```sql
CREATE TABLE criminals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    alias TEXT,
    image BLOB NOT NULL,
    mime_type TEXT DEFAULT 'image/jpeg',
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
```

**Database Operations:**
```python
# Insert criminal record
def _insert_criminal(name, alias, notes, image_bytes, mime_type):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO criminals (name, alias, image, mime_type, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (name, alias, image_bytes, mime_type, notes))
        conn.commit()
```

---

## External Services & APIs

### 1. Nominatim (Geocoding)

| Aspect | Details |
|--------|---------|
| **Service** | OpenStreetMap's Nominatim |
| **Purpose** | Convert addresses to coordinates |
| **Cost** | Free (with usage limits) |

**Why Nominatim?**
- **Free**: No API key required for light usage
- **Comprehensive**: Global address coverage
- **Open Data**: OpenStreetMap community-maintained

---

### 2. Google Apps Script Proxy

| Aspect | Details |
|--------|---------|
| **Purpose** | Proxy for Gemini API calls |
| **Benefit** | Bypass regional restrictions |

**Architecture:**
```
Client → Flask Backend → Apps Script → Gemini API
```

```python
payload = {
    "apiKey": api_key,
    "systemPrompt": system_prompt,
    "userPrompt": user_prompt,
    "temperature": temperature,
    "maxOutputTokens": max_output_tokens,
}
resp = requests.post(appscript_url, json=payload, timeout=25)
```

---

## Reporting & Communication

### 1. ReportLab (PDF Generation)

| Aspect | Details |
|--------|---------|
| **Package** | `reportlab` |
| **Purpose** | Generate investigation PDF reports |

**Why ReportLab?**
- **Pure Python**: No external dependencies
- **Full Control**: Precise layout control
- **Professional Output**: Publication-quality PDFs

**PDF Contents:**
- FIR cleaned text
- Character-based questions
- General investigative questions
- Character profiles
- Investigation roadmap
- Detected locations

```python
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def _generate_pdf_report(...):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    # Add sections...
    pdf.save()
    return buffer.read()
```

---

### 2. SMTP (Email Delivery)

| Aspect | Details |
|--------|---------|
| **Package** | `smtplib`, `email` (Python standard library) |
| **Server** | Gmail SMTP (`smtp.gmail.com:587`) |
| **Purpose** | Email PDF reports to recipients |

**Why SMTP Direct?**
- **No Dependencies**: Uses Python standard library
- **Gmail Integration**: Widely accessible
- **Attachment Support**: Native PDF attachment handling

```python
def _send_report_email(recipient, pdf_bytes):
    msg = EmailMessage()
    msg["Subject"] = "JUSTICE AI - FIR Investigation Report"
    msg.add_attachment(pdf_bytes, maintype="application", 
                       subtype="pdf", filename="fir_report.pdf")
    
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
```

---

## Development Tools

### 1. Requests (HTTP Client)

| Aspect | Details |
|--------|---------|
| **Package** | `requests` |
| **Purpose** | External API calls |

**Why Requests?**
- **Simple API**: Human-friendly HTTP for Python
- **Timeout Support**: Prevents hanging connections
- **JSON Handling**: Built-in JSON encoding/decoding

---

### 2. Project Structure

```
Avishkaar/
├── run.py                 # Flask entrypoint
├── config.py              # Configuration class
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (not in git)
├── app/
│   ├── __init__.py        # App factory
│   ├── routes.py          # All route handlers + utilities
│   ├── models.py          # Database models
│   ├── services/          # Business logic (FIR analyzer)
│   ├── templates/         # Jinja2 HTML templates
│   └── static/            # CSS, JS, assets
├── database/
│   └── data.db            # SQLite database
└── doc/
    └── TECH_STACK.md      # This documentation
```

---

## Feature-to-Tech Mapping

### Quick Reference Table

| Feature | Primary Tech | Supporting Tech | Why This Stack |
|---------|--------------|-----------------|----------------|
| **FIR Image Upload** | Flask | Werkzeug | Secure file handling, easy integration |
| **OCR Extraction** | Tesseract | OpenCV (preprocessing) | Free, accurate with LSTM, multi-language |
| **Text Cleanup** | OpenAI GPT-4o-mini | - | High-quality text understanding |
| **Question Generation** | OpenAI GPT-4o-mini | JSON mode | Structured, relevant investigative questions |
| **Character Profiles** | OpenAI GPT-4o-mini | JSON mode | Extract structured person data |
| **Legal Sections** | OpenAI GPT-4o-mini | - | Indian law knowledge built-in |
| **Investigation Roadmap** | OpenAI GPT-4o-mini | - | Actionable step generation |
| **Location Extraction** | OpenAI GPT-4o-mini | - | NLP-based address detection |
| **Map Visualization** | MapLibre GL | Nominatim | Free, open-source, interactive |
| **AI Chat Advisor** | Gemini + OpenAI | Apps Script proxy | Redundancy, cost distribution |
| **Voice Responses** | OpenAI TTS | - | Natural, real-time speech |
| **Audio Transcription** | Whisper | - | State-of-the-art accuracy |
| **Video Analysis** | OpenCV | GPT-4o-mini (vision) | Frame extraction + expression analysis |
| **Criminal Database** | SQLite | - | Zero-config, BLOB support |
| **Face Matching** | DeepFace (VGG-Face) | OpenCV (detection) | Pre-trained, high accuracy |
| **Fallback Face Matching** | LBP + ORB | NumPy, OpenCV | Works without TensorFlow |
| **PDF Reports** | ReportLab | - | Pure Python, full control |
| **Email Delivery** | SMTP | Gmail | No dependencies, reliable |
| **3D Background** | Spline | - | Professional visual impact |
| **Responsive UI** | CSS3 | Google Fonts | Modern, glassmorphism design |

---

## Technology Selection Rationale

### Why This Stack Works for JUSTICE AI

1. **Hackathon-Optimized**
   - Flask's minimal setup time
   - SQLite's zero configuration
   - Pre-trained AI models (no training needed)

2. **Cost-Effective**
   - GPT-4o-mini: Lower cost, sufficient quality
   - Tesseract/OpenCV: Free, open-source
   - SQLite: No database hosting costs

3. **Accuracy-Focused**
   - DeepFace: 97%+ face recognition accuracy
   - Whisper: Industry-leading transcription
   - CLAHE + denoising: Optimized OCR preprocessing

4. **Redundancy Built-In**
   - Gemini ↔ OpenAI fallback for chat
   - DeepFace ↔ LBP/ORB fallback for faces
   - Binary ↔ Grayscale OCR comparison

5. **Real-World Deployment Ready**
   - Environment-based configuration
   - Secure file handling
   - Email delivery for reports

---

## Version Information

| Component | Version/Model |
|-----------|---------------|
| Python | 3.10+ |
| Flask | Latest |
| OpenAI Model | gpt-4o-mini |
| Whisper | whisper-1 |
| TTS | gpt-4o-mini-tts |
| DeepFace Model | VGG-Face |
| MapLibre GL | 3.6.1 |
| Spline Viewer | 1.12.27 |
| Tesseract | 4.x+ (LSTM) |

---

## Conclusion

JUSTICE AI leverages a modern, cost-effective tech stack optimized for rapid hackathon development while maintaining production-quality features. The combination of Flask's simplicity, OpenAI's powerful models, DeepFace's accurate recognition, and open-source tools like Tesseract and MapLibre creates a robust FIR analysis platform.

The architecture emphasizes:
- **Modularity**: Easy to extend or swap components
- **Fallbacks**: Multiple redundancy paths
- **Security**: Environment-based secrets, secure file handling
- **User Experience**: Modern UI, real-time AI responses

---

*Document generated for JUSTICE AI Hackathon Project*
*Last Updated: December 2024*
