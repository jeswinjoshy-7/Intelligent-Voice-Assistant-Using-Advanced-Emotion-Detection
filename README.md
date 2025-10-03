Neural Voice: A Full-Stack Empathetic Conversational AI Platform
<img width="255" height="345" alt="Screenshot from 2025-10-04 02-42-58" src="https://github.com/user-attachments/assets/67bef30a-fcf5-40e3-a9ae-31a8238b0ce4" />


Neural Voice is a **full-stack, emotion-aware conversational AI assistant** engineered to elevate human-AI interactions. Utilizing a robust **REST architecture** and high-performance inference services, the system delivers contextually appropriate, empathetic responses synthesized with professional, emotion-adaptive voice technology.

-----

1\. Core Capabilities

The platform's core differentiator is its ability to process and adapt to human emotional states in real-time, delivering a truly personalized conversational experience.


| Feature | Technical Insight | Value Proposition |
| :--- | :--- | :--- |
| **Advanced Emotion Detection** | Multi-layered analysis (Keyword + Pattern + LLM) supporting **25+ distinct emotions** and **3 intensity levels**. | Robust, high-fidelity emotional understanding that surpasses basic sentiment analysis. |
| **Professional Voice Synthesis** | Integration with **Murf AI** (en-US-natalie) with dynamic parameter adaptation for natural, empathetic speech. | Delivers a high-quality, professional, and emotionally expressive voice output. |
| **Empathetic AI Response** | Powered by **Groq Llama 3.1** with sophisticated emotion-aware prompt engineering. | Ensures contextually and emotionally appropriate conversational flow. |
| **Real-Time Processing** | Optimized **HTTP REST pipeline** for Speech-to-Text (STT) $\rightarrow$ Emotion $\rightarrow$ LLM $\rightarrow$ Text-to-Speech (TTS) workflow. | Low-latency, stable interaction loop for seamless voice communication. |

-----

2\. Technical Architecture

The backend implements a deterministic, sequential emotion-aware pipeline, prioritizing emotional accuracy and empathetic fidelity over raw speed.
             <img width="1024" height="1024" alt="Gemini_Generated_Image_pytegepytegepyte" src="https://github.com/user-attachments/assets/f9e53a04-8c36-486a-97f1-3465af859a87" />

2.1. Processing Workflow

1.  **Input Acquisition:** Audio is uploaded via `multipart/form-data`.
2.  **Speech-to-Text (STT):** Transcription via **Google Speech Recognition**.
3.  **Emotion Detection:** Text analysis leveraging a multi-layered model (keyword matching, pattern recognition, LLM inference).
4.  **LLM Generation:** **Groq Llama 3.1** generates emotionally attuned responses using the detected context.
5.  **Text-to-Speech (TTS):** **Murf AI** synthesizes the reply, dynamically adjusting speech rate, pitch, and tone based on the detected emotion and its intensity level.
6.  **Audio Delivery:** Playback of the natural, empathetic response.

2.2. Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend / Orchestration** | **Python (FastAPI)** | Core REST endpoints, pipeline management, emotion processing. |
| **Inference (LLM)** | **Groq Llama 3.1 8B Instant** | High-speed, context-aware response generation. |
| **Emotion Engine** | **Multi-layered Analysis** | Keyword, Pattern, and LLM-based emotion classification. |
| **STT** | **Google Speech Recognition** | High-accuracy voice transcription. |
| **TTS** | **Murf AI (en-US-natalie)** | Professional voice synthesis with advanced emotional parameter control. |
| **Frontend / UI** | **HTML5, JavaScript, Web Audio API** | Voice recording, real-time visualization, and conversation interface. |
| **Audio Utilities** | **FFmpeg, SpeechRecognition** | Audio format conversion and real-time processing. |

-----

3\. Deployment and Setup

3.1. Prerequisites

  * **Python 3.8+** and `pip`
  * **FFmpeg** (System-level installation for audio processing)
  * **Modern Web Browser** with microphone support.
  * **API Keys** for Groq and Murf AI services.

3.2. Repository Setup

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/neural-voice.git
    cd neural-voice
    ```

2.  **Environment Setup:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **System Dependencies (FFmpeg):**

    | OS | Command |
    | :--- | :--- |
    | **Ubuntu/Debian** | `sudo apt update && sudo apt install ffmpeg portaudio19-dev` |
    | **macOS** | `brew install ffmpeg portaudio` |
    | **Windows** | Download FFmpeg and add the executable to the system **PATH**. |

4.  **Configuration File (`.env`):**
    Create a `.env` file based on `.env.example` and populate the API keys.

    ```
    GROQ_API_KEY="gsk_your_groq_api_key_here"
    MURF_API_KEY="your_murf_api_key_here"
    LLM_MODEL="llama-3.1-8b-instant"
    # ... other configuration settings
    ```

3.3. Running Locally

1.  **Start the Backend (FastAPI):**
    ```bash
    cd neural-voice
    source venv/bin/activate
    python backend.py
    # Server runs on http://localhost:8000
    ```
2.  **Start the Frontend:**
    ```bash
    python -m http.server 3000 # Option 1: Python HTTP server
    # or: npx http-server -p 3000 # Option 2: Node.js http-server
    ```
3.  **Access:** Navigate to `http://localhost:3000` in your browser.

-----

4\. API Reference

**Base URL:** `http://localhost:8000`

4.1. Complete Emotion-Aware Pipeline

**POST /chat**

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `text` | string | The user's input text. |

**Purpose:** Executes STT $\rightarrow$ Emotion $\rightarrow$ LLM $\rightarrow$ TTS workflow. Accepts text, returns response text, emotion analysis, and audio URL.

**Example Request:**

```bash
curl -s -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{"text": "I am so frustrated with this problem!"}'
```

**Example Response (200 OK):**

```json
{
  "response_text": "I understand that feeling anxious before a presentation is completely normal. You've got this!",
  "emotion_data": {
    "emotion": "anxiety",
    "confidence": 0.87,
    "intensity": "high",
    "methods": {
      "keyword": ["anxiety", 2.1],
      "llm": ["anxiety", 0.9]
    }
  },
  "audio_url": "https://murf.ai/user-upload/temp/response-audio.wav"
}
```

4.2. Supporting Endpoints

| Endpoint | Method | Purpose | Data Type |
| :--- | :--- | :--- | :--- |
| **`/process_speech`** | POST | Transcribes audio file and performs emotion detection. | `multipart/form-data` |
| **`/detect_emotion`** | POST | Performs multi-layered emotion detection on provided text only. | `application/json` |
| **`/synthesize_speech`** | POST | Generates emotional speech given text, emotion, and intensity. | `application/json` |
| **`/health`** | GET | Checks service availability (backend, Groq, Murf AI). | None |

-----

5\. Directory Structure

A standardized, organized file structure for rapid development and maintenance.

```
neural-voice/
├── backend.py                   # FastAPI orchestration and REST endpoints
├── index.html                   # Core frontend interface
├── requirements.txt             # Project dependencies
├── .env                         # Environment variables (Sensitive configuration)
├── README.md                    # This documentation
├── docs/
│   ├── api-reference.md         # Extended API documentation
│   └── deployment.md            # Production deployment guide
└── tests/
    ├── test_api.py              # API endpoint unit tests
    └── test_integration.py      # End-to-end pipeline tests
```

-----

6\. Production Deployment Notes

For robust, production-ready deployment, adhere to the following best practices:

  * **Service Hosting:** Deploy the backend using **Gunicorn + Uvicorn workers**, tuning the worker count to the host CPU cores.
  * **Security:** Enforce **HTTPS** for all client-side interactions to ensure microphone access and data security.
  * **API Management:** Implement strict **rate limiting** and request size controls at the ASGI or gateway layer.
  * **Observability:** Log emotion analytics and system events using **structured logging** for conversation insights and error tracking.
  * **Optimization:** Implement **audio response caching** to reduce latency and API costs for frequently triggered emotional phrases.

-----

7\. Roadmap and Future Development

The platform is evolving towards a more personalized and integrated conversational experience.

  * **Real-time Streaming:** Implement **WebSocket** for streaming emotion detection and low-latency response delivery.
  * **Contextual Memory:** Integrate **Conversation Memory** for multi-turn emotion tracking and deeper personalization.
  * **Advanced Profiling:** Develop **Psychological Profiling** based on emotion patterns for mental health and customer service applications.
  * **Mobile Integration:** Release **Mobile SDKs** for iOS and Android with optimized on-device audio preprocessing.
