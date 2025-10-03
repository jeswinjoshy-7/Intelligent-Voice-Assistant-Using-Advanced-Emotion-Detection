# backend.py - FIXED VERSION
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import requests
import json
import speech_recognition as sr
from groq import Groq
import tempfile
import time
import re
import io
from typing import Dict, Any, Optional
import subprocess

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_NAME = 'llama-3.1-8b-instant'
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
MURF_API_KEY = os.getenv('MURF_API_KEY', '')
MURF_GENERATE_URL = "https://api.murf.ai/v1/speech/generate"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NatalieBackend')

app = FastAPI(title="Natalie Voice Assistant API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class TextInput(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    intensity: str
    methods: Dict[str, Any]

class VoiceResponse(BaseModel):
    audio_url: Optional[str] = None
    emotion_data: Dict[str, Any]

class ChatResponse(BaseModel):
    response_text: str
    emotion_data: Dict[str, Any]
    audio_url: Optional[str] = None  # FIXED: Made optional

class AdvancedEmotionDetector:
    def __init__(self):
        self.emotion_keywords = {
            'joy': ['happy', 'joyful', 'glad', 'pleased', 'cheerful', 'delighted', 'content'],
            'excitement': ['excited', 'thrilled', 'pumped', 'energized', 'hyped', 'enthusiastic', 'amazing', 'fantastic', 'incredible'],
            'love': ['love', 'adore', 'cherish', 'treasure', 'appreciate', 'care about', 'fond of'],
            'gratitude': ['thank you', 'grateful', 'thankful', 'appreciate', 'blessed', 'thanks'],
            'pride': ['proud', 'accomplished', 'achieved', 'succeeded', 'victory', 'triumph'],
            'sadness': ['sad', 'unhappy', 'down', 'blue', 'gloomy', 'dejected', 'melancholy'],
            'grief': ['devastated', 'heartbroken', 'mourning', 'loss', 'grieving', 'bereaved'],
            'disappointment': ['disappointed', 'let down', 'failed', 'expected more', 'underwhelmed'],
            'loneliness': ['lonely', 'alone', 'isolated', 'abandoned', 'solitary', 'forsaken'],
            'despair': ['hopeless', 'despairing', 'defeated', 'given up', 'helpless'],
            'anger': ['angry', 'mad', 'furious', 'livid', 'outraged', 'irate', 'pissed'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'aggravated', 'bothered', 'irked'],
            'resentment': ['resentful', 'bitter', 'grudge', 'unfair', 'wronged', 'betrayed'],
            'rage': ['enraged', 'infuriated', 'seething', 'boiling', 'explosive'],
            'fear': ['scared', 'afraid', 'frightened', 'terrified', 'petrified'],
            'anxiety': ['anxious', 'worried', 'nervous', 'stressed', 'tense', 'uneasy', 'concerned'],
            'panic': ['panicked', 'overwhelmed', 'frantic', 'desperate', 'crisis'],
            'insecurity': ['insecure', 'uncertain', 'doubtful', 'unsure', 'vulnerable'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'wow'],
            'confusion': ['confused', 'puzzled', 'bewildered', 'perplexed', 'don\'t understand'],
            'curiosity': ['curious', 'interested', 'intrigued', 'wondering', 'fascinated'],
            'embarrassment': ['embarrassed', 'ashamed', 'humiliated', 'awkward', 'uncomfortable'],
            'guilt': ['guilty', 'regret', 'sorry', 'remorseful', 'shouldn\'t have'],
            'envy': ['jealous', 'envious', 'wish I had', 'not fair', 'they have'],
            'nostalgia': ['miss', 'remember', 'used to', 'back then', 'nostalgic', 'reminds me'],
            'boredom': ['bored', 'boring', 'dull', 'uninteresting', 'tedious', 'nothing to do'],
            'contempt': ['disgusted', 'revolted', 'sick of', 'can\'t stand', 'pathetic'],
            'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'at ease'],
            'confident': ['confident', 'sure', 'certain', 'self-assured', 'capable', 'strong'],
            'hopeful': ['hopeful', 'optimistic', 'positive', 'looking forward', 'bright future'],
            'determined': ['determined', 'motivated', 'focused', 'committed', 'driven', 'will do'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'usual', 'regular'],
            'professional': ['business', 'work', 'formal', 'official', 'meeting', 'presentation']
        }
        
        self.intensifiers = {
            'high': ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely', 'utterly', 'so', 'really'],
            'medium': ['quite', 'fairly', 'somewhat', 'rather', 'pretty', 'kind of'],
            'low': ['a bit', 'a little', 'slightly', 'somewhat']
        }
        
        self.negations = ['not', 'never', 'don\'t', 'won\'t', 'can\'t', 'couldn\'t', 'wouldn\'t', 'shouldn\'t', 'isn\'t', 'aren\'t']

    def detect_emotion_intensity(self, text):
        text_lower = text.lower()
        
        for intensifier in self.intensifiers['high']:
            if intensifier in text_lower:
                return 'high'
        
        if any(word.isupper() for word in text.split() if len(word) > 2):
            return 'high'
        
        if text.count('!') >= 2:
            return 'high'
        elif text.count('!') == 1:
            return 'medium'
        
        for intensifier in self.intensifiers['medium']:
            if intensifier in text_lower:
                return 'medium'
        
        for intensifier in self.intensifiers['low']:
            if intensifier in text_lower:
                return 'low'
        
        return 'medium'

    def keyword_based_detection(self, text):
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    weight = len(keyword.split())
                    score += weight
            
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            return primary_emotion, emotion_scores[primary_emotion]
        
        return 'neutral', 0

    def llm_emotion_detection(self, text):
        if not groq_client:
            return None, 0, 'medium'
        
        try:
            emotion_list = list(self.emotion_keywords.keys())
            emotion_prompt = f"""
Analyze the emotional tone and intensity of this text. Consider:
- Primary emotion being expressed
- Intensity level (low, medium, high)

Text: "{text}"

Available emotions: {', '.join(emotion_list)}

Respond in this exact format:
Primary: [emotion]
Intensity: [low/medium/high]
Confidence: [0.1-1.0]"""

            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": emotion_prompt}],
                model=MODEL_NAME,
                temperature=0.3,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            
            primary_match = re.search(r'Primary:\s*(\w+)', result)
            intensity_match = re.search(r'Intensity:\s*(\w+)', result)
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', result)
            
            if primary_match:
                emotion = primary_match.group(1).lower()
                intensity = intensity_match.group(1) if intensity_match else 'medium'
                confidence = float(confidence_match.group(1)) if confidence_match else 0.8
                
                if emotion in emotion_list:
                    return emotion, confidence, intensity
            
            return None, 0, 'medium'
            
        except Exception as e:
            logger.error(f"LLM emotion detection failed: {e}")
            return None, 0, 'medium'

    def detect_comprehensive_emotion(self, text):
        keyword_emotion, keyword_score = self.keyword_based_detection(text)
        llm_emotion, llm_confidence, llm_intensity = self.llm_emotion_detection(text)
        detected_intensity = self.detect_emotion_intensity(text)
        
        final_scores = {}
        
        if keyword_score > 0:
            final_scores[keyword_emotion] = keyword_score * 0.4
        
        if llm_emotion and llm_confidence > 0.5:
            if llm_emotion in final_scores:
                final_scores[llm_emotion] += llm_confidence * 0.6
            else:
                final_scores[llm_emotion] = llm_confidence * 0.6
        
        if final_scores:
            final_emotion = max(final_scores, key=final_scores.get)
            final_confidence = final_scores[final_emotion]
            final_intensity = llm_intensity if llm_emotion else detected_intensity
        else:
            final_emotion = 'neutral'
            final_confidence = 0.5
            final_intensity = 'medium'
        
        return {
            'emotion': final_emotion,
            'confidence': final_confidence,
            'intensity': final_intensity,
            'methods': {
                'keyword': (keyword_emotion, keyword_score),
                'llm': (llm_emotion, llm_confidence)
            }
        }

class VoiceSynthesizer:
    def __init__(self):
        self.emotion_voice_settings = {
            'joy': {'rate': 1.15, 'pitch': 1.08},
            'excitement': {'rate': 1.25, 'pitch': 1.12},
            'love': {'rate': 0.95, 'pitch': 1.03},
            'gratitude': {'rate': 1.0, 'pitch': 1.05},
            'pride': {'rate': 1.1, 'pitch': 1.06},
            'sadness': {'rate': 0.8, 'pitch': 0.92},
            'grief': {'rate': 0.7, 'pitch': 0.88},
            'disappointment': {'rate': 0.85, 'pitch': 0.94},
            'loneliness': {'rate': 0.82, 'pitch': 0.93},
            'despair': {'rate': 0.75, 'pitch': 0.90},
            'anger': {'rate': 0.9, 'pitch': 0.98},
            'frustration': {'rate': 0.95, 'pitch': 1.02},
            'resentment': {'rate': 0.88, 'pitch': 0.96},
            'rage': {'rate': 1.05, 'pitch': 1.0},
            'fear': {'rate': 0.85, 'pitch': 1.05},
            'anxiety': {'rate': 0.9, 'pitch': 1.08},
            'panic': {'rate': 1.15, 'pitch': 1.15},
            'insecurity': {'rate': 0.88, 'pitch': 1.02},
            'surprise': {'rate': 1.2, 'pitch': 1.15},
            'confusion': {'rate': 0.95, 'pitch': 1.05},
            'curiosity': {'rate': 1.05, 'pitch': 1.08},
            'embarrassment': {'rate': 0.85, 'pitch': 1.0},
            'guilt': {'rate': 0.8, 'pitch': 0.95},
            'envy': {'rate': 0.92, 'pitch': 0.98},
            'nostalgia': {'rate': 0.9, 'pitch': 1.02},
            'boredom': {'rate': 0.85, 'pitch': 0.95},
            'contempt': {'rate': 0.9, 'pitch': 0.94},
            'calm': {'rate': 0.95, 'pitch': 1.0},
            'confident': {'rate': 1.05, 'pitch': 1.03},
            'hopeful': {'rate': 1.0, 'pitch': 1.05},
            'determined': {'rate': 1.08, 'pitch': 1.04},
            'neutral': {'rate': 1.0, 'pitch': 1.0},
            'professional': {'rate': 1.0, 'pitch': 1.0}
        }

    def adjust_voice_for_intensity(self, base_settings, intensity):
        settings = base_settings.copy()
        
        if intensity == 'high':
            settings['rate'] = min(settings['rate'] * 1.1, 1.4)
            settings['pitch'] = min(settings['pitch'] * 1.05, 1.2)
        elif intensity == 'low':
            settings['rate'] = max(settings['rate'] * 0.95, 0.7)
            settings['pitch'] = max(settings['pitch'] * 0.98, 0.85)
        
        return settings

    def synthesize_speech(self, text, emotion_data):
        if not MURF_API_KEY:
            logger.warning("No Murf API key available")
            return None
        
        emotion = emotion_data['emotion']
        intensity = emotion_data['intensity']
        
        base_settings = self.emotion_voice_settings.get(emotion, self.emotion_voice_settings['neutral'])
        settings = self.adjust_voice_for_intensity(base_settings, intensity)
        
        payload = {
            "voiceId": "en-US-natalie",
            "style": "conversational",
            "text": text,
            "rate": settings['rate'],
            "pitch": settings['pitch'],
            "format": "WAV",
            "sampleRate": 44100
        }
        
        headers = {
            "api-key": MURF_API_KEY,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(MURF_GENERATE_URL, json=payload, headers=headers, timeout=15)  # Reduced timeout
            
            if response.status_code == 200:
                response_data = response.json()
                audio_url = response_data.get('audioFile')
                return audio_url
            else:
                logger.error(f"Murf API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Murf API timeout")
            return None
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return None

# FIXED: Audio format conversion function
def convert_audio_to_wav(input_file_path, output_file_path):
    """Convert audio file to WAV format using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_file_path, 
            '-ar', '16000',  # Sample rate
            '-ac', '1',      # Mono
            '-f', 'wav',     # Output format
            output_file_path,
            '-y'             # Overwrite output file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return False

# Initialize components
emotion_detector = AdvancedEmotionDetector()
voice_synthesizer = VoiceSynthesizer()
recognizer = sr.Recognizer()

@app.post("/detect_emotion", response_model=EmotionResponse)
async def detect_emotion(input_data: TextInput):
    """Detect emotion from text input"""
    try:
        emotion_data = emotion_detector.detect_comprehensive_emotion(input_data.text)
        return EmotionResponse(**emotion_data)
    except Exception as e:
        logger.error(f"Emotion detection error: {e}")
        raise HTTPException(status_code=500, detail="Emotion detection failed")

@app.post("/synthesize_speech")
async def synthesize_speech(input_data: TextInput, emotion: str = "neutral", intensity: str = "medium"):
    """Synthesize speech from text with emotion"""
    try:
        emotion_data = {
            'emotion': emotion,
            'intensity': intensity,
            'confidence': 1.0
        }
        
        audio_url = voice_synthesizer.synthesize_speech(input_data.text, emotion_data)
        
        return {"audio_url": audio_url, "emotion_data": emotion_data}  # FIXED: Returns None if failed
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        raise HTTPException(status_code=500, detail="Speech synthesis failed")

@app.post("/generate_response")
async def generate_response(input_data: TextInput):
    """Generate empathetic response based on detected emotion"""
    try:
        # Detect emotion
        emotion_data = emotion_detector.detect_comprehensive_emotion(input_data.text)
        
        # Generate response
        if groq_client:
            system_prompt = f"""
You are Natalie, an empathetic AI assistant. The user's emotional state:
- Primary emotion: {emotion_data['emotion']}
- Intensity: {emotion_data['intensity']} 
- Detection confidence: {emotion_data['confidence']:.2f}

Respond appropriately to their emotional state. Keep response 15-35 words. Be natural and empathetic."""

            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_data.text}
                ],
                model=MODEL_NAME,
                temperature=0.7,
                max_tokens=70
            )
            
            response_text = response.choices[0].message.content.strip()
        else:
            response_text = "I understand what you're saying. How can I help you with that?"
        
        return ChatResponse(response_text=response_text, emotion_data=emotion_data, audio_url=None)
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        raise HTTPException(status_code=500, detail="Response generation failed")

@app.post("/process_speech")
async def process_speech(audio_file: UploadFile = File(...)):
    """FIXED: Process speech audio and return transcription and emotion"""
    temp_input_path = None
    temp_wav_path = None
    
    try:
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(await audio_file.read())
            temp_input_path = temp_audio.name
        
        # Convert to WAV format
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name
        
        # Convert audio format
        if not convert_audio_to_wav(temp_input_path, temp_wav_path):
            raise Exception("Audio format conversion failed")
        
        # Transcribe speech
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Detect emotion
        emotion_data = emotion_detector.detect_comprehensive_emotion(text)
        
        return {"transcription": text, "emotion_data": emotion_data}
        
    except sr.UnknownValueError:
        logger.error("Could not understand audio")
        raise HTTPException(status_code=400, detail="Could not understand audio")
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        raise HTTPException(status_code=500, detail="Speech recognition service error")
    except Exception as e:
        logger.error(f"Speech processing error: {e}")
        raise HTTPException(status_code=500, detail="Speech processing failed")
    finally:
        # Clean up temp files
        if temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

@app.post("/chat", response_model=ChatResponse)
async def chat(input_data: TextInput):
    """FIXED: Complete chat pipeline"""
    try:
        # Detect emotion
        emotion_data = emotion_detector.detect_comprehensive_emotion(input_data.text)
        
        # Generate empathetic response
        if groq_client:
            system_prompt = f"""
You are Natalie, an empathetic AI assistant. The user's emotional state:
- Primary emotion: {emotion_data['emotion']}
- Intensity: {emotion_data['intensity']} 
- Detection confidence: {emotion_data['confidence']:.2f}

Respond appropriately to their emotional state. Keep response 15-35 words. Be natural and empathetic."""

            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_data.text}
                ],
                model=MODEL_NAME,
                temperature=0.7,
                max_tokens=70
            )
            
            response_text = response.choices[0].message.content.strip()
        else:
            response_text = "I understand what you're saying. How can I help you with that?"
        
        # Try to synthesize speech (but don't fail if it doesn't work)
        audio_url = None
        try:
            audio_url = voice_synthesizer.synthesize_speech(response_text, emotion_data)
        except Exception as e:
            logger.warning(f"Voice synthesis failed, continuing without audio: {e}")
        
        return ChatResponse(
            response_text=response_text,
            emotion_data=emotion_data,
            audio_url=audio_url  # This can be None now
        )
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")

@app.get("/")
async def root():
    return {"message": "Natalie Voice Assistant API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "groq_available": groq_client is not None,
        "murf_available": bool(MURF_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
