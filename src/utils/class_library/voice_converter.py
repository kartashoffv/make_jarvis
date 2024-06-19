import os
import requests

from dotenv import load_dotenv
load_dotenv()


XI_API_KEY = os.getenv('XI_API_KEY')
BASE_PATH = os.getenv('BASE_PATH')

CHUNK_SIZE = 1024 
VOICE_ID = "XrExE9yKIg1WjnnlVkGX"
OUTPUT_PATH = "{BASE_PATH}/make_jarvis/data/processed/output.mp3"

tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

headers = {
    "Accept": "application/json",
    "xi-api-key": XI_API_KEY
}


def save_text_to_speak(text_to_speak):
    data = {
        "text": text_to_speak,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    response = requests.post(tts_url, headers=headers, json=data, stream=True)

    if response.ok:
        with open(OUTPUT_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        print("Audio stream saved successfully.")
    else:
        print(response.text)
