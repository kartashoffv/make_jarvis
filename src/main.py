import os
from termcolor import colored

import speech_recognition as sr

from utils.class_library.voice_converter import save_text_to_speak
from utils.class_library.voice_gen import play_mp3
from utils.agent_openai import run_agent
BASE_PATH = os.getenv('BASE_PATH')

def listen_and_detect(keyword='сима', stopword="стоп"):
    recognizer = sr.Recognizer()
    start_processing = False

    while True:
        with sr.Microphone() as source:
            if not start_processing:
                print(colored(f"[Для начала работы скажите '{keyword}']", 'green'))
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            # whisper работает оффлайн, но проблемы с галюцинациями и скоростью
            # text = recognizer.recognize_whisper(audio, language="ru", model='small')
            text = recognizer.recognize_google(audio, language="ru-RU")
            print(colored(f"[DEBUG MODE] Вы сказали : {text}", 'red'))
    
            if not start_processing:
                if keyword.lower() in text.lower():
                    print(colored(f"[Услышал {keyword} - начинаю работу.]", 'green'))
                    start_processing = True
            else:
                if stopword.lower() in text.lower():
                    print(colored(f"[Услышал {stopword} - завершаю работу.]", 'green'))
                    start_processing = False
                else:
                    print(colored(f"[Распознал текст: {text}]", 'blue'))
                    result = run_agent(text)
                    save_text_to_speak(result['output'])
                    print(colored(f"[Сгенерированный ответ: {result['output']}]", 'blue'))
                    play_mp3(f"{BASE_PATH}/make_jarvis/data/processed/output.mp3")

        except sr.UnknownValueError:
            print("[PING] -- аудиоряд отсутствует или не распознан корректно")
        except sr.RequestError as e:
            print(f"Could not request results from audio recognition; {e}.")


def main():
    listen_and_detect()


if __name__ == "__main__":
    main()
