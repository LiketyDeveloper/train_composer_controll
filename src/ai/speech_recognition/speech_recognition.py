import os
import json

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from vosk import SetLogLevel

from src.config.speech_recognition import SR_MODEL_PATH
from src.config.neural_net import VOCAB_PATH

class SpeechRecognition:
    
    _hyperparameters = {
        "sample_rate": 16000,
        "model_path": SR_MODEL_PATH,
    }
    
    def __init__(self, show_logs=False):  

        if not show_logs:
            SetLogLevel(-1)

        with open(VOCAB_PATH, "r", encoding="utf-8") as file:
            self._vocab = json.load(file)
        
        model_path = self._hyperparameters["model_path"]
        
        # Load the model from file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = Model(model_path)

        self._recognizer = KaldiRecognizer(model, self._hyperparameters["sample_rate"], str(self._vocab).replace("'", '"'))


    def recognize(self, file_path: str) -> str:
        """
        Recognize speech in a given audio file.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            str: Recognized text.
        """
        audio_file = AudioSegment.from_file(file_path)

        # Change sample rate, which is the sample rate of the model
        audio_file = audio_file.set_frame_rate(self._hyperparameters["sample_rate"])
        raw_data = audio_file.raw_data

        # Recognize speech
        self._recognizer.AcceptWaveform(raw_data)

        # Get the final result
        result = self._recognizer.FinalResult()
        
        return json.loads(result)['text']