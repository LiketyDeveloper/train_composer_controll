from loguru import logger
import json

import torch 
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import pymorphy3

from src.ai import DEVICE

from src.config.neural_net import MODEL_FILE_PATH, STEMMED_VOCAB_PATH
from src.util.nltk import preprocess_text, get_number_from_text
from src.util import id2label

from .speech_recognition import SpeechRecognition


class CommandIdentifier(nn.Module):
    """Neural Network class to identify commands from text input"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()            
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.ReLU()
        
        with open(STEMMED_VOCAB_PATH, "r", encoding="utf-8") as file:
            self.vocabulary = json.load(file)
                    
        logger.debug("Neural Network model initialized")

        
    def forward(self, x):
        out = self.l1(x)
        out = self.activation(out)
        out = self.l2(out)
        out = self.activation(out)
        out = self.l3(out)
        
        return out

    def invoke(self, query):        
        morph = pymorphy3.MorphAnalyzer()
        query = preprocess_text(query, morph)
        cv = CountVectorizer(ngram_range=(2,2), vocabulary=self.vocabulary)
        
        text_vector = cv.transform([query]).toarray()[0]
        text_vector = torch.Tensor(text_vector).to(DEVICE)
        text_vector = text_vector.view(1, -1)
        res = self(text_vector)
                
        return res.argmax()
    

class CommandPipeline:
    
    def __init__(self):
        self.speech_recognition = SpeechRecognition()
        self.model = self.load_model()
    
    @classmethod
    def load_model(cls):
        """Load command identifier model from file"""

        data = torch.load(MODEL_FILE_PATH, weights_only=True)

        model = CommandIdentifier(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            num_classes=data["output_size"]
        )

        model.load_state_dict(data["model_state"])
        model.eval()
        model.to(DEVICE)

        return model
    
    def predict(self, audio_path):
        text = self.speech_recognition.recognize(audio_path)
        label_id = int(self.model.invoke(text))
        
        if "(количество)" in id2label(label_id):
            attribute = get_number_from_text(text)
            if not attribute:
                attribute = -1
        else:
            attribute = -1
        
        return {
            "text": text,
            "label": label_id,
            "attribute": attribute
            }
