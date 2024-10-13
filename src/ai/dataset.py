import json
import os
from loguru import logger
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
import pymorphy3

from src.util.nltk import preprocess_text, get_text_from_number
from src.ai import DEVICE

from src.config.neural_net import ANNOTATIONS_PATH, STEMMED_VOCAB_PATH, VOCAB_PATH


class CommandDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.annotations = []
        self.train_annotations = []
        self.test_annotations = []
        
        for filename in os.listdir(ANNOTATIONS_PATH):
            if filename.endswith(".json") and "luga" not in filename :
                with open(os.path.join(ANNOTATIONS_PATH, filename), "r", encoding="utf-8") as file:
                    data = json.load(file)
                    self.annotations.extend(data)

        self.data = []

        self.morph = pymorphy3.MorphAnalyzer()
        
        docs = []
        stemmed_docs = []
        # Creating vocabulary with all words
        for annotation in (pbar := tqdm(self.annotations)):
            docs.append(annotation["text"])
            
            stemmed_text = preprocess_text(annotation["text"], self.morph)
            stemmed_docs.append(stemmed_text)
            
            pbar.set_description("Creating vocabulary")
        
        self.cv = CountVectorizer()
        self.cv.fit_transform(docs)
        self.vocabulary = sorted(self.cv.vocabulary_.keys())
        self.vocabulary.extend([get_text_from_number(i) for i in range(140)])
        
        self.cv = CountVectorizer(ngram_range=(2, 2))
        self.cv.fit_transform(docs)
        self.vocabulary.extend(sorted(self.cv.vocabulary_.keys()))
        
        self.cv = CountVectorizer(ngram_range=(2, 2))
        self.cv.fit_transform(stemmed_docs)

        self.stemmed_vocabulary = sorted(self.cv.vocabulary_.keys())
        
        with open(VOCAB_PATH, "w", encoding="utf8") as filename:
            json.dump(self.vocabulary, filename, ensure_ascii=False, indent=4)
            logger.success(f"Vocabulary saved to {VOCAB_PATH}, {len(self.vocabulary)} words")
        
        # Saving vocabulary to file
        with open(STEMMED_VOCAB_PATH, "w", encoding="utf8") as filename:
            json.dump(self.stemmed_vocabulary, filename, ensure_ascii=False, indent=4)
            logger.success(f"Stemmed vocabulary saved to {STEMMED_VOCAB_PATH}, {len(self.stemmed_vocabulary)} words")
        
        
        for annotation in (pbar := tqdm(self.annotations)):
            text = preprocess_text(annotation["text"], self.morph)
            vector = torch.tensor(self.cv.transform([text]).toarray()[0], dtype=torch.float32).to(DEVICE)
            
            label = torch.tensor(annotation["label"]).to(DEVICE)
            
            self.data.append((vector, label))    
            pbar.set_description(f"Getting dataset")
        
        self.n_samples = len(self.annotations)
        
        logger.success(f"Dataset successfully created: {self.n_samples} samples")
    
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.n_samples