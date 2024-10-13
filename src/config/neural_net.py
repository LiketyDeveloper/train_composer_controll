from src.util import get_path

"""Command Identifier Neural Netword Settings"""

MODEL_FILE_PATH = get_path("ai", "data", "model.pth")
ANNOTATIONS_PATH = get_path("ai", "dataset", "annotation")
VOCAB_PATH = get_path("ai", "data", "vocabulary.json")
STEMMED_VOCAB_PATH = get_path("ai", "data", "stem_vocabulary.json")
NGRAM_RANGE = 1
NN_TRAIN_EPOCHS = 12
