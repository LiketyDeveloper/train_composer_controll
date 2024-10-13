import nltk
# nltk.download('stopwords')    # Only if not installed
# nltk.download('punkt_tab')    # Only if not installed
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import pymorphy3
import numpy as np


def get_stopwords():
    return stopwords.words("russian")

 
def tokenize(text: str) -> list:
    """
    Split a string into meaningful units (tokens). These units are separated by whitespace, punctuation, or other special characters.

    ### For example:
    "hello world, how are you?" 
    
    -> 
    
    ["hello", "world,", "how", "are", "you", "?"]

    **return**: list of tokens
    """
    return nltk.word_tokenize(text)


def stem(word: str) -> str:
    """
    Returns the root form of a word, also known as the word's lemma.

    ### For example:
    "running" -> "run"

    **return**: root word
    """
    stemmer = SnowballStemmer("russian")
    return stemmer.stem(word)
    
    
def preprocess_text(text: str, morph: pymorphy3.MorphAnalyzer) -> list:
    """Returns a list of stemmed tokens from given string"""
    result = []
    for token in tokenize(text):
        if morph.parse(token)[0].tag.POS != 'NUMR' and token not in get_stopwords():
            result.append(stem(token))
    if len(result) == 1:
       result.append("[PAD]")
    
    return " ".join(result)

def bag_of_words(tokenized_words: list, all_words: list) -> np.ndarray:
    """
    Creates the bag of words based on given list of tokenized words and list of all words.
    
    ### For example: 
    ["hello", "world", "how", "are", "you"]
    
    "How are you" -> [0, 0, 1, 1, 1]
    
    **return**: bag of words
    """
    
    tokenized_words = [stem(w) for w in tokenized_words]
    
    # bag = np.zeros(len(all_words), dtype=np.float32)
    bag = [0.0 for _ in range(len(all_words))]
    for idx, w in enumerate(all_words):
        if w in tokenized_words:
            bag[idx] = 1.0
            
    return bag       


def get_text_from_number(number):
    """
    Returns the Russian name for the given number.
    """
    if number < 0 or number > 200:
        raise ValueError(f"Number cannot be greater than 200 or less than 0. Received: {number}")


    ones = [
        '',
        'один',
        'два',
        'три',
        'четыре',
        'пять',
        'шесть',
        'семь',
        'восемь',
        'девять',
        'десять',
        'одиннадцать',
        'двенадцать',
        'тринадцать',
        'четырнадцать',
        'пятнадцать',
        'шестнадцать',
        'семнадцать',
        'восемнадцать',
        'девятнадцать'
    ]

    tens = [
        '',
        '',
        'двадцать',
        'тридцать',
        'сорок',
        'пятьдесят',
        'шестьдесят',
        'семьдесят',
        'восемьдесят',
        'девяносто'
    ]


    if number < 20:
        return ones[number]
    elif number < 100:
        return tens[number // 10] + ('' if number % 10 == 0 else ' ' + ones[number % 10])
    elif 120 > number >= 100:
        return 'сто ' + ones[number % 100] if number % 100 != 0 else 'сто'
    else:
        return 'сто ' + tens[(number % 100) // 10] + ' ' + ones[number % 10]


def get_number_from_text(text):
    units = [
        'нол', 'один', 'два', 'три', 'четыр', 'пят', 'шест', 'сем', 'восем', 'девя', 'деся', 
        'одиннадца', 'двенадца', 'тринадца', 'четырнадца', 'пятнадца', 'шестнадца', 'семнадца', 'восемнадца', 'девятнадца'
    ]

    tens = ['двадца', 'тридца', 'сорок', 'пятьдес', 'шестьдес', 'семьдес', 'восемьдес', 'девян']

    hundreds = ["сто", "двест", "трист"]


    result = 0
    
    for word in tokenize(text):
        stemmed_word = stem(word)
        
        if stemmed_word in units:
            result += units.index(stemmed_word)
        elif stemmed_word in tens:
            result += (tens.index(stemmed_word) + 2) * 10
        elif stemmed_word in hundreds:
            result += (hundreds.index(stemmed_word) + 1) * 100
            
            
    return result 

def get_all_numbers():
    result = {}


    
    
