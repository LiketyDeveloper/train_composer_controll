import json
from tqdm import tqdm

import pymorphy3
from src.ai.model import CommandPipeline
from src.util import get_path

def calculate_wer(reference, hypothesis):
    morph = pymorphy3.MorphAnalyzer()
    ref_words = [morph.parse(word)[0].normal_form for word in reference.split()]
    hyp_words = [morph.parse(word)[0].normal_form for word in hypothesis.split()]
	# Counting the number of substitutions, deletions, and insertions
    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
 
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
 
	# Total number of words in the reference text
    total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)
    wer = (substitutions + deletions + insertions) / total_words
    return wer


def wer():
	all_wers = []

	with open(get_path("ai", "dataset", "annotation", "luga.json"), "r", encoding="utf-8") as file:
		annotations = json.load(file)
	
	pipeline = CommandPipeline()
	with open(get_path("metrics", "wer.log"), "w", encoding="utf-8") as file:
		for annotation in (pbar := tqdm(annotations)):
			predicted = pipeline.predict(get_path("ai", "dataset", "luga", annotation["audio_filepath"]))
			current_wer = calculate_wer(annotation["text"], predicted["text"])
			all_wers.append(current_wer)

			file.write(f"{annotation["text"]} {predicted['text']} >> {current_wer}\n")

			pbar.set_description(f"Текущий wer: {sum(all_wers)/len(all_wers):.4f}")
  
	return sum(all_wers)/len(all_wers)
    