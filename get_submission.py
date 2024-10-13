from __future__ import annotations

import os
import argparse
import json
import time

from memory_profiler import profile

from tqdm import tqdm

from src import CommandPipeline


class Predictor:
    """Class for your model's predictions.

    You are free to add your own properties and methods
    or modify existing ones, but the output submission
    structure must be identical to the one presented.

    Examples:
        >>> python -m get_submission --src input_dir --dst output_dir
    """

    def __init__(self):
        self.model = CommandPipeline()
        
    # @profile                                              # while measuring the average memory consumption was 542.10MiB
    def __call__(self, audio_path: str):
        prediction = self.model.predict(audio_path)
        result = {
            "audio": os.path.basename(audio_path),          # Audio file base name
            "text": prediction.get("text", -1),             # Predicted text
            "label": prediction.get("label", -1),           # Text class
            "attribute": prediction.get("attribute", -1),   # Predicted attribute (if any, or -1)
        }
        return result



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Get submission.")
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the source audio files.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to the output submission.",
    )
    args = parser.parse_args()
    predictor = Predictor()

    results = []
    processing_times = []
    
    for audio_path in (pbar := tqdm(os.listdir(args.src))):
        pbar.set_description(f"Processing file {audio_path}")
        
        start = time.time()
        result = predictor(os.path.join(args.src, audio_path))
        end = time.time()
        
        execution_time = (end - start) * 1000
        processing_times.append(execution_time)    
        
        results.append(result)
    pbar.set_description(f"Processed all files!")
    
    print(f"Minimum processing time: {min(processing_times):.4f} ms")
    print(f"Average processing time: {(sum(processing_times) / len(processing_times)):.4f} ms")
    print(f"Maximum processing time: {max(processing_times):.4f} ms")

    with open(
        os.path.join(args.dst, "submission.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)
        
