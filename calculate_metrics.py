from src.metrics.wer import wer


if __name__ == "__main__":
    # The model value of WER is: ~ 0.1456
    print(f"Результат WER(Word Error Rate): {wer()}")   
    