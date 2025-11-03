import numpy as np

def rule_label(word_count: int, readability: float) -> str:
    if word_count > 1500 and 50 <= readability <= 70:
        return "High"
    if word_count < 500 or readability < 30:
        return "Low"
    return "Medium"

def thin_content(word_count: int) -> bool:
    return word_count < 500

def label_to_int(label: str) -> int:
    return {"Low":0, "Medium":1, "High":2}[label]

def int_to_label(i: int) -> str:
    return {0:"Low", 1:"Medium", 2:"High"}[i]
