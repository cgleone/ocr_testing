from difflib import SequenceMatcher
import Levenshtein as lev

actual = "Hello I'm Cathleen and I'm testing this thing 123"


def string_similarity(converted, actual):
    return SequenceMatcher(None, converted, actual).ratio()

def levenshtein_similarity(converted, actual):
    return lev.ratio(converted, actual)