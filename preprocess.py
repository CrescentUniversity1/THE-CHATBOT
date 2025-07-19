# preprocess.py

import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import json

# Load abbreviation dictionary
with open("data/abbreviations.json", "r") as f:
    ABBREVIATIONS = json.load(f)

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def expand_abbreviations(text):
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"\b{abbr}\b", full, text, flags=re.IGNORECASE)
    return text

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = expand_abbreviations(text)
    text = correct_spelling(text)
    return text
