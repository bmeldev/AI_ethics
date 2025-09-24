import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util

# MiniLM L12 multilingual used for Spanish
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# readibility: Szigriszt-Pazos Perspicuity Index
def readibility(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = len(sentences)
    
    words = re.findall(r'\w+', text)
    num_words = len(words)
    
    syllables = sum(count_syllables(w) for w in words)
    
    if num_sentences == 0 or num_words == 0:
        return 0.0
    
    # formula Szigriszt-Pazos Perspicuity Index:
    readibility = 206.84 - (62.3 * (syllables / num_words)) - (num_words / num_sentences)
    return (readibility + 100) / 300  # scale to [0,1]


# readibility: average number of syllables per word
def avg_syl(text):
    words = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    if not words:
        return 0.0

    total_syllables = sum(count_syllables(word) for word in words)
    return total_syllables / len(words)


def count_syllables(word):
    word = word.lower()
    return len(re.findall(r'[aeiouáéíóúü]', word))


def features_style(text):
    words = text.split()
    num_words = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    vocab_richness = len(set(words)) / num_words if num_words else 0
    punct_count = len(re.findall(r'[.,;:!?]', text)) / len(text) if len(text) > 0 else 0
    
    capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    newline_ratio = text.count("\n") / len(text) if len(text) > 0 else 0
    tab_ratio = text.count("\t") / len(text) if len(text) > 0 else 0
    
    readibility_score = readibility(text)
    
    features = np.array([
        avg_word_len,
        vocab_richness,
        punct_count,
        capital_ratio,
        newline_ratio,
        tab_ratio,
        readibility_score
    ])
    
    # features capital letters/newline/tab 3x more important
    weights = np.array([1, 1, 1, 3, 3, 3, 1])
    return features * weights


def similarity(text1, text2):
    # Similarity semantics
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    semantic_sim = util.cos_sim(emb1, emb2).item()
    
    # Similarity style
    f1 = features_style(text1)
    f2 = features_style(text2)
    if np.linalg.norm(f1) == 0 or np.linalg.norm(f2) == 0:
        style_sim = 0
    else:
        style_sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
    
    combined = (0.7 * semantic_sim + 0.3 * style_sim)
    
    # Sigmoid normalizes to [0,1]
    probability = 1 / (1 + np.exp(-5 * (combined - 0.5)))
    return probability
