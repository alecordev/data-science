import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download required NLTK data files
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# Example document
text = """Artificial Intelligence and machine learning are advancing rapidly. These technologies impact various industries."""

# Tokenize and filter out stopwords
stop_words = set(stopwords.words("english"))
words = word_tokenize(text)
filtered_words = [
    word for word in words if word.isalpha() and word.lower() not in stop_words
]

# Part-of-Speech Tagging
tagged_words = pos_tag(filtered_words)

# Extract nouns (can be extended for verbs, adjectives, etc.)
keywords = [word for word, pos in tagged_words if pos.startswith("NN")]

# Get keyword frequency
keyword_freq = Counter(keywords)
print("Keywords:", keyword_freq)

from nltk.corpus import wordnet

nltk.download("wordnet")

# Function to get synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


# Example for a keyword
keyword = "intelligence"
synonyms = get_synonyms(keyword)
print(f"Synonyms for {keyword}: {synonyms}")


def get_related_terms(word):
    related_terms = set()
    for syn in wordnet.synsets(word):
        for hypernym in syn.hypernyms():
            related_terms.update(lemma.name() for lemma in hypernym.lemmas())
    return related_terms


related_terms = get_related_terms(keyword)
print(f"Related terms for {keyword}: {related_terms}")
