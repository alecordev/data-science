import spacy
from nltk.corpus import wordnet

# Load a spaCy model
nlp = spacy.load(
    "en_core_web_md"
)  # Make sure you have a medium or larger model for word vectors


def find_similar_words(word: str, top_n: int = 5):
    """
    Identify similar words to the given input word based on word embeddings from spaCy.

    Parameters:
    - word (str): The word for which to find similar words.
    - top_n (int): The number of similar words to return.

    Returns:
    - List of tuples with similar words and their similarity scores.
    """
    # Initialize a blank doc
    word_token = nlp.vocab[word]

    if not word_token.has_vector:
        raise ValueError(
            f"The word '{word}' is not in the vocabulary or doesn't have a vector."
        )

    # Find the top_n most similar words in the vocabulary
    similar_words = sorted(
        word_token.vocab, key=lambda w: word_token.similarity(w), reverse=True
    )

    # Filter out non-alphabetic words and return top N similar words
    similar_words = [
        (w.text, word_token.similarity(w))
        for w in similar_words[: top_n + 1]
        if w.is_alpha and w.has_vector
    ]

    return similar_words[
        1 : top_n + 1
    ]  # Exclude the first result since it's the input word itself


def extract_named_entities(text: str):
    """
    Identify named entities (people, organizations, locations, etc.) from a text using spaCy's NER.

    Parameters:
    - text (str): The input text from which to extract named entities.

    Returns:
    - Dictionary containing entity types as keys and lists of entities as values.
    """
    doc = nlp(text)
    entities = {}

    # Loop through the recognized entities
    for ent in doc.ents:
        ent_label = ent.label_

        # Group entities by their label (e.g., 'PERSON', 'ORG', 'GPE', etc.)
        if ent_label not in entities:
            entities[ent_label] = []

        entities[ent_label].append(ent.text)

    return entities


# Example usage
if __name__ == "__main__":
    # Synonym/Similar words example
    word = "happy"
    print(f"Words similar to '{word}':")
    similar_words = find_similar_words(word)
    for similar_word, score in similar_words:
        print(f"Word: {similar_word}, Similarity: {score:.4f}")

    # Named Entity Recognition example
    text = "Google was founded by Larry Page and Sergey Brin in 1998."
    print("\nNamed entities in the text:")
    named_entities = extract_named_entities(text)
    print(named_entities)
