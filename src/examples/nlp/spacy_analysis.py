import subprocess
import spacy


# subprocess.run("python -m spacy download en_core_web_sm", shell=True)
# subprocess.run("python -m spacy download en_core_web_md", shell=True)
nlp = spacy.load("en_core_web_lg")

text = """Artificial Intelligence and machine learning are advancing rapidly. These technologies impact various industries. Michael Jordan is a famous data scientist."""
doc = nlp(text)

# Extract nouns and named entities (can customize for verbs or adjectives)
keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
entities = [ent.text for ent in doc.ents]

print("Keywords:", keywords)
print("Named Entities:", entities)


def similar_words():
    # nlp = spacy.load("en_core_web_lg")
    # nlp = spacy.load("en_core_web_sm")

    # Define a keyword to find similar words
    keyword = "intelligence"
    keyword_token = nlp(keyword)

    # Find words similar to the keyword in the document
    similar_words = []
    for token in doc:
        if token.has_vector:  # Ensure token has word vector
            similarity = keyword_token.similarity(token)
            if similarity > 0.6:  # Set a threshold for similarity
                similar_words.append((token.text, similarity))

    print("Similar words:", similar_words)


if __name__ == "__main__":
    similar_words()
