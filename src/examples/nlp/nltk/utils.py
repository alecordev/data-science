from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer  # print(SnowballStemmer.languages)


def get_stem(word="working"):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


def get_synonyms(word):
    syn = wordnet.synsets(word)
    return syn


def get_antonyms(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return antonyms


if __name__ == '__main__':
    # print(get_synonyms("example"))
    w = get_synonyms("example")
    print(w[0].definition())
    print(w[0].examples())

    print(get_antonyms("small"))
    print(get_stem())
