import nltk
from collections import defaultdict, Counter

corpus = nltk.corpus.brown

def a():
    tags = defaultdict(int)
    for (_, tag) in corpus.tagged_words():
        tags[tag] += 1
    common = Counter(tags).most_common(5)
    print("{} are the most 5 common tags".format(common))

def get_unique_tags():
    wordtags = defaultdict(set)
    for (word, tag) in corpus.tagged_words():
        wordtags[word].add(tag)
    return wordtags

def get_ambiguous_words():
    wordtags = get_unique_tags()
    ambig = len([word for word in wordtags if len(wordtags[word]) > 1])
    return ambig

def b():
    print("{} words are ambiguous".format(get_ambiguous_words()))

def c():
    ambiguous_words = get_ambiguous_words()
    total_words = len(set(corpus.words()))
    percentage = round(ambiguous_words / total_words, 4) * 100
    print("{}% of words are ambiguous".format(percentage))

def d():
    wordtags = get_unique_tags()

    set_length = lambda item : len(item[1])
    sorted_wordtags = sorted(wordtags.items(), key=set_length, reverse=True)

    conditions = lambda w : w.isalpha() and len(w) > 4

    words_to_keep = 5
    top_words = [[w, list(tags)] for (w, tags) in sorted_wordtags if conditions(w)][:words_to_keep]

    selected_word = top_words[3]
    selected_word  # arbitrary word
    # generate all combination of (word, tag)-tuples

    def prettyprint(tagged_sent):
        print(" ".join([w for (w, _) in tagged_sent]))

    wordtags = [(selected_word[0], tag) for tag in selected_word[1]]
    for wt in wordtags:
        print(wt)
        for sent in corpus.tagged_sents():
            if wt in sent:
                prettyprint(sent)
                break

a()
b()
c()
d()