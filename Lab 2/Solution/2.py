import nltk
from nltk import DefaultTagger
from collections import defaultdict, Counter
from utils import data_splits, make_backoff_taggers

def most_common_tag(corpus):
    # use this from task 1
    tags = defaultdict(int)
    for (_, tag) in corpus.tagged_words():
        tags[tag] += 1
    return Counter(tags).most_common(1)[0]

def a():
    nps_common = most_common_tag(nltk.corpus.nps_chat)
    brown_common = most_common_tag(nltk.corpus.brown)

    print("Most common in NPS:", nps_common)
    print("Most common in Brown:", brown_common)

    nps_tagger = DefaultTagger(nps_common)
    brown_tagger = DefaultTagger(brown_common)

    return nps_tagger, brown_tagger

def b():
    default_tagger = DefaultTagger("NN")  # NN is the most common tag 

    for id, (train, test) in data_splits().items():
        print("Evaluating", id)
        for tagger_info, tagger in make_backoff_taggers(train, default_tagger).items():
            print("{}: {}".format(tagger_info, tagger.accuracy(test)))


def c():
    # select the tagger based on data from brown 90:10 split
    default_tagger = DefaultTagger("NN")

    train, test = data_splits().get("brown 90/nps 10")
    taggers = make_backoff_taggers(train, default_tagger)

    for tagger_info, tagger in taggers.items():
        print("{}\n{}".format(
                tagger_info,
                tagger.evaluate_per_tag(test, truncate=5, sort_by_count=True)
            )
        )

def lookup_tagger(non_tagged, tagged, amount):
    fd = nltk.FreqDist(non_tagged())
    cfd = nltk.ConditionalFreqDist(tagged())
    most_freq_words = fd.most_common(amount)
    likely_tags = dict(
        (word, cfd[word].max()) for (word, _) in most_freq_words
    )
    baseline_tagger = nltk.UnigramTagger(model=likely_tags)
    return baseline_tagger

def brown_lookup():
    corpus = nltk.corpus.brown
    most_common = 200
    return lookup_tagger(corpus.words, corpus.tagged_words, most_common)

def d():
    for id, (train, test) in data_splits().items():
        print("Evaluating", id)
        for tagger_info, tagger in make_backoff_taggers(train, brown_lookup()).items():
            print("{}: {}".format(tagger_info, tagger.accuracy(test)))

def e():
    text = "With an arbitrary text from another corpus (or an article you scraped in Lab 1), use the tagger you just created and print a few tagged sentences."
    tokens = nltk.word_tokenize(text)

    train, _ = data_splits().get("brown 90/nps 10")
    brown_lookup_combined = make_backoff_taggers(train, brown_lookup(), return_last=True)

    print(brown_lookup_combined.tag(tokens))

a()
b()
c()
d()
e()
