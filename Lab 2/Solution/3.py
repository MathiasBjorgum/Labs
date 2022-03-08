# markov models
import nltk
from nltk.corpus import brown

# define distinguishable start/end tuples of tag/word
# used to mark sentences
START = ("START", "START")
END = ("END", "END")

def get_tags(corpus):
    tags_words = []
    for sent in corpus.tagged_sents():
        tags_words.append(START)
        # shorten tags to 2 characters each for simplicity
        tags_words.extend([(tag[:2], word) for (word, tag) in sent])
        tags_words.append(END)

    return tags_words

get_tags(brown)

# see https://nltk.readthedocs.io/en/latest/api/nltk.html
def probDist(corpus, probability_distribution, tag_observation_fn):
    tag_words = get_tags(corpus)
    tags = [tag for (tag, _) in tag_words]

    # conditional frequency distribution over tag/word
    cfd_tagwords = nltk.ConditionalFreqDist(tag_words)
    cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, probability_distribution)

    # conditional frequency distribution of observations:
    cfd_tags = nltk.ConditionalFreqDist(tag_observation_fn(tags))
    cpd_tags = nltk.ConditionalProbDist(cfd_tags, probability_distribution)

    return cpd_tagwords, cpd_tags

def a():
    return probDist(
        brown,
        probability_distribution=nltk.MLEProbDist,
        tag_observation_fn=nltk.bigrams
    )

a()

def a():
    corpus = brown

    # maximum likelihood estimate to create a probability distribution 
    probability_distribution = None
    # a function to create tag observations. Hint: can we observe anything with unigrams?
    tag_observation_fn = None
    
    return probDist(corpus, probability_distribution, tag_observation_fn)

def b():
    tagwords, tags = a()
    
    prob_verb_is_run = tagwords["VB"].prob("run")
    prob_vb_follows_pp = tags["PP"].prob("VB")

    def prettify(prob):
        return "{}%".format(round(prob * 100, 4))

    print("Prob. of a Verb(VB) being 'run' is", prettify(prob_verb_is_run))
    print("Prob. of a Preposition(PP) being followed by a Verb(VB) is", prettify(prob_vb_follows_pp))

def c():
    tagwords, _ = a()
    tags_to_check = ["NN", "VB", "JJ"]
    for n in tags_to_check:
        print(n, tagwords[n].freqdist().most_common(10))
        

def d():
    tagwords, tags = a()

    sentence = "I can code some code"
    proposed_tags = "PP VB VB DT NN"
    
    words_with_tags = list(zip(sentence.split(), proposed_tags.split()))
    print(words_with_tags)
    
    # get the probability of the sentence STARTing with "PP" (first POS-tag)
    prob = tags["START"].prob(words_with_tags[0][1])
    
    # we need the index to access future tags
    for idx, (word, tag) in enumerate(words_with_tags):
        if idx < len(words_with_tags) - 1:
            next_tag = words_with_tags[idx + 1][1]
        else:
            next_tag = "END"

        prob *= tagwords[tag].prob(word) * tags[tag].prob(next_tag)


    print(prob)

d()