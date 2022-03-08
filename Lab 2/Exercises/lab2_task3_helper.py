import nltk
from nltk.corpus import brown

class lab2_helper:

    import nltk
    from nltk.corpus import brown
    # see https://nltk.readthedocs.io/en/latest/api/nltk.html
    # define distinguishable start/end tuples of tag/word
    # used to mark sentences
    global START
    START = ("START", "START")
    global END
    END = ("END", "END")

    @staticmethod
    def get_tags(corpus):
        tags_words = []
        for sent in corpus.tagged_sents():
            tags_words.append(START)
            # shorten tags to 2 characters each for simplicity
            tags_words.extend([(tag[:2], word) for (word, tag) in sent])
            tags_words.append(END)

        return tags_words
    @staticmethod
    def probDist(corpus, probability_distribution, tag_observation_fn):
        tag_words = lab2_helper.get_tags(corpus)
        tags = [tag for (tag, _) in tag_words]
        # conditional frequency distribution over tag/word
        cfd_tagwords = nltk.ConditionalFreqDist(tag_words)
        cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, probability_distribution)
        # conditional frequency distribution of observations:
        cfd_tags = nltk.ConditionalFreqDist(tag_observation_fn(tags))
        cpd_tags = nltk.ConditionalProbDist(cfd_tags, probability_distribution)
        
        return cpd_tagwords, cpd_tags

    @staticmethod
    def task3a():
        corpus = brown
        # maximum likelihood estimate to create a probability distribution 
        probability_distribution = nltk.MLEProbDist # IMPLEMENT
        # a function to create tag observations. Hint: can we observe anything with unigrams?
        tag_observation_fn = nltk.bigrams  # IMPLEMENT
        return lab2_helper.probDist(corpus, probability_distribution, tag_observation_fn)
        
    @staticmethod
    def prettify(prob):
        return "{}%".format(round(prob * 100, 4))
        
    @staticmethod
    def task3b():
        tagwords, tags = lab2_helper.task3a()

        prob_verb_is_run = tagwords["VB"].prob("run")  # IMPLEMENT
        prob_v_follows_p = tags["PP"].prob("VB")  # IMPLEMENT
        print("Prob. of a Verb(VB) being 'run' is", lab2_helper.prettify(prob_verb_is_run))
        print("Prob. of a Preposition(PP) being followed by a Verb(VB) is", lab2_helper.prettify(prob_v_follows_p))

def main():
    
    lab2_helper.task3b()



if __name__ == "__main__":
    main()