from nltk.corpus import brown, nps_chat
from nltk.tag import RegexpTagger, UnigramTagger, BigramTagger, TrigramTagger
from sklearn.model_selection import train_test_split as split

brown = brown.tagged_sents()
nps = nps_chat.tagged_posts()


def data_splits():
    b50_train, b50_test = split(brown, test_size=0.5)
    b90_train, b10_test = split(brown, test_size=0.9)
    n50_train, n50_test = split(nps, test_size=0.5)
    n90_train, n10_test = split(nps, test_size=0.9)

    return {
        "brown 50/nps 50": [b50_train, n50_test],
        "brown 90/nps 10": [b90_train, n10_test],
        "nps 50/brown 50": [n50_train, b50_test],
        "nps 90/brown 10": [n90_train, b10_test],
    }

patterns = [
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # simple past
    (r'.*es$', 'VBZ'),                 # 3rd singular present
    (r'.*ould$', 'MD'),                # modals
    (r'.*\'s$', 'NN$'),                # possessive nouns
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
]

def make_backoff_taggers(train_data, default, return_last=False):
    regex = RegexpTagger(patterns, backoff=default)
    uni = UnigramTagger(train=train_data, backoff=regex)
    bi = BigramTagger(train=train_data, backoff=uni)
    tri = TrigramTagger(train=train_data, backoff=bi)

    if return_last:
        return tri
    else:
        return {
            "Default": default,
            "Regex (backoff: def)": regex,
            "Unigram (backoff: regex)": uni,
            "Bigram (backoff: uni)": bi,
            "Trigram (backoff: bi)": tri
        }