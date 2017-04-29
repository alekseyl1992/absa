import nltk
from nltk import WordPunctTokenizer

from utils import load_w2v, get_ote_ds, load_dataset

import numpy as np
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from sklearn.metrics import accuracy_score


class Tokenizer:
    def tokenize(self, sent):
        return nltk.tokenize.word_tokenize(sent)


class OTE:
    def __init__(self, w2v):
        self.w2v = w2v
        self.tokenizer = None
        self.mlb = None
        self.clf = None

        print('Loading tokenizer...')
        self.tokenizer = Tokenizer()

        self.stop_words = [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ]

    def get_ote_features(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        vectors = []

        for token in tokens:
            # token = self.stemmer.stem(token)
            if token in self.w2v.vocab and token not in self.stop_words:
                vector = self.w2v.word_vec(token)
                vectors.append(vector)

        return np.array(vectors)

    def train(self):
        print('-- OTE:')

        print('Loading dataset...')
        ds_train = load_dataset(r'data/restaurants_train.xml')
        x_train, y_train = get_ote_ds(ds_train, self.get_ote_features, self.tokenizer)
        ds_test = load_dataset(r'data/restaurants_test.xml')
        x_test, y_test = get_ote_ds(ds_test, self.get_ote_features, self.tokenizer)

        model = ChainCRF()
        ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
        ssvm.fit(x_train, y_train)

        predicted = ssvm.predict(x_test)
        accuracy = accuracy_score(y_test, predicted)

        print('Accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    w2v = load_w2v(use_mock=True)
    ote = OTE(w2v)
    ote.train()
