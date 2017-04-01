import numpy as np
import math
from nltk import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC

from acd import ACD
from utils import load_dataset, split_ds, load_w2v, get_pd_ds, load_core_nlp_parser, split_on_sents


class PD:
    def __init__(self, w2v, acd):
        self.w2v = w2v
        self.tokenizer = None
        self.stemmer = None
        self.parser = None
        self.clf = None
        self.acd = acd

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

    def _get_pd_features_ignore_category(self, tokens):
        vectors = []

        for token in tokens:
            if token in self.w2v.vocab and token not in self.stop_words:
                vector = self.w2v.word_vec(token)
                vectors.append(vector)

        if len(vectors) > 0:
            vectors = np.array(vectors)
            result = np.average(vectors, axis=0)
            result /= np.linalg.norm(result)
        else:
            result = np.zeros(300)

        return result

    def get_pd_features_ignore_category(self, text, category, cats_len):
        tokens = self.tokenizer.tokenize(text.lower())
        return self._get_pd_features_ignore_category(tokens)

    def get_pd_features_append_category(self, text, category, cats_len):
        text_vector = self.get_pd_features_ignore_category(text, category, cats_len)

        category_tokens = category.lower().split('#')
        category_vector = self._get_pd_features_ignore_category(category_tokens)

        return np.concatenate([text_vector, category_vector])

    def get_pd_features_insert_category(self, text, category, cats_len):
        entity, aspect = category.lower().split('#')
        text = text + ' ' + entity + ' ' + aspect

        return self.get_pd_features_ignore_category(text, category, cats_len)

    def get_pd_features_map_linear_cut_off(self, text, category, cats_len):
        tokens = self.tokenizer.tokenize(text)
        position, ote = self.acd.predict_ote_for_category(tokens, category)
        print((text, category, ote, position))

        if cats_len != 1 and ote is not None and not category.endswith('GENERAL'):
            distance_limit = int((len(tokens) / cats_len))
            tokens = tokens[
                     position - distance_limit:
                     position + distance_limit + 1]

        return self._get_pd_features_ignore_category(tokens)

    def get_pd_features_map_linear_weighted(self, text, category, cats_len):
        tokens = self.tokenizer.tokenize(text)
        tokens_len = len(tokens)

        (position, ote), word2score = self.acd.predict_ote_for_category(tokens, category)
        # print((text, category, ote, position))

        max_distance = max([position, tokens_len - position - 1]) if tokens_len != 1 else 1
        assert max_distance != 0, '{}'.format((position, tokens_len, text))
        tokens = [(token, 0.1 + 1. - math.fabs(position - i) / max_distance)
                  for i, token in enumerate(tokens)]

        vectors = []

        for i, (token, weight) in enumerate(tokens):
            if token in self.w2v.vocab and token not in self.stop_words:
                cat_prob = 1 + word2score[i][1]
                vector = self.w2v.word_vec(token) * weight * cat_prob
                vectors.append(vector)

        if len(vectors) > 0:
            vectors = np.array(vectors)
            result = np.average(vectors, axis=0)
            result /= np.linalg.norm(result)
        else:
            result = np.zeros(300)

        return result

    def text_category_prob(self, sent, category):
        distribution = self.acd.predict(sent)
        for cat, prob in distribution:
            if cat == category:
                return prob

        return 0

    def get_pd_features_map_core_nlp_cut_off(self, text, category, cats_len, sents):
        if len(sents) == 1 or cats_len == 1:
            return self.get_pd_features_ignore_category(text, category, cats_len)

        max_prob = 0
        max_prob_sent = None

        for sent in sents:
            prob = self.text_category_prob(sent, category)
            if prob > max_prob:
                max_prob = prob
                max_prob_sent = sent

        # TODO: maybe check max_prob > some_threshold
        if max_prob_sent is None or max_prob < 0.5:
            return self.get_pd_features_ignore_category(text, category, cats_len)

        return self.get_pd_features_ignore_category(max_prob_sent, category, cats_len)

    def train_pd(self):
        print('-- PD:')
        print('Loading tokenizer...')
        self.tokenizer = WordPunctTokenizer()

        print('Loading stemmer...')
        self.stemmer = PorterStemmer()

        print('Loading parser...')
        self.parser = load_core_nlp_parser()

        print('Loading dataset...')
        ds = load_dataset('data/laptops_train.xml')
        x, y = get_pd_ds(ds, self.get_pd_features_map_core_nlp_cut_off, self.parser)
        x_train, x_test, y_train, y_test = split_ds(x, y)

        # clf = linear_model.LogisticRegression(C=1.5)

        for c in [1, 2, 5, 7, 10, 13, 15, 17, 20, 23, 25, 30, 25, 40, 45, 50]:
            print('SVC(C={})'.format(c))

            clf = SVC(kernel='rbf', C=c)

            print('  Training...')
            clf.fit(x_train, y_train)

            print('  Evaluating...')
            predictions = clf.predict(x_test)

            f1 = f1_score(y_test, predictions, average='micro')
            print('  F1: {}'.format(f1))
            accuracy = accuracy_score(y_test, predictions)
            print('  Accuracy: {}'.format(accuracy))

        # return clf


if __name__ == '__main__':
    w2v = load_w2v()
    acd = ACD(w2v)
    acd.train_acd()

    pd = PD(w2v, acd)
    pd.train_pd()