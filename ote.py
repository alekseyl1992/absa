from utils import load_w2v, get_ote_ds, load_dataset

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pystruct.learners import FrankWolfeSSVM
from pystruct.models import ChainCRF
from sklearn.model_selection import GridSearchCV


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

    def w2v_safe(self, token):
        if token in self.w2v.vocab and token not in self.stop_words:
            return self.w2v.word_vec(token)
        else:
            return np.zeros(300)

    def get_ote_features(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        return np.array(list(map(self.w2v_safe, tokens)))

    def get_ote_features_window(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        vectors = []

        prev_vec = np.zeros(300)
        for i, token in enumerate(tokens):
            if i + 1 < len(tokens):
                next_vec = self.w2v_safe(tokens[i+1])
            else:
                next_vec = np.zeros(300)

            cur_vec = self.w2v_safe(token)

            vector = np.concatenate([
                prev_vec, cur_vec, next_vec
            ])

            prev_vec = cur_vec

            vectors.append(vector)

        return np.array(vectors)

    def calc_f1(self, actuals, predictions):
        # count aspect terms in each sentence
        tp, tn, fp, fn = 0, 0, 0, 0

        actuals = map(self.iob_to_positional, actuals)
        predictions = map(self.iob_to_positional, predictions)

        for actual, prediction in zip(actuals, predictions):
            actual = set(actual)
            prediction = set(prediction)

            tp += len(actual.intersection(prediction))
            fp += len(prediction - actual)
            fn += len(actual - prediction)

        if tp != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0

        return f1, precision, recall

    @staticmethod
    def iob_to_positional(iob):
        positions = []
        begin = 0
        prev_label = 0
        for i, label in enumerate(iob):
            if prev_label == 0:
                if label == 1:
                    begin = i
            elif prev_label == 1 or prev_label == 2:
                if label == 0:
                    positions.append((begin, i - 1))
                elif label == 1:
                    positions.append((begin, i - 1))
                    begin = i

        if iob[-1] != 0:
            positions.append((begin, len(iob) - 1))

        return positions

    def train(self):
        print('-- OTE:')

        print('Loading dataset...')
        ds_train = load_dataset(r'data/restaurants_train.xml')
        x_train, y_train = get_ote_ds(ds_train, self.get_ote_features, self.tokenizer)
        ds_test = load_dataset(r'data/restaurants_test.xml')
        x_test, y_test = get_ote_ds(ds_test, self.get_ote_features, self.tokenizer)

        print('Fitting...')
        model = ChainCRF()
        ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
        ssvm.fit(x_train, y_train)

        predicted = ssvm.predict(x_test)
        f1 = self.calc_f1(y_test, predicted)

        print('{}'.format(f1))

    def scoring_fun(self, clf, x, y):
        predicted = clf.predict(x)
        f1 = self.calc_f1(y, predicted)
        return f1[0]

    def print_scores(self, scores, param_name, param_title, value_name):
        xs = map(lambda el: el[param_name], scores['params'])
        ys = map(lambda score: score, scores['mean_test_score'])

        df = pd.DataFrame(
            columns=[param_title, value_name],
            data=np.array(list(zip(xs, ys))))

        print(df)

    def grid_search(self):
        print('Loading dataset...')
        ds_train = load_dataset(r'data/restaurants_train.xml')
        x, y = get_ote_ds(ds_train, self.get_ote_features_window, self.tokenizer)

        print('Fitting...')
        model = ChainCRF()
        ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)

        params = {
            'C': np.linspace(0.005, 0.1, 10) + np.linspace(0.15, 1, 10)
        }

        grid_cv = GridSearchCV(ssvm, param_grid=params, scoring=self.scoring_fun)
        grid_cv.fit(x, y)

        scores = grid_cv.cv_results_

        plt.figure()

        plt_x_name = list(params.keys())[0]
        param_title = plt_x_name.replace('estimator__', '')

        plt_y_name = 'Mean F1'

        self.print_scores(scores, plt_x_name, param_title, plt_y_name)

        plt_x = list(map(
            lambda s: s[plt_x_name],
            scores['params']
        ))
        plt_y = list(map(
            lambda s: s,
            scores['mean_test_score']
        ))

        plt.title('SVM CRF')
        plt.xlabel(param_title)
        plt.ylabel(plt_y_name)

        plt.plot(plt_x, plt_y)
        plt.show()


if __name__ == '__main__':
    w2v = load_w2v(use_mock=False)
    ote = OTE(w2v)
    ote.grid_search()
