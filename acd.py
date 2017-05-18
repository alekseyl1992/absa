import string

import pickle
import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import pprint
from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split
from sklearn.model_selection import validation_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.utils import shuffle

from base import BaseTask
from plotter import plot_learning_curve
from utils import load_dataset, get_acd_ds, get_f1, category_fdist, load_w2v


class ACD(BaseTask):
    def __init__(self, w2v):
        super().__init__(w2v)

        self.stemmer = None
        self.mlb = None
        self.plot_f1s = False
        self.clf = None

        self.x_test_hand = None
        self.x_train_hand = None

    def get_acd_features(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        vectors = []

        for token in tokens:
            # token = self.stemmer.stem(token)
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

    def fit_and_evaluate(self, clf, x_train, y_train, x_test, y_test, mlb):
        clf.fit(x_train, y_train)

        print('Evaluating...')
        classes = mlb.classes_
        predictions = clf.predict_proba(x_test)

        for step in [0.005, 0.01, 0.05, 0.07, 0.08, 0.09,
                     0.1, 0.13, 0.15, 0.17, 0.2, 0.23, 0.25, 0.27, 0.3, 0.5]:
            f1 = get_f1(predictions, classes, y_test, step)
            print('F1: {}, step: {}'.format(f1, step))

    def learning_curve(self, clf, x_train, y_train, x_test, y_test):
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        plot_learning_curve(
            estimator=clf,
            X=x_train,
            y=y_train,
            cv=cv,
            title='Learning curve',
            n_jobs=4
        )

    def scoring_fun(self, clf, x, y):
        classes = self.mlb.classes_
        predictions = clf.predict_proba(x)

        assert len(predictions[0]) == len(classes)

        f1s = []
        steps = np.arange(0.01, 0.5, 0.05)
        for step in steps:
            actuals = self.mlb.inverse_transform(y)
            f1 = get_f1(predictions, classes, actuals, step)
            f1s.append(f1['f1'])

        if self.plot_f1s:
            df = pd.DataFrame(
                columns=['Step', 'F1'],
                data=np.array(list(zip(steps, f1s))))

            print(df)

            plt.figure()
            plt.grid(True)
            plt_x_name = 'Step'
            plt_y_name = 'F1'

            plt.title('F1 vs Step')
            plt.xlabel(plt_x_name)
            plt.ylabel(plt_y_name)

            plt.plot(steps, f1s)

            plt.show()

        max_f1 = np.max(f1s)
        return max_f1

    def validation_curve(self, clf, x, y, param_name, param_range):
        train_scores, valid_scores = validation_curve(
            clf, x, y,
            param_name=param_name,
            param_range=param_range,
            scoring=self.scoring_fun)

        print('Train Scores:')
        pprint(dict(zip(param_range, train_scores)))

        print('Valid Scores:')
        pprint(dict(zip(param_range, valid_scores)))

    def print_scores(self, scores, param_name, param_title, value_name):
        xs = map(lambda el: el[param_name], scores['params'])
        ys = map(lambda score: score, scores['mean_test_score'])

        df = pd.DataFrame(
            columns=[param_title, value_name],
            data=np.array(list(zip(xs, ys))))

        print(df)

    def print_results(self, results):
        data = []
        for name, scores, param_name, param_title in results:
            max_score = np.max(scores['mean_test_score'])
            max_score_id = np.argmax(scores['mean_test_score'])
            data.append([name, param_title, scores['params'][max_score_id][param_name], max_score])

        df = pd.DataFrame(
            columns=[
                'Классификатор', 'Оптимизируемый параметр',
                'Значение параметра', 'Max Mean F1'],
            data=np.array(data))

        print(df)

    def grid_search_acd(self, x, y, n_jobs=2):
        self.mlb = MultiLabelBinarizer()

        tasks = [
            {
                'clf': MLPClassifier(max_iter=500,
                                     hidden_layer_sizes=(156,),
                                     activation='logistic',
                                     alpha=0.001,
                                     learning_rate='adaptive',
                                     random_state=1),
                'name': 'MLP (156)',
                'params': {
                    'max_iter': np.arange(50, 1200, step=100)
                }
            }
        ]

        y = self.mlb.fit_transform(y)

        import warnings
        warnings.filterwarnings("ignore")

        results = []

        for _, task in enumerate(tasks):
            name = task['name']
            clf = task['clf']
            params = task['params']
            print('Running {}...'.format(name))

            grid_cv = GridSearchCV(clf, param_grid=params, scoring=self.scoring_fun, n_jobs=n_jobs)
            grid_cv.fit(x, y)

            scores = grid_cv.cv_results_

            plt.figure(_)

            plt_x_name = list(params.keys())[0]
            param_title = plt_x_name.replace('estimator__', '')

            plt_y_name = 'Mean F1'

            self.print_scores(scores, plt_x_name, param_title, plt_y_name)

            results.append((name, scores, plt_x_name, param_title))

            plt_x = list(map(
                lambda s: s[plt_x_name],
                scores['params']
            ))
            plt_y = list(map(
                lambda s: s,
                scores['mean_test_score']
            ))

            plt.title(name)
            plt.xlabel(param_title)
            plt.ylabel(plt_y_name)

            plt.plot(plt_x, plt_y)

        print('Grid search results:')
        self.print_results(results)

        plt.show()

    def train(self):
        print('-- ACD:')

        print('Loading dataset...')
        ds_train = load_dataset(r'data/restaurants_train.xml')
        fdist = category_fdist(ds_train)
        x_train, y_train = get_acd_ds(ds_train, fdist, self.get_acd_features)
        x_train_hand = pickle.load(
            open(r'data/hand/acd/acd-rest-train.pickle', 'rb'), encoding='latin1')

        ds_test = load_dataset(r'data/restaurants_test.xml')
        fdist = category_fdist(ds_test)
        x_test, y_test = get_acd_ds(ds_test, fdist, self.get_acd_features)
        x_test_hand = pickle.load(
            open(r'data/hand/acd/acd-rest-test.pickle', 'rb'), encoding='latin1')

        x_train, x_test, y_train, y_test = self.resplit(x_train, x_test, y_train, y_test)
        x_train_hand, x_test_hand, _, _ = self.resplit(x_train_hand, x_test_hand, x_train_hand, x_test_hand)

        merged_train = np.concatenate([x_train, x_train_hand], axis=1)
        merged_test = np.concatenate([x_test, x_test_hand], axis=1)

        self.mlb = MultiLabelBinarizer()
        clf = OneVsRestClassifier(
            SVC(kernel='linear', C=0.478947368421, probability=True, random_state=1))

        y_train = self.mlb.fit_transform(y_train)
        y_test = self.mlb.fit_transform(y_test)

        print('Training...')
        clf.fit(merged_train, y_train)

        print('Evaluating...')
        self.plot_f1s = True
        score = self.scoring_fun(clf, merged_test, y_test)
        print('F1: {}'.format(score))

        self.clf = clf

    def predict(self, sent, sample_id=None):
        features = self.get_acd_features(sent)

        if sample_id is not None:
            if self.x_test_hand is None:
                self.x_test_hand = pickle.load(
                    open(r'data/hand/acd/acd-rest-test.pickle', 'rb'), encoding='latin1')

            if self.x_train_hand is None:
                self.x_train_hand = pickle.load(
                    open(r'data/hand/acd/acd-rest-train.pickle', 'rb'), encoding='latin1')

            if sample_id[0] == 'test':
                features = np.concatenate([features, self.x_test_hand[sample_id[1]]])
            else:
                features = np.concatenate([features, self.x_train_hand[sample_id[1]]])

        predicted = self.clf.predict_proba([features])[0]
        classes = self.mlb.classes_

        return sorted(zip(classes, predicted), key=lambda tup: -tup[1])

    def predict_many(self, sents, just_top_one=True, sample_id=None):
        results = []
        for sent in sents:
            if sent is None:
                results.append((None, []))
                continue

            prediction = self.predict(sent, sample_id)
            if just_top_one:
                results.append((sent, prediction[0]))
            else:
                results.append((sent, prediction))
        return results

    # predicts all OTEs
    def predict_ote(self, sent, step=0.31):
        sent_cats = self.predict(sent)

        # get list of sentence's cats
        top_sent_cats = []
        for cat in sent_cats:
            if cat[1] > step:
                top_sent_cats.append(cat[0])

        tokens = self.tokenizer.tokenize(sent)
        per_word_cats = self.predict_many(tokens, False)

        results = []
        for sent_cat in top_sent_cats:
            ote = self.find_word_with_highest_score(per_word_cats, sent_cat)
            results.append((sent_cat, ote))

        return results

    def predict_ote_for_category(self, tokens, cat, sample_id):
        per_word_cats = self.predict_many(tokens, False, sample_id)
        _, word2score = self.find_word_with_highest_score(per_word_cats, cat)

        tokens = self.filter_nouns(tokens)
        per_word_cats = self.predict_many(tokens, False, sample_id)
        ote, _ = self.find_word_with_highest_score(per_word_cats, cat)

        return ote, word2score

    def get_word2score(self, tokens, cat, sample_id):
        per_word_cats = self.predict_many(tokens, False, sample_id)
        _, word2score = self.find_word_with_highest_score(per_word_cats, cat)
        return word2score

    def is_punct(self, word):
        return word[0] in string.punctuation

    def filter_nouns(self, tokens):
        tagged = nltk.pos_tag(tokens)
        return map(lambda t: t[0]
                   if t[1] in ['NN', 'PRP', 'NNP', 'JJ', 'NNS'] and not self.is_punct(t[0])
                   else None,
                   tagged)

    def find_word_with_highest_score(self, per_word_cats, sent_cat):
        word2score = []
        for i, (word, distribution) in enumerate(per_word_cats):
            found_prob = 0
            for cat, prob in distribution:
                if cat == sent_cat:
                    found_prob = prob
                    break

            word2score.append(((i, word), found_prob))

        return sorted(word2score, key=lambda tup: -tup[1])[0][0], word2score


if __name__ == '__main__':
    w2v = load_w2v(True)
    acd = ACD(w2v)
    acd.train()
