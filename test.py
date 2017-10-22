import gensim
import numpy as np
from nltk import PorterStemmer, pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC, LinearSVC
from nltk.tokenize import WordPunctTokenizer

from ds_loader import load_dataset, split_ds, get_acd_ds, get_f1, category_fdist
from plotter import plot_learning_curve

from matplotlib import pyplot as plt


def test_load_dataset():
    ds = load_dataset('data/laptops_train.xml')
    assert len(ds) > 0


def test_w2v():
    # Load Google's pre-trained Word2Vec model.
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        'pretrained/GoogleNews-vectors-negative300.bin', binary=True)

    assert len(w2v.word_vec('computer')) == 300
    assert w2v.most_similar(positive=['woman', 'king'], negative=['man'])[0][0] == 'queen'


class W2VMock:
    def __init__(self):
        self.vocab = []

    def word_vector(self, word):
        return np.zeros(300)


class ACD:
    def __init__(self):
        self.w2v = None
        self.tokenizer = None
        self.stemmer = None
        self.mlb = None

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
        for step in np.arange(0.01, 0.5, 0.05):
            actuals = self.mlb.inverse_transform(y)
            f1 = get_f1(predictions, classes, actuals, step)
            f1s.append(f1['f1'])

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

    def test_acd(self):
        print('Loading w2v...')

        self.w2v = W2VMock()
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format('pretrained/GoogleNews-vectors-negative300.bin', binary=True)

        print('Loading tokenizer...')
        self.tokenizer = WordPunctTokenizer()

        print('Loading stemmer...')
        self.stemmer = PorterStemmer()

        print('Loading dataset...')
        ds = load_dataset('data/laptops_train.xml')
        fdist = category_fdist(ds)
        x, y = get_acd_ds(ds, fdist, self.get_acd_features)
        # x_train, x_test, y_train, y_test = split_ds(x, y)

        self.mlb = MultiLabelBinarizer()
        # y_train = self.mlb.fit_transform(y_train)
        # y_test = self.mlb.fit_transform(y_test)

        # clf = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
        # clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
        # clf = OneVsRestClassifier(GaussianNB())
        # clf = MLPClassifier(max_iter=500,
        #                     hidden_layer_sizes=(20,),
        #                     activation='logistic',
        #                     learning_rate='adaptive')

        tasks = [
            {
                'clf': OneVsRestClassifier(SVC(kernel='rbf', probability=True)),
                'name': 'SVM (метод опорных векторов)',
                'params': {
                    'estimator__C': ('C', np.linspace(0.1, 5.0, num=20))
                }
            },
            {
                'clf': OneVsRestClassifier(RandomForestClassifier(n_estimators=10)),
                'name': 'Random Forest (случайный лес)',
                'params': {
                    'estimator__n_estimators': ('n_estimators (количество деревьев)', np.arange(1, 60, step=2))
                }
            },
            {
                'clf': OneVsRestClassifier(GaussianNB()),
                'name': 'Gaussian NB (наивный байесовский классификатор)',
                'params': {
                    'n_jobs': ('n_jobs', [1, 1])
                }
            },
            {
                'clf': MLPClassifier(max_iter=500,
                                     hidden_layer_sizes=(20,),
                                     activation='logistic',
                                     learning_rate='adaptive'),
                'name': 'MLP (20) (многослойный персептрон)',
                'params': {
                    'max_iter': ('max_iter (число итераций обучения)', np.arange(100, 2000, step=100))
                }
            },
            {
                'clf': MLPClassifier(max_iter=500,
                                     hidden_layer_sizes=(156,),
                                     activation='logistic',
                                     learning_rate='adaptive'),
                'name': 'MLP (156) (многослойный персептрон)',
                'params': {
                    'max_iter': ('max_iter (число итераций обучения)', np.arange(100, 3000, step=200))
                }
            }
        ]

        y = self.mlb.fit_transform(y)

        import warnings
        warnings.filterwarnings("ignore")

        for _, task in enumerate(tasks):
            name = task['name']
            clf = task['clf']
            params2values = {k: v[1] for k, v in task['params'].items()}
            params2titles = {k: v[0] for k, v in task['params'].items()}
            print('Running {}...'.format(name))

            grid_cv = GridSearchCV(clf, param_grid=params2values, scoring=self.scoring_fun)
            grid_cv.fit(x, y)

            scores = grid_cv.grid_scores_
            pprint(scores)

            plt.figure(_)

            param = list(params2titles.keys())[0]

            plt_x = list(map(
                lambda s: s.parameters[param],
                scores
            ))
            plt_y = list(map(
                lambda s: s.mean_validation_score,
                scores
            ))

            plt_x_name = list(params2titles.values())[0]
            plt_y_name = 'F1'

            plt.title(name)
            plt.xlabel(plt_x_name)
            plt.ylabel(plt_y_name)

            plt.grid()
            plt.plot(plt_x, plt_y)
        plt.show()

        return

        print('Training...')
        # self.fit_and_evaluate(clf, x_train, y_train, x_test, y_test, self.mlb)

        y = self.mlb.fit_transform(y)
        # self.validation_curve(clf, x, y,
        #                       param_name='max_iter',
        #                       param_range=np.arange(200, 2000, 100, dtype=int))
        # self.validation_curve(clf, x, y,
        #                       param_name='estimator__C',
        #                       param_range=np.linspace(0.1, 1.0, num=10))
        self.validation_curve(clf, x, y,
                              param_name='estimator__priors',
                              param_range=[None])


if __name__ == '__main__':
    acd = ACD()
    acd.test_acd()
