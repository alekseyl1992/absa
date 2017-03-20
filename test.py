import gensim
import numpy as np
from nltk import PorterStemmer
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from nltk.tokenize import WordPunctTokenizer

from ds_loader import load_dataset, split_ds, get_acd_ds, get_f1, category_fdist
from plotter import plot_learning_curve


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

    def test_acd(self):
        print('Loading w2v...')

        self.w2v = W2VMock()
        # self.w2v = gensim.models.KeyedVectors.load_word2vec_format(
        #     'pretrained/GoogleNews-vectors-negative300.bin', binary=True)

        print('Loading tokenizer...')
        self.tokenizer = WordPunctTokenizer()

        print('Loading stemmer...')
        self.stemmer = PorterStemmer()

        print('Loading dataset...')
        ds = load_dataset('data/laptops_train.xml')
        fdist = category_fdist(ds)
        x, y, labels = get_acd_ds(ds, fdist, self.get_acd_features)
        x_train, x_test, y_train, y_test = split_ds(x, y)

        # clf = SVC(kernel='rbf', probability=True)
        clf = MLPClassifier(max_iter=2000,
                            hidden_layer_sizes=(20,),
                            activation='logistic',
                            learning_rate='adaptive')

        print('Training...')
        # clf.fit(x_train, y_train)

        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        plot_learning_curve(
            estimator=clf,
            X=x_train,
            y=y_train,
            cv=cv,
            title='Learning curve',
            n_jobs=4
        )

        # print('Evaluating...')
        # classes = clf.classes_
        # predictions = clf.predict_proba(x_test)
        #
        # for step in [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5]:
        #     f1 = get_f1(predictions, classes, labels, step)
        #     print('F1: {}, step: {}'.format(f1, step))


if __name__ == '__main__':
    acd = ACD()
    acd.test_acd()
