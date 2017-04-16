import math

import numpy as np
np.random.seed(1337)  # for reproducibility

import keras
from keras import layers
from keras.layers import Dense, Dropout, GlobalMaxPooling1D, Convolution1D
from keras.layers import Input
from keras.models import Model
import pickle
from nltk import PorterStemmer, re, pprint
from nltk.tokenize import WordPunctTokenizer
from nltk.tree import ParentedTree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC

from utils import load_dataset, split_ds, load_w2v, get_pd_ds, load_core_nlp_parser


class PD:
    def __init__(self, w2v, acd):
        self.w2v = w2v
        self.tokenizer = None
        self.stemmer = None
        self.parser = None
        self.clf = None
        self.acd = acd

        self.max_sentence_len = 30
        self.average_vectors = True

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

    def reshape(self, vectors):
        if self.average_vectors:
            if len(vectors) > 0:
                vectors = np.array(vectors)
                result = np.average(vectors, axis=0)
                result /= np.linalg.norm(result)
            else:
                result = np.zeros(300)

            return result
        else:
            # while len(vectors) < self.max_sentence_len:
            #     vectors.append(np.zeros(300))

            if len(vectors) == 0:
                vectors = [np.zeros(300)]

            while len(vectors) < self.max_sentence_len:
                vectors += vectors[:self.max_sentence_len - len(vectors)]

            if len(vectors) != self.max_sentence_len:
                vectors = vectors[:self.max_sentence_len]

            return np.array(vectors)

    def _get_pd_features_ignore_category(self, tokens):
        vectors = []

        for token in tokens:
            if token in self.w2v.vocab and token not in self.stop_words:
                vector = self.w2v.word_vec(token)
                vectors.append(vector)

        return self.reshape(vectors)

    def get_pd_features_ignore_category(self, text, *args, **kwargs):
        tokens = self.tokenizer.tokenize(text.lower())
        return self._get_pd_features_ignore_category(tokens)

    def get_pd_features_append_category(self, text, category, cats_len, sents, ote):
        text_vector = self.get_pd_features_map_tree_distance(text, category, cats_len, sents, ote)

        category_tokens = category.lower().split('#')
        category_vector = self._get_pd_features_ignore_category(category_tokens)

        return np.concatenate([text_vector, category_vector * 3])

    def get_pd_features_insert_category(self, text, category, cats_len, sents, ote_):
        entity, aspect = category.lower().split('#')
        text = text + ' ' + entity + ' ' + aspect

        return self.get_pd_features_ignore_category(text, category, cats_len)

    def get_pd_features_map_linear_cut_off(self, text, category, cats_len, sents, ote_):
        tokens = self.tokenizer.tokenize(text)
        (position, ote), word2score = self.acd.predict_ote_for_category(tokens, category)

        if cats_len != 1 and ote is not None:
            distance_limit = 2
            tokens = tokens[
                     position - distance_limit:
                     position + distance_limit + 1]

        return self._get_pd_features_ignore_category(tokens)

    def get_pd_features_map_linear_weighted(self, text, category, cats_len, sents, ote):
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

    def get_pd_features_map_core_nlp_cut_off(self, text, category, cats_len, sents, ote):
        max_prob = 0
        max_prob_sent = text

        sents.append((text, None))

        for sent, tree in sents:
            prob = self.text_category_prob(sent, category)
            if prob > max_prob:
                max_prob = prob
                max_prob_sent = sent

        return np.concatenate([
            self.get_pd_features_ignore_category(text, category, cats_len),
            self.get_pd_features_ignore_category(max_prob_sent, category, cats_len),
        ])

    def get_pd_features_map_core_nlp_ote(self, text, category, cats_len, sents, ote):
        max_prob = 0
        max_prob_sent = text

        for sent, tree in sents:
            prob = self.text_category_prob(sent, category)
            if prob > max_prob and ote in sent:
                max_prob = prob
                max_prob_sent = sent

        return np.concatenate([
            self.get_pd_features_append_category(text, category, cats_len, sents, ote),
            self.get_pd_features_ignore_category(text, category, cats_len, sents, ote),
            self.get_pd_features_ignore_category(max_prob_sent, category, cats_len),
        ])

    def get_pd_features_map_core_nlp_append_adjp(self, text, category, cats_len, sents, ote):
        text_vector = self.get_pd_features_ignore_category(text, category, cats_len)

        max_prob = 0
        max_prob_sent = None
        max_prob_tree = None

        for sent, tree in sents:
            prob = self.text_category_prob(sent, category)
            if prob > max_prob:
                max_prob = prob
                max_prob_sent = sent
                max_prob_tree = tree

        if max_prob_tree is not None:
            adjps = list(max_prob_tree.subtrees(lambda t: t.label() == 'ADJP'))
            if len(adjps) == 0:
                adjps = list(max_prob_tree.subtrees())

            adjp_str = ' '.join([
                tokens_to_sent(adjp.leaves())
                for adjp in adjps
            ])
            adjp_vec = self.get_pd_features_ignore_category(adjp_str, category, cats_len)
            return adjp_vec

        return text_vector

    def get_tree_distance(self, ptree, from_id, to_id):
        from_location = ptree.leaf_treeposition(from_id)
        to_location = ptree.leaf_treeposition(to_id)

        matching_distance = 0
        for i, (x, y) in enumerate(zip(from_location, to_location)):
            if x == y:
                matching_distance += 1
            else:
                break

        distance = max([len(from_location), len(to_location)]) - matching_distance

        # ignore pre-last tree level
        if distance != 0:
            distance -= 1

        return distance

    def get_tree_distance_matrix(self, tree):
        ptree = ParentedTree.fromstring(str(tree))

        leaf_values = ptree.leaves()
        distance_matrix = []
        for leaf_id_from, leaf_from in enumerate(leaf_values):
            distance_matrix.append([])
            for leaf_id_to, leaf_to in enumerate(leaf_values):
                distance = self.get_tree_distance(ptree, leaf_id_from, leaf_id_to)
                distance_matrix[leaf_id_from].append(distance)

        return distance_matrix

    def get_pd_features_map_tree_distance(self, text, category, cats_len, sents, ote):
        tree = sents[0][1]

        tokens = list(map(lambda token: token.lower(), tree.leaves()))

        distance_matrix = np.array(self.get_tree_distance_matrix(tree))
        h = tree.height()
        word2score = self.acd.get_word2score(tokens, category)
        p_i = np.array([score
                        for (_, score) in word2score])

        p_ij = np.transpose(p_i * np.exp(-distance_matrix ** 2 / (2 * h)))
        p_j = np.sum(p_ij, axis=0)
        p_j = p_j / np.linalg.norm(p_j) + 0.5

        tokens = zip(tokens, p_j)

        vectors = []
        for i, (token, weight) in enumerate(tokens):
            if token in self.w2v.vocab:
                word_vec = self.w2v.word_vec(token)
                word_vec /= np.linalg.norm(word_vec)
                vector = word_vec * weight
                vectors.append(vector)

        return self.reshape(vectors)

    def get_linear_distance_matrix(self, tokens):
        distance_matrix = []
        for i, token_from in enumerate(tokens):
            distance_matrix.append([0] * len(tokens))
            for j, token_to in enumerate(tokens):
                distance_matrix[i][j] = math.fabs(i - j)

        return distance_matrix

    def get_pd_features_map_linear_distance(self, text, category, cats_len, sents, ote):
        tokens = self.tokenizer.tokenize(text)

        distance_matrix = np.array(self.get_linear_distance_matrix(tokens))
        h = len(tokens)
        word2score = self.acd.get_word2score(tokens, category)
        p_i = np.array([score
                        for (_, score) in word2score])

        p_ij = np.transpose(p_i * np.exp(-distance_matrix ** 2 / (2 * h)))
        p_j = np.sum(p_ij, axis=0)
        p_j = p_j / np.linalg.norm(p_j) + 0.5

        tokens = zip(tokens, p_j)

        vectors = []
        for i, (token, weight) in enumerate(tokens):
            if token in self.w2v.vocab:
                word_vec = self.w2v.word_vec(token)
                word_vec /= np.linalg.norm(word_vec)
                vector = word_vec * weight
                vectors.append(vector)

        return self.reshape(vectors)

    def prepare_data(self, feature_extractor, average_vectors=False):
        self.average_vectors = average_vectors
        print('-- PD:')
        print('Loading tokenizer...')
        self.tokenizer = WordPunctTokenizer()

        print('Loading stemmer...')
        self.stemmer = PorterStemmer()

        print('Loading parser...')
        self.parser = load_core_nlp_parser()

        print('Loading datasets...')
        ds_train = load_dataset(r'data/restaurants_train.xml')
        x_train, y_train = get_pd_ds(
            ds_train, feature_extractor, self.parser, my_split_on_sents, r'data/restaurants_train.xml')
        ds_test = load_dataset(r'data/restaurants_test.xml')
        x_test, y_test = get_pd_ds(
            ds_test, feature_extractor, self.parser, my_split_on_sents, r'data/restaurants_test.xml')
        return x_train, x_test, y_train, y_test

    def train_pd(self):
        x_train, x_test, y_train, y_test = self.prepare_data(self.get_pd_features_map_tree_distance, False)

        x_train = x_train.reshape(len(x_train), self.max_sentence_len * 300)
        x_test = x_test.reshape(len(x_test), self.max_sentence_len * 300)

        max_accuracy = 0

        for c in np.arange(0.01, 0.25, 0.02):
            # for c in np.arange(0.00001, 0.0001, 0.00002):
            clf = SVC(kernel='rbf', C=c, random_state=1, probability=True)

            # print('  Training...')
            clf.fit(x_train, y_train)

            # print('  Evaluating...')
            predictions = clf.predict_proba(x_test)
            accuracy = self.calc_accuracy(y_test, predictions, clf.classes_)
            print('  {}: {}'.format(c, accuracy))

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                self.clf = clf

        return self.clf, max_accuracy

    def train_pd_keras(self, data=None):
        if data:
            x_train, x_test, y_train, y_test = data
        else:
            x_train, x_test, y_train, y_test = self.prepare_data(self.get_pd_features_map_tree_distance, False)

        x_train_hand = pickle.load(
            open(r'../aueb-absa/polarity_detection/hand-features-train.pickle', 'rb'), encoding='latin1')
        x_test_hand = pickle.load(
            open(r'../aueb-absa/polarity_detection/hand-features-test.pickle', 'rb'), encoding='latin1')

        x_train_hand = np.array(x_train_hand)
        x_test_hand = np.array(x_test_hand)

        batch_size = 50
        num_classes = 3
        epochs = 100

        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_test = lb.transform(y_test)

        input_shape = (self.max_sentence_len, 300)

        input = Input(input_shape)

        convs = []
        for kernel_size in [(3,), (4,), (5,)]:
            conv = Convolution1D(filters=200,
                                 kernel_size=kernel_size,
                                 activation='relu')(input)
            pool = GlobalMaxPooling1D()(conv)
            convs.append(pool)

        convs = layers.concatenate(convs)
        dropout = Dropout(0.3)(convs)

        pre_output = Dense(50, activation='sigmoid')(dropout)
        input_hand = Input(shape=(59,))

        merged = layers.concatenate([pre_output, input_hand])

        output = Dense(num_classes, activation='softmax')(merged)

        model = Model(inputs=[input, input_hand], outputs=[output])

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        history = model.fit([x_train, x_train_hand], y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=([x_test, x_test_hand], y_test))

        val_acc = history.history['val_acc']
        print('Max val_acc: {} (epoch: {})'.format(np.max(val_acc), np.argmax(val_acc) + 1))

    def train_pd1(self):
        print('-- PD:')
        print('Loading tokenizer...')
        self.tokenizer = WordPunctTokenizer()

        print('Loading stemmer...')
        self.stemmer = PorterStemmer()

        print('Loading parser...')
        self.parser = load_core_nlp_parser()

        print('Loading dataset...')
        # ds = load_dataset('data/laptops_train.xml')
        ds = load_dataset(
            r'C:\Projects\ML\aueb-absa\polarity_detection\restaurants\ABSA16_Restaurants_Train_SB1_v2.xml')
        x, y = get_pd_ds(ds, self.get_pd_features_ignore_category, self.parser, my_split_on_sents)
        x_train, x_test, y_train, y_test = split_ds(x, y)

        max_accuracy = 0
        for c in np.arange(0.05, 0.7, 0.05):
            # print('SVC(C={})'.format(c))

            clf = SVC(kernel='rbf', C=c, random_state=1, probability=True)

            # print('  Training...')
            clf.fit(x_train, y_train)

            # print('  Evaluating...')
            predictions = clf.predict_proba(x_test)
            accuracy = self.calc_accuracy(y_test, predictions, clf.classes_)
            # print('  Accuracy: {}'.format(accuracy))

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                self.clf = clf

        return self.clf, max_accuracy, x_test, y_test

    def train_pd2(self):
        print('-- PD:')
        print('Loading tokenizer...')
        self.tokenizer = WordPunctTokenizer()

        print('Loading stemmer...')
        self.stemmer = PorterStemmer()

        print('Loading parser...')
        self.parser = load_core_nlp_parser()

        print('Loading dataset...')
        # ds = load_dataset('data/laptops_train.xml')
        ds = load_dataset(
            r'C:\Projects\ML\aueb-absa\polarity_detection\restaurants\ABSA16_Restaurants_Train_SB1_v2.xml')
        x, y = get_pd_ds(ds, self.get_pd_features_insert_category, self.parser, my_split_on_sents)
        x_train, x_test, y_train, y_test = split_ds(x, y)

        max_accuracy = 0
        for c in np.arange(0.1, 1.1, 0.1):
            # print('SVC(C={})'.format(c))

            clf = SVC(kernel='rbf', C=c, random_state=1, probability=True)

            # print('  Training...')
            clf.fit(x_train, y_train)

            # print('  Evaluating...')
            predictions = clf.predict_proba(x_test)
            accuracy = self.calc_accuracy(y_test, predictions, clf.classes_)
            # print('  Accuracy: {}'.format(accuracy))

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                self.clf = clf

        return self.clf, max_accuracy, x_test, y_test

    def calc_accuracy(self, y_test, predictions, classes):
        predictions = [
            classes[np.argmax(prediction)]
            for prediction in predictions
        ]

        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def train_two(self):
        print('PD1...')
        pd1, acc1, x_test1, y_test1 = self.train_pd1()
        print('PD1 Accuracy: {}'.format(acc1))

        print('PD2...')
        pd2, acc2, x_test2, y_test2 = self.train_pd2()
        print('PD2 Accuracy: {}'.format(acc2))

        predictions1 = pd1.predict_proba(x_test1)
        predictions2 = pd2.predict_proba(x_test2)

        predictions = np.average([predictions1, predictions2], axis=0)
        accuracy = self.calc_accuracy(y_test1, predictions, pd1.classes_)
        print('Result Accuracy: {}'.format(accuracy))

    def predict_polarity(self, sent):
        features = self.get_pd_features_ignore_category(sent, None, None)
        polarity = self.clf.predict([features])
        return polarity


def my_split_on_sents(tree, source_sent):
    root = list(tree)[0]
    children = root.subtrees(lambda t: len(t.leaves()) > 4)

    sents = [
        (tokens_to_sent(sent.leaves()), sent)
        for sent in children
    ]

    sents = [(source_sent, root)] + sents
    return sents


regex = re.compile(r' ([.,\'!;])')


def tokens_to_sent(tokens):
    s = ' '.join(tokens)
    s = s.replace('-LRB-', '(')
    s = s.replace('-RRB-', ')')
    s = s.replace(' n\'t', 'n\'t')
    s = regex.sub(r'\1', s)

    return s


if __name__ == '__main__':
    w2v = load_w2v(use_mock=True)

    acd = None
    # acd = ACD(w2v)
    # acd.train_acd()

    pd = PD(w2v, acd)
    pd.train_pd()
