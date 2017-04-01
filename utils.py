import os
from pprint import pprint
import xml.etree.ElementTree as ET

import gensim
import numpy as np
from nltk import FreqDist
from nltk.parse.stanford import StanfordParser


class Entry:
    def __init__(self, text, opinions):
        self.text = text
        self.opinions = opinions


class Opinion:
    def __init__(self, category, polarity=None):
        self.category = category
        self.polarity = polarity


def load_dataset(path):
    # parse XML
    tree = ET.parse(path)
    root = tree.getroot()

    dataset = []
    for review in root.findall('Review'):
        for sentences in review.findall('sentences'):
            for sentence in sentences.findall('sentence'):
                text = sentence.find('text')
                opinions_list = sentence.find('Opinions')

                if not opinions_list:
                    continue

                opinions = opinions_list.findall('Opinion')

                parsed_opinions = []
                for opinion in opinions:
                    parsed_opinion = Opinion(
                        category=opinion.attrib['category'],
                        polarity=opinion.attrib['polarity'])

                    parsed_opinions.append(parsed_opinion)

                entry = Entry(
                    text=text.text,
                    opinions=parsed_opinions)
                dataset.append(entry)

    return dataset


def get_acd_ds(source_ds, fdist, feature_extractor):
    features, labels = [], []

    common_categories = list(map(lambda pair: pair[0], fdist.most_common(19)))

    for source_entry in source_ds:
        opinion_labels = []
        features.append(feature_extractor(source_entry.text))
        for opinion in source_entry.opinions:
            # if opinion.category not in common_categories:
            #     opinion.category = 'OTHER#OTHER'

            # collect all labels for the entry
            opinion_labels.append(opinion.category)

        labels.append(opinion_labels)

    return np.array(features), np.array(labels)


def get_pd_ds(source_ds, feature_extractor, parser=None):
    features, labels = [], []

    ds_len = len(source_ds)

    preprocessed = []
    if parser:
        print('Core NLP Parser preprocessing...')

        texts = []
        for source_entry in source_ds:
            texts.append(source_entry.text)

        trees = parser.raw_parse_sents(texts)
        preprocessed = [
            split_on_sents(tree, source_sent)
            for tree, source_sent in zip(trees, texts)
        ]

    for i, source_entry in enumerate(source_ds):
        cats_len = len(source_entry.opinions)
        for opinion in source_entry.opinions:
            if parser:
                features.append(feature_extractor(source_entry.text, opinion.category, cats_len, preprocessed[i]))
            else:
                features.append(feature_extractor(source_entry.text, opinion.category, cats_len))
            labels.append(opinion.polarity)

        if i % 100 == 0:
            print('get_pd_ds progress: {}/{}'.format(i, ds_len))

    return np.array(features), np.array(labels)


def split_ds(x, y, test_size=0.2):
    assert len(x) == len(y)

    train_idx = int(len(x) * (1.0 - test_size))

    x_train = x[:train_idx]
    y_train = y[:train_idx]

    x_test = x[train_idx:]
    y_test = y[train_idx:]

    return x_train, x_test, y_train, y_test


def category_fdist(ds):
    fdist = FreqDist()
    for entry in ds:
        for opinion in entry.opinions:
            fdist[opinion.category] += 1

    return fdist


def get_f1(predictions, classes, actuals, step):
    tn, tp, fp, fn = 0, 0, 0, 0

    assert len(predictions) == len(actuals), \
        'len(predictions)={}, len(actuals)={}'.format(len(predictions), len(actuals))

    for prediction_id, prediction in enumerate(predictions):
        predicted_classes = []
        for i, p in enumerate(prediction):
            if p > step:
                predicted_classes.append(classes[i])
        predicted_classes = set(predicted_classes)

        actual_classes = set(actuals[prediction_id])

        tp += len(predicted_classes.intersection(actual_classes))
        fp += len(predicted_classes - actual_classes)
        fn += len(actual_classes - predicted_classes)

    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        precision = 0
        recall = 0
        f1 = 0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


class W2VMock:
    def __init__(self):
        self.vocab = []

    def word_vector(self, word):
        return np.zeros(300)


def load_w2v(use_mock=False):
    if use_mock:
        print('Loading mock w2v...')
        w2v = W2VMock()
    else:
        print('Loading real w2v...')
        w2v = gensim.models.KeyedVectors.load_word2vec_format(
            'pretrained/GoogleNews-vectors-negative300.bin', binary=True)

    print('Done')
    return w2v


def load_core_nlp_parser():
    cwd = os.path.dirname(os.path.realpath(__file__))
    stanford_parser_dir = os.path.join(cwd, 'core_nlp')
    eng_model_path = os.path.join(stanford_parser_dir,
                                  'models',
                                  'english\\edu\\stanford\\nlp\\models\\lexparser',
                                  'wsjRNN.ser.gz')

    path_to_models_jar = os.path.join(stanford_parser_dir, 'stanford-parser-3.7.0-models.jar')
    path_to_jar = os.path.join(stanford_parser_dir, 'stanford-parser.jar')

    parser = StanfordParser(
        model_path=eng_model_path,
        path_to_models_jar=path_to_models_jar,
        path_to_jar=path_to_jar)

    return parser


def split_on_sents(tree, source_sent):
    root = list(tree)[0]
    children = root[0]
    sents = [
        tokens_to_sent(sent.leaves())
        for sent in children
        if sent.label() == 'S'
    ]

    if len(sents) == 0:
        return [source_sent]

    return sents


def draw(parsed):
    for line in parsed:
        print(line)
        line.draw()


def tokens_to_sent(tokens):
    s = ' '.join(tokens)
    s.replace(' ,', ',')
    s.replace(' .', '.')

    return s