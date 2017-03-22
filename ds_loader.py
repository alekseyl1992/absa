import xml.etree.ElementTree as ET

import numpy as np
from nltk import FreqDist
from sklearn.model_selection import train_test_split


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

    assert len(predictions) == len(actuals)

    for prediction_id, prediction in enumerate(predictions):
        predicted_classes = set(prediction)
        actual_classes = set(actuals[prediction_id])

        tp += len(predicted_classes.intersection(actual_classes))
        fp += len(predicted_classes - actual_classes)
        fn += len(actual_classes - predicted_classes)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = (2 * precision * recall) / (precision + recall)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }
