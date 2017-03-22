from pprint import pprint

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer

x = [
    (1, 0),
    (0, 1),
    (1, 1),
]

y = [
    ['cat'],
    ['dog'],
    ['cat', 'dog'],
]

mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(y)

clf = MLPClassifier(hidden_layer_sizes=(3,), max_iter=2000)
clf.fit(x, y_bin)

pred = clf.predict_proba(x)
pprint(pred)
