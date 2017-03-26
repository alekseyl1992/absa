from pprint import pprint

from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
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
grid_cv = GridSearchCV(clf, param_grid={
    'max_iter': [1, 1]
})
grid_cv.fit(x, y_bin)

pprint(grid_cv.grid_scores_)

# clf.fit(x, y_bin)
#
# pred = clf.predict(x)
# pprint(pred)
#
# all_labels = mlb.inverse_transform(pred)
# pprint(all_labels)
