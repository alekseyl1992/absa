import gensim

from ds_loader import load_dataset


def test_load_dataset():
    ds = load_dataset('data/laptops_train.xml')
    assert len(ds) > 0


def test_w2v():
    # Load Google's pre-trained Word2Vec model.
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        'pretrained/GoogleNews-vectors-negative300.bin', binary=True)

    assert len(w2v.word_vec('computer')) == 300
    assert w2v.most_similar(positive=['woman', 'king'], negative=['man'])[0][0] == 'queen'


if __name__ == '__main__':
    test_w2v()
