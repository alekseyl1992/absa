from pprint import pprint

from utils import load_dataset, load_dataset_2014


def dataset_stats(source_ds):
    stats = {
        'total_sents': len(source_ds),
        'total_opinions': 0,
        'positive': 0,
        'neutral': 0,
        'negative': 0,
        'different': 0,
        'pos_and_neg': 0,
        'cats': 0
    }

    cats = []

    for source_entry in source_ds:
        polarities = []
        for opinion in source_entry.opinions:
            stats[opinion.polarity] += 1
            polarities.append(opinion.polarity)
            stats['total_opinions'] += 1
            cats += opinion.category

        if len(set(polarities)) > 1:
            stats['different'] += 1

        if 'positive' in polarities and 'negative' in polarities:
            stats['pos_and_neg'] += 1

    stats['cats'] = len(set(cats))

    stats['positive'] = (stats['positive'], round(stats['positive'] / stats['total_opinions'] * 100, 2))
    stats['neutral'] = (stats['neutral'], round(stats['neutral'] / stats['total_opinions'] * 100, 2))
    stats['negative'] = (stats['negative'], round(stats['negative'] / stats['total_opinions'] * 100, 2))
    stats['different'] = (stats['different'], round(stats['different'] / stats['total_sents'] * 100, 2))
    stats['pos_and_neg'] = (stats['pos_and_neg'], round(stats['pos_and_neg'] / stats['total_sents'] * 100, 2))

    return stats


if __name__ == '__main__':
    for domain in ['laptops', 'restaurants']:
        for part in ['train', 'test']:
            print('data/{}_{}.xml'.format(domain, part))
            ds = load_dataset('data/{}_{}.xml'.format(domain, part))
            stats = dataset_stats(ds)
            pprint(stats)
            print()
