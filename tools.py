from pprint import pprint

from utils import load_dataset, load_dataset_2014


def dataset_stats(source_ds):
    stats = {
        'total': len(source_ds),
        'positive': 0,
        'neutral': 0,
        'negative': 0,
        'conflict': 0,
        'different': 0,
        'pos_and_neg': 0
    }

    for source_entry in source_ds:
        polarities = []
        for opinion in source_entry.opinions:
            stats[opinion.polarity] += 1
            polarities.append(opinion.polarity)

        if len(set(polarities)) > 1:
            stats['different'] += 1

        if 'positive' in polarities and 'negative' in polarities:
            stats['pos_and_neg'] += 1

    return stats


if __name__ == '__main__':
    # ds = load_dataset('data/laptops_train.xml')
    # ds = load_dataset(r'C:\Projects\ML\aueb-absa\polarity_detection\restaurants\ABSA16_Restaurants_Train_SB1_v2.xml')
    ds = load_dataset_2014('data/laptops_train_2014.xml')
    stats = dataset_stats(ds)
    pprint(stats)
