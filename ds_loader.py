import xml.etree.ElementTree as ET


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

                opinions = opinions_list.find('Opinion')

                parsed_opinions = []
                for opinion in opinions:
                    parsed_opinion = Opinion(
                        category=opinion.attrib['category'],
                        polarity=opinion.attrib['polarity'])

                    parsed_opinions.append(parsed_opinion)

                entry = Entry(
                    text=text.text,
                    opinions=opinions)
                dataset.append(entry)

    return dataset
