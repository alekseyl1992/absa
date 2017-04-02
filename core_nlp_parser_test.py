from pprint import pprint

from utils import load_core_nlp_parser, draw

parser = load_core_nlp_parser('lexparser/englishRNN.ser.gz')


source_sents = [
    'The screen resolution is crystal clear, the speakers are amazing, and the track pad is easy to use.',
    'Despite the fact that there\'s no optical drive, everything is perfect.',
    'The home page/startup is easy to navigate which is the one thing I was mainly concerned about.'
]

for source_sent in source_sents:
    print(source_sent)
    tree = parser.raw_parse(source_sent)
    draw(tree)
    print()
