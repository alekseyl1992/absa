import os
from pprint import pprint

from nltk.parse.stanford import StanfordParser


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

sent = 'W7 Pro with W8 pro upgrade is nice, but it frequently freezes for a few seconds here and there.'
parsed = parser.raw_parse(sent)
for line in parsed:
    print(line)
    line.draw()
