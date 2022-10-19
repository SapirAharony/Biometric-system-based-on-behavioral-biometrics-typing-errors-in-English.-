import system_features_extractor, os, string_metrics
import pandas as pd
from json import load
import numpy as np
from gensim.models import Word2Vec

################################## exctract data from original files

def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = load(f)
    return data


source_file_dir = 'C:\\Users\\user\\Desktop\\inz_wyniki'
dest_file_dir = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files'

# for file in os.listdir(source_file_dir):
#     if file[-4:] == 'json':
#         system_features_extractor.extract_data(source_file_dir + '\\' + file, dest_file_dir + '\\done_' + file, file[:-5].capitalize())

# print(string_metrics.Distances('test', 'ter', is_tokenized=True).__dict__)

lista = []
print(system_features_extractor.Word('tert', 'NN','tests', 'NNP').__dict__['distance'].__dict__)
# dystans = string_metrics.Distances('test', 'ter', is_tokenized=True)
#

vec_size = 1
pos_tags = [
    ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
     'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
     'WP', 'WP$', 'WRB']]
pos_tag_word2vec = Word2Vec(pos_tags, min_count=1, vector_size=vec_size)

for k in pos_tags[0]:
    print(pos_tag_word2vec.wv.get_vector(k))


# for k in string_metrics.Distances('test', 'ter',  is_tokenized=True).__dict__:
#     if isinstance(k, float) or isinstance(k, int):
#         lista.append(k)

# print(lista + dystans.operations.)


