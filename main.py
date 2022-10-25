import system_features_extractor, os
from json import load


################################## exctract data from original files

def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = load(f)
    return data


# source_file_dir = 'C:\\Users\\user\\Desktop\\inz_wyniki'
# dest_file_dir = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files'
#
# for file in os.listdir(source_file_dir):
#     if file[-4:] == 'json':
#         system_features_extractor.extract_data(source_file_dir + '\\' + file, dest_file_dir + '\\done_' + file, file[:-5].capitalize())
#
# for file in os.listdir(dest_file_dir):
#     system_features_extractor.add_str_to_json(dest_file_dir+'\\'+file, file[5:-5].capitalize())

import random
import keras.utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import os
from json import load
import numpy as np
from itertools import combinations

# assign ids to pos_tags
pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
            'POS',
            'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
            'WP', 'WP$', 'WRB']
tmp = {}
i = 0
for tag in pos_tags:
    tmp[tag] = i
    i += 1
pos_tags = tmp
del tmp
directory = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\'

def load_data(files_directory: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for file in os.listdir(files_directory):
        df = pd.concat([df, get_misspelled_words_df_from_json(files_directory + '\\' + file, labeled=True)],
                       ignore_index=True)

    # drop None (clear data)
    df = df.dropna().reset_index()
    del df['index']
    return df
user_names = {}
def get_misspelled_words_df_from_json(file_path: str, labeled: bool = True, use_tags: bool = True):
    cols = [
        # edit ops
        'damerau_levenshtein_distance',
        'jaro_winkler_ns',
        # # # # token based
        'gestalt_ns',
        'sorensen_dice_ns',
        'overlap',
        # # phonetic
        'mra_ns',
        # # # seq based
        'lcsstr',
        'ml_type_id',
        'ml_operation_subtype_id',
        'ml_det0',
        'ml_det1',
        'ml_det2',
        'ml_det3',
        'ml_det4',
        'ml_det5'
    ]
    global user_names
    misspelled = pd.DataFrame(columns=cols)
    for dictionary in read_json_file(file_path)['Sentence']:
        distances_ml = []
        if 'misspelled_words' in dictionary.keys() and len(dictionary['misspelled_words']) > 0:
            id = 0
            for misspell in dictionary['misspelled_words']:
                if use_tags:
                    tags = [pos_tags[misspell['pos_tag']], pos_tags[misspell['corrected_word_tag']]]
                if 'distance' in misspell.keys() and 'operations' in misspell['distance'].keys():
                    id += 1
                    for dist in cols:
                        if dist in misspell['distance'].keys() and id < 2:
                            if isinstance(misspell['distance'][dist], float) or isinstance(dist, int):
                                distances_ml.append(misspell['distance'][dist])
                    for op in misspell['distance']['operations']:

                        tmp = [float(k) for k in op['ml_repr']]
                        if not use_tags:
                            misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp], columns=cols)],
                                                   ignore_index=True)
                        else:
                            misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp + tags],
                                                                             columns=cols + ['pos_tag_org',
                                                                                             'pos_tag_corrected'])],
                                                   ignore_index=True)
    if labeled:
        name = read_json_file(file_path)['Name']
        if not user_names:
            user_names[name] = 0
        else:
            if name not in user_names.keys():
                user_names[name] = max(user_names.values()) + 1
        misspelled['user_label'] = [user_names[name] for _ in range(misspelled.shape[0])]
    return misspelled


df = load_data(directory)
print(df)








