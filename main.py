# import system_features_extractor, os, string_metrics
# import pandas as pd
# from json import load
# import numpy as np
# from gensim.models import Word2Vec
#
# ################################## exctract data from original files
#
# def read_json_file(path_to_file):
#     with open(path_to_file, 'r') as f:
#         data = load(f)
#     return data
#
#
# source_file_dir = 'C:\\Users\\user\\Desktop\\inz_wyniki'
# dest_file_dir = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files'
#
# # for file in os.listdir(source_file_dir):
# #     if file[-4:] == 'json':
# #         system_features_extractor.extract_data(source_file_dir + '\\' + file, dest_file_dir + '\\done_' + file, file[:-5].capitalize())
#
# # print(string_metrics.Distances('test', 'ter', is_tokenized=True).__dict__)
#
# lista = []
# print(system_features_extractor.Word('tert', 'NN','tests', 'NNP').__dict__['distance'].__dict__)
# # dystans = string_metrics.Distances('test', 'ter', is_tokenized=True)
# #
#
# vec_size = 1
# pos_tags = [
#     ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
#      'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
#      'WP', 'WP$', 'WRB']]
# pos_tag_word2vec = Word2Vec(pos_tags, min_count=1, vector_size=vec_size)
#
# for k in pos_tags[0]:
#     print(pos_tag_word2vec.wv.get_vector(k))
#
#
# # for k in string_metrics.Distances('test', 'ter',  is_tokenized=True).__dict__:
# #     if isinstance(k, float) or isinstance(k, int):
# #         lista.append(k)
#
# # print(lista + dystans.operations.)

dcitionary = {}
print(dcitionary.__str__())

import shutil
import os


# Function to create new folder if not exists
def make_new_folder(folder_name, parent_folder):
    # Path
    path = os.path.join(parent_folder, folder_name)

    # Create the folder
    # 'new_folder' in
    # parent_folder
    try:
        # mode of the folder
        mode = 0o777

        # Create folder
        os.mkdir(path, mode)
    except OSError as error:
        print(error)


# current folder path
current_folder = "C:\\Users\\user\\PycharmProjects\\bio_system\\rocs\\14"

# list of folders to be merged
list_dir = [name for name in os.listdir("C:\\Users\\user\\PycharmProjects\\bio_system\\rocs\\14") if os.path.isdir(os.path.join("C:\\Users\\user\\PycharmProjects\\bio_system\\rocs\\14", name))]
# enumerate on list_dir to get the
# content of all the folders ans store
# it in a dictionary
content_list = {}
for index, val in enumerate(list_dir):
    path = os.path.join(current_folder, val)
    content_list[list_dir[index]] = os.listdir(path)

# folder in which all the content will
# be merged
merge_folder = "merge_folder"

# merge_folder path - current_folder
# + merge_folder
merge_folder_path = os.path.join(current_folder, merge_folder)

# create merge_folder if not exists
make_new_folder(merge_folder, current_folder)

# loop through the list of folders
for sub_dir in content_list:

    # loop through the contents of the
    # list of folders
    for contents in content_list[sub_dir]:
        # make the path of the content to move
        path_to_content = sub_dir + "/" + contents

        # make the path with the current folder
        dir_to_move = os.path.join(current_folder, path_to_content)

        # move the file
        shutil.move(dir_to_move, merge_folder_path)


