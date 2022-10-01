from system_features_extractor import read_json_file
import pandas as pd
import os

file_path = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\'


def get_correct_words(file_path: str, labeled: bool = False):

    words = pd.DataFrame()
    for dictionary in read_json_file(file_path)['Sentence']:
        for word in dictionary['all_words']:
            if word['corrected_word'] is None:
                words = pd.concat([words, pd.json_normalize(word)], ignore_index=True)
    if labeled:
        words['label'] = [read_json_file(file_path)['Name'] for _ in range(words.shape[0])]
    return words


def get_misspelled_words_df_from_file(file_path: str, labeled: bool = False):
    misspelled = pd.DataFrame()
    for dictionary in read_json_file(file_path)['Sentence']:
        if 'misspelled_words' in dictionary.keys() and dictionary['misspelled_words']:
            for k in dictionary['misspelled_words']:
                if 'distance' in k.keys() and k['distance']['operations']:
                    k['distance']['operations'] = k['distance']['operations'][0]['ml_repr']
                misspelled = pd.concat([misspelled, pd.json_normalize(k)], ignore_index=True)
    if labeled:
        misspelled['label'] = [read_json_file(file_path)['Name'] for _ in range(misspelled.shape[0])]
    return misspelled



data = pd.DataFrame()
for file in os.listdir(file_path):
    data = pd.concat([data, get_misspelled_words_df_from_file(file_path + '\\' + file, labeled=True)],
                         ignore_index=True)


print(data.info())
print(data[['original_word', 'pos_tag', 'distance.damerau_levenshtein_distance','distance.operations']])