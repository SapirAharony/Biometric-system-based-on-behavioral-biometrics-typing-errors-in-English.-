# from RealTimeListenerModule import RealTimeKeyListener
#
# real_time_listener = RealTimeKeyListener()
from types import SimpleNamespace

import System_features_extractor, textblob, os, json
from spellchecker import SpellChecker
import nltk

sentence = "Sh likes playing fotball."
destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"


# print('txt' , corrected[0])
def correct_spelling_txt_blb(sentence) -> str:
    """ A function which returns corrected spelling (by TextBlob)"""
    # Making our first textblob
    return textblob.TextBlob(sentence).correct()  # Correcting the text


def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = json.load(f)
    return data

def write_word_to_json_file(path_to_file, key, main_dictionary):
    pass




list_of_words = System_features_extractor.ListOfWords(sentence)




# print("\n\n\nTRY: ", convert2serialize(list_of_words),"\n\n\n")
import dictfier

# query = [
#     'original_word',
#     'lemmatized_word',
#     'corrected_word_txt_blb',
#     'corrected_candidates_spell_chck',
#     'distances_txt_blb': {
#     'levenshtein_distance',
#     'type_of_lev_operations',
#     'damerau_levenshtein_distance'
#     },
#     'distances_spell_chck' ,
#     'pos_tag'
#
#
# ]
# std_info = dictfier.dictfy(list_of_words, query)

# print("LIST_DICT: " ,list_of_words.__dict__)
# print("WORD_DICT: " ,list_of_words.words[0].__dict__)

print(System_features_extractor.Distances('football', 'fotball'))
print(System_features_extractor.Distances('frotball', 'fotball'))
print(System_features_extractor.Distances('footbal', 'fotoball').__dict__)
