# from RealTimeListenerModule import RealTimeKeyListener
#
# real_time_listener = RealTimeKeyListener()
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


def write_list_of_words_to_json_file(path_to_file, key, dictionary):
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        # open file
        data = read_json_file(path_to_file)
        print('data', data, type(data))
        # clear file
        open(path_to_file, 'w').close()
        # add data
        file = open(path_to_file, 'a+')
        data[key].append(dictionary)
        file.seek(0)
        json.dump(data, file, indent=4)
    else:
        file = open(path_to_file, 'w+')
        tmp = {key: [dictionary]}
        json.dump(tmp, file, indent=4)
    file.close()


list_of_words = System_features_extractor.ListOfWords(sentence)
print(list_of_words)
print(System_features_extractor.Distances('football', 'fotball'))
print(System_features_extractor.Distances('frotball', 'fotball'))
print(System_features_extractor.Distances('footbal', 'fotoball').__dict__)
