from RealTimeListenerModule import RealTimeKeyListener
from OfflineListenerModule import OfflineListener
offline_lstnr = OfflineListener()
offline_lstnr.read_text_file('C:/Users/user/Desktop/tmp.txt')
# real_time_listener = RealTimeKeyListener()
from types import SimpleNamespace

import System_features_extractor, textblob, os, json
from spellchecker import SpellChecker
import nltk

sentence = "Sh likes playing fotball."
destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"
sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[\W_]', gaps=True)
def correct_spelling_txt_blb(sentence) -> str:
    """ A function which returns corrected spelling (by TextBlob)"""
    # Making our first textblob
    return textblob.TextBlob(sentence).correct()  # Correcting the text



corrected_by_txt_blb = correct_spelling_txt_blb(" ".join(
            (sentence_tokenizer.tokenize(sentence)))).split()


# print("LIST_DICT: " , corrected_by_txt_blb)
# print("WORD_DICT: " ,list_of_words.words[0].__dict__)
