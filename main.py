# from RealTimeListenerModule import RealTimeKeyListener
#
# real_time_listener = RealTimeKeyListener()
import System_features_extractor, textblob
from spellchecker import SpellChecker
import nltk

sentence = "Sh likes playing fotball."
# print('txt' , corrected[0])
def correct_spelling_txt_blb(sentence) -> str:
    """ A function which returns corrected spelling (by TextBlob)"""
    # Making our first textblob
    return textblob.TextBlob(sentence).correct()  # Correcting the text
# print(System_features_extractor.ListOfWords(sentence))

print(System_features_extractor.Distances('football', 'fotball'))
print(System_features_extractor.Distances('frotball', 'fotball'))
print(System_features_extractor.Distances('footbal', 'fotoball').__dict__)




