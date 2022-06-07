# from RealTimeListenerModule import RealTimeKeyListener
#
# real_time_listener = RealTimeKeyListener()
import System_features_extractor, textblob
from spellchecker import SpellChecker
import nltk

sentence = "She likes playing fotball."
# corrected = textblob.TextBlob(sentence).correct()
# print('txt' , corrected[0])
print(System_features_extractor.ListOfWords(sentence))




