# from RealTimeListenerModule import RealTimeKeyListener
#
# real_time_listener = RealTimeKeyListener()
import System_features_extractor, textblob
from spellchecker import SpellChecker
import nltk

sentence = "She likes playing fotball."
# corrected = textblob.TextBlob(sentence).correct()
# print('txt' , corrected[0])
def correct_spelling_txt_blb(sentence) -> str:
    """ A function which returns corrected spelling (by TextBlob)"""
    # Making our first textblob
    return textblob.TextBlob(sentence).correct()  # Correcting the text
print(System_features_extractor.ListOfWords(sentence))
sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[\W_]', gaps=True)
print(sentence_tokenizer.tokenize(correct_spelling_txt_blb(sentence).stripped))




