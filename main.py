# from key_listener import RealTimeKeyListener
#
# real_time_listener = RealTimeKeyListener()

# position = -100
import nltk
sentence = 'testowe: /działanie. aplikacji,'
print(sentence.split())
__word_tokenizer = nltk.tokenize.RegexpTokenizer('[\s,:/\"]', gaps=True)
print(__word_tokenizer.tokenize(sentence))
# print(sentence)
# print(sentence[:-1])
# sentence = sentence[:position-1] + sentence[position:]
# print(sentence)
# # sentence = sentence[:position] + ' ' + sentence [position+1:]
# print(sentence)
# sentence = sentence[:position-1] + sentence[position:]
# print(sentence)
# sentence = sentence[:position-1] + sentence[position:]
# print(sentence)



