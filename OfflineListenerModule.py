import nltk

class OfflineListener:
    """ A class which should be used for 'offline' files. """
    # list of ListOfWords objects
    __sentences = []
    # tokenizer to separate new context (mostly sentences)
    __sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[.;!?\n]', gaps=True)
    # tokenizer to separate words
    __word_tokenizer = nltk.tokenize.RegexpTokenizer('[\s,:/\"]', gaps=True)

    # def read_text_file(self, path_to_file):
    #     if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
    #         text = open(path_to_file, 'r').read()
    #         for sentence in self.__sentence_tokenizer.tokenize(text):  # for sentence in file
    #             __list_of_words = s_f_extractor.ListOfWords(self.__word_tokenizer.tokenize(sentence))
    #             __list_of_words.is_from_file = True
    #             self.__sentences.append(__list_of_words)
