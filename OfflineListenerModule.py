import nltk, os, System_features_extractor

class OfflineListener:
    """ A class which should be used for 'offline' files. """
    # list of ListOfWords objects
    __sentences = []
    # tokenizer to separate new context (mostly sentences)
    __sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[.;!?\n]', gaps=True)
    # tokenizer to separate words
    # __word_tokenizer = nltk.tokenize.RegexpTokenizer('[\s,:/\"]', gaps=True)
    destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"


    def read_text_file(self, path_to_file):
        if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
            text = open(path_to_file, 'r').read()
            for sentence in self.__sentence_tokenizer.tokenize(text):  # for sentence in file
                __list_of_words = System_features_extractor.ListOfWords(sentence)
                __list_of_words.is_from_file = True
                System_features_extractor.write_object_to_json_file(self.destination_json_file_path, 'Sentences',__list_of_words)
                __list_of_words.clear_list()
