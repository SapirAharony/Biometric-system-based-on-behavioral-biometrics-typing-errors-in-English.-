import nltk, os, System_features_extractor, re

class OfflineListener:
    """ A class which should be used for 'offline' files. """
    # list of ListOfWords objects
    __sentences = []
    # tokenizer to separate new context (mostly sentences)
    __sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[.;!?\n]', gaps=True)
    # tokenizer to separate words
    # __word_tokenizer = nltk.tokenize.RegexpTokenizer('[\s,:/\"]', gaps=True)
    destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"
    # source_txt_file = "C:/Users/user/Desktop/destination_file.json"


    def read_text_file(self, source_path_to_file):
        if os.path.isfile(source_path_to_file) and os.path.getsize(source_path_to_file) > 0:
            with open(source_path_to_file) as f:
                text = f.read()
            for sentence in self.__sentence_tokenizer.tokenize(text):
                __list_of_words = System_features_extractor.ListOfWords(sentence)
                __list_of_words.is_from_file = True
                print(sentence)
                System_features_extractor.write_object_to_json_file(self.destination_json_file_path, 'Sentences',
                                                            System_features_extractor.object_to_dicts(__list_of_words))
                __list_of_words.clear_list()

        # for sentence in text:  # for sentence in file
            #     __list_of_words = System_features_extractor.ListOfWords(sentence)
            #     __list_of_words.is_from_file = True
            #     print(sentence)
            #     System_features_extractor.write_object_to_json_file(self.destination_json_file_path, 'Sentences',__list_of_words)
            #     __list_of_words.clear_list()
