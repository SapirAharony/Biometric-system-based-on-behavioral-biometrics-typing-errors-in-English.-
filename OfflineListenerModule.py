import PyPDF2
import nltk, os, System_features_extractor, re
from docx import Document


class OfflineListener:
    """ A class which should be used for 'offline' files. """
    # tokenizer to separate new context (mostly sentences)
    __sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[.;!?\n]', gaps=True)
    # tokenizer to separate words
    # __word_tokenizer = nltk.tokenize.RegexpTokenizer('[\s,:/\"]', gaps=True)
    destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"
    source_txt_file = "C:/Users/user/Desktop/destination_file.json"
    file_types = ['txt', 'pdf', 'docx']


    def read_text_file(self, source_path_to_file):
        if os.path.isfile(source_path_to_file) and os.path.getsize(source_path_to_file) > 0:
            text = ''
            if source_path_to_file[-3:] == 'txt':
                with open(source_path_to_file) as f:
                    text = f.read()
            elif source_path_to_file[-4:] == 'docx':
                doc = Document(source_path_to_file)
                text = ''
                for paragraph in doc.paragraphs:
                    text += paragraph.text
            elif source_path_to_file[-4:] == 'pdf':
                pdf_file = open(source_path_to_file, 'rb')
                pdf_reader = PyPDF2.PdfFileReader(pdf_file)
                for page_num in range(0, pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)

                    text += page.extractText()
            for sentence in self.__sentence_tokenizer.tokenize(text):
                __list_of_words = System_features_extractor.ListOfWords(sentence)
                __list_of_words.is_from_file = True
                System_features_extractor.write_object_to_json_file(self.destination_json_file_path, 'Sentences',
                                                            System_features_extractor.object_to_dicts(__list_of_words))
                __list_of_words.clear_list()

