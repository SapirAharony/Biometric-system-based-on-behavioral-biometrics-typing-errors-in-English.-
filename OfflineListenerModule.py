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
    source_txt_file_path = "C:/Users/user/Desktop/destination_file.json"
    file_types = ['txt', 'pdf', 'docx']


    def read_text_file(self):
        if os.path.isfile(self.source_txt_file_path) and os.path.getsize(self.source_txt_file_path) > 0:
            text = ''
            if self.source_txt_file_path[-3:] == 'txt':
                with open(self.source_txt_file_path) as f:
                    text = f.read()
            elif self.source_txt_file_path[-4:] == 'docx':
                doc = Document(self.source_txt_file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text
            elif self.source_txt_file_path[-4:] == 'pdf':
                pdf_file = open(self.source_txt_file_path, 'rb')
                pdf_reader = PyPDF2.PdfFileReader(pdf_file)
                for page_num in range(0, pdf_reader.numPages):
                    page = pdf_reader.getPage(page_num)

                    text += page.extractText()
            for sentence in self.__sentence_tokenizer.tokenize(text):
                __list_of_words = System_features_extractor.ListOfWords(sentence)
                __list_of_words.is_from_file = True
                if __list_of_words.words:
                    System_features_extractor.write_object_to_json_file(self.destination_json_file_path, 'Sentences',
                                                            System_features_extractor.object_to_dicts(__list_of_words))
                __list_of_words.clear_list()