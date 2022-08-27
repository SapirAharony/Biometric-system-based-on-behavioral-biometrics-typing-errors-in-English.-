import nltk, os, System_features_extractor, PyPDF2
from docx import Document
from tika import parser

def read_text_file(source_txt_file_path) -> str:
    """A method that reads a file .txt, .docx or .pdf and returns text as string. """
    if os.path.isfile(source_txt_file_path) and os.path.getsize(source_txt_file_path) > 0:
        text = ''
        if source_txt_file_path[-3:] == 'txt':
            with open(source_txt_file_path, encoding='utf8') as f:
                text = f.read()
        elif source_txt_file_path[-4:] == 'docx':
            doc = Document(source_txt_file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text
        elif source_txt_file_path[-3:] == 'pdf':
            # pdf_file = open(source_txt_file_path, 'rb')
            # pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            # for page_num in range(0, pdf_reader.numPages):
            #     page = pdf_reader.getPage(page_num)
            #     text += page.extractText()
            text = parser.from_file(source_txt_file_path)['content']
        return text

pdf = read_text_file("C:/Users/user/Desktop/test.pdf")
txt = read_text_file("C:/Users/user/Desktop/test.txt")
docx = read_text_file("C:/Users/user/Desktop/test.docx")
print('pdf', len(pdf), pdf)
print(4 * '\n')
print('txt', len(txt), txt)
print(4 * '\n')
print('docx', len(docx), docx)

tmp = {'Button.left': 10, 'Key.shift': 3, "'L'": 1, "'e'": 6, "'t'": 6, "'s'": 3, 'Key.space': 11, "'c'": 2, "'h'": 1,
       "'k'": 2, "'i'": 3, "'f'": 1, "'w'": 1, "'o'": 5, "'r'": 3, "'d'": 1, "'n'": 2, "'.'": 1, "'B'": 1, "'u'": 1,
       "'m'": 1, "'a'": 2, "'y'": 2, "'b'": 2, 'Key.shift_r': 1, "'?'": 1, "'M'": 1, 'Key.backspace': 1, 'Key.esc': 1,
       'Key.f4': 1}

# from System_features_extractor import add_simple_dict_to_json_file, grammar_check
# add_simple_dict_to_json_file( "C:/Users/user/Desktop/destination_file.json", "Keys", tmp)

# print(grammar_check('I loves my firend'))
# import nltk
#
# sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[^(\'\-)\w]', gaps=True)
# wrong_sent, correct_sent = 'I likes a lot', 'I like a lut'
# for cs, ws in zip(sentence_tokenizer.tokenize(correct_sent), sentence_tokenizer.tokenize(wrong_sent)):
#     print(cs, ws)
#
# from System_features_extractor import get_damerau_levenshtein_distance_matrix, get_string_oprations
#
# print(get_string_oprations(wrong_sent, correct_sent))
# print(len(get_string_oprations(wrong_sent, correct_sent)))

from OfflineListenerModule import OfflineListener

# luistener = OfflineListener()
#
# print(luistener.read_text_file())
