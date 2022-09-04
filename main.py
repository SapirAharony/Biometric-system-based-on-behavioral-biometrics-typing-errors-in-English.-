import nltk, os, System_features_extractor, PyPDF2
# from docx import Document
# from tika import parser
#
# def read_text_file(source_txt_file_path) -> str:
#     """A method that reads a file .txt, .docx or .pdf and returns text as string. """
#     if os.path.isfile(source_txt_file_path) and os.path.getsize(source_txt_file_path) > 0:
#         text = ''
#         if source_txt_file_path[-3:] == 'txt':
#             with open(source_txt_file_path, encoding='utf8') as f:
#                 text = f.read()
#         elif source_txt_file_path[-4:] == 'docx':
#             doc = Document(source_txt_file_path)
#             for paragraph in doc.paragraphs:
#                 text += paragraph.text
#         elif source_txt_file_path[-3:] == 'pdf':
#             # pdf_file = open(source_txt_file_path, 'rb')
#             # pdf_reader = PyPDF2.PdfFileReader(pdf_file)
#             # for page_num in range(0, pdf_reader.numPages):
#             #     page = pdf_reader.getPage(page_num)
#             #     text += page.extractText()
#             text = parser.from_file(source_txt_file_path)['content']
#         return text

# pdf = read_text_file("C:/Users/user/Desktop/test.pdf")
# txt = read_text_file("C:/Users/user/Desktop/test.txt")
# docx = read_text_file("C:/Users/user/Desktop/test.docx")
# print('pdf', len(pdf), pdf)
# print(4 * '\n')
# print('txt', len(txt), txt)
# print(4 * '\n')
# print('docx', len(docx), docx)

tmp = ['Button.left', 'Key.shift', "'L'", "'e'", "'t'", "'s'", 'Key.space', "'c'", "'h'", "'k'", "'i'", "'f'", "'w'",
       "'o'", "'r'", "'d'", "'n'", "'.'", "'B'", "'u'", "'m'", "'a'", "'y'", "'b'", 'Key.shift_r', "'?'", "'M'",
       'Key.backspace', 'Key.esc', 'Key.f4']
import json
from System_features_extractor import add_simple_dict_to_json_file, read_json_file, add_list_to_json_file
file_json = "C:/Users/user/Desktop/destination_file.json"
# add_simple_dict_to_json_file(file_json, "All keys", json.dumps(tmp))
add_list_to_json_file(file_json, "All keys", tmp)

print(read_json_file(file_json).keys())



# print(type(parsed))
# print(parsed)
# print(parsed.keys())
# print(parsed['Sentence'][0].keys())




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


from System_features_extractor import ListOfWords

#
# sent = ListOfWords('I love playing football here.')
# sentence = nltk.pos_tag(sent.sentence_tokenizer.tokenize(sent.original_sentence))
# lista = [tag[1] for tag in nltk.pos_tag(sent.sentence_tokenizer.tokenize(sent.original_sentence))]
# print(sentence)
# print(lista)
# grammar = """NP: {<DT>?<JJ>*<NN>}"
# VP: {<>}
#           """
# cp=nltk.RegexpParser(grammar)
# result = cp.parse(sentence)
# print(result)
# result.draw()