# import re
#
# import nltk, os, System_features_extractor, PyPDF2
#
# # from docx import Document
# # from tika import parser
# #
# # def read_text_file(source_txt_file_path) -> str:
# #     """A method that reads a file .txt, .docx or .pdf and returns text as string. """
# #     if os.path.isfile(source_txt_file_path) and os.path.getsize(source_txt_file_path) > 0:
# #         text = ''
# #         if source_txt_file_path[-3:] == 'txt':
# #             with open(source_txt_file_path, encoding='utf8') as f:
# #                 text = f.read()
# #         elif source_txt_file_path[-4:] == 'docx':
# #             doc = Document(source_txt_file_path)
# #             for paragraph in doc.paragraphs:
# #                 text += paragraph.text
# #         elif source_txt_file_path[-3:] == 'pdf':
# #             # pdf_file = open(source_txt_file_path, 'rb')
# #             # pdf_reader = PyPDF2.PdfFileReader(pdf_file)
# #             # for page_num in range(0, pdf_reader.numPages):
# #             #     page = pdf_reader.getPage(page_num)
# #             #     text += page.extractText()
# #             text = parser.from_file(source_txt_file_path)['content']
# #         return text
#
# # pdf = read_text_file("C:/Users/user/Desktop/test.pdf")
# # txt = read_text_file("C:/Users/user/Desktop/test.txt")
# # docx = read_text_file("C:/Users/user/Desktop/test.docx")
# # print('pdf', len(pdf), pdf)
# # print(4 * '\n')
# # print('txt', len(txt), txt)
# # print(4 * '\n')
# # print('docx', len(docx), docx)
#
# tmp = ['Button.left', 'Key.shift', "'L'", "'e'", "'t'", "'s'", 'Key.space', "'c'", "'h'", "'k'", "'i'", "'f'", "'w'",
#        "'o'", "'r'", "'d'", "'n'", "'.'", "'B'", "'u'", "'m'", "'a'", "'y'", "'b'", 'Key.shift_r', "'?'", "'M'",
#        'Key.backspace', 'Key.esc', 'Key.f4']
# import json
# from System_features_extractor import add_simple_dict_to_json_file, read_json_file, add_list_to_json_file
#
# file_json = "C:/Users/user/Desktop/destination_file.json"
# # add_simple_dict_to_json_file(file_json, "All keys", json.dumps(tmp))
# add_list_to_json_file(file_json, "All keys", tmp)
#
# # print(grammar_check('I loves my firend'))
# # import nltk
# #
# # sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[^(\'\-)\w]', gaps=True)
# # wrong_sent, correct_sent = 'I likes a lot', 'I like a lut'
# # for cs, ws in zip(sentence_tokenizer.tokenize(correct_sent), sentence_tokenizer.tokenize(wrong_sent)):
# #     print(cs, ws)
# #
# # from System_features_extractor import get_damerau_levenshtein_distance_matrix, get_string_oprations
# #
# # print(get_string_oprations(wrong_sent, correct_sent))
# # print(len(get_string_oprations(wrong_sent, correct_sent)))
#
# from OfflineListenerModule import OfflineListener
#
# # luistener = OfflineListener()
# #
# # print(luistener.read_text_file())
#
#
# from System_features_extractor import ListOfWords
#
# #
# # sent = ListOfWords('I love playing football here.')
# # sentence = nltk.pos_tag(sent.sentence_tokenizer.tokenize(sent.original_sentence))
# # lista = [tag[1] for tag in nltk.pos_tag(sent.sentence_tokenizer.tokenize(sent.original_sentence))]
# # print(sentence)
# # print(lista)
# # grammar = """NP: {<DT>?<JJ>*<NN>}"
# # VP: {<>}
# #           """
# # cp=nltk.RegexpParser(grammar)
# # result = cp.parse(sentence)
# # print(result)
# # result.draw()
#
#
# import difflib, Levenshtein, textdistance, nltk
#
# first_string = "Tet setence showing distance"
# second_string = "Test sentence showing distance"
# s = difflib.SequenceMatcher(isjunk=None, a=first_string, b=second_string)
# print(s.ratio(), Levenshtein.ratio(first_string, second_string), Levenshtein.jaro_winkler(first_string, second_string))
#
# print(s.get_opcodes())
# print(s.get_matching_blocks())
#
#
#
# print(3 * '\n')
# print('edit based')
# print('hamming: ', textdistance.hamming(first_string, second_string))
# print('d_l: ', textdistance.damerau_levenshtein(first_string, second_string))
# print('lev: ', textdistance.levenshtein(first_string, second_string))
# print('mlipns: ', textdistance.mlipns(first_string, second_string))
# print('strcmp95: ', textdistance.strcmp95(first_string, second_string))
# print('needleman_wunsch: ', textdistance.needleman_wunsch(first_string, second_string))
# print('gotoh: ', textdistance.gotoh(first_string, second_string))
# print('smith_waterman: ', textdistance.smith_waterman(first_string, second_string))
#
# print(3 * '\n')
# print('Sequence based')
# print('lcsseq: ', textdistance.lcsseq(first_string, second_string))
# print('lcsstr: ', textdistance.lcsstr(first_string, second_string))
# print('ratcliff_obershelp: ', textdistance.ratcliff_obershelp(first_string, second_string))
#
# print(3 * '\n')
# print('phonetic based')
# print('mra: ', textdistance.mra(first_string, second_string))
# print('editex: ', textdistance.editex(first_string, second_string))
#
# print(3 * '\n')
# print('simple based')
# print('prefix: ', textdistance.prefix(first_string, second_string))
# print('postfix: ', textdistance.postfix(first_string, second_string))
# print('length: ', textdistance.length(first_string, second_string))
# print('identity: ', textdistance.identity(first_string, second_string))
# print('matrix: ', textdistance.matrix(first_string, second_string))
#
# sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[^(\'\-)\w]', gaps=True)
# first_string_tokenized = sentence_tokenizer.tokenize(first_string)
# second_string_tokenized = sentence_tokenizer.tokenize(second_string)
#
# print(3 * '\n')
# print('token based')
# print('jaccard: ', textdistance.jaccard(first_string, second_string),
#       textdistance.jaccard(first_string_tokenized, second_string_tokenized))
# print('sorensen_dice: ', textdistance.sorensen_dice(first_string, second_string),
#       textdistance.sorensen_dice(first_string_tokenized, second_string_tokenized))
# print('tversky: ', textdistance.tversky(first_string, second_string),
#       textdistance.tversky(first_string_tokenized, second_string_tokenized))
# print('overlap: ', textdistance.overlap(first_string, second_string),
#       textdistance.overlap(first_string_tokenized, second_string_tokenized))
# print('tanimoto: ', textdistance.tanimoto(first_string, second_string),
#       textdistance.tanimoto(first_string_tokenized, second_string_tokenized))
# print('cosine: ', textdistance.cosine(first_string, second_string),
#       textdistance.cosine(first_string_tokenized, second_string_tokenized))
# print('monge_elkan: ', textdistance.monge_elkan(first_string, second_string),
#       textdistance.monge_elkan(first_string_tokenized, second_string_tokenized))
# print('bag: ', textdistance.bag(first_string, second_string),
#       textdistance.bag(first_string_tokenized, second_string_tokenized))
import textdistance, difflib

import System_features_extractor


def unique_values(lista: list):
    tmp = []
    for k in lista:
        if k not in tmp:
            tmp.append(k)
    return tmp


# print(textdistance.ratcliff_obershelp('bartek', 'bratek'))
# print(textdistance.ratcliff_obershelp('test bartek', 'bartek test'))
# print(textdistance.ratcliff_obershelp('bartek', 'Bartek'))
# print(textdistance.lcsseq('bartek', 'bratek'))
# print(textdistance.lcsseq('test bartek', 'bartek test'))
# print(textdistance.sorensen_dice('bartek a lo', 'batek'))
# print(textdistance.sorensen_dice('bartek', 'Bartek'))
# print(textdistance.jaccard('bartek', 'batek'))
# print(textdistance.jaccard('bartek', 'Bartek'))
# word_1, word_2 = 'bartek', 'brtek'
# operations = System_features_extractor.get_string_oprations(word_1, word_2)
#
# word_1, word_2 = 'bartek', 'bratek'
# operations = System_features_extractor.get_string_oprations(word_1, word_2)
#
#
# word_1, word_2 = 'brtek', 'bartek'
# operations = System_features_extractor.get_string_oprations(word_1, word_2)


word_1, word_2 = 'rarrer', 'bartek'
operations = System_features_extractor.get_string_oprations(word_1, word_2)
print(operations)

ops = []



