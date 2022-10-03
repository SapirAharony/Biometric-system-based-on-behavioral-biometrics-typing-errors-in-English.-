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

import system_features_extractor
import string_metrics


def unique_values(lista: list):
    tmp = []
    for k in lista:
        if k not in tmp:
            tmp.append(k)
    return tmp



ops = []

# print((System_features_extractor.read_json_file('karola_dest.json').keys()))
#
# print(System_features_extractor.read_json_file('karola_dest.json')['Sentence'][3])

from system_features_extractor import ListOfWords
import os,system_features_extractor

# def extract_data(source_file_path, dest_file_path):
#     keys = system_features_extractor.read_json_file(source_file_path).keys()
#     print(keys)
#     sentences = system_features_extractor.read_json_file(source_file_path)['Sentence']
#     for sentence in sentences:
#         if sentence is not None and isinstance(sentence['original_sentence'], str):
#             list_of_words = ListOfWords(sentence['original_sentence'], sentence['add_by_left_click'],
#                                         sentence['is_from_file'])
#             system_features_extractor.write_object_to_json_file(dest_file_path, 'Sentence',
#                                                                 system_features_extractor.object_to_dicts(list_of_words))
#     if 'Keys' in keys:
#         system_features_extractor.add_simple_dict_to_json_file(dest_file_path, 'Keys', system_features_extractor.read_json_file(source_file_path)['Keys'])
#     if 'No printable keys' in keys:
#         system_features_extractor.add_simple_dict_to_json_file(dest_file_path, 'No printable keys',
#                                                                    system_features_extractor.read_json_file(source_file_path)['No printable keys'])
#     if 'Pressed keys' in keys:
#         system_features_extractor.add_list_to_json_file(dest_file_path, 'Pressed keys',
#                                                             system_features_extractor.read_json_file(source_file_path)['Pressed keys'])
#     if 'Digraphs' in keys:
#         system_features_extractor.add_list_to_json_file(dest_file_path, 'Digraphs', system_features_extractor.read_json_file(source_file_path)['Digraphs'])


tuples = [('test', 'test'), ('my', 'my'),('rollout', 'rollout'),('piece', 'piece'),('understood', 'understood'), ('oooo', 'oooo'),('hohoh', 'haha')]
for k in tuples:
    print(system_features_extractor.Distances(k[0], k[1], is_tokenized=True).__dict__)
for k in tuples:
    print(system_features_extractor.Distances(k[0], k[1]).__dict__)

################################## exctract data from original files


source_file_dir = 'C:\\Users\\user\\Desktop\\inz_wyniki'
dest_file_dir = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files'
for file in os.listdir(source_file_dir):
    if file[-4:] == 'json':
        print(file, end=' ')
        system_features_extractor.extract_data(source_file_dir + '\\' + file, dest_file_dir + '\\done_' + file, file[:-5].capitalize())

text = 'Appare ntly it\'s kept checking up on her feeding her and so on.'
print(system_features_extractor.ListOfWords(text).__dict__)
print(system_features_extractor.ListOfWords(text).__dict__['original_sentence_structure'])
print(system_features_extractor.ListOfWords(text).__dict__['corrected_sentence_structure'])

text = 'Apparentlyit\'s kept checking up on her feeding her and so on.'
print(system_features_extractor.ListOfWords(text).__dict__)
print(system_features_extractor.ListOfWords(text).__dict__['original_sentence_structure'])
print(system_features_extractor.ListOfWords(text).__dict__)

#
# word = 'test'
# list_of_words = ['text', 'tst', 'polak', 'lkocha']
# print(difflib.get_close_matches(word, list_of_words, n=1)[0])
#
# print(system_features_extractor.Word('aa','','b').__dict__)
# print(string_metrics.Distances('aa','ba'))



