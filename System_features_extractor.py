# spell checker
import os
import language_tool_python
from autocorrect import Speller
import json
import nltk
import Levenshtein
import re

language_tool = language_tool_python.LanguageTool('en-US')


def get_damerau_levenshtein_distance_matrix(word_1: str, word_2: str, is_damerau: bool = False):
    distance_matrix = [[0 for _ in range(len(word_2) + 1)] for _ in range(len(word_1) + 1)]
    for i in range(len(word_1) + 1):
        distance_matrix[i][0] = i
    for j in range(len(word_2) + 1):
        distance_matrix[0][j] = j
    for i in range(len(word_1)):
        for j in range(len(word_2)):
            if word_1[i] == word_2[j]:
                cost = 0
            else:
                cost = 1
            distance_matrix[i + 1][j + 1] = min(distance_matrix[i][j + 1] + 1,  # insert
                                                distance_matrix[i + 1][j] + 1,  # delete
                                                distance_matrix[i][j] + cost)  # replace
            if is_damerau:
                if i and j and word_1[i] == word_2[j - 1] and word_1[i - 1] == word_2[j]:
                    distance_matrix[i + 1][j + 1] = min(distance_matrix[i + 1][j + 1],
                                                        distance_matrix[i - 1][j - 1] + cost)  # transpose
    return distance_matrix


def get_string_oprations(word_1, word_2, is_damerau=True):
    dist_matrix = get_damerau_levenshtein_distance_matrix(word_1, word_2, is_damerau=is_damerau)
    i, j = len(dist_matrix), len(dist_matrix[0])
    i -= 1
    j -= 1
    operations_list = []
    while i != -1 and j != -1:
        if is_damerau and i > 1 and j > 1 and word_1[i - 1] == word_2[j - 2] and word_1[i - 2] \
                == word_2[j - 1]:
            if dist_matrix[i - 2][j - 2] < dist_matrix[i][j]:
                operations_list.insert(0, ('transpose', i - 1, i - 2))
                i -= 2
                j -= 2
                continue
        tmp = [dist_matrix[i - 1][j - 1], dist_matrix[i][j - 1], dist_matrix[i - 1][j]]
        index = tmp.index(min(tmp))
        if index == 0:
            if dist_matrix[i][j] > dist_matrix[i - 1][j - 1]:
                operations_list.insert(0, ('replace', i - 1, j - 1))
            i -= 1
            j -= 1
        elif index == 1:
            operations_list.insert(0, ('insert', i - 1, j - 1))
            j -= 1
        elif index == 2:
            operations_list.insert(0, ('delete', i - 1, i - 1))
            i -= 1
    return operations_list


class Distances:
    levenshtein_distance = 0
    type_of_d_l_operations = None
    damerau_levenshtein_distance = 0
    __word_1 = None
    __word_2 = None

    def __init__(self, str_1, str_2):
        self.__word_1 = str_1
        self.__word_2 = str_2
        self.damerau_levenshtein_distance = len(get_string_oprations(self.__word_1, self.__word_2, is_damerau=True))
        self.type_of_d_l_operations = {"insert": 0,
                                       "replace": 0,
                                       "delete": 0,
                                       "transpose": 0}
        self.levenshtein_distance = Levenshtein.distance(self.__word_1, self.__word_2)
        self.jaro_winkler = Levenshtein.jaro_winkler(self.__word_1, self.__word_2)
        self.set_operations()

    def set_operations(self, is_damerau=True):
        for k in get_string_oprations(self.__word_1, self.__word_2, is_damerau):
            self.type_of_d_l_operations[k[0]] += 1


def correct_spelling_autocorrect(sentence) -> str:
    """ A function which returns corrected spelling (by Speller from autocorrect)"""
    return Speller()(sentence)


def correct_language_tool(sentence: str) -> str:
    """ A function which returns corrected sentence (by language_tool from language_tool_python)"""
    return language_tool.correct(sentence)


class Word:
    """ A class which includes words, which are separated by NEXT_WORD_KEYS"""
    lev_threshold = 0.5

    def __init__(self, word: str, pos_tag: str, corrected_word: str = None, corrected_word_tag: str = None,
                 use_treshold: bool = True):
        self.original_word = word
        self.pos_tag = pos_tag
        self.corrected_word = corrected_word
        self.corrected_word_tag = corrected_word_tag
        if corrected_word is not None:
            if (use_treshold and (int(Levenshtein.distance(self.original_word, self.corrected_word)) / len
                (self.original_word)) <= self.lev_threshold) or not use_treshold:
                self.distance = Distances(self.original_word, self.corrected_word)


class ListOfWords:
    """ A class which includes words, which are separated by NEXT_WORD_COMBINATION"""
    sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[^(\'\-)\w]', gaps=True)
    add_by_left_click = None
    is_from_file = None
    lev_threshold = 0.35
    __sentence_reg_pattern = r'^([ \W_]+[^A-z])'

    def __init__(self, sentence: str, add_by_left_click: bool = False, is_from_file: bool = False,
                 use_treshold: bool = True):
        self.original_sentence = re.sub(self.__sentence_reg_pattern, '', sentence)
        self.corrected_sentence = correct_language_tool(self.original_sentence)
        if self.original_sentence and (self.original_sentence[0].islower() and self.corrected_sentence[0].isupper()):
            self.original_sentence = self.original_sentence[:1].upper() + self.original_sentence[1:]
        self.all_words = []
        self.misspelled_words = []
        self.original_sentence_structure = [tag[1] for tag in nltk.pos_tag(
            self.sentence_tokenizer.tokenize(self.original_sentence))]
        if self.original_sentence != self.corrected_sentence:
            self.corrected_sentence_structure = [tag[1] for tag in nltk.pos_tag(
                self.sentence_tokenizer.tokenize(self.corrected_sentence))]
            if self.corrected_sentence_structure == self.original_sentence_structure:
                self.corrected_sentence_structure = None
            if (use_treshold and int(Levenshtein.distance(self.original_sentence, self.corrected_sentence)) / (len(
                    self.original_sentence)) <= self.lev_threshold) or not use_treshold:
                self.sentence_distances = Distances(self.original_sentence, self.corrected_sentence)
        else:
            self.corrected_sentence_structure = None
            self.sentence_distances = None
        i = 0
        for original_word, correct_word in zip(self.sentence_tokenizer.tokenize(self.original_sentence),
                                               self.sentence_tokenizer.tokenize(self.corrected_sentence)):
            tag = self.original_sentence_structure[i]
            if original_word != correct_word:
                if self.corrected_sentence_structure is not None:
                    correct_word_tag = self.corrected_sentence_structure[i]
                else:
                    correct_word_tag = None
                self.all_words.append(Word(original_word, tag, correct_word, correct_word_tag))
                self.misspelled_words.append(Word(original_word, tag, correct_word, correct_word_tag))
            else:
                self.all_words.append(Word(original_word, tag))
            i += 1
        self.add_by_left_click = add_by_left_click
        self.is_from_file = is_from_file

    def set_left_click(self):
        self.add_by_left_click = True

    def clear_list(self):
        self.add_by_left_click = False
        self.all_words.clear()
        self.misspelled_words.clear()
        self.original_sentence = None
        self.corrected_sentence = None
        self.is_from_file = None


def get_freq_word(list_of_words: ListOfWords, freq_dict: dict):
    if isinstance(list_of_words, ListOfWords) and isinstance(freq_dict, dict):
        for word_istance in list_of_words.all_words:
            if word_istance.original_word not in freq_dict.keys():
                freq_dict[word_istance.original_word] = 1
            else:
                freq_dict[word_istance.original_word] += 1
    return freq_dict


def object_to_dicts(objct):
    if isinstance(objct, dict):
        return {k: object_to_dicts(v) for k, v in objct.items()}
    elif not isinstance(objct, str) and hasattr(objct, "__iter__"):
        return [object_to_dicts(v) for v in objct]
    elif hasattr(objct, "_ast"):
        return object_to_dicts(objct._ast())
    elif hasattr(objct, "__dict__"):
        return {
            key: object_to_dicts(value)
            for key, value in objct.__dict__.items()
            if not callable(value) and not key.startswith('_')
        }
    else:
        return objct


def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = json.load(f)
    return data


def write_object_to_json_file(path_to_file: str, key: str, main_dictionary: dict):
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        # open file
        data = read_json_file(path_to_file)
        # clear file
        open(path_to_file, 'w').close()
        # add data
        file = open(path_to_file, 'a+')
        data[key].append(main_dictionary)
        file.seek(0)
        json.dump(data, file, indent=4)
    else:
        file = open(path_to_file, 'w+')
        tmp = {key: [main_dictionary]}
        json.dump(tmp, file, indent=4)
    file.close()


def add_simple_dict_to_json_file(path_to_file: str, key: str, dict_obj: dict):
    # check if is empty
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        data = read_json_file(path_to_file)
        open(path_to_file, 'w').close()
        file = open(path_to_file, 'a+')
        if isinstance(dict_obj, dict) and dict_obj.keys():
            if key in data.keys():
                for k in dict_obj.keys():
                    if k not in data[key].keys():
                        data[key][k] = dict_obj[k]
                    elif k in data[key].keys() and (isinstance(data[key][k], int) or isinstance(data[key][k], float)):
                        data[key][k] += dict_obj[k]
            else:
                data[key] = dict_obj
        json.dump(data, file, indent=4)
        file.close()


def add_list_to_json_file(path_to_file: str, key: str, list_obj: list):
    # check if is empty
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        data = read_json_file(path_to_file)
        open(path_to_file, 'w').close()
        file = open(path_to_file, 'a+')
        if key in data.keys() and isinstance(data[key], list):
            data[key].extend(list_obj)
        elif key not in data.keys():
            data[key] = list_obj
        json.dump(data, file, indent=4)
        file.close()

