# spell checker
import os
import language_tool_python
from autocorrect import Speller
import json
import nltk
import Levenshtein
import re
import textdistance
import difflib


def correct_spelling_autocorrect(sentence) -> str:
    """ A function which returns corrected spelling (by Speller from autocorrect)"""
    return Speller()(sentence)

language_tool = language_tool_python.LanguageTool('en-US')

def correct_language_tool(sentence: str) -> str:
    """ A function which returns corrected sentence (by language_tool from language_tool_python)"""
    return language_tool.correct(sentence)



class Word:
    """ A class which includes words, which are separated by NEXT_WORD_KEYS"""
    lev_threshold = 0.5

    def __init__(self, word: str, pos_tag: str, corrected_word: str = None, corrected_word_tag: str = None):
        self.original_word = word
        self.pos_tag = pos_tag
        self.corrected_word = corrected_word
        self.corrected_word_tag = corrected_word_tag
        # if corrected_word is not None:
        #     if (use_treshold and (int(Levenshtein.distance(self.original_word, self.corrected_word)) / len
        #         (self.original_word)) <= self.lev_threshold) or not use_treshold:
        #         self.distance = Distances(self.original_word, self.corrected_word)


class ListOfWords:
    """ A class which includes words, which are separated by NEXT_WORD_COMBINATION"""
    sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[^(\'\-)\w]', gaps=True)
    lev_threshold = 0.4
    __sentence_reg_pattern = r'^([ \W_]*[^A-z])'

    def __init__(self, sentence: str, add_by_left_click: bool = False, is_from_file: bool = False):
        self.original_sentence = re.sub(self.__sentence_reg_pattern, '', sentence)
        self.corrected_sentence = correct_language_tool(self.original_sentence)
        if self.original_sentence and (
                self.original_sentence[0].islower() and self.corrected_sentence[0].isupper()):
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
            # if (use_treshold and int(Levenshtein.distance(self.original_sentence, self.corrected_sentence)) / (len(
            #         self.original_sentence)) <= self.lev_threshold) or not use_treshold:
            #     self.sentence_distances = Distances(self.original_sentence, self.corrected_sentence)
        else:
            self.corrected_sentence_structure = None
            # self.sentence_distances = None
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

    @classmethod
    def initiate_from_json_file(cls):
        pass


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
