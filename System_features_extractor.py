# spell checker
import os

from spellchecker import SpellChecker
from autocorrect import Speller
import textblob
import json
import nltk
from nltk import collections
import Levenshtein
from gingerit.gingerit import GingerIt


class Distances:
    levenshtein_distance = 0
    type_of_lev_operations = None
    damerau_levenshtein_distance = 0
    __word_1 = None
    __word_2 = None

    def __init__(self, str_1, str_2):
        self.__word_1 = str_1
        self.__word_2 = str_2
        self.type_of_lev_operations = {"insert": 0,
                                       "replace": 0,
                                       "delete": 0,
                                       "transpose": 0}
        self.levenshtein_distance = Levenshtein.distance(str_1, str_2)
        self.__set_operations()
        self.damerau_levenshtein_distance = len(self.get_string_oprations())

    def __str__(self):
        return '1. ' + self.__word_1 + '\t2. ' + self.__word_2 + '\n\t-D-L distance: ' + str(
            self.damerau_levenshtein_distance) + '\n\t-L distance: ' + str(
            self.levenshtein_distance) + "\n\t-Operations:" + str(self.type_of_lev_operations)

    def __set_operations(self, is_damerau=True):
        for k in self.get_string_oprations(is_damerau):
            self.type_of_lev_operations[k[0]] += 1

    def get_string_oprations(self, is_damerau=True):
        dist_matrix = self.__get_damerau_levenshtein_distance_matrix(is_damerau=is_damerau)
        i, j = len(dist_matrix), len(dist_matrix[0])
        i -= 1
        j -= 1
        operations_list = []
        while i != -1 and j != -1:
            if is_damerau and i > 1 and j > 1 and self.__word_1[i - 1] == self.__word_2[j - 2] and self.__word_1[i - 2] \
                    == self.__word_2[j - 1]:
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

    def __get_damerau_levenshtein_distance_matrix(self, is_damerau=False):
        distance_matrix = [[0 for _ in range(len(self.__word_2) + 1)] for _ in range(len(self.__word_1) + 1)]
        for i in range(len(self.__word_1) + 1):
            distance_matrix[i][0] = i
        for j in range(len(self.__word_2) + 1):
            distance_matrix[0][j] = j
        for i in range(len(self.__word_1)):
            for j in range(len(self.__word_2)):
                if self.__word_1[i] == self.__word_2[j]:
                    cost = 0
                else:
                    cost = 1
                distance_matrix[i + 1][j + 1] = min(distance_matrix[i][j + 1] + 1,  # insert
                                                    distance_matrix[i + 1][j] + 1,  # delete
                                                    distance_matrix[i][j] + cost)  # replace
                if is_damerau:
                    if i and j and self.__word_1[i] == self.__word_2[j - 1] and self.__word_1[i - 1] == self.__word_2[j]:
                        distance_matrix[i + 1][j + 1] = min(distance_matrix[i + 1][j + 1],
                                                            distance_matrix[i - 1][j - 1] + cost)  # transpose
        return distance_matrix


def correct_spelling_spell_checker(word_or_list_of_words):
    """ A function which returns corrected spelling (by SpellChecker)"""
    # return SpellChecker().correction(word_or_list_of_words.lower())
    if isinstance(word_or_list_of_words, str) and ' ' not in word_or_list_of_words and len(word_or_list_of_words) > 1:
        return str(SpellChecker().correction(word_or_list_of_words.lower()))
    elif isinstance(word_or_list_of_words, list):
        return [SpellChecker().correction(word.lower()) for word in
                list(filter(lambda x: len(x) > 1, word_or_list_of_words))]


def candidates_to_correct_spelling_spell_checker(word_or_list_of_words):
    """ A class which returns canditates spelling (by SpellChecker)"""
    # return SpellChecker().candidates(word_or_list_of_words.lower())
    if isinstance(word_or_list_of_words, str) and ' ' not in word_or_list_of_words and len(word_or_list_of_words) > 1:
        return SpellChecker().candidates(word_or_list_of_words.lower())
    elif isinstance(word_or_list_of_words, list):
        return [SpellChecker().candidates(word.lower()) for word in
                list(filter(lambda x: len(x) > 1, word_or_list_of_words))]


def correct_spelling_autocorrect(sentence) -> str:
    """ A function which returns corrected spelling (by Speller from autocorrect)"""
    return Speller()(sentence)


def correct_spelling_txt_blb(sentence) -> str:
    """ A function which returns corrected spelling (by TextBlob)"""
    return textblob.TextBlob(sentence).correct()


def get_pos_for_word(word: str) -> str:
    """ Method that returns a POS tag for lemmatization """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)


def lemmatize_word(word: str) -> str:
    return nltk.stem.WordNetLemmatizer().lemmatize(word, get_pos_for_word(word))


def grammar_check (sentence: str, justResult=True):
    if justResult:
        return str(GingerIt().parse(sentence)['result'])
    else:
        return GingerIt().parse(sentence)


class Word:
    original_word = ''
    lemmatized_word = ''
    corrected_word_txt_blb = ''
    corrected_word_spell_chck = ''
    corrected_word_autocorrect = ''
    distances_txt_blb = None
    distances_spell_chck = None
    distances_autocorrect = None
    pos_tag = ''

    def __init__(self, word):
        self.original_word = word
        self.lemmatized_word = lemmatize_word(word)

    def __str__(self):
        tmp = ''
        if self.lemmatized_word is not None:
            tmp += "\n\tLemmatized word: " + self.lemmatized_word + ' '
        if self.corrected_word_spell_chck:
            tmp += "\n\tCorrected word by SpellChecker: " + self.corrected_word_spell_chck + ',\n\t-distance: ' + str(
                self.distances_spell_chck)
        if self.corrected_word_txt_blb:
            tmp += "\n\tCorrected word by TextBlob: " + self.corrected_word_txt_blb + ',\n\t-distance: ' + str(
                self.distances_txt_blb)
        if tmp:
            return 'Word: ' + self.original_word + "\n\tpos tag: " + self.pos_tag + tmp
        else:
            return 'Word: ' + self.original_word + "\n\tpos tag: " + self.pos_tag


class ListOfWords:
    """ A class which includes words, which are separated by NEXT_WORD_COMBINATION"""
    sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[^(\'\-)\w]', gaps=True)
    __add_by_left_click = False
    __original_sentence = None
    __grammarly_corrected_sentence = None
    __is_from_file = None
    words = None
    pos_tags_counter = None
    # misspelled_words_txt_blb = None
    # misspelled_words_spell_chckr = None

    def __init__(self, sentence, add_by_left_click=False, is_from_file=False, ):
        self.words = []
        # self.misspelled_words_txt_blb = []
        # self.misspelled_words_spell_chckr = []
        # self.misspelled_words_autocorrect=[]
        self.__original_sentence = sentence
        # corrected_by_txt_blb = correct_spelling_txt_blb(" ".join(self.sentence_tokenizer.tokenize(sentence))).split()
        corrected_by_txt_blb = self.sentence_tokenizer.tokenize(str(correct_spelling_txt_blb(sentence.lower())))
        corrected_by_autocorrect = self.sentence_tokenizer.tokenize(correct_spelling_autocorrect(sentence.lower()))
        i = 0
        for word in self.sentence_tokenizer.tokenize(sentence):
            self.words.append(Word(word))
            self.words[len(self.words) - 1].pos_tag = \
                nltk.pos_tag(self.sentence_tokenizer.tokenize(sentence.lower()))[i][1]
            if correct_spelling_spell_checker(word) != word.lower() and correct_spelling_spell_checker(word):
                self.words[i].corrected_word_spell_chck = correct_spelling_spell_checker(word)
                self.words[i].distances_spell_chck = Distances(correct_spelling_spell_checker(word), word)
                # self.misspelled_words_spell_chckr.append(self.words[i])
            if corrected_by_txt_blb[i] != word.lower():
                self.words[i].corrected_word_txt_blb = corrected_by_txt_blb[i]
                self.words[i].distances_txt_blb = Distances(corrected_by_txt_blb[i], word)
                # self.misspelled_words_txt_blb.append(self.words[i])
            if corrected_by_autocorrect[i] != word.lower():
                self.words[i].corrected_word_autocorrect = corrected_by_autocorrect[i]
                self.words[i].distances_autocorrect = Distances(corrected_by_autocorrect[i], word)
                # self.misspelled_words_autocorrect.append(self.words[i])

            i += 1
        self.pos_tags_counter = collections.Counter(
            tag for word, tag in (nltk.pos_tag(self.sentence_tokenizer.tokenize(sentence.lower()))))
        self.__add_by_left_click = add_by_left_click
        self.__is_from_file = is_from_file

    def __str__(self):
        print("Words: ")
        for word in self.words:
            print("-", word)
        return '\n- Add by left click: ' + str(
            self.__add_by_left_click) + '\n- Is from file: ' + str(self.__is_from_file) + '\n- POS Tags counter' + str(
            self.pos_tags_counter)

    def set_left_click(self):
        self.__add_by_left_click = True

    def get_click(self):
        return self.__add_by_left_click

    def clear_list(self):
        self.__add_by_left_click = False
        self.words.clear()
        self.__original_sentence = None
        self.__is_from_file = None
        self.pos_tags_counter = None
        # self.misspelled_words_spell_chckr = None
        # self.misspelled_words_txt_blb = None


def get_freq_word(list_of_words: ListOfWords, freq_dict: dict):
    if isinstance(list_of_words, ListOfWords) and isinstance(freq_dict, dict):
        for word_istance in list_of_words.words:
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


def write_object_to_json_file(path_to_file, key, main_dictionary):
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


def add_simple_dict_to_json_file(path_to_file, key, dict_obj):
    # check if is empty
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        data = read_json_file(path_to_file)
        open(path_to_file, 'w').close()
        file = open(path_to_file, 'a+')
        if isinstance(dict_obj, dict) and dict_obj.keys() and key in data.keys():
            for k in dict_obj.keys():
                if k not in data[key].keys():
                    data[key][k] = dict_obj[k]
                elif k in data[key].keys() and (isinstance(data[key][k], int) or isinstance(data[key][k], float)):
                    data[key][k] += dict_obj[k]
        json.dump(data, file, indent=4)
        file.close()