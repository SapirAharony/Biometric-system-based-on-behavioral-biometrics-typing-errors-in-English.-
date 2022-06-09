# spell checker
import os

from spellchecker import SpellChecker
from textblob import TextBlob
import json
# import os
import nltk
from nltk import collections
import Levenshtein


class Distances:
    levenshtein_distance = 0
    type_of_lev_operations = {"insert": 0,
                              "replace": 0,
                              "delete": 0,
                              "transpose": 0}
    damerau_levenshtein_distance = 0
    __word_1 = None
    __word_2 = None

    def __init__(self, str_1, str_2):
        self.__word_1 = str_1
        self.__word_2 = str_2
        self.levenshtein_distance = Levenshtein.distance(str_1, str_2)
        self.__set_operations(str_1, str_2)
        self.damerau_levenshtein_distance = len(self.get_string_oprations(str_1, str_2))

    def __str__(self):
        return '1. ' + self.__word_1 + '\t2. ' + self.__word_2 + '\n\t-D-L distance: ' + str(
            self.damerau_levenshtein_distance) + '\n\t-L distance: ' + str(
            self.levenshtein_distance) + "\n\t-Operations:" + str(self.type_of_lev_operations)

    def __set_operations(self, word_1, word_2, is_damerau=True):
        for k in self.get_string_oprations(word_1, word_2, is_damerau):
            self.type_of_lev_operations[k[0]] += 1

    def get_string_oprations(self, string_1, string_2, is_damerau=True):
        dist_matrix = self.__get_damerau_levenshtein_distance_matrix(string_1, string_2, is_damerau=is_damerau)
        i, j = len(dist_matrix), len(dist_matrix[0])
        i -= 1
        j -= 1
        operations_list = []
        while i != -1 and j != -1:
            if is_damerau:
                if i > 1 and j > 1 and string_1[i - 1] == string_2[j - 2] and string_1[i - 2] == string_2[j - 1]:
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

    def __get_damerau_levenshtein_distance_matrix(self, string1, string2, is_damerau=False):
        distance_matrix = [[0 for _ in range(len(string2) + 1)] for _ in range(len(string1) + 1)]
        for i in range(len(string1) + 1):
            distance_matrix[i][0] = i
        for j in range(len(string2) + 1):
            distance_matrix[0][j] = j
        for i in range(len(string1)):
            for j in range(len(string2)):
                if string1[i] == string2[j]:
                    cost = 0
                else:
                    cost = 1
                distance_matrix[i + 1][j + 1] = min(distance_matrix[i][j + 1] + 1,  # insert
                                                    distance_matrix[i + 1][j] + 1,  # delete
                                                    distance_matrix[i][j] + cost)  # replace
                if is_damerau:
                    if i and j and string1[i] == string2[j - 1] and string1[i - 1] == string2[j]:
                        distance_matrix[i + 1][j + 1] = min(distance_matrix[i + 1][j + 1],
                                                            distance_matrix[i - 1][j - 1] + cost)  # transpose
        return distance_matrix


def correct_spelling_spell_checker(word_or_list_of_words):
    """ A function which returns corrected spelling (by SpellChecker)"""
    if isinstance(word_or_list_of_words, str) and ' ' not in word_or_list_of_words:
        return SpellChecker().correction(word_or_list_of_words)
    elif isinstance(word_or_list_of_words, list):
        return [SpellChecker().correction(word) for word in word_or_list_of_words]


def candidates_to_correct_spelling_spell_checker(word_or_list_of_words):
    """ A class which returns canditates spelling (by SpellChecker)"""
    if isinstance(word_or_list_of_words, str) and ' ' not in word_or_list_of_words:
        return SpellChecker().correction(word_or_list_of_words)
    elif isinstance(word_or_list_of_words, list):
        return [SpellChecker().correction(word) for word in word_or_list_of_words]


def correct_spelling_txt_blb(sentence) -> str:
    """ A function which returns corrected spelling (by TextBlob)"""
    # Making our first textblob
    return TextBlob(sentence).correct()  # Correcting the text


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


class Word:
    word = ''
    lemmatized_word = ''
    corrected_word_txt_blb = ''
    corrected_candidates_spell_chck = ''
    distances_txt_blb = None
    distances_spell_chck = None
    pos_tag = ''

    def __init__(self, word):
        self.word = word
        self.lemmatized_word = lemmatize_word(word)

    def __str__(self):
        tmp = ''
        if self.lemmatized_word is not None:
            tmp += "\n\tLemmatized word: " + self.lemmatized_word + ' '
        if self.corrected_candidates_spell_chck:
            tmp += "\n\tCorrected word by SpellChecker: " + self.corrected_candidates_spell_chck + ',\n\t-distance: ' + str(
                self.distances_spell_chck)
        if self.corrected_word_txt_blb:
            tmp += "\n\tCorrected word by TextBlob: " + self.corrected_word_txt_blb + ',\n\t-distance: ' + str(
                self.distances_txt_blb)
        if tmp:
            return 'Word: ' + self.word + "\n\tpos tag: " + self.pos_tag + tmp
        else:
            return 'Word: ' + self.word + "\n\tpos tag: " + self.pos_tag

    def __repr__(self):
        return str(self)


def write_list_of_words_to_json_file(path_to_file, key, dictionary):
    if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
        # open file
        data = read_json_file(path_to_file)
        print('data', data, type(data))
        # clear file
        open(path_to_file, 'w').close()
        # add data
        file = open(path_to_file, 'a+')
        data[key].append(dictionary)
        file.seek(0)
        json.dump(data, file, indent=4)
    else:
        file = open(path_to_file, 'w+')
        tmp = {key: [dictionary]}
        json.dump(tmp, file, indent=4)
    file.close()


def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = json.load(f)
    return data


class ListOfWords:
    """ A class which includes words, which are separated by NEXT_WORD_COMBINATION"""
    sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[\W_]', gaps=True)
    __add_by_left_click = False
    __original_sentence = ''
    words = []
    is_from_file = False
    pos_tags_counter = None

    def __init__(self, sentence, add_by_left_click=False):
        self.__original_sentence = sentence
        corrected_by_txt_blb = self.sentence_tokenizer.tokenize(
            correct_spelling_txt_blb(self.__original_sentence).stripped)
        i = 0
        for word in self.sentence_tokenizer.tokenize(sentence):
            self.words.append(Word(word))
            self.words[len(self.words) - 1].pos_tag = \
                nltk.pos_tag(self.sentence_tokenizer.tokenize(sentence.lower()))[i][1]
            if correct_spelling_spell_checker(word) != word:
                self.words[i].corrected_candidates_spell_chck = correct_spelling_spell_checker(word)
                self.words[i].distances_spell_chck = Distances(correct_spelling_spell_checker(word), word)
            if corrected_by_txt_blb[i] != word:
                self.words[i].corrected_word_txt_blb = corrected_by_txt_blb[i]
                self.words[i].distances_txt_blb = Distances(corrected_by_txt_blb[i], word)
            i += 1
        self.pos_tags_counter = collections.Counter(
            tag for word, tag in (nltk.pos_tag(self.sentence_tokenizer.tokenize(sentence.lower()))))
        self.__add_by_left_click = add_by_left_click

    def __repr__(self):
        return str(self)

    def __str__(self):
        print("Words: ")
        for word in self.words:
            print("-", word)
        return '\n- Add by left click: ' + str(
            self.__add_by_left_click) + '\n- Is from file: ' + str(self.is_from_file) + '\n- POS Tags counter' + str(
            self.pos_tags_counter)

    def set_left_click(self):
        self.__add_by_left_click = True

    def get_click(self):
        return self.__add_by_left_click

    def clear_list(self):
        self.__add_by_left_click = False
        self.words.clear()


def get_freq_word(list_of_words: ListOfWords, freq_dict: dict):
    if isinstance(list_of_words, ListOfWords) and isinstance(freq_dict, dict):
        for word_istance in list_of_words.words:
            if word_istance.word not in freq_dict.keys():
                freq_dict[word_istance.word] = 1
            else:
                freq_dict[word_istance.word] += 1
    return freq_dict
