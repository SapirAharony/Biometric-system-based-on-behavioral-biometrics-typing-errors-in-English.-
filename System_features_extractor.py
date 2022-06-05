# spell checker
from spellchecker import SpellChecker
from textblob import TextBlob
import json
# import os
import nltk
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
        self.__set_operations()
        self.damerau_levenshtein_distance = len(self.get_string_oprations(str_1, str_2))

    def __set_operations(self, is_damerau=True):
        for k in self.get_string_oprations(self.__word_1, self.__word_2, is_damerau):
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


def get_freq_word(list_of_words, freq_dict):
    if isinstance(list_of_words, ListOfWords) and isinstance(freq_dict, dict):
        for word in list_of_words.words:
            if word not in freq_dict.keys():
                freq_dict[word] = 1
            else:
                freq_dict[word] += 1
    return freq_dict


def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = json.load(f)
    return data


class FeatureExtractor:
    """ A class that  contains correctors, pos_tags"""
    spell_checker = SpellChecker()

    def correct_spelling_spell_checker(self, word):
        """ A function which returns corrected spelling (by SpellChecker)"""
        correct_word = self.spell_checker.correction(word)
        return correct_word

    def candidates_to_correct_spelling_spell_checker(self, word):
        """ A class which returns canditates spelling (by SpellChecker)"""
        return self.spell_checker.candidates(word)

    def correct_spelling_txt_blb(self, sentence):
        """ A function which returns corrected spelling (by TextBlob)"""
        # Making our first textblob
        return TextBlob(sentence).correct()  # Correcting the text

    def __get_pos_for_word(self, word):
        """ Method that returns a POS tag for lemmatization """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                    "N": nltk.corpus.wordnet.NOUN,
                    "V": nltk.corpus.wordnet.VERB,
                    "R": nltk.corpus.wordnet.ADV}
        return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

    def lemmatized_word(self, word):
        return nltk.stem.WordNetLemmatizer().lemmatize(word, self.__get_pos_for_word(word))

    # def set_update_pos_tags(self, word):
    #     return nltk.pos_tag(word)


class Word:
    __word = ''
    __lemmatized_word = ''
    __corrected_word_txt_blb = ''
    __corrected_candidates_spell_chck = ''
    __distances_txt_blb = None
    __distances_spell_chck = None
    __was_edit = False
    _pos_tag = ''
    featureextr = FeatureExtractor()

    def __init__(self, word):
        self.__word = word

    def __str__(self):
        return self.__word + " " + self.__lemmatized_word + " " + self.__corrected_word_txt_blb + " " + self.__distances_txt_blb

    def get_word(self, word):
        return word


class ListOfWords:
    """ A class which includes words, which are separated by NEXT_WORD_COMBINATION"""
    __add_by_left_click = False
    __sentence = ''
    words = []
    pos_tags = []
    is_from_file = False
    feature_extractor = FeatureExtractor

    def __init__(self, sentence):
        self.sentence = sentence
        for word in self.sentence.split():
            self.words.append(Word(word))
        self.__add_by_left_click = False

    def __str__(self):
        return 'words: ' + " ".join(self.words) + '\nAdd by left click: ' + str(
            self.__add_by_left_click) + '\nIs from file: ' + str(self.is_from_file)

    def set_left_click(self, value):
        if value:
            self.__add_by_left_click = True
        elif not value:
            self.__add_by_left_click = False

    def get_click(self):
        return self.__add_by_left_click

    def clear_list(self):
        self.__add_by_left_click = False
        self.words.clear()

    # def write_list_of_words_to_json_file(self, path_to_file, key, dictionary):
    #     self.set_update_pos_tags()
    #     if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
    #         # open file
    #         data = read_json_file(path_to_file)
    #         print('data', data, type(data))
    #         # clear file
    #         open(path_to_file, 'w').close()
    #         # add data
    #         file = open(path_to_file, 'a+')
    #         data[key].append(dictionary)
    #         file.seek(0)
    #         json.dump(data, file, indent=4)
    #     else:
    #         file = open(path_to_file, 'w+')
    #         tmp = {key: [dictionary]}
    #         json.dump(tmp, file, indent=4)
    #     file.close()
