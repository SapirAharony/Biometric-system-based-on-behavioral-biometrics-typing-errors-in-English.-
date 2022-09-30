import language_tool_python, Levenshtein, textdistance, difflib
from autocorrect import Speller


class EditOperation:
    def __init__(self, operation_type_name: str,operation_type_id: int, idx: int, previous_char: str, next_char: str):
        self.operation_type_name = operation_type_name
        self.previous_char = previous_char
        self.next_char = next_char
        self.char_idx = idx
        self.ml_repr = [operation_type_id]
        if next_char:
            self.ml_repr.append(ord(next_char))
        else:
            self.ml_repr.append(-1)
        if self.previous_char:
            self.ml_repr.append(ord(previous_char))
        else:
            self.ml_repr.append(-1)

class Insert(EditOperation):
    def __init__(self, new_char: str, idx: int, previous_char: str, next_char: str):
        self.inserted_char = new_char
        super().__init__(operation_type_name='Insert', operation_type_id=1, idx=idx, previous_char=previous_char, next_char=next_char)
        self.ml_repr.append(ord(self.inserted_char))
        self.ml_repr.append(self.char_idx)

class Delete(EditOperation):
    def __init__(self, deleted_char: str, idx: int, previous_char: str, next_char: str):
        self.deleted_char = deleted_char
        super().__init__(operation_type_name='Delete', operation_type_id=2, idx=idx, previous_char=previous_char, next_char=next_char)
        self.ml_repr.append(ord(self.deleted_char))
        self.ml_repr.append(self.char_idx)

class Replace(EditOperation):
    def __init__(self, old_char: str, new_char: str, idx: int, previous_char: str, next_char: str):
        self.old_char = old_char
        self.new_char = new_char
        super().__init__(operation_type_name='Replace', operation_type_id=3, idx=idx, previous_char=previous_char, next_char=next_char)
        self.ml_repr.append(ord(self.old_char))
        self.ml_repr.append(self.char_idx)
        self.ml_repr.append(ord(self.new_char))

class Transpose(EditOperation):
    def __init__(self, left_char: str, right_char: str, idx_left: int, idx_right: int, previous_char: str,
                 next_char: str):
        self.left_char = left_char
        self.right_char = right_char
        self.idx_right = idx_right
        super().__init__(operation_type_name='Transpose', operation_type_id=4, idx=idx_left, previous_char=previous_char, next_char=next_char)
        self.ml_repr.append(ord(self.left_char))
        self.ml_repr.append(self.char_idx)
        self.ml_repr.append(ord(self.right_char))
        self.ml_repr.append(self.idx_right)

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
    lev_thresholds = {'short_word': (4, 0.3), 'longer_words': (5, 0.6)}

    def __init__(self, str_1, str_2, use_treshold: bool = True, is_tokenized: bool = False):
        # check condtitions to calculate Distances (thresholds for shorts and longer words)
        if use_treshold and (((((len(str_1)) <= self.lev_thresholds['short_word'][0] or len(str_2)) <= self.lev_thresholds['short_word'][0]) and
                              textdistance.damerau_levenshtein.normalized_similarity(str_1, str_2) >= self.lev_thresholds['short_word'][1]) or
                             ((len(str_1)) >= self.lev_thresholds['longer_words'][0] and
                              textdistance.damerau_levenshtein.normalized_similarity(str_1, str_2) >= self.lev_thresholds['longer_words'][1])):
            self.__word_1 = str_1
            self.__word_2 = str_2
            self.operations = []
            self.set_operations()

            # edit
            self.damerau_levenshtein_distance = textdistance.damerau_levenshtein.normalized_similarity(self.__word_1, self.__word_2)
            self.jaro_winkler_ns = textdistance.jaro_winkler.normalized_similarity(self.__word_1, self.__word_2)

            # token based
            if is_tokenized:
                self.gestalt_ns = textdistance.ratcliff_obershelp.normalized_similarity(self.__word_1, self.__word_2)
                self.sorensen_dice_ns = textdistance.sorensen_dice.normalized_similarity(self.__word_1, self.__word_2)
                self.cosine_ns = textdistance.cosine.normalized_similarity(self.__word_1, self.__word_2)
                self.overlap = textdistance.overlap.normalized_similarity(self.__word_1, self.__word_2)

            # phonetic
            self.mra_ns = textdistance.mra.normalized_similarity(self.__word_1, self.__word_2)

            # Sequence based
            self.lcsstr = textdistance.lcsstr.normalized_similarity(self.__word_1, self.__word_2)

        # self.__seq_matcher = difflib.SequenceMatcher(isjunk=None, a=self.__word_1, b=self.__word_2)
        # self.seq_matcher_opcodes = self.__seq_matcher.get_opcodes()
        # self.get_matching_blocks = self.__seq_matcher.get_matching_blocks()
        # # if isTokenizedWord:
        # #     self.overlap = textdistance.overlap(self.__word_1, self.__word_2)

    def set_operations(self, is_damerau=True):
        for operation in get_string_oprations(self.__word_1, self.__word_2, is_damerau):
            if operation[0] == 'delete':
                if operation[1] - 1 >= 0 and operation[1] + 1 < len(self.__word_1) - 1:
                    self.operations.append(Delete(deleted_char=self.__word_1[operation[1]], idx=operation[1],
                                                  previous_char=self.__word_1[operation[1] - 1],
                                                  next_char=self.__word_1[operation[1] + 1]))
                elif operation[1] - 1 < 0 and operation[1] + 1 > len(self.__word_1) - 1:
                    self.operations.append(Delete(deleted_char=self.__word_1[operation[1]], idx=operation[1],
                                                  previous_char='', next_char=''))
                elif operation[1] + 1 > len(self.__word_1) - 1:
                    self.operations.append(Delete(deleted_char=self.__word_1[operation[1]], idx=operation[1],
                                                  previous_char=self.__word_1[operation[1] - 1], next_char=''))
                elif operation[1] - 1 < 0:
                    self.operations.append(Delete(deleted_char=self.__word_1[operation[1]], idx=operation[1],
                                                  previous_char='', next_char=self.__word_1[operation[1] + 1]))

            elif operation[0] == 'replace':
                if operation[1] - 1 >= 0 and operation[1] + 1 < len(self.__word_1) - 1:
                    self.operations.append(
                        Replace(old_char=self.__word_1[operation[1]], new_char=self.__word_2[operation[2]],
                                idx=operation[1],
                                previous_char=self.__word_1[operation[1] - 1],
                                next_char=self.__word_1[operation[1] + 1]))
                elif operation[1] - 1 < 0 and operation[1] + 1 > len(self.__word_1) - 1:
                    self.operations.append(
                        Replace(old_char=self.__word_1[operation[1]], new_char=self.__word_2[operation[2]],
                                idx=operation[1],
                                previous_char='',
                                next_char=''))

                elif operation[1] + 1 > len(self.__word_1) - 1:
                    self.operations.append(
                        Replace(old_char=self.__word_1[operation[1]], new_char=self.__word_2[operation[2]],
                                idx=operation[1],
                                previous_char=self.__word_1[operation[1] - 1],
                                next_char=''))

                elif operation[1] - 1 < 0:
                    self.operations.append(
                        Replace(old_char=self.__word_1[operation[1]], new_char=self.__word_2[operation[2]],
                                idx=operation[1],
                                previous_char='',
                                next_char=self.__word_1[operation[1] + 1]))

            elif operation[0] == 'insert':
                if operation[1] >= 0 and operation[1] + 1 <= len(self.__word_1) - 1:
                    self.operations.append(Insert(new_char=self.__word_2[operation[2]], idx=operation[1] + 1,
                                                  previous_char=self.__word_1[operation[1]],
                                                  next_char=self.__word_1[operation[1] + 1]))
                elif operation[1] < 0 and operation[1] + 1 > len(self.__word_1) - 1:
                    self.operations.append(Insert(new_char=self.__word_2[operation[2]], idx=operation[1] + 1,
                                                  previous_char='',
                                                  next_char=""))
                elif operation[1] + 1 > len(self.__word_1) - 1:
                    self.operations.append(Insert(new_char=self.__word_2[operation[2]], idx=operation[1] + 1,
                                                  previous_char=self.__word_1[operation[1]],
                                                  next_char=''))
                elif operation[1] < 0:
                    self.operations.append(Insert(new_char=self.__word_2[operation[2]], idx=operation[1] + 1,
                                                  previous_char="", next_char=self.__word_1[operation[1] + 1]))

            else:
                if operation[2] - 1 >= 0 and operation[1] + 1 < len(self.__word_1) - 1:
                    self.operations.append(Transpose(left_char=self.__word_1[operation[2]], idx_left=operation[2],
                                                     right_char=self.__word_1[operation[1]], idx_right=operation[1],
                                                     previous_char=self.__word_1[operation[2] - 1],
                                                     next_char=self.__word_1[operation[1] + 1]))
                elif operation[2] - 1 < 0 and operation[1] + 1 > len(self.__word_1) - 1:
                    self.operations.append(Transpose(left_char=self.__word_1[operation[2]], idx_left=operation[2],
                                                     right_char=self.__word_1[operation[1]], idx_right=operation[1],
                                                     previous_char='', next_char=''))
                elif operation[1] + 1 > len(self.__word_1) - 1:
                    self.operations.append(Transpose(left_char=self.__word_1[operation[2]], idx_left=operation[2],
                                                     right_char=self.__word_1[operation[1]], idx_right=operation[1],
                                                     previous_char=self.__word_1[operation[2] - 1], next_char=''))
                elif operation[2] - 1 < 0:
                    self.operations.append(Transpose(left_char=self.__word_1[operation[2]], idx_left=operation[2],
                                                     right_char=self.__word_1[operation[1]], idx_right=operation[1],
                                                     previous_char="", next_char=self.__word_1[operation[1] + 1]))

