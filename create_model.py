import os, pandas, json, nltk
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import pandas as pd
import os


def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = json.load(f)
    return data


directory = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\'
file_pth = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\done_miki.json'

# data = read_json_file(file_pth)['Sentence']
# for list_of_words in data:
#     if list_of_words and 'misspelled_words' in list_of_words.keys() and list_of_words['misspelled_words']:
#         for k in list_of_words['misspelled_words']:
#             if 'distance' in k.keys() and k['distance']['operations']:
#                 print(k)
#                 print(k['distance']['operations'][0]['ml_repr'])
#                 k['distance']['operations'] = k['distance']['operations'][0]['ml_repr']




def get_correct_words_from_json(file_path: str, labeled: bool = False):
    words = pd.DataFrame()

    for dictionary in read_json_file(file_path)['Sentence']:
        for word in dictionary['all_words']:
            if word['corrected_word'] is None:
                words = pd.concat([words, pd.json_normalize(word)], ignore_index=True)
    if labeled:
        words['label'] = [read_json_file(file_path)['Name'] for _ in range(words.shape[0])]
    return words


user_names = {}

def get_misspelled_words_df_from_json(file_path: str, labeled: bool = False):
    global user_names
    misspelled = pd.DataFrame()
    for dictionary in read_json_file(file_path)['Sentence']:
        if 'misspelled_words' in dictionary.keys() and len(dictionary['misspelled_words']) > 0:
            for k in dictionary['misspelled_words']:
                if 'distance' in k.keys() and k['distance']['operations']:
                    k['distance']['operations'] = np.array(k['distance']['operations'][0]['ml_repr'])
                misspelled = pd.concat([misspelled, pd.json_normalize(k)], ignore_index=True)
    if labeled:
        name = read_json_file(file_path)['Name']
        if not user_names:
            user_names[name] = 0
        else:
            if name not in user_names.keys():
                user_names[name] = max(user_names.values()) + 1
        misspelled['label'] = [user_names[name] for _ in range(misspelled.shape[0])]
    return misspelled


pos_tags = [
    ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
     'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
     'WP', 'WP$', 'WRB']]


def get_tokenized_sentences(file_path: str, is_merged: bool = False):
    corrected_sentences = []
    original_sentences = []
    for dictionary in read_json_file(file_path)['Sentence']:
        if 'tokenized_corrected_sentence' in dictionary.keys() and dictionary['tokenized_corrected_sentence']:
            corrected_sentences = corrected_sentences + dictionary['tokenized_corrected_sentence']
        if 'tokenized_original_sentence' in dictionary.keys() and dictionary['tokenized_original_sentence']:
            original_sentences = original_sentences + dictionary['tokenized_original_sentence']
    if is_merged:
        return original_sentences + corrected_sentences
    else:
        return original_sentences, corrected_sentences


# join data from files
df = pd.DataFrame()
for file in os.listdir(directory):
    df = pd.concat([df, get_misspelled_words_df_from_json(directory + '\\' + file, labeled=True)],
                   ignore_index=True)


is_merged = False
# merged
if is_merged:
    merged_tokenized_senteces = []
    for file in os.listdir(directory):
        merged_tokenized_senteces = merged_tokenized_senteces + get_tokenized_sentences(file, is_merged=True)

    original_word_word2vec = corrected_word_word2vec = Word2Vec([merged_tokenized_senteces], min_count=1)

else:
    original_tokenized_senteces, corrected_tokenized_senteces = [], []
    for file in os.listdir(directory):
        original_tokenized_senteces = original_tokenized_senteces + get_tokenized_sentences(directory+file, is_merged=False)[0]
        corrected_tokenized_senteces = corrected_tokenized_senteces + get_tokenized_sentences(directory+file, is_merged=False)[1]
    original_word_word2vec = Word2Vec([original_tokenized_senteces], min_count=1)
    corrected_word_word2vec = Word2Vec([corrected_tokenized_senteces], min_count=1)

# clear data
df = df.dropna().reset_index()
del df['index']


if 'label' in df.columns.tolist():
    y = df['label'].to_numpy()
    del df['label']


# define interesting metrics
cols = [
        # 'original_word',
        # 'pos_tag',
        # 'corrected_word',
        # 'corrected_word_tag',
        'distance.operations',
        # edit ops
        'distance.damerau_levenshtein_distance',
        'distance.jaro_winkler_ns',
        # token based
        'distance.gestalt_ns',
        'distance.sorensen_dice_ns',
        'distance.cosine_ns',
        'distance.overlap',
        # phonetic
        'distance.mra_ns',
        # seq based
        'distance.lcsstr']

float_vals = ['distance.damerau_levenshtein_distance', 'distance.jaro_winkler_ns', 'distance.gestalt_ns',
          'distance.sorensen_dice_ns', 'distance.cosine_ns', 'distance.overlap', 'distance.mra_ns', 'distance.lcsstr']


df = df.loc[:, cols]
# drop rows with None values
df = df.dropna()

pos_tag_word2vec = Word2Vec(pos_tags, min_count=1)

# vectorize pos_tags
if 'corrected_word_tag' in df.columns.tolist():
    df['corrected_word_tag'] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x, norm=True))

if 'pos_tag' in df.columns.tolist():
    df['pos_tag'] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x, norm=True))

# vectorize words
if 'original_word' in df.columns.tolist():
    df['original_word'] = df['original_word'].apply(lambda x: original_word_word2vec.wv.get_vector(x, norm=True))

if 'corrected_word' in df.columns.tolist():
    df['corrected_word'] = df['corrected_word'].apply(lambda x: corrected_word_word2vec.wv.get_vector(x, norm=True))

# split it to features

X = df.to_numpy()


print(X.shape)
print(y.shape)

print(X[0])


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

svm = svm.SVC()
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)
acc = metrics.accuracy_score(y_test, prediction)
print("predictions: ", prediction)
print("accuracy: ", acc)

