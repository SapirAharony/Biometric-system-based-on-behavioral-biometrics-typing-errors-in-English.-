import os, pandas, json, nltk
import numpy as np
from sklearn import svm, metrics, linear_model
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize, StandardScaler
import pandas as pd
import os

#### LOAD DATA ####
def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = json.load(f)
    return data


directory = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\'
file_pth = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\done_miki.json'

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

# def get_misspelled_words_df_from_json(file_path: str, labeled: bool = False):
#     global user_names
#     misspelled = pd.DataFrame()
#     for dictionary in read_json_file(file_path)['Sentence']:
#         if 'misspelled_words' in dictionary.keys() and len(dictionary['misspelled_words']) > 0:
#             for k in dictionary['misspelled_words']:
#                 if 'distance' in k.keys() and k['distance']['operations']:
#                     k['distance']['operations'] = k['distance']['operations'][0]['ml_repr']
#                     print(type(k['distance']['operations']))
#                 misspelled = pd.concat([misspelled, pd.json_normalize(k)], ignore_index=True)
#     if labeled:
#         name = read_json_file(file_path)['Name']
#         if not user_names:
#             user_names[name] = 0
#         else:
#             if name not in user_names.keys():
#                 user_names[name] = max(user_names.values()) + 1
#         misspelled['label'] = [user_names[name] for _ in range(misspelled.shape[0])]
#     return misspelled


def get_misspelled_words_df_from_json(file_path: str, labeled: bool = False):
    global user_names
    misspelled = pd.DataFrame()
    for dictionary in read_json_file(file_path)['Sentence']:
        if 'misspelled_words' in dictionary.keys() and len(dictionary['misspelled_words']) > 0:
            for k in dictionary['misspelled_words']:
                if 'distance' in k.keys() and k['distance']['operations']:
                    a = 0
                    for i in k['distance']['operations'][0]['ml_repr']:
                        k['distance']['operations'+str(a)] = i
                        print(i)
                        a += 1
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


print(df.shape)
print(df.dtypes)
df = df.dropna().reset_index()
del df['index']
del df['distance.operations']
if 'label' in df.columns.tolist():
    y = df['label'].to_numpy()
    del df['label']

vec_size = 10
pos_tags = [
    ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
     'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
     'WP', 'WP$', 'WRB']]
pos_tag_word2vec = Word2Vec(pos_tags, min_count=1, vector_size=10)


# define interesting metrics
cols = [
        'pos_tag',
        'corrected_word_tag',
        'distance.operations0',
        'distance.operations1',
        'distance.operations2',
        'distance.operations3',
        'distance.operations4',
        'distance.operations5',
        'distance.operations6',
        'distance.operations7',
        # edit ops
        'distance.damerau_levenshtein_distance',
        # #'distance.jaro_winkler_ns',
        # # # token based
        'distance.gestalt_ns',
        # 'distance.sorensen_dice_ns',
        # 'distance.cosine_ns',
        # 'distance.overlap',
        # # # phonetic
        'distance.mra_ns',
        # # # seq based
        'distance.lcsstr']

float_vals = ['distance.damerau_levenshtein_distance', 'distance.jaro_winkler_ns', 'distance.gestalt_ns',
          'distance.sorensen_dice_ns', 'distance.cosine_ns', 'distance.overlap', 'distance.mra_ns', 'distance.lcsstr']


df = df.loc[:, cols]
# drop rows with None values
df = df.dropna()

del df['pos_tag']
del df['corrected_word_tag']

# vectorize pos_tags
if 'corrected_word_tag' in df.columns.tolist():
    df['corrected_word_tag'+str(0)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[0])
    df['corrected_word_tag'+str(1)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[1])
    df['corrected_word_tag'+str(2)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[2])
    df['corrected_word_tag'+str(3)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[3])
    df['corrected_word_tag'+str(4)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[4])
    df['corrected_word_tag'+str(5)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[5])
    df['corrected_word_tag'+str(6)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[6])
    df['corrected_word_tag'+str(7)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[7])
    df['corrected_word_tag'+str(8)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[8])
    df['corrected_word_tag'+str(9)] = df['corrected_word_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[9])

if 'pos_tag' in df.columns.tolist():
    df['pos_tag'+str(0)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[0])
    df['pos_tag'+str(1)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[1])
    df['pos_tag'+str(2)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[2])
    df['pos_tag'+str(3)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[3])
    df['pos_tag'+str(4)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[4])
    df['pos_tag'+str(5)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[5])
    df['pos_tag'+str(6)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[6])
    df['pos_tag'+str(7)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[7])
    df['pos_tag'+str(8)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[8])
    df['pos_tag'+str(9)] = df['pos_tag'].apply(lambda x: pos_tag_word2vec.wv.get_vector(x)[9])

# features_df = pd.DataFrame()
#
# features_df['result'] = df.apply(lambda row: np.vectorize(row), axis=1)
# print(features_df)
# split it to features

X = df.to_numpy()

print(X.shape)
print(y.shape)

print(X[0])



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


svm = svm.SVC()
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)
acc = metrics.accuracy_score(y_test, prediction)
print("predictions: ", prediction)
print("accuracy: ", acc)


'''
print(10*"\n")

l_reg = linear_model.LinearRegression()


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)


model = l_reg.fit(X_train, Y_train)
preds = model.predict(X_test)
print("Predictions: ", preds)

print("R^2: ", l_reg.score(X,y))

'''