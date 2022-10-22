from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import random
import keras.utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import os
from json import load
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from keras import callbacks

# assign ids to pos_tags
pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
            'POS',
            'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
            'WP', 'WP$', 'WRB']
tmp = {}
i = 0
for tag in pos_tags:
    tmp[tag] = i
    i += 1
pos_tags = tmp
del tmp

# source_files
directory = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\'
file_pth = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\done_miki.json'

# define users and columns to read
user_names = {}


is_ver_sim = True
is_ver_sim = False

num_of_features = 8


# load data
def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = load(f)
    return data


def get_misspelled_words_df_from_json(file_path: str, labeled: bool = True, use_tags: bool = True):
    cols = [
        # edit ops
        'damerau_levenshtein_distance',
        'jaro_winkler_ns',
        # # # # token based
        'gestalt_ns',
        'sorensen_dice_ns',
        'overlap',
        # # phonetic
        'mra_ns',
        # # # seq based
        'lcsstr',
        'ml_type_id',
        'ml_operation_subtype_id',
        'ml_det0',
        'ml_det1',
        'ml_det2',
        'ml_det3',
        'ml_det4',
        'ml_det5'
    ]
    global user_names
    misspelled = pd.DataFrame(columns=cols)
    for dictionary in read_json_file(file_path)['Sentence']:
        distances_ml = []
        if 'misspelled_words' in dictionary.keys() and len(dictionary['misspelled_words']) > 0:
            id = 0
            for misspell in dictionary['misspelled_words']:
                if use_tags:
                    tags = [pos_tags[misspell['pos_tag']], pos_tags[misspell['corrected_word_tag']]]
                if 'distance' in misspell.keys() and misspell['distance']['operations']:
                    id += 1
                    for dist in cols:
                        if dist in misspell['distance'].keys() and id < 2:
                            if isinstance(misspell['distance'][dist], float) or isinstance(dist, int):
                                distances_ml.append(misspell['distance'][dist])
                    for op in misspell['distance']['operations']:

                        tmp = [float(k) for k in op['ml_repr']]
                        if not use_tags:
                            misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp], columns=cols)],
                                                   ignore_index=True)
                        else:
                            misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp + tags],
                                                                             columns=cols + ['pos_tag_org',
                                                                                             'pos_tag_corrected'])],
                                                   ignore_index=True)
    if labeled:
        name = read_json_file(file_path)['Name']
        if not user_names:
            user_names[name] = 0
        else:
            if name not in user_names.keys():
                user_names[name] = max(user_names.values()) + 1
        misspelled['label'] = [user_names[name] for _ in range(misspelled.shape[0])]
    return misspelled


def load_data(files_directory: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for file in os.listdir(files_directory):
        df = pd.concat([df, get_misspelled_words_df_from_json(files_directory + '\\' + file, labeled=True)],
                       ignore_index=True)

    # drop None (clear data)
    df = df.dropna().reset_index()
    del df['index']
    return df


def create_data_for_v_a_simulation(correct_user_name: str, data_frame: pd.DataFrame, number_of_samples: int = 10):
    user_data = data_frame[data_frame['label'] == user_names[correct_user_name]]
    sample = user_data.sample(n=number_of_samples, replace=False)
    user_data = user_data[user_data.index.isin(sample.index.tolist()) == False]
    user_data.reindex()
    sample.reindex()
    data_frame = pd.concat([data_frame[data_frame['label'] != user_names[correct_user_name]], user_data])
    del user_data
    return data_frame, sample


def plot_result(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

cols = [
    # edit ops
    'damerau_levenshtein_distance',
    'jaro_winkler_ns',
    # # # # token based
    'gestalt_ns',
    'sorensen_dice_ns',
    'overlap',
    # # # phonetic
    'mra_ns',
    # # # seq based
    'lcsstr',
    'ml_type_id',
    'ml_operation_subtype_id',
    'ml_det0',
    'ml_det1',
    'ml_det2',
    'ml_det3',
    'ml_det4',
    'ml_det5',
    'pos_tag_org',
    'pos_tag_corrected',
    'label']
df = load_data(directory)[cols]


# split data --> create a sample for simulation
for k in range(2, len(cols)):
    num_of_features = k
    if is_ver_sim:
        correct_user_name = random.choice(list(user_names.keys()))
        df, sample = create_data_for_v_a_simulation(correct_user_name, df, 5)

    # standarization
    scaler = StandardScaler()

    # scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(df.iloc[:, :-1])

    # define X and y
    y = df[df.columns[-1]].values
    # choose features
    selector = SelectKBest(f_classif, k=num_of_features)
    X = selector.fit_transform(X, y)
    f_score_column_indexes = (-selector.scores_).argsort()[:num_of_features]  # choosen featuers indexes
    chosen_cols = [cols[k] for k in f_score_column_indexes]

    # split data

    if not is_ver_sim:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        # y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))


    else:
        X_train = X
        y_train = y

    # y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))

    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
    poly_pred = poly.predict(X_test)
    rbf_pred = rbf.predict(X_test)
    poly_accuracy = accuracy_score(y_test, poly_pred)
    poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    print(k)
    print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy * 100))
    print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1 * 100))
    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy * 100))
    print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1 * 100))

    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB().fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)
    print('\n')
    # accuracy on X_test
    gnb_accuracy = gnb.score(X_test, y_test)
    print('GNB', gnb_accuracy * 100)
    gnb_f1 = f1_score(y_test, gnb_predictions, average='weighted')
    print('Accuracy: ', "%.2f" % (gnb_accuracy * 100))
    print('F1: ', "%.2f" % (gnb_f1 * 100))

    # training a KNN classifier
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    print('\n')
    # accuracy on X_test
    knn_accuracy = knn.score(X_test, y_test)
    knn_predictions = knn.predict(X_test)
    print('KNN', knn_accuracy)
    knn_f1 = f1_score(y_test, knn_predictions, average='weighted')
    print('Accuracy: ', "%.2f" % (knn_accuracy * 100))
    print('F1: ', "%.2f" % (knn_f1 * 100))
    print(10*'\n')



