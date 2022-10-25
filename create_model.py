import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from json import load
import numpy as np
from itertools import combinations

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

is_ver_sim = False

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
                if 'distance' in misspell.keys() and 'operations' in misspell['distance'].keys():
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
        misspelled['user_label'] = [user_names[name] for _ in range(misspelled.shape[0])]
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
    data_frame = pd.concat([data_frame[data_frame['user_label'] != user_names[correct_user_name]], user_data])
    del user_data
    return data_frame, sample

scaler = StandardScaler()

n_gram_size = 8


def create_ngrams(data_frame, n_gram_size=n_gram_size, num_of_vecs_per_user=1000):
    users_original_data, user_n_grams, result_X, result_Y = {}, {}, [], []
    for lab in data_frame['user_label'].unique():
        users_original_data[lab] = data_frame[data_frame['user_label'] == lab]
        users_original_data[lab] = scaler.fit_transform(users_original_data[lab].iloc[:, :-1])
    for u_id in users_original_data.keys():
        user_n_grams[u_id] = np.array(
            random.sample(list(combinations(users_original_data[u_id], n_gram_size)), num_of_vecs_per_user))
        for vec_id in range(len(user_n_grams[u_id])):
            result_X.append(user_n_grams[u_id][vec_id].flatten())
        for _ in range(num_of_vecs_per_user):
            result_Y.append(u_id)
    del user_n_grams
    return result_X, result_Y


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
    'user_label']

df = load_data(directory)
df = load_data(directory)[cols]


X, y = create_ngrams(df, n_gram_size)

y = np.array(y)

X = scaler.fit_transform(X)

# df = df[df.label == 1]
# df.reindex(df.index, copy=False)
# print(df)
# # split data --> create a sample for simulation
# # standarization
# scaler = StandardScaler()
# for lab in df.label.unique():
#     users_data[lab] = scaler.fit_transform(df.iloc[:, :-1])
# X = scaler.fit_transform(df.iloc[:, :-1])
#
# # define X and y
# y = df[df.columns[-1]].values
#
# print(X)
# print(5*'\n')
# print(y)

"""

# choose features
selector = SelectKBest(f_classif, k=num_of_features)
X = selector.fit_transform(X, y)
f_score_column_indexes = (-selector.scores_).argsort()[:num_of_features]  # choosen featuers indexes
chosen_cols = [cols[k] for k in f_score_column_indexes]


# split data

if not is_ver_sim:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))


else:
    X_train = X
    y_train = y

y_train_categorical = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))



# # define callbacks
# logger = tf.keras.callbacks.TensorBoard(log_dir='rocs',
#                                         write_graph=True,
#                                         histogram_freq=1)
# 
# earlystopping = callbacks.EarlyStopping(monitor='val_loss',
#                                         mode='min',
#                                         patience=7,
#                                         restore_best_weights=True)
# 
# # define model
# 
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_1', input_dim=num_of_features))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(32, activation='relu', name='layer_2'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(32, activation='relu', name='layer_3'))
# model.add(tf.keras.layers.BatchNormalization(momentum=0.95,
#                                              epsilon=0.005,
#                                              beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
#                                                 stddev=0.05),
#                                              gamma_initializer=tf.keras.initializers.Constant(value=0.9)
#                                              ))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(len(user_names.keys()), activation='softmax', name='output_layer'))
# 
# sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
# 
# history = model.fit(tf.expand_dims(X_train, axis=-1), y_train, validation_split=0.4, epochs=1000,
#                         batch_size=128, callbacks=[earlystopping, logger])
# 
# model_summary = model.summary()
# print(model_summary)
# 
# if not is_ver_sim:
#     plot_result(history, "loss")
#     plot_result(history, "accuracy")
#     score = model.evaluate(X_test, y_test, batch_size=128)
#     print(y_test)
#     print("Score: ", score)
#     # plot_result("accuracy")
# else:
#     sample_X = sample.iloc[:, :-1]
#     sample_y = sample[sample.columns[-1]].values
#     prediction = model.predict(sample_X, batch_size=None, verbose=0, steps=None)
#     print(prediction)
"""
