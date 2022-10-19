"""

# import keras.callbacks
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import os
# from json import load
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from explore_data import plot_df
#
#
#
#
# pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
#             'POS',
#             'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
#             'WP', 'WP$', 'WRB']
# tmp = {}
# i = 0
# for tag in pos_tags:
#     tmp[tag] = i
#     i += 1
# pos_tags = tmp
# del tmp
#
# directory = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\'
# file_pth = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\done_miki.json'
# user_names = {}
# cols = [
#     # edit ops
#     # 'damerau_levenshtein_distance',
#     'jaro_winkler_ns',
#     # # # # token based
#     'gestalt_ns',
#     'sorensen_dice_ns',
#     # 'overlap',
#     # # # phonetic
#     'mra_ns',
#     # # # seq based
#     # 'lcsstr',
#     'ml_type_id',
#     'ml_operation_subtype_id'
# ]
#
# for k in range(6):
#     cols.append('ml_det' + str(k))
#
#
# # load data
# def read_json_file(path_to_file):
#     with open(path_to_file, 'r') as f:
#         data = load(f)
#     return data
#
#
# def get_misspelled_words_df_from_json(file_path: str, cols: list, labeled: bool = False, use_tags: bool = True):
#     global user_names
#     misspelled = pd.DataFrame(columns=cols)
#     for dictionary in read_json_file(file_path)['Sentence']:
#         distances_ml = []
#         if 'misspelled_words' in dictionary.keys() and len(dictionary['misspelled_words']) > 0:
#             id = 0
#             for misspell in dictionary['misspelled_words']:
#                 if use_tags:
#                     tags = [pos_tags[misspell['pos_tag']], pos_tags[misspell['corrected_word_tag']]]
#                 if 'distance' in misspell.keys() and misspell['distance']['operations']:
#                     id += 1
#                     for dist in cols:
#                         if dist in misspell['distance'].keys() and id < 2:
#                             if isinstance(misspell['distance'][dist], float) or isinstance(dist, int):
#                                 distances_ml.append(misspell['distance'][dist])
#                     for op in misspell['distance']['operations']:
#
#                         tmp = [float(k) for k in op['ml_repr']]
#                         if not use_tags:
#                             misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp], columns=cols)],
#                                                    ignore_index=True)
#                         else:
#                             misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp + tags],
#                                                                              columns=cols + ['pos_tag_org',
#                                                                                              'pos_tag_corrected'])],
#                                                    ignore_index=True)
#     if labeled:
#         name = read_json_file(file_path)['Name']
#         if not user_names:
#             user_names[name] = 0
#         else:
#             if name not in user_names.keys():
#                 user_names[name] = max(user_names.values()) + 1
#         misspelled['label'] = [user_names[name] for _ in range(misspelled.shape[0])]
#     return misspelled
#
#
# df = pd.DataFrame()
# for file in os.listdir(directory):
#     df = pd.concat([df, get_misspelled_words_df_from_json(directory + '\\' + file, cols=cols, labeled=True)],
#                    ignore_index=True)
#
# print(df.shape)
# # print(df.dtypes)
# df = df.dropna().reset_index()
# print(df.shape)
# del df['index']
#
# print(df.head())
# for i in range(len(df.columns[:-1])):
#     label = df.columns[i]
# path = 'C:\\Users\\user\\PycharmProjects\\bio_system\\graphs\\'
#
#
# # plot_df(df, path, amount_of_labels=3)
#
#
# # cols = [
# #     # edit ops
# #     'damerau_levenshtein_distance',
# #     'jaro_winkler_ns',
# #     # # # token based
# #     'gestalt_ns',
# #     # 'sorensen_dice_ns',
# #      # 'overlap',
# #     # # # phonetic
# #     'mra_ns',
# #     # # # seq based
# #     'lcsstr',
# #     'ml_type_id',
# #     'ml_operation_subtype_id'
# # ]
#
# df = df[cols + ['pos_tag_org', 'pos_tag_corrected', 'label']]
# print(df.columns)
# X = df[df.columns[:-1]].values
# y = df[df.columns[-1]].values
#
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# data = np.hstack((X, np.reshape(y, (-1, 1))))
#
# transformed_df = pd.DataFrame(data, columns=df.columns)
#
#
#
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
# print(X_train.shape, X_temp.shape, y_train.shape, y_temp.shape)
# X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
# print(X_valid.shape, X_test.shape, y_valid.shape, y_test.shape)
#
# model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu', name='layer_1'),
#                              tf.keras.layers.Dense(100, activation='relu', name='layer_2'),
#                              # tf.keras.layers.Dense(180, activation='relu', name='layer_3'),
#                              tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')])
#
#
#
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])
# model.add(tf.keras.layers.Flatten())
#
#
# logger = tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, )
#
# # history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), shuffle=True, verbose=2,
# # callbacks=[logger])
#
# history = model.fit(X_train, y_train, epochs=9, validation_data=(X_valid, y_valid), shuffle=True, verbose=2, )
#
# print(model.evaluate(X_test, y_test))
#
#
# def plot_result(item):
#     plt.plot(history.history[item], label=item)
#     plt.plot(history.history["val_" + item], label="val_" + item)
#     plt.xlabel("Epochs")
#     plt.ylabel(item)
#     plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
#     plt.legend()
#     plt.grid()
#     plt.show()
#
#
# plot_result("loss")
# plot_result("accuracy")
"""
import keras.utils
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from json import load
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


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

# load data
def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = load(f)
    return data


def get_misspelled_words_df_from_json(file_path: str, cols: list, labeled: bool = False, use_tags: bool = True):
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

# read data
df = pd.DataFrame()
for file in os.listdir(directory):
    df = pd.concat([df, get_misspelled_words_df_from_json(directory + '\\' + file, cols=cols, labeled=True)],
                   ignore_index=True)

# drop None (clear data)
df = df.dropna().reset_index()
print(df.shape)
del df['index']

print(df.head())
for i in range(len(df.columns[:-1])):
    label = df.columns[i]
path = 'C:\\Users\\user\\PycharmProjects\\bio_system\\graphs\\'


# define interesting cols
# cols = [
#     # edit ops
#     'damerau_levenshtein_distance',
#     'jaro_winkler_ns',
#     # # # token based
#     'gestalt_ns',
#     # 'sorensen_dice_ns',
#      # 'overlap',
#     # # # phonetic
#     'mra_ns',
#     # # # seq based
#     'lcsstr',
#     'ml_type_id',
#     'ml_operation_subtype_id'
# ]
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
    #'lcsstr',
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


# selector = SelectKBest(f_classif, k=10)
# selected_features = selector.fit_transform(train_features, train_labels)
# f_score_indexes = (-selector.scores_).argsort()[:10]

df = df[cols]
print(df.columns)
print(df.describe())
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
data = np.hstack((X, np.reshape(y, (-1, 1))))

# transformed_df = pd.DataFrame(data, columns=df.columns)




X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=0)
print(X_train.shape, X_temp.shape, y_train.shape, y_temp.shape)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
print(X_valid.shape, X_test.shape, y_valid.shape, y_test.shape)
y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))
y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_1', input_dim=len(cols)-1))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', name='layer_2'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', name='layer_3'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(user_names.keys()), activation='softmax', name='output_layer'))


sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.compile(
#     loss="binary_crossentropy", optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]
# )
# model.add(tf.keras.layers.Flatten())


logger = tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, )

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1200, batch_size=128)

score = model.evaluate(X_test[:-1], y_test[:-1], batch_size=128)
print(score)
print(model.predict(X_test[-1:], batch_size=None, verbose=0, steps=None))
print(y_test[-1:])


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("accuracy")

