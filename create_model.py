import random

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



# load data
def read_json_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = load(f)
    return data


def get_misspelled_words_df_from_json(file_path: str, labeled: bool = False, use_tags: bool = True):
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




test_mode = False
summary = ([], [])
for k in range(150):
    df = pd.DataFrame()
    for file in os.listdir(directory):
        df = pd.concat([df, get_misspelled_words_df_from_json(directory + '\\' + file, labeled=True)],
                       ignore_index=True)

    # drop None (clear data)
    df = df.dropna().reset_index()
    del df['index']

    # declare user_name
    correct_user_name = random.choice(list(user_names.keys()))

    user_data = df[df['label'] == user_names[correct_user_name]]
    sample = user_data.sample(n=5, replace=False)


    user_data = user_data[user_data.index.isin(sample.index.tolist()) == False]
    user_data.reindex()
    sample.reindex()
    df = pd.concat([df[df['label'] != user_names[correct_user_name]], user_data])
    del user_data

    for i in range(len(df.columns[:-1])):
        label = df.columns[i]

    path = 'C:\\Users\\user\\PycharmProjects\\bio_system\\graphs\\'



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
        # 'ml_det0',
        # 'ml_det1',
        # 'ml_det2',
        # 'ml_det3',
        # 'ml_det4',
        # 'ml_det5',
        'pos_tag_org',
        'pos_tag_corrected',
        'label']

    num_of_features = 10
    if num_of_features > len(cols):
        num_of_features = len(cols)
    df = df[cols]

    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    sample_X = sample.iloc[:, :-1]
    sample_Y = sample[sample.columns[-1]].values

    scaler = StandardScaler()
    data = np.hstack((X, np.reshape(y, (-1, 1))))

    num_of_features = 10
    if num_of_features > len(cols):
        num_of_features = len(cols)
    selector = SelectKBest(f_classif, k=num_of_features)
    X = selector.fit_transform(X, y)
    f_score_column_indexes = (-selector.scores_).argsort()[:10]  # choosen featuers indexes

    choosen_cols = [cols[k] for k in f_score_column_indexes]
    print(choosen_cols)

    sample_X = sample[sample.columns.intersection(choosen_cols)]
    sample_X = scaler.fit_transform(sample_X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=0)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
    y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
    y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))
    sample_Y = keras.utils.to_categorical(sample_Y, num_classes=len(user_names.keys()))
    print('X_train', X_train, 3 * '\n', 'X_temp', X_temp, 3 * '\n', 'y_train', y_train, 3 * '\n', 'y_valid',
          y_valid, 3 * '\n', 'y_test', y_test, 3 * '\n', 'sample_Y', sample_Y)

    #
    # model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.compile(
    #     loss="binary_crossentropy", optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]
    # )
    # model.add(tf.keras.layers.Flatten())

    logger = tf.keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', name='layer_1', input_dim=num_of_features))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_3'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.95,
                                                 epsilon=0.005,
                                                 beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                     stddev=0.05),
                                                 gamma_initializer=tf.keras.initializers.Constant(value=0.9)
                                                 ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(user_names.keys()), activation='softmax', name='output_layer'))
    print(model.summary())

    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    earlystopping = callbacks.EarlyStopping(monitor="val_accuracy",
                                            mode="max", patience=7,
                                            restore_best_weights=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    history = model.fit(tf.expand_dims(X_train, axis=-1), y_train, validation_data=(X_valid, y_valid), epochs=1800,
                        batch_size=128, callbacks=earlystopping)

    score = model.evaluate(X_test, y_test, batch_size=128)
    print(y_test)
    print("Score: ", score)

    prediction = model.predict(sample_X, batch_size=None, verbose=0, steps=None)
    print(prediction)

    result = {k: [] for k in user_names.keys()}
    for single_pred in prediction:
        for el_id in range(len(single_pred)):
            key = [k for k, v in user_names.items() if v == el_id][0]
            result[key].append(single_pred[el_id])

    for key in result:
        result[key] = sum(result[key])
        print(key, result[key])

    predicted_user = [k for k, v in result.items() if v == max(result.values())][0]
    print(predicted_user, correct_user_name, predicted_user == correct_user_name)
    summary[0].append(predicted_user == correct_user_name)
    summary[1].append((predicted_user, correct_user_name))



for tup in summary:
    print(tup)
print(len([x for x in summary[0] if x is True]))
print(len(summary[0]))
print(len([x for x in summary[0] if x is True])/len(summary[0]))





# read data


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


# plot_result("loss")
# plot_result("accuracy")
