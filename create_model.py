from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler # to avoid biasing due to sample length ,e.g. one person has 2 times more samples than another
import pandas as pd
import os
from json import load
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
     'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
     'WP', 'WP$', 'WRB']
tmp = {}
i = 0
for tag in pos_tags:
    tmp[tag] = i
    i += 1
pos_tags = tmp
del tmp


directory = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\'
file_pth = 'C:\\Users\\user\\PycharmProjects\\bio_system\\json_files\\done_miki.json'
user_names = {}
cols = [
    # edit ops
    'damerau_levenshtein_distance',
    'jaro_winkler_ns',
    # # # token based
    'gestalt_ns',
    'sorensen_dice_ns',
     'overlap',
    # # # phonetic
    'mra_ns',
    # # # seq based
    #'lcsstr',
    'ml_type_id',
    'ml_operation_subtype_id'
]

for k in range(6):
    cols.append('ml_det' + str(k))



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
                            misspelled = pd.concat([misspelled, pd.DataFrame([distances_ml + tmp + tags], columns=cols+['pos_tag_org', 'pos_tag_corrected'])],
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





df = pd.DataFrame()
for file in os.listdir(directory):
    df = pd.concat([df, get_misspelled_words_df_from_json(directory + '\\' + file, cols=cols, labeled=True)],
                   ignore_index=True)

print(df.shape)
# print(df.dtypes)
df = df.dropna().reset_index()
print(df.shape)
del df['index']


print(df.head())
for i in range(len(df.columns[:-1])):
    label = df.columns[i]
path = 'C:\\Users\\user\\PycharmProjects\\bio_system\\graphs\\'
# print()
# plot df
# for col in df.columns[:-1]:
#     print(col, end=' ')
#     plt.hist(df[df['label'] == 0][col], color='blue', label='0', stacked=True, alpha=0.3, density=True, bins=15)
#     plt.hist(df[df['label'] == 1][col], color='black', label='1', stacked=True, alpha=0.3, density=True, bins=15)
#     plt.hist(df[df['label'] == 2][col], color='green', label='2', stacked=True, alpha=0.3, density=True, bins=15)
#     plt.title(col)
#     plt.ylabel('Probability')
#     plt.xlabel(col)
#     plt.legend()
#     plt.savefig(path+'orig\\'+col+'.png')

cols = [
    # edit ops
    # 'damerau_levenshtein_distance',
    'jaro_winkler_ns',
    # # # token based
    # 'gestalt_ns',
    'sorensen_dice_ns',
     # 'overlap',
    # # # phonetic
    'mra_ns',
    # # # seq based
    #'lcsstr',
    'ml_type_id',
    'ml_operation_subtype_id'
]

df = df[cols+['pos_tag_org', 'pos_tag_corrected', 'label']]
print(df.columns)
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
data = np.hstack((X, np.reshape(y, (-1, 1))))
# over = RandomOverSampler()
# X, y = over.fit_resample(X, y)
# data = np.hstack((X, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)
print(len(transformed_df[transformed_df['label'] == 0]))
print(len(transformed_df[transformed_df['label'] == 1]))
print(len(transformed_df[transformed_df['label'] == 2]))

# plot transformed df

# for col in df.columns[:-1]:
#     print(col, end=' ')
#     plt.hist(transformed_df[transformed_df['label'] == 0][col], color='blue', label='0', stacked=True, alpha=0.3, density=True, bins=15)
#     plt.hist(transformed_df[transformed_df['label'] == 1][col], color='black', label='1', stacked=True, alpha=0.3, density=True, bins=15)
#     plt.hist(transformed_df[transformed_df['label'] == 2][col], color='green', label='2', stacked=True, alpha=0.3, density=True, bins=15)
#     plt.title(col)
#     plt.ylabel('Probability')
#     plt.xlabel(col)
#     plt.legend()
#     plt.savefig(path+'transformed\\'+col+'.png')


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
print(X_train.shape, X_temp.shape, y_train.shape, y_temp.shape)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
print(X_valid.shape, X_test.shape, y_valid.shape, y_test.shape)

model = tf.keras.Sequential([tf.keras.layers.Dense(8, activation='relu'),
                             tf.keras.layers.Dense(32, activation='relu'),
                            # tf.keras.layers.Dense(32, activation='relu'),
                             tf.keras.layers.Dense(1, activation='softmax')])
model.add(tf.keras.layers.Flatten())


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=5, validation_data=(X_valid, y_valid))

print(model.evaluate(X_test, y_test))



