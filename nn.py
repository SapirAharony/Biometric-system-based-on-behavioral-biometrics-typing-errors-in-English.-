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
from create_model import X, y, cols, user_names

def plot_result_nn(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()



# if not is_ver_sim:
#         sample_X = sample.iloc[:, :-1]
#         sample_Y = sample[sample.columns[-1]].values

scaler = StandardScaler()

data = np.hstack((X, np.reshape(y, (-1, 1))))

num_of_features = 10
if num_of_features > len(cols):
    num_of_features = len(cols)

selector = SelectKBest(f_classif, k=num_of_features)
X = selector.fit_transform(X, y)
f_score_column_indexes = (-selector.scores_).argsort()[:num_of_features]  # choosen featuers indexes
choosen_cols = [cols[k] for k in f_score_column_indexes]
print(choosen_cols)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))
y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))
logger = tf.keras.callbacks.TensorBoard(log_dir='rocs', write_graph=True, histogram_freq=1, )
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu', name='layer_1', input_dim=num_of_features))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_2'))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_3'))
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
                        batch_size=128, callbacks=[earlystopping, logger])
score = model.evaluate(X_test, y_test, batch_size=128)
print(y_test)
print("Score: ", score)

plot_result_nn(history, "loss")
plot_result_nn(history, "accuracy")
