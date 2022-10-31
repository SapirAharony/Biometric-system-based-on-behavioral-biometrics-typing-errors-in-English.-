import keras.utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import callbacks
from create_model import X, y, user_names, program_n_gram_size, program_is_ver_sim, features_cols
from draw_results import draw_roc_curve, plot_result_nn


if program_is_ver_sim:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    from create_model import y_test, X_test
else:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=0)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.66, random_state=0)

y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))
y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

logger = tf.keras.callbacks.TensorBoard(log_dir='rocs', write_graph=True, histogram_freq=1, )
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_1', input_dim=X_train.shape[1]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', name='layer_2'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_3'))
model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                             beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                                             gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(user_names.keys()), activation='softmax', name='output_layer'))
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
earlystopping = callbacks.EarlyStopping(monitor="val_accuracy",
                                        mode="max", patience=7,
                                        restore_best_weights=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
history = model.fit(tf.expand_dims(X_train, axis=-1), y_train, validation_data=(X_valid, y_valid), epochs=150,
                    batch_size=128, callbacks=[earlystopping, logger])

pred = model.predict(X_test)

title = f"Some extension of Receiver operating characteristic to multiclass using {program_n_gram_size}-graphs."
file_title = f"{len(user_names.keys())}_{program_n_gram_size}_gram_{features_cols}feaures"
plot_result_nn(history, "loss")
plot_result_nn(history, "accuracy")

draw_roc_curve(y_test, pred, plot_title=title, classes=user_names, file_title="rocs\\" + file_title)

