from tensorflow import expand_dims, keras
from sklearn.model_selection import train_test_split
from keras import callbacks
from create_model import X, y, user_names, program_n_gram_size, program_is_ver_sim, features_cols
from draw_results import draw_roc_curve, plot_result_nn, get_info_readme, plot_confusion_metrics
from os import path, makedirs
from tensorflow import config
import numpy as np

print(config.list_physical_devices('GPU'))

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
print(X_test.shape[0] / X_train.shape[0] * 100)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)
print(y_test.shape[0] / y_train.shape[0] * 100)

logger = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, )
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', name='layer_1', input_dim=X_train.shape[1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation='relu', name='layer_2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu', name='layer_3'))
model.add(keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                          beta_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                                          gamma_initializer=keras.initializers.Constant(value=0.9)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(user_names.keys()), activation='softmax', name='output_layer'))
sgd = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
earlystopping = callbacks.EarlyStopping(monitor="val_accuracy",
                                        mode="max", patience=7,
                                        restore_best_weights=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
history = model.fit(expand_dims(X_train, axis=-1), y_train, validation_data=(X_valid, y_valid), epochs=250,
                    batch_size=128, callbacks=[earlystopping, logger])

pred = model.predict(X_test)
threshold = 0.5
indexes_to_delete = []
pred_rejected = []
y_deleted = []
if len(pred) == len(y_test):
    for i in range(len(pred)):
        if pred[i].max() < threshold:
            pred_rejected.append(pred[i])
            y_deleted.append(y_test[i])
            indexes_to_delete.append(i)
for i in range(len(indexes_to_delete)):
    y_test = np.delete(y_test, indexes_to_delete[i] - i, 0)
    pred = np.delete(pred, indexes_to_delete[i] - i, 0)
del indexes_to_delete

title = f"Receiver operating characteristic using {program_n_gram_size}-graphs."
file_title = f"{len(user_names.keys())}_{program_n_gram_size}_gram_{len(features_cols)}features"
directory = f"graphs\\{len(user_names.keys())}_{program_n_gram_size}_gram_{len(features_cols)}features"

if not path.exists(directory):
    makedirs(directory)

plot_result_nn(history, file_title=directory + '\\data_loss' + file_title)
draw_roc_curve(y_test, pred, plot_title=title, classes=user_names, file_title=directory + '\\roc_curve_' + file_title)
plot_confusion_metrics(y_test, pred, list(sorted(user_names.values())),
                       list(dict(sorted(user_names.items(), key=lambda item: item[1])).keys()))

with open(directory + '\\readme.txt', 'a+') as f:
    f.truncate(0)
    f.write("Features: " + get_info_readme(features_cols))
    f.write(3 * "\n")
    f.write("X_train: " + str(X_train.shape[0]) + "(" + str(
        X_train.shape[0] / (X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])))
    f.write("\nX_valid: " + str(X_valid.shape[0]) + "(" + str(
        X_valid.shape[0] / (X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])))
    f.write("\nX_test: " + str(X_test.shape[0]) + "(" + str(
        X_test.shape[0] / (X_train.shape[0] + X_valid.shape[0] + X_test.shape[0])))

with open(directory + '\\pred.txt', 'a+') as f:
    f.truncate(0)
    for prediction in pred:
        f.write(str(prediction) + ",")

with open(directory + '\\y_test.txt', 'a+') as f:
    f.truncate(0)
    for y in y_test:
        f.write(str(y) + ",")

model.save(directory + '\\my_model')
