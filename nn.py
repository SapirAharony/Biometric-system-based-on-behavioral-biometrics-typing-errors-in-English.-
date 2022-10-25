import keras.utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from keras import callbacks
from create_model import X, y, cols, user_names, n_gram_size


def plot_result_nn(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


scaler = StandardScaler()

data = np.hstack((X, np.reshape(y, (-1, 1))))

num_of_features = 10
if num_of_features > len(cols):
    num_of_features = len(cols)

selector = SelectKBest(f_classif, k=num_of_features)
X = selector.fit_transform(X, y)

# f_score_column_indexes = (-selector.scores_).argsort()[:num_of_features]  # choosen featuers indexes
# print(f_score_column_indexes)
# choosen_cols = [cols[k] for k in f_score_column_indexes]
# print(choosen_cols)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.66, random_state=0)
y_test = keras.utils.to_categorical(y_test, num_classes=len(user_names.keys()))
y_train = keras.utils.to_categorical(y_train, num_classes=len(user_names.keys()))
y_valid = keras.utils.to_categorical(y_valid, num_classes=len(user_names.keys()))
logger = tf.keras.callbacks.TensorBoard(log_dir='rocs', write_graph=True, histogram_freq=1, )
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_1', input_dim=num_of_features))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_2'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation='relu', name='layer_3'))
model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005,
                                             beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                                             gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
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
history = model.fit(tf.expand_dims(X_train, axis=-1), y_train, validation_data=(X_valid, y_valid), epochs=50,
                    batch_size=128, callbacks=[earlystopping, logger])


from sklearn.metrics import roc_curve, auc
from itertools import cycle
pred = model.predict(X_test)
lw = 2
n_classes = len(user_names.keys())
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(_)


fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC
mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle='dotted',
    linewidth=5,
)

# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format([name for name, id in user_names.items() if id == i][0], roc_auc[i]),
    )


plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Some extension of Receiver operating characteristic to multiclass using {n_gram_size}-graphs.")
plt.legend(loc="lower right")
plt.show()


#1360