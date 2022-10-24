from sklearn import svm
from sklearn.metrics import auc, roc_auc_score, roc_curve, plot_roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from create_model import cols, X, y, user_names
import operator
from itertools import cycle
import matplotlib
from os import path, makedirs


matplotlib.use('Agg')



def get_results(classif, X_tr, y_tr, X_tst, y_tst):
    classifier = classif.fit(X_tr, y_tr)
    predictions = classifier.predict(X_tst)
    acc = round(accuracy_score(predictions, y_tst)*100, 2)
    plot_multiclass_roc(classifier, X_tst, y_tst,len(user_names.keys()), acc)
    if hasattr(classifier, 'predict_proba'):
        roc_auc = roc_auc_score(y_tst, classifier.predict_proba(X_tst), multi_class='ovr')
        return (roc_auc, acc)
    else: return acc


def write_dict_to_file(dest_file: str, title:str, dictionary: dict):
    with open(dest_file, 'a+') as f:
        f.write(title + str(dictionary) + '\n')
        # for k in dictionary:
        #     f.write(str(k) + '->' + str(dictionary[k]) + '\n')

# def multiplot():
#     plt.figure(figsize=(12, 8))
#     bins = [i / 20 for i in range(20)] + [1]
#     classes = model_multiclass.classes_
#     roc_auc_ovr = {}
#     for i in range(len(classes)):
#         # Gets the class
#         c = classes[i]
#
#         # Prepares an auxiliar dataframe to help with the plots
#         df_aux = X_test.copy()
#         df_aux['class'] = [1 if y == c else 0 for y in y_test]
#         df_aux['prob'] = y_proba[:, i]
#         df_aux = df_aux.reset_index(drop=True)
#
#         # Plots the probability distribution for the class and the rest
#         ax = plt.subplot(2, 3, i + 1)
#         sns.histplot(x="prob", data=df_aux, hue='class', color='b', ax=ax, bins=bins)
#         ax.set_title(c)
#         ax.legend([f"Class: {c}", "Rest"])
#         ax.set_xlabel(f"P(x = {c})")
#
#         # Calculates the ROC Coordinates and plots the ROC Curves
#         ax_bottom = plt.subplot(2, 3, i + 4)
#         tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
#         plot_roc_curve(tpr, fpr, scatter=False, ax=ax_bottom)
#         ax_bottom.set_title("ROC Curve OvR")
#
#         # Calculates the ROC AUC OvR
#         roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])
#     plt.tight_layout()


def plot_multiclass_roc(classifier, X_tst, y_tst, num_of_classes, acc, dest_driectory= f"C:\\Users\\user\\PycharmProjects\\bio_system\\rocs\\{len(cols)-1}"):
    fpr, tpr, roc_auc = {}, {}, {}
    y_tst = label_binarize(y_tst, classes=[_ for _ in range(num_of_classes)])
    for i in range(num_of_classes):
        if hasattr(classifier,'predict_proba'):
            fpr[i], tpr[i], _ = roc_curve(y_tst[:, i], classifier.predict_proba(X_tst)[:, i])
        elif hasattr(classifier, 'decision_function'):
            fpr[i], tpr[i], _ = roc_curve(y_tst[:, i], classifier.decision_function(X_tst)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    # Plot of a ROC curve for a specific class
    plt.close()
    plt.figure(figsize=(7.5, 5))
    plt.rcParams.update({'font.size': 8})
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    textstr = f'accuracy{acc}'
    # plt.plot(all_fpr, all_tpr, label='Label')
    plt.text(0.05, 0.95, textstr, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    for i in range(num_of_classes):
        plt.plot(fpr[i], tpr[i], next(linecycler), label="{} ROC curve (area = ' {})".format([name for name, id in user_names.items() if id == i][0], roc_auc[i]))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {classifier}')
        plt.legend(loc="lower right")

    if not path.exists(dest_driectory + '\\' + str(classifier)):
        makedirs(dest_driectory + '\\' + str(classifier))
    plt.show()
    # plt.savefig(dest_driectory + '\\' + str(classifier) + '\\' + str(X_tst.shape[1]) + '.png')
    return plt


# split data --> create a sample for simulation
svc_linear, svc_poly, svc_rbf, gnb, knn_three, knn_four, knn_five, knn_six, knn_seven, knn_eight, knn_nine, features = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
# comparision
for k in range(2, len(cols)):
    num_of_features = k
    X_for_model = X
    # choose features
    selector = SelectKBest(f_classif, k=num_of_features)
    X_for_model = selector.fit_transform(X_for_model, y)
    f_score_column_indexes = (-selector.scores_).argsort()[:num_of_features]  # choosen featuers indexes
    chosen_cols = [cols[k] for k in f_score_column_indexes]
    # split data

    X_train, X_test, y_train, y_test = train_test_split(X_for_model, y, test_size=0.3, random_state=0)
    svc_poly[k] = get_results(svm.SVC(kernel='poly', degree=3, C=1, probability=True), X_train, y_train, X_test, y_test)
    svc_rbf[k] = get_results(svm.SVC(kernel='rbf', gamma=0.5, C=0.1, probability=True), X_train, y_train, X_test, y_test)
    svc_linear[k] = get_results(svm.LinearSVC(multi_class='crammer_singer', C=1), X_train, y_train, X_test, y_test)
    gnb[k] = get_results(GaussianNB(), X_train, y_train, X_test, y_test)
    knn_nine[k] = get_results(KNeighborsClassifier(n_neighbors=9), X_train, y_train, X_test, y_test)
    knn_eight[k] = get_results(KNeighborsClassifier(n_neighbors=8), X_train, y_train, X_test, y_test)
    knn_seven[k] = get_results(KNeighborsClassifier(n_neighbors=7), X_train, y_train, X_test, y_test)
    knn_six[k] = get_results(KNeighborsClassifier(n_neighbors=6), X_train, y_train, X_test, y_test)
    knn_five[k] = get_results(KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_test, y_test)
    knn_four[k] = get_results(KNeighborsClassifier(n_neighbors=4), X_train, y_train, X_test, y_test)
    knn_three[k] = get_results(KNeighborsClassifier(n_neighbors=3), X_train, y_train, X_test, y_test)
    features[k] = chosen_cols

svc_linear = dict(sorted(svc_linear.items(), key=operator.itemgetter(1), reverse=True))
svc_poly = dict(sorted(svc_poly.items(), key=operator.itemgetter(1), reverse=True))
svc_rbf = dict(sorted(svc_rbf.items(), key=operator.itemgetter(1), reverse=True))
knn_three = dict(sorted(knn_three.items(), key=operator.itemgetter(1), reverse=True))
knn_four = dict(sorted(knn_four.items(), key=operator.itemgetter(1), reverse=True))
knn_five = dict(sorted(knn_five.items(), key=operator.itemgetter(1), reverse=True))
knn_six = dict(sorted(knn_six.items(), key=operator.itemgetter(1), reverse=True))
knn_seven = dict(sorted(knn_seven.items(), key=operator.itemgetter(1), reverse=True))
knn_eight = dict(sorted(knn_eight.items(), key=operator.itemgetter(1), reverse=True))
knn_nine = dict(sorted(knn_nine.items(), key=operator.itemgetter(1), reverse=True))
gnb = dict(sorted(gnb.items(), key=operator.itemgetter(1), reverse=True))

print('SVC_linear', svc_linear)
print('SVC_poly', svc_poly)
print('svc_rbf', svc_rbf)
print('knn_three', knn_three)
print('knn_four', knn_four)
print('knn_five', knn_five)
print('knn_six', knn_six)
print('knn_seven', knn_seven)
print('knn_eight', knn_eight)
print('knn_nine', knn_nine)
print('gnb', gnb)
features['all'] = cols
print(features)

write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "features", features)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "svc_linear", svc_linear)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "svc_poly", svc_poly)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "svc_rbf", svc_rbf)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "knn_three", knn_three)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "knn_four", knn_four)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "knn_five", knn_five)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "knn_six", knn_six)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "knn_seven", knn_seven)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "knn_eight", knn_eight)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "knn_nine", knn_nine)
write_dict_to_file(f"/rocs/{len(cols)-1}/features.txt", "gnb", gnb)

