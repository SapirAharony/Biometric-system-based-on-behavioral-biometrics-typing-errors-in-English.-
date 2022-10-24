from sklearn import svm
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from create_model import cols, X, y, user_names
import operator
from itertools import cycle
import matplotlib

matplotlib.use('Agg')
from os import path, makedirs


def get_results(classif, X_tr, y_tr, X_tst, y_tst):
    classifier = classif.fit(X_tr, y_tr)
    predictions = classifier.predict(X_tst)
    plot_multiclass_roc(classifier, X_tst, y_tst, len(user_names.keys()))
    if hasattr(classifier, 'predict_proba'):
        roc_auc = roc_auc_score(y_tst, classifier.predict_proba(X_tst), multi_class='ovr')
        return (roc_auc, round(accuracy_score(predictions, y_tst)*100, 2))
    else: return round(accuracy_score(predictions, y_tst)*100, 2)






def plot_multiclass_roc(classifier, X_tst, y_tst, num_of_classes, dest_driectory= "C:\\Users\\user\\PycharmProjects\\bio_system\\rocs"):
    fpr, tpr, roc_auc = {}, {}, {}
    y_tst = label_binarize(y_tst, classes=[_ for _ in range(num_of_classes)])
    print(X_tst.shape)
    for i in range(num_of_classes):
        if hasattr(classifier,'predict_proba'):
            fpr[i], tpr[i], _ = roc_curve(y_tst[:, i], classifier.predict_proba(X_tst)[:, i])
        elif hasattr(classifier, 'decision_function'):
            fpr[i], tpr[i], _ = roc_curve(y_tst[:, i], classifier.decision_function(X_tst)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_of_classes)]))
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    # Plot of a ROC curve for a specific class
    plt.close()
    plt.figure(figsize=(7.5, 5))
    plt.rcParams.update({'font.size': 8})
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    for i in range(num_of_classes):
        plt.plot(fpr[i], tpr[i], next(linecycler), label="{} ROC curve (area = ' {})".format([name for name, id in user_names.items() if id == i][0], roc_auc[i]))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {classifier}')
        plt.legend(loc="lower right")
    print(dest_driectory + '\\' + str(classifier) + '\\' + str(X_tst.shape[1]) + '.png')
    plt.savefig(dest_driectory + '\\' + str(classifier) + '\\' + str(X_tst.shape[1]) + '.png')
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
print(features)
# rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
# poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
# poly_pred = poly.predict(X_test)
# rbf_pred = rbf.predict(X_test)
# svc_poly.append(accuracy_score(y_test, poly_pred))

# rbf_accuracy = accuracy_score(y_test, rbf_pred)


# from sklearn.naive_bayes import GaussianNB
#
# gnb = GaussianNB().fit(X_train, y_train)
# gnb_predictions = gnb.predict(X_test)
# # accuracy on X_test
# gnb_accuracy = accuracy_score(X_test, y_test)
#
#
#
#
#
#
# # training a KNN classifier
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
# # accuracy on X_test
# knn_accuracy = knn.score(X_test, y_test)
# knn_predictions = knn.predict(X_test)
#
#
#
# # training a KNN classifier
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
# # accuracy on X_test
# knn_accuracy = knn.score(X_test, y_test)
# knn_prediction = knn.predict(X_test)


'''


result = confusion_matrix(y_test, knn_predictions)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, knn_predictions)
print("Classification Report:")
print(result1)
print(12*'\n*')


result = confusion_matrix(y_test, gnb_predictions)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, gnb_predictions)
print("Classification Report:")
print(result1)
print('\n')

result = confusion_matrix(y_test, gnb_predictions)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, gnb_predictions)
print("Classification Report:")
print(result1)
print('\n')

result = confusion_matrix(y_test, rbf_pred)
print("Confusion Matrix:")
print(result)

result1 = classification_report(y_test, rbf_pred)
print("Classification Report:")
print(result1)
print('\n')
print(result)
result1 = classification_report(y_test, poly_pred)
'''
