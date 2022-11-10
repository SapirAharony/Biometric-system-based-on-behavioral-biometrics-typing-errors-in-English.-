from create_model import cols
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np


def draw_roc_curve(y_test, y_score, classes, plot_title, file_title=None, print_classes=True):
    lw = 2
    n_classes = len(classes)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])


    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    if print_classes:
        colors = cycle(["aqua", "darkorange", "cornflowerblue", 'b', 'g', 'c', 'r', 'm', 'y', 'k'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(
                    [name for name, id in classes.items() if id == i][0], roc_auc[i]),
                # label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )


    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontdict={'size': 8})
    plt.ylabel("True Positive Rate", fontdict={'size': 8})
    plt.title(plot_title, fontdict={'size': 10})
    plt.legend(loc="lower right")
    plt.legend(loc="lower right", fontsize=8)
    if file_title is not None:
        plt.savefig(file_title)
    else:
        plt.show()

# def draw_roc_curve(y_test, pred, classes, plot_title, file_title):
#     lw = 2
#     n_classes = len(classes)
#     fpr, tpr, roc_auc = {}, {}, {}
#     for i in range(n_classes):
#         fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], pred[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     print(10 * '\n')
#     print(fpr)
#     print(10*'\n')
#     print(tpr)
#     print(10 * '\n')
#     fpr["micro"], tpr["micro"], thresholds = roc_curve(y_test.ravel(), pred.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#
#     # Plot all ROC curves
#     plt.clf()
#     plt.figure(figsize=(8, 6))
#     plt.plot(
#         fpr["micro"],
#         tpr["micro"],
#         label="average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#         color="deeppink",
#         linestyle='dotted',
#         linewidth=5, marker='.'
#     )
#
#     colors = cycle(["aqua", "darkorange", "cornflowerblue", 'r', 'g', 'b', 'y'])
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(
#             fpr[i],
#             tpr[i],
#             color=color,
#             lw=lw, marker='.',
#             label="ROC curve of class {0} (area = {1:0.2f})".format(
#                 [name for name, id in classes.items() if id == i][0], roc_auc[i]),
#         )
#
#     plt.plot([0, 1], [0, 1], "k--", lw=lw,  marker=".")
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate", fontdict=font_8)
#     plt.ylabel("True Positive Rate", fontdict=font_8)
#     plt.title(plot_title)
#     plt.legend(loc="lower right", fontsize=8)
#     plt.savefig(file_title)

# def plot_result_nn(history, item, file_title = None):
#     plt.clf()
#     plt.figure(figsize=(8, 6))
#
#
#
#     plt.plot(history.history[item], label=item)
#     plt.plot(history.history["val_" + item], label="val_" + item)
#     plt.xlabel("Epochs".clf()
#     plt.ylabel(item, 'FontSize', 9)
#     plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
#     plt.legend(font_size=10)
#     plt.grid()
#     if file_title:
#         plt.savefig(file_title)
#     else:
#         plt.show()

def plot_result_nn(history, file_title=None):
    fig, axs = plt.subplots(2, 1)
    x = [k + 1 for k in range(len(history.history["loss"]))]
    axs[0].plot(x, history.history["val_loss"], label="Validation data loss",  marker='.')
    axs[0].plot(x, history.history["loss"], label="Train data loss",  marker='.')
    axs[0].set_title('Train and validation data loss over epochs.', fontsize=10)
    axs[0].set_ylabel('Data loss', fontsize=8)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].plot(x, history.history["val_accuracy"], label="Validation accuracy",  marker='.')
    axs[1].plot(x, history.history["accuracy"], label="Train accuracy",  marker='.')
    axs[1].set_title('Train and validation accuracy over epochs.', fontsize=10)
    axs[1].set_ylabel('Accuracy', fontsize=8)
    axs[1].legend(loc="lower right", fontsize=8)


    for ax in axs.flat:
        ax.set(xlabel='Epochs')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    if file_title is not None:
        plt.savefig(file_title)
    else:
        plt.show()


def get_info_readme(list_of_features):
    infos = ''
    for id in list_of_features:
        infos = infos + '\n' + cols[id]
    return infos


def plot_confusion_metrics(y_test, y_pred, labels: list, display_labels: list, file_title=None):
    system_confusion_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1),
                                               labels=labels)
    cm_display = ConfusionMatrixDisplay(system_confusion_matrix, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(10,10))
    cm_display.plot(cmap="RdYlGn", ax=ax)
    if file_title is not None:
        plt.savefig(file_title)
    else:
        plt.show()



