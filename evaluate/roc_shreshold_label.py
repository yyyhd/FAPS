import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd


def gene_roc_curve(actual_val_list, score_list, pos_label):
    fpr, tpr, threshold = roc_curve(actual_val_list, score_list, pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # set the size of letter
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    for i in range(0,fpr.shape[0],10):
        x = fpr[i]
        y = tpr[i]
        text = threshold[i]
        plt.text(x, y + 0.01, '%.4f' % text)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


path = ''
df = pd.read_csv(path)
actual_val_list = list(df['gt1'])
score_list = list(df['prob1'])
pos_label = 1
gene_roc_curve(actual_val_list, score_list, pos_label)
