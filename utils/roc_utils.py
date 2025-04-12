import numpy as np
import matplotlib.pyplot as plt

def compute_roc_curve(true_mask, pred_probs, thresholds):
    tpr_list = []
    fpr_list = []

    true_flat = true_mask.flatten()
    pred_flat = pred_probs.flatten()

    for thresh in thresholds:
        pred_bin = (pred_flat >= thresh).astype(int)

        TP = np.sum((pred_bin == 1) & (true_flat == 1))
        FP = np.sum((pred_bin == 1) & (true_flat == 0))
        FN = np.sum((pred_bin == 0) & (true_flat == 1))
        TN = np.sum((pred_bin == 0) & (true_flat == 0))

        TPR = TP / (TP + FN + 1e-6)
        FPR = FP / (FP + TN + 1e-6)

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return fpr_list, tpr_list

def plot_roc_curve(fpr, tpr, thresholds):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, marker='o')
    for i, th in enumerate(thresholds):
        plt.text(fpr[i], tpr[i], f"{th:.1f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.show()