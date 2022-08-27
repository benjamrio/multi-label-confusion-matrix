import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def compute_MLconfusion_matrix(y_true, y_pred):
    """Compute the multi-label confusion matrix between groud truth labels and predictions

    Arguments:
        y_true {list} -- list of ground truth label (one-hot encoded)
        y_pred {list} -- list of predictions (one-hot encoded). Should be the same "shape" as y_true.

    Returns:
        ndarray -- 2D array corresponding to the confusion matrix
    """
    assert len(y_true) == len(
        y_pred), f"Not the same amount of predictions ({len(y_pred)}) as ground truth labels ({len(y_true)})"

    assert all(map(lambda x: len(x) == len(y_true[0]), y_true)) and all(
        map(lambda x: len(x) == len(y_true[0]), y_pred)), "The multi-labels must be of same length"

    num_classes = len(y_true[0])
    cm = np.zeros((num_classes, num_classes))

    for true_labels, pred_labels in zip(y_true, y_pred):
        # label presnt in both ground truth and predictions
        right_preds = [k & l for k, l in zip(true_labels, pred_labels)]
        right_preds_indices = [i for i, x in enumerate(
            right_preds) if x == 1]  # their indices
        # label that are presetn in either ground truth or prediction
        diff_preds = [k - l for k, l in zip(true_labels, pred_labels)]
        # indices of ground truth label absent from prediction
        fn_indices = [i for i, x in enumerate(diff_preds) if x == 1]
        # indices of predictions that are not ground truth
        fp_indices = [i for i, x in enumerate(diff_preds) if x == -1]
        # for i in right_preds_indices: #perfect predictions
        #       conf[ 0, i, i] += 1
        if right_preds_indices == []:  # if pure confusion, do half the work
            for gt_idx in fn_indices:
                for p_idx in [i for i, x in enumerate(pred_labels) if x == 1]:
                    cm[gt_idx, p_idx] += 2  # score of 2 if pure cofusion
        else:
            for gt_idx in fn_indices:  # false negative : gt absent from preds
                for p_idx in [i for i, x in enumerate(pred_labels) if x == 1]:
                    cm[gt_idx, p_idx] += 1
            for p_idx in fp_indices:  # false positive : pred absent from gt
                for gt_idx in [i for i, x in enumerate(true_labels) if x == 1]:
                    cm[gt_idx, p_idx] += 1

    return cm


def plot_MLconfusion_matrix(cm, labels):
    """Quick plotting function for the multi-label confusion matrix

    Arguments:
        matrix {ndarray} -- multi-label 2D confusion matrix
        labels {list[str]} -- list of the labels. Should be in the right order.
    """
    g = sns.heatmap(cm, xticklabels=labels, yticklabels=labels)
    g.title.set_text("Mitigated confusion")
    g.set_ylabel("Actual value")
    g.set_xlabel("Predicted value")
    plt.show()
