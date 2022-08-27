# ML

In multi-label classification tasks, knowing what pairs of labels are often swapped (e.g. the ground truth is labeled "A" and predicted is labeled "B").
Here, I propose a synthetic confusion matrix that computes such model evaluation. With such a function, one can easily see what labels are often swapped.

Some projects have a similar aim but don't exactly achieve such a result:

* `sklearn.metrics.multilabel_confusion_matrix` computes a 2x2 confusion matrix for each class which doesn't convey this information of confusion among different classes

## Usage

## Details

## Next steps

* support other input types : `ndarray`, sets (not one-encoded)

*
