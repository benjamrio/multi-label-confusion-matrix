from main import compute_MLconfusion_matrix


def main():
    a = ([[1, 0, 1]], [[1, 0, 0]])
    b = ([[1, 0, 0]], [[1, 0, 1]])
    c = ([[1, 0, 0]], [[1, 0, 0]])
    d = ([[1, 0, 0]], [[0, 1, 1]])

    print(f"""Example of ML confusion matrix:\n
        >> Less preds than ground truths:\n
    Input: {a}\n
    Conf matrix:\n {compute_MLconfusion_matrix(*a)}\n
        >> More preds than ground truths:\n
    Input: {b}\n
    Conf matrix:\n {compute_MLconfusion_matrix(*b)}\n
        >> Perfect preds:\n
    Input: {c}\n
    Conf matrix:\n {compute_MLconfusion_matrix(*c)}\n
        >> Bad preds:\n
    Input: {d}\n
    Conf matrix:\n {compute_MLconfusion_matrix(*d)}\n
    """)


if __name__ == "__main__":
    main()
