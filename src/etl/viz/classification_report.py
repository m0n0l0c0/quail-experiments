import json
import argparse


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--metrics", type=str, required=True,
        help="Classification data to print report"
    )
    parser.add_argument(
        "-d", "--digits", type=int, required=False, default=2,
        help="Number of digits to round floats"
    )
    parser.add_argument(
        "-t", "--tabulated", action="store_true",
        help="Whether to print data tabulated or just plain space-formatted"
    )
    return parser.parse_args()


# extracted from:
# https://github.com/scikit-learn/scikit-learn/
# blob/0fb307bf3/sklearn/metrics/_classification.py#L1825
def classification_report(data_dict, digits=2, tabulated=False):
    """Build a text report showing the main classification metrics.
    Read more in the :ref:`User Guide <classification_report>`.
    Parameters
    ----------
    report : string
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy otherwise.
        See also :func:`precision_recall_fscore_support` for more details
        on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    """

    non_label_keys = ["accuracy", "macro avg", "weighted avg"]
    y_type = "binary"

    target_names = [
        "%s" % key for key in data_dict.keys() if key not in non_label_keys
    ]

    # labelled micro average
    micro_is_accuracy = (y_type == "multiclass" or y_type == "binary")

    headers = ["precision", "recall", "f1-score", "support"]
    p = [data_dict[lab][headers[0]] for lab in target_names]
    r = [data_dict[lab][headers[1]] for lab in target_names]
    f1 = [data_dict[lab][headers[2]] for lab in target_names]
    s = [data_dict[lab][headers[3]] for lab in target_names]

    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    if not tabulated:
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        row_fmt_accuracy = "{:>{width}s} " + " {:>9.{digits}}" * 2 \
            + " {:>9.{digits}f}" + " {:>9}\n"
    else:
        head_fmt = "\t{}" * len(headers) + "\n"
        report = head_fmt.format(*headers)
        row_fmt = "{}" + "\t{:>9.{digits}f}" * 3 + "\t{}\n"
        row_fmt_accuracy = "{} " + "\t{:>9.{digits}}" * 2 \
            + "\t{:>9.{digits}f}" + "\t{}\n"

    for row in rows:
        if not tabulated:
            report += row_fmt.format(*row, width=width, digits=digits)
            report += "\n"
        else:
            report += row_fmt.format(*row, digits=digits)

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        if line_heading == "accuracy":
            avg = [data_dict[line_heading], sum(s)]
            if not tabulated:
                report += row_fmt_accuracy.format(
                    line_heading, "", "",
                    *avg, width=width, digits=digits
                )
            else:
                report += row_fmt_accuracy.format(
                    line_heading, "", "",
                    *avg, digits=digits
                )
        else:
            avg = list(data_dict[line_heading].values())
            if not tabulated:
                report += row_fmt.format(
                    line_heading, *avg, width=width, digits=digits
                )
            else:
                report += row_fmt.format(line_heading, *avg, digits=digits)

    return report


if __name__ == '__main__':
    args = parse_flags()
    data = json.load(open(args.metrics))
    print(classification_report(
        data, digits=args.digits, tabulated=args.tabulated
    ))
