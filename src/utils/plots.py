from textwrap import wrap

import numpy as np
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)


GT_COLS = [
    "The exercise description matched the selected theme (Yes/No)",
    "The exercise description matched the selected topic (Yes/No)",
    "Included concepts that were too advanced (Yes/No)",
]


PRED_COLS = ["themeCorrect", "topicCorrect", "usesAdditionalConcepts"]


LABELS = ["yes", "no"]

POS_LABELS = ["yes", "yes", "no"]


def normalize(series, pos_label=None):
    if pos_label is not None:
        return (
            series.astype(str)
            .str.strip('"')
            .str.lower()
            .map(lambda x: 1 if x == pos_label else 0)
        )

    try:
        pos_label = "no" if series.name == GT_COLS[2] else "yes"
    except:
        pos_label = "yes"

    return (
        series.astype(str)
        .str.strip('"')
        .str.lower()
        .map(lambda x: 1 if x == pos_label else 0)
    )


def wrap_text(text, num_chars=20):
    if isinstance(text, str):
        return "\n".join(wrap(text, num_chars))

    return ["\n".join(wrap(t, num_chars)) for t in text]


def calculate_accuracy(df):

    gt_theme = normalize(df[GT_COLS[0]])
    gt_topic = normalize(df[GT_COLS[1]])
    gt_concept = normalize(df[GT_COLS[2]])

    pred_theme = normalize(df[PRED_COLS[0]])
    pred_topic = normalize(df[PRED_COLS[1]])
    pred_concept = normalize(df[PRED_COLS[2]])

    # Accuracy calculation
    theme_acc = gt_theme == pred_theme
    topic_acc = gt_topic == pred_topic
    concept_acc = gt_concept == pred_concept

    return {
        "theme_accuracy": theme_acc.mean(),
        "topic_accuracy": topic_acc.mean(),
        "concept_accuracy": concept_acc.mean(),
        "total_accuracy": (theme_acc & topic_acc & concept_acc).mean(),
    }


def plot_confusion_matrices(df, axes, labels=LABELS, cols1=GT_COLS, cols2=PRED_COLS):
    fig = axes[0].figure

    for i, (ax, col1, col2) in enumerate(zip(axes, cols1, cols2)):

        y_true = normalize(df[col1], POS_LABELS[i])
        y_pred = normalize(df[col2], POS_LABELS[i])
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        cm = cm / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set_title(wrap_text(f"{col1.upper()} Confusion", 20))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    return fig, axes


def plot_accuracy(df, ax, cols1=GT_COLS, cols2=PRED_COLS):
    fig = ax.figure

    acc = [
        (normalize(df[col1]) == normalize(df[col2])).mean()
        for col1, col2 in zip(cols1, cols2)
    ]

    #    acc = acc + [reduce(lambda x, y: x & y, acc)]

    sns.barplot(x=cols1, y=acc, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Label")
    ax.set_xticklabels([wrap_text(col1, 20) for col1 in cols1])

    return fig, ax


def plot_distributions(df, axes, labels=LABELS, cols1=GT_COLS, cols2=PRED_COLS):
    fig = axes[0].figure

    for i, (col1, col2) in enumerate(zip(cols1, cols2)):
        sns.countplot(x=df[col1], order=labels, ax=axes[i])
        axes[i].set_title(wrap_text(f"{col1.upper()} True", 20))

        sns.countplot(x=df[col2], order=labels, ax=axes[i])
        axes[i].set_title(wrap_text(f"{col1.upper()} Predicted", 20))

    return fig, axes


def plot_metric_heatmap(df, ax, cols1=GT_COLS, cols2=PRED_COLS):
    metrics = []
    for i, (c1, c2) in enumerate(zip(cols1, cols2)):
        y_true = normalize(df[c1], POS_LABELS[i])
        y_pred = normalize(df[c2], POS_LABELS[i])

        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        metrics.append([p, r, f1])

    metrics = np.array(metrics)

    sns.heatmap(
        metrics,
        annot=True,
        square=True,
        xticklabels=["Precision", "Recall", "F1"],
        yticklabels=wrap_text(GT_COLS, 20),
        cmap="Blues",
        ax=ax,
    )

    ax.set_title("Per-label Metrics")


def plot_precision_recall_curves(df, ax, cols1=GT_COLS, cols2=PRED_COLS):
    for i, (c1, c2) in enumerate(zip(cols1, cols2)):
        y_true = normalize(df[c1], POS_LABELS[i])
        y_pred = normalize(df[c2], POS_LABELS[i])

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        ax.plot(recall, precision, label=wrap_text(f"{c1} (AP={ap:.2f})"))

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend()
    ax.grid()


def plot_cooccurrence(df, ax, cols=GT_COLS):
    y = df[cols].apply(normalize, axis=1).values
    co_occurrence = np.dot(y.T, y)
    # co_occurrence = co_occurrence / co_occurrence.sum(axis=1, keepdims=True)

    sns.heatmap(
        co_occurrence,
        annot=True,
        fmt="d",
        xticklabels=wrap_text(cols),
        yticklabels=wrap_text(cols),
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Label Co-occurrence")


def plot_error_distribution(df, ax, cols1=GT_COLS, cols2=PRED_COLS):
    y_true = df[cols1].apply(normalize, axis=1).values
    y_pred = df[cols2].apply(normalize, axis=1).values

    errors_per_sample = np.sum(y_true != y_pred, axis=1)

    ax.hist(errors_per_sample, bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Number of Incorrect Labels")
    ax.set_ylabel("Count")
    ax.set_title("Errors per Sample")
