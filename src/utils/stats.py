import pandas as pd

GT_COLS = [
    "The exercise description matched the selected theme (Yes/No)",
    "The exercise description matched the selected topic (Yes/No)",
    "Included concepts that were too advanced (Yes/No)"
]

PRED_COLS = ["themeCorrect", "topicCorrect", "usesAdditionalConcepts"]


def normalize(series):
    return series.astype(str).str.strip('"').str.lower()


def calculate_accuracy(df):

    gt_theme = normalize(df[GT_COLS[0]])
    gt_topic = normalize(df[GT_COLS[1]])
    gt_concept = normalize(df[GT_COLS[2]])

    pred_theme = normalize(df[PRED_COLS[0]])
    pred_topic = normalize(df[PRED_COLS[1]])
    pred_concept = normalize(df[PRED_COLS[2]])
    
    # Accuracy calculation
    theme_acc = (gt_theme == pred_theme)
    topic_acc = (gt_topic == pred_topic)
    concept_acc = (gt_concept == pred_concept)

    return {
        "theme_accuracy": theme_acc.mean(),
        "topic_accuracy": topic_acc.mean(),
        "concept_accuracy": concept_acc.mean(),
        "total_accuracy": (theme_acc & topic_acc & concept_acc).mean()
    }