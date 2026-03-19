import pandas as pd

def calculate_accuracy(df, predictions):

    # Ground truth columns
    gt_theme = df['The exercise description matched the selected theme (Yes/No)']
    gt_topic = df['The exercise description matched the selected topic (Yes/No)']
    gt_concept = df['Included concepts that were too advanced (Yes/No)']

    # Normalize values
    def normalize(series):
        return series.astype(str).str.strip('"').str.lower()

    gt_theme = normalize(gt_theme)
    gt_topic = normalize(gt_topic)
    gt_concept = normalize(gt_concept)

    pred_theme = normalize(pd.Series(predictions['themeCorrect']))
    pred_topic = normalize(pd.Series(predictions['topicCorrect']))
    pred_concept = normalize(pd.Series(predictions['usesAdditionalConcepts']))
    
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