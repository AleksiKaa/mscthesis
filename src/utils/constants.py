DEFAULT_DATA = "/home/kaariaa3/mscthesis/data/complete_dataset.csv"

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

PIPE_RETURN_FULL_TEXT = False

PIPE_MAX_NEW_TOKENS = 500

MODEL_TEMPERATURE = 0.3

DEFAULT_JUDGE_RESULT = {
    "themeCorrect": "no",
    "topicCorrect": "no",
    "conceptCorrect": "no",
    "explanation": "PARSE ERROR",
}

DEFAULT_AUGMENT_RESULT = {
    "AugmentedDescription": "PARSE ERROR",
    "AugmentedSolution": "PARSE ERROR",
}

ERROR_RESULT = {"Error": "PARSE ERROR"}

BATCH_SIZE = 8
