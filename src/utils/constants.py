DEFAULT_DATA = "/home/kaariaa3/mscthesis/data/complete_dataset.csv"

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

PIPE_RETURN_FULL_TEXT = False

PIPE_MAX_NEW_TOKENS = 1000

MODEL_TEMPERATURE = 0.3

DEFAULT_JUDGE_RESULT = {
    "themeCorrect": "no",
    "topicCorrect": "no",
    "conceptCorrect": "no",
    "explanation": "PARSE ERROR",
}

DEFAULT_AUGMENT_RESULT = {
    "augmentedProblemDescription": "PARSE ERROR",
    "augmentedExampleSolution": "PARSE ERROR",
}

ERROR_RESULT = {"Error": "PARSE ERROR"}

BATCH_SIZE = 4
