print("Importing libraries...")
import pandas as pd
import numpy as np
from transformers import pipeline
import json

DATAPATH = "/home/kaariaa3/mscthesis/data/out.csv"

SYSTEM_PROMPT = """I want you to act as a programming teacher for an in-
troductory Dart course. Your students are programming
novices. I will provide some coding example exercises,
and it will be your job to critique them. Your responses 
should be written in simple English. Do not cite music 
lyrics or books. Do not include any greetings, be concise. 
Do not mention trigger words associated with mental or 
physical disorders, for example, weight loss or diet."""

CRITIQUE_TEMPLATE = """You are evaluating a programming exercise.

Intended theme: $THEME$
Intended topic: $TOPIC$
Intended programming concept: $CONCEPT$

Exercise description:
$TEXT$

Example solution:
$CODE$

Analyze the exercise step by step.

Step 1 — Identify what the exercise is actually about.
Step 2 — Check alignment with the theme.
Step 3 — Check alignment with the topic.
Step 4 — Check alignment with the programming concept.
Step 5 — Provide a final explanation of failures.

Return a JSON of form 
{
    "Correct" : "yes" or "no"
    "Explanation": reasoning
}
"""

USED_MODEL = "Qwen/Qwen2.5-14B-Instruct"
# USED_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Model parameters
params = {
    "model": USED_MODEL,
    "device_map": 0,  # Force GPU
    "max_new_tokens": 500,
    "temperature": 0.3,
}
print(f"Model parameters: {params}")

print("Initializing pipeline...")
# Initialize the pipeline
pipe = pipeline("text-generation", **params)


# Functions
def make_prompt(row):
    _, topic, theme, concept, problem_description, example_solution, *_ = row
    return (
        CRITIQUE_TEMPLATE.replace("$THEME$", theme)
        .replace("$TOPIC$", topic)
        .replace("$CONCEPT$", concept)
        .replace("$TEXT$", problem_description)
        .replace("$CODE$", example_solution)
    )


def run_model(data):
    response = pipe(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data["prompt"]},
        ],
        return_full_text=False,
        max_new_tokens=500,
    )

    result = response[0]["generated_text"]
    result_as_json = json.loads(result)

    data["Correct"] = result_as_json["Correct"]
    data["Explanation"] = result_as_json["Explanation"]

    return data


print("Reading input data...")
# Read CSV
df = pd.read_csv(DATAPATH, sep=";")

# Get rows with "no" labels
# label_col = df.columns[-1]
# cond = df[label_col] == "no"
# wrongs = df[cond]

# eval_df = wrongs
eval_df = df

print("Creating prompts...")
eval_df["prompt"] = eval_df.apply(make_prompt, axis=1)

print("Generating responses...\n")
eval_df = eval_df.apply(run_model, axis=1)

eval_df.to_csv("out.csv", sep=";", index=False)
