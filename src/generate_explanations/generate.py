print("Importing libraries...")
import pandas as pd
import numpy as np
from transformers import pipeline

DATAPATH = "/home/kaariaa3/mscthesis/data/out.csv"

SYSTEM_PROMPT = """I want you to act as a programming teacher for an in-
troductory Dart course. Your students are programming
novices. I will provide some coding example exercises,
and it will be your job to critique them. Your responses 
should be written in simple English. Do not cite music 
lyrics or books. Do not include any greetings, be concise. 
Do not mention trigger words associated with mental or 
physical disorders, for example, weight loss or diet."""

CRITIQUE_TEMPLATE = """Please generate an explanation why the following exercise description and Dart
code are not faithful to the provided topic, theme or concept. Your response
should be a string literal.

Input example is not faithful to:
$Topic$Theme$Concept
Exercise description: $problemDescription

Code: $exampleSolution"""


USED_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Model parameters
params = {
    "model": USED_MODEL,
    "task": "text-generation",
    "device_map": "auto",
    "max_new_tokens": 1000,
    "temperature": 0.7,
}
print(f"Model parameters: {params}")

print("Initializing pipeline...")
# Initialize the pipeline
pipe = pipeline(**params)

print("Reading input data...")
# Read CSV
df = pd.read_csv(DATAPATH, sep=";")

# Get rows with "no" labels
label_col = df.columns[-1]
cond = df[label_col] == "no"
wrongs = df[cond]

system_message = {"role": "system", "content": SYSTEM_PROMPT}

explanations = {}

print("Creating prompts...")
# Generate prompts
for idx, row in wrongs.iterrows():
    title, topic, theme, concept, problem_description, example_solution, *evals = row

    prompt = CRITIQUE_TEMPLATE

    # Map from index to template label
    label = {0: "$Theme", 1: "$Topic", 2: "$Concept"}

    # Map template label to content
    rep = {"$Theme": theme, "$Topic": topic, "$Concept": concept}

    # Replace template labels with content
    for i, evaluation in enumerate(evals):
        key = label.get(i, "")
        rep_str = rep.get(key, "")
        if key == "" or rep_str == "":
            continue

        if evaluation == "yes":
            prompt = prompt.replace(key, "")
            continue

        prompt = prompt.replace(key, f"{key[1:]}: " + rep_str + "\n")

    prompt = prompt.replace("$problemDescription", problem_description).replace(
        "$exampleSolution", example_solution
    )

    print("\n" + prompt + "\n")

    # Generate text and print the response
    print("Generating response...\n")
    response = pipe(
        [system_message, {"role": "user", "content": prompt}], return_full_text=False
    )

    response_text = response[0]["generated_text"]
    explanations[idx] = response_text
    print(response_text)


df = df.assign(Explanation=explanations).replace({np.nan: ""})

df.to_csv("out.csv", sep=";", index=False)
