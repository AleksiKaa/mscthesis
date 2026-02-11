# Imports

print("imports")
import pandas as pd
import numpy as np
import os
from dotenv import dotenv_values
from transformers import pipeline

print("config")
config = dotenv_values(".env")

SYSTEM_PROMPT = """
I want you to act as a programming teacher for an in-
troductory Dart course. Your students are programming
novices. I will provide some coding example exercises,
and it will be your job to critique them. Your responses 
should be written in simple English. Do not cite music 
lyrics or books. Do not include any greetings, be concise. 
Do not mention trigger words associated with mental or 
physical disorders, for example, weight loss or diet.
"""

CRITIQUE_TEMPLATE = """
Please generate an explanation why the following exercise description and Dart
code are not faithful to the provided topic, theme or concept. Your response
should be a JSON string.

$Topic$Theme$Concept
Exercise description: $problemDescription

Code: $exampleSolution
"""

print("pipe")
# Initialize the pipeline
pipe = pipeline(
  "text-generation", # Task type
  model="mistralai/Mistral-7B-Instruct-v0.3", # Model name
  device_map="auto", # Let the pipeline automatically select best available device
  max_new_tokens=1000
)

print("df")
# Read CSV
df = pd.read_csv(config["DATAPATH"], sep=";")

# Get rows with "no" labels
label_col = df.columns[-1]
cond = df[label_col] == "no"
wrongs = df[cond]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
]


print("prompts")
# Generate prompts
for idx, row in wrongs.iterrows():
    _, topic, theme, concept, problem_description, example_solution, *evals = row

    p = CRITIQUE_TEMPLATE

    # Map from index to template label
    idx = {
        1: "$Theme",
        2: "$Topic",
        3: "$Concept"
    }

    # Map template label to content
    rep = {
        "$Theme": theme,
        "$Topic": topic,
        "$Concept": concept
    }

    # Replace template labels with content
    for i, evaluation in enumerate(evals):
        key = idx.get(i, "")
        rep_str = rep.get(key, "")
        if key == "" or rep_str == "":
            continue

        if evaluation == "yes":
            p = p.replace(key, "")
            continue

        p = p.replace(key, f"{key[1:]}: " +  rep_str + "\n")

    p = p \
        .replace("$problemDescription", problem_description) \
        .replace("$exampleSolution", example_solution)
    
    # Add prompt to list of messages
    messages.append({"role": "user", "content": p})
    break
    
# Generate text and print the response
response = pipe(messages, return_full_text=False)
print(response)

