# Imports
import pandas as pd
import numpy as np
import os
from dotenv import dotenv_values

config = dotenv_values(".env")

system_prompt = """
I want you to act as a programming teacher for an in-
troductory Dart course. Your students are programming
novices. I will provide some coding example exercises,
and it will be your job to critique them. Your responses 
should be written in simple English. Do not cite music 
lyrics or books. Do not include any greetings, be concise. 
Do not mention trigger words associated with mental or 
physical disorders, for example, weight loss or diet.
"""

critique_template = """
Please generate an explanation why the following exercise description and Dart
code are not faithful to the provided topic, theme or concept. Your response
should be a JSON string.

$Topic$Theme$Concept
Exercise description: $problemDescription

Code: $exampleSolution
"""

def main():
    # Read CSV
    path = os.relpath(os.getcwd(), config["DATA_PATH"])
    df = pd.read_csv("../../data/out.csv", sep=";")

    # Get rows with "no" labels
    label_col = df.columns[-1]
    cond = df[label_col] == "no"
    wrongs = df[cond]

    # Generate prompts
    for idx, row in wrongs.iterrows():
        _, topic, theme, concept, problemDescription, exampleSolution, *evals = row
    
        p = critique_template
        
        idx = {
            1: "$Theme",
            2: "$Topic",
            3: "$Concept"
        }
    
        rep = {
            "$Theme": theme,
            "$Topic": topic,
            "$Concept": concept
        }
        
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
            .replace("$problemDescription", problemDescription) \
            .replace("$exampleSolution", exampleSolution)
    
        print(system_prompt + p)
    
        break

if __name__ == "__main__":
    main()