print("Importing libraries...")
import argparse
import pandas as pd
import transformers
import json

from ..utils.prompts import SYSTEM_PROMPT, CRITIQUE_TEMPLATE

DEFAULT_DATA = "/home/kaariaa3/mscthesis/data/cleaned.csv"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"


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


def run_model(pipe, data):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default=DEFAULT_DATA)
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("-s", "--skipgen", type=bool, default=False)

    args = parser.parse_args()

    # Print CL arguments
    print(args)

    # Skip flag
    if args.skipgen:
        return

    # Model parameters
    params = {
        "model": args.model,
        "device_map": 0,  # Force GPU
        "max_new_tokens": 500,
        "temperature": 0.3,
    }
    print(f"Model parameters: {params}")

    print("Initializing pipeline...")
    # Initialize the pipeline
    pipeline = transformers.pipeline("text-generation", **params)

    print("Reading input data...")
    # Read CSV
    eval_df = pd.read_csv(args.file, sep=";")

    print("Creating prompts...")
    eval_df["prompt"] = eval_df.apply(make_prompt, axis=1)

    print("Generating responses...\n")
    eval_df = eval_df.apply(lambda row: run_model(pipeline, row), axis=1)

    eval_df.to_csv(
        "../../outputs/results/generate_explanations_result.csv", sep=";", index=False
    )


if __name__ == "__main__":
    main()
