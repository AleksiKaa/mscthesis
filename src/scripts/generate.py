import os
import sys
import argparse
import json
import pandas as pd
import transformers

print("Libraries imported")


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add parent directory to path

from utils.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_TEMPLATE,
)

from utils.constants import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    PIPE_RETURN_FULL_TEXT,
    PIPE_MAX_NEW_TOKENS,
    MODEL_TEMPERATURE,
)

# Task type to system prompt
system_prompts = {
    "judge": JUDGE_SYSTEM_PROMPT,
}


def get_task_type(tasktype):
    match tasktype:
        case "judge" | "j":
            task = "judge"
        case "zeroshot" | "z":
            task = "zeroshot"
        case "fewshot" | "f":
            task = "fewshot"
        case "explicit" | "e":
            task = "explicit"
        case "implicit" | "i":
            task = "implicit"

    return task


# Functions
def make_prompt(row, task_type):
    _, problem_description, example_solution, _, _, theme, topic, concept, *_ = row

    match task_type:
        case "judge":
            return (
                JUDGE_TEMPLATE.replace("$THEME$", theme)
                .replace("$TOPIC$", topic)
                .replace("$CONCEPT$", concept)
                .replace("$TEXT$", problem_description)
                .replace("$CODE$", example_solution)
            )
        case "zeroshot":
            pass
        case "fewshot":
            pass
        case "explicit":
            pass
        case "implicit":
            pass
        case _:
            raise ValueError(f"Task type '{_}' not recognised as valid task type!")


def run_model(pipe, data, task_type):
    system_prompt = system_prompts.get(task_type, None)

    if system_prompt is None:
        raise ValueError(f"Task type '{task_type}' not recognised as valid task type!")

    response = pipe(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data["prompt"]},
        ],
        return_full_text=PIPE_RETURN_FULL_TEXT,
    )

    result = response[0]["generated_text"]
    result = result.replace("```json", "").replace(
        "```", ""
    )  # Strip markdown around actual content

    result_dict = json.loads(result)

    for k, v in result_dict.items():
        data[k] = v

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobid")
    parser.add_argument("-f", "--file", type=str, default=DEFAULT_DATA)
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("-s", "--skipgen", type=bool, default=False)
    parser.add_argument("-c", "--csv", type=bool, default=True)
    parser.add_argument("-a", "--append_result", type=bool, default=True)
    parser.add_argument("-n", "--n_rows", type=int, default=1)
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=[
            "judge",
            "j",
            "zeroshot",
            "z",
            "fewshot",  # Not implemented
            "f",
            "explicit",  # Not implemented
            "e",
            "implicit",  # Not implemented
            "i",
        ],
        required=True,
    )

    args = parser.parse_args()

    # Print CL arguments
    print(args)

    # Skip flag
    if args.skipgen:
        return

    task = get_task_type(args.type)

    # Model parameters
    params = {
        "model": args.model,
        "device_map": 0,  # Force GPU
        "max_new_tokens": PIPE_MAX_NEW_TOKENS,
        "temperature": MODEL_TEMPERATURE,
    }
    print(f"Model parameters: {params}")

    print("Initializing pipeline...")
    # Initialize the pipeline
    pipeline = transformers.pipeline("text-generation", **params)

    print("Reading input data...")
    # Read CSV
    df = pd.read_csv(args.file, sep=";")
    # eval_df = df.loc[0 : args.n_rows - 1]
    eval_df = df.iloc[[83, 121, 158, 220]]

    print("Creating prompts...")
    eval_df["prompt"] = eval_df.apply(lambda row: make_prompt(row, task), axis=1)

    print("Generating responses...\n")
    result = eval_df.apply(lambda row: run_model(pipeline, row, task), axis=1)

    if args.csv:
        result.to_csv(
            f"../../outputs/results/generate_{task}_result_{args.jobid}.csv",
            sep=";",
            index=False,
        )


if __name__ == "__main__":
    main()
