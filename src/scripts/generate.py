import os
import sys
import argparse
from collections import defaultdict
import transformers
from datasets import load_dataset

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
    BATCH_SIZE,
)

from utils.helpers import parse_output

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
    match task_type:
        case "judge":
            return (
                JUDGE_TEMPLATE.replace("$THEME$", row["theme"])
                .replace("$TOPIC$", row["topic"])
                .replace("$CONCEPT$", row["concept"])
                .replace("$TEXT$", row["problemDescription"])
                .replace("$CODE$", row["exampleSolution"])
            )
        case _:
            raise ValueError(f"Task type '{_}' not recognised as valid task type!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobid")
    parser.add_argument("-f", "--file", type=str, default=DEFAULT_DATA)
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("-c", "--csv", type=bool, default=True)
    parser.add_argument("-n", "--n_rows", type=int, default=None)
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["judge", "j"],
        required=True,
    )

    args = parser.parse_args()

    # Print CL arguments
    print(args)

    # Load Data
    print("Reading input data...")
    task = get_task_type(args.type)
    dataset = load_dataset("csv", data_files=DEFAULT_DATA, split="train", sep=";")

    if args.n_rows is not None and args.n_rows > 0:
        dataset = dataset.select(range(args.n_rows))

    # Make prompts
    print("Forming prompts...")
    dataset = dataset.map(lambda row: {"prompt": make_prompt(row, task)})

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
    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    pipeline.model.config.pad_token_id = pipeline.model.config.eos_token_id

    print("Generating responses...\n")

    results = defaultdict(list)
    for prompt in dataset["prompt"]:
        print(prompt)
        output = pipeline(
            [
                {"role": "system", "content": system_prompts.get(task)},
                {"role": "user", "content": prompt},
            ],
            batch_size=BATCH_SIZE,
            return_full_text=PIPE_RETURN_FULL_TEXT,
        )

        text = output[0]["generated_text"]
        print(text)

        parsed = parse_output(text)
        for k, v in parsed.items():  # Map to named lists
            results[k].append(v)

    for k, v in results.items():  # Add named lists as columns
        dataset = dataset.add_column(k, v)

    if args.csv:
        dataset.to_pandas().to_csv(
            f"../../outputs/results/batch_{args.jobid}_result.csv",
            sep=";",
            index=False,
        )


if __name__ == "__main__":
    main()
