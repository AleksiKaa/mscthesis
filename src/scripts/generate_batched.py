import os
import sys
import argparse
import transformers
from datasets import load_dataset
import json

print("Libraries imported")


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add parent directory to path

from utils.constants import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    PIPE_RETURN_FULL_TEXT,
    PIPE_MAX_NEW_TOKENS,
    MODEL_TEMPERATURE,
    BATCH_SIZE,
)

from utils.helpers import (
    parse_output,
    get_task_type,
    get_default_response,
    get_system_prompt,
    make_prompt,
)


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
        choices=["judge", "j", "augment", "a"],
        required=True,
    )

    args = parser.parse_args()

    # Print CL arguments
    print(args)

    # Load Data
    print("Reading input data...")
    task = get_task_type(args.type)
    dataset = load_dataset("csv", data_files=args.file, split="train", sep=";")

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

    system_prompt = get_system_prompt(task)
    results = {key: [] for key in get_default_response(task).keys()}
    default_response = get_default_response(task)

    batch_size = BATCH_SIZE
    prompts = dataset["prompt"]
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Build batched conversation inputs
        batch_inputs = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in batch_prompts
        ]

        # Single batched forward pass
        outputs = pipeline(
            batch_inputs,
            return_full_text=PIPE_RETURN_FULL_TEXT,
        )

        # Process each output in the batch
        for output in outputs:
            text = output[0]["generated_text"]
            parsed = parse_output(text)

            for key, value in default_response.items():
                results[key].append(json.dumps(parsed.get(key, value)))

    # For debugging purposes
    print(results)

    # Add named lists as columns
    for column_name, column_data in results.items():
        dataset = dataset.add_column(column_name, column_data)

    if args.csv:
        dataset.to_pandas().to_csv(
            f"./outputs/results/batch_{args.jobid}_result.csv",
            sep=";",
            index=False,
        )


if __name__ == "__main__":
    main()
