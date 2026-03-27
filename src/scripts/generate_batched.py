import sys
import os
import argparse
import transformers
from datasets import load_dataset, disable_caching
import json
import pandas as pd
import numpy as np

sys.path.append("./src/")  # Add module directory to path

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
    sample_dataset,
)

print("Libraries imported")

print("Constants loaded from: " + sys.modules["utils.constants"].__file__)
print("Helper functions loaded from: " + sys.modules["utils.helpers"].__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobid")
    parser.add_argument("-f", "--file", type=str, default=DEFAULT_DATA)
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("-c", "--csv", type=int, default=1, choices=[0, 1])
    parser.add_argument("-n", "--n_rows", type=int, default=None)
    parser.add_argument(
        "-us", "--use_instructions", type=int, default=1, choices=[0, 1]
    )
    parser.add_argument(
        "-tof", "--type_of_demonstrations", type=int, choices=[-1, 0, 1], default=0
    )
    parser.add_argument("-nd", "--number_of_demonstrations", type=int, default=0)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["detect", "d", "augment", "a"],
        required=True,
    )
    parser.add_argument("-v", "--version", type=str, required=True)

    args = parser.parse_args()

    # Output directory
    outdir = f"./outputs/{args.version}/{args.model}/{args.jobid}"

    # Ensure directory exists
    os.makedirs(outdir, exist_ok=True)

    # Save config
    with open(f"{outdir}/config.json", "w", encoding="utf-8") as file:
        config_json = json.dumps(vars(args))
        file.write(config_json)

    # Print CL arguments
    print(args)

    # Disable dataset caching
    disable_caching()

    # Load Data
    print("Reading input data...")
    task = get_task_type(args.type)
    dataset = load_dataset("csv", data_files=args.file, split="train", sep=";")
    dataset = dataset.shuffle(seed=args.seed)

    # Make prompts
    print("Forming prompts...")

    demonstrations = sample_dataset(
        dataset, args.seed, args.number_of_demonstrations, args.type_of_demonstrations
    )
    system_prompt = get_system_prompt(task, demonstrations, bool(args.use_instructions))

    dataset = dataset.map(
        lambda row: {
            "user_prompt": make_prompt(row, task),
            "system_prompt": system_prompt,
        },
    )

    # Select n rows
    if args.n_rows is not None and args.n_rows > 0:
        dataset = dataset.select(range(args.n_rows))

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

    print(f"Generating responses for {dataset.num_rows} prompts...\n")

    results = {key: [] for key in get_default_response(task).keys()}
    default_response = get_default_response(task)

    batch_size = BATCH_SIZE
    prompts = dataset["user_prompt"]
    system_prompts = dataset["system_prompt"]
    for i in range(0, len(prompts), batch_size):

        # Build batched conversation inputs
        batch_prompts = zip(
            system_prompts[i : i + batch_size], prompts[i : i + batch_size]
        )

        batch_inputs = [
            [
                {"role": "system", "content": sp},
                {"role": "user", "content": up},
            ]
            for sp, up in batch_prompts
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

    # Write results to csv
    if bool(args.csv):
        dataset.to_pandas().to_csv(
            f"{outdir}/result.csv",
            sep=";",
            index=False,
        )


if __name__ == "__main__":
    main()
