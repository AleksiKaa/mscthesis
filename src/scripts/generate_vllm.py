import sys
import os
import argparse

from vllm import LLM, SamplingParams
from datasets import load_dataset, disable_caching
import json
import pandas as pd
import numpy as np

sys.path.append("./src/")  # Add module directory to path

from utils.constants import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    PIPE_MAX_NEW_TOKENS,
    MODEL_TEMPERATURE,
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

    print("Initializing model...")

    # Initialize the model
    mode = "mistral" if "mistral" in args.model.lower() else "auto"

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        max_num_seqs=16,
        enforce_eager=True,
        tokenizer_mode=mode,
        config_format=mode,
        load_format=mode,
    )

    sampling_params = SamplingParams(temperature=0.3, max_tokens=250)

    prompts = dataset["user_prompt"]
    system_prompts = dataset["system_prompt"]
    complete_prompts = [
        [
            {"role": "system", "content": sp},
            {"role": "user", "content": up},
        ]
        for sp, up in zip(system_prompts, prompts)
    ]

    print(f"Generating responses for {dataset.num_rows} prompts...\n")

    default_response = get_default_response(task)
    results = {key: [] for key in default_response.keys()}

    # Single batched forward pass
    outputs = llm.chat(
        complete_prompts,
        sampling_params=sampling_params,
        use_tqdm=True if args.n_rows is not None else False,
    )

    # Process each output in the batch
    for out in outputs:
        text = out.outputs[0].text
        parsed = parse_output(text)
        for key, value in default_response.items():
            results[key].append(json.dumps(parsed.get(key, value)))

    for column_name, column_data in results.items():
        dataset = dataset.add_column(column_name, column_data)

    # For debugging purposes
    print(results)

    # Write results to csv
    if bool(args.csv):
        dataset.to_pandas().to_csv(
            f"{outdir}/result.csv",
            sep=";",
            index=False,
        )


if __name__ == "__main__":
    main()
