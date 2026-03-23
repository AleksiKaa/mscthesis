import sys
import argparse
import transformers
from datasets import load_dataset
import json
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
    sample_dataset_indices,
    create_demonstrations_set,
)

print("Libraries imported")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobid")
    parser.add_argument("-f", "--file", type=str, default=DEFAULT_DATA)
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("-c", "--csv", type=bool, default=True)
    parser.add_argument("-n", "--n_rows", type=int, default=None)
    parser.add_argument("-rd", "--random_demos", type=int, default=0)
    parser.add_argument("-ufd", "--use_fixed_demos", type=bool, default=False)
    parser.add_argument(
        "-fdi",
        "--fixed_demos_indices",
        type=list[int],
        default=[
            534,
            60,
            272,
            253,
            114,
            446,
        ],
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["detect", "d", "augment", "a"],
        required=True,
    )

    args = parser.parse_args()

    # Print CL arguments
    print(args)

    # Load Data
    print("Reading input data...")
    task = get_task_type(args.type)
    dataset = load_dataset("csv", data_files=args.file, split="train", sep=";")
    dataset = dataset.shuffle(seed=42)

    fixed_demos_idx = np.array([], dtype=np.int64)
    if args.use_fixed_demos:
        fixed_demos_idx = np.array(args.fixed_demos_indices, dtype=np.int64)

    # Make prompts
    print("Forming prompts...")

    demonstrations_rng = np.random.default_rng(seed=10)
    random_indices = sample_dataset_indices(
        demonstrations_rng, dataset.num_rows, args.random_demos
    )

    dataset = dataset.map(
        lambda row, i: {
            "user_prompt": make_prompt(row, task),
            "system_prompt": get_system_prompt(
                task,
                create_demonstrations_set(
                    dataset,
                    random_indices[i] if random_indices is not None else None,
                    fixed_demos_idx,
                ),
            ),
        },
        with_indices=True,
    )

    dataset = (
        dataset.select(  # Exclude fixed demonstrations from dataset if used in prompts
            [i for i in range(len(dataset)) if i not in fixed_demos_idx]
        )
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

    if args.csv:
        dataset.to_pandas().to_csv(
            f"./outputs/results/batch_{args.jobid}_result.csv",
            sep=";",
            index=False,
        )


if __name__ == "__main__":
    main()
