import sys
import os
import argparse
import transformers
from vllm import LLM, SamplingParams
from datasets import load_dataset, disable_caching
import json
import pandas as pd
import numpy as np

sys.path.append("./src/")  # Add module directory to path

from utils.constants import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    PIPE_RETURN_FULL_TEXT,
    MAX_GENERATED_TOKENS,
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
    ############################
    #     Argument parsing     #
    ############################
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
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--engine", type=str, required=True, choices=["vllm", "transformers"]
    )
    parser.add_argument("--max_number_of_sequences", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("Unknown arguments: " + str(unknown))

    # Output directory
    outdir = f"./outputs/{args.version}/{args.model}/{args.jobid}"

    ############################
    #   Write config & setup   #
    ############################

    # Ensure directory exists
    os.makedirs(outdir, exist_ok=True)

    # Save config
    with open(f"{outdir}/config.json", "w", encoding="utf-8") as file:
        config_json = json.dumps(vars(args))
        file.write(config_json)

    # Print CL arguments
    print(args)

    ############################
    # Load data & make prompts #
    ############################

    # Disable dataset caching
    disable_caching()

    # Load Data
    print("Reading input data...")
    task = get_task_type(args.type)
    dataset = load_dataset("csv", data_files=args.file, split="train", sep=";")
    dataset = dataset.shuffle(seed=args.seed)

    # Make prompts
    print("Forming prompts...")

    demonstrations, demo_indices = sample_dataset(
        dataset, args.seed, args.number_of_demonstrations, args.type_of_demonstrations
    )
    system_prompt = get_system_prompt(task, demonstrations, bool(args.use_instructions))

    dataset = dataset.map(
        lambda row: {
            "user_prompt": make_prompt(row, task),
            "system_prompt": system_prompt,
        },
    )

    if args.number_of_demonstrations > 0:
        # Remove demonstration examples from the dataset to avoid data leakage
        dataset = dataset.select(
            [idx for idx in range(len(dataset)) if idx not in demo_indices]
        )

    # Select n rows
    if args.n_rows is not None and args.n_rows > 0:
        dataset = dataset.select(range(args.n_rows))

    user_prompts = dataset["user_prompt"]
    system_prompts = dataset["system_prompt"]

    prompts = [
        [
            {"role": "system", "content": sp},
            {"role": "user", "content": up},
        ]
        for sp, up in zip(system_prompts, user_prompts)
    ]

    # Disable thinking mode
    if "qwen3" in args.model.lower():
        for prompt in prompts:
            prompt.append(
                {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
            )

    ###############################
    #     Initialize model &      #
    #     and generate responses  #
    ###############################

    top_p = 0.8
    top_k = 20
    min_p = 0.0

    results = {key: [] for key in get_default_response(task).keys()}
    default_response = get_default_response(task)

    if args.engine == "vllm":
        # Initialize the model
        mode = "mistral" if "mistral-small" in args.model.lower() else "auto"
        llm = LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=4096,  # Max length of prompt + output
            max_num_seqs=args.max_number_of_sequences,
            enforce_eager=True,  # Disable cuda graph for lower VRAM consumption
            tokenizer_mode=mode,
            config_format=mode,
            load_format=mode,
        )

        # All models have same params
        sampling_params = SamplingParams(
            temperature=MODEL_TEMPERATURE,
            max_tokens=MAX_GENERATED_TOKENS,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

        tqdm_flag = True if args.n_rows is not None else False
        if "mistral-small" not in args.model.lower():
            print("Using tokenization with chat template for generation...")
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
            prompts = [
                tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prompt in prompts
            ]

            outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=tqdm_flag,
            )
        else:
            # Installed autotokenizer version does not support mistral small, use chat template
            # without tokenization and generate with chat method instead of generate method
            print(
                "Using chat template without tokenization for generation with chat method..."
            )
            outputs = llm.chat(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=tqdm_flag,
            )
    else:  # Transformers pipeline
        # Model parameters
        params = {
            "model": args.model,
            "device_map": 0,  # Force GPU
            "max_new_tokens": MAX_GENERATED_TOKENS,
            "temperature": MODEL_TEMPERATURE,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }
        print(f"Model parameters: {params}")

        print("Initializing pipeline...")
        # Initialize the pipeline
        pipeline = transformers.pipeline("text-generation", **params)
        pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
        pipeline.model.config.pad_token_id = pipeline.model.config.eos_token_id

        print(f"Generating responses for {dataset.num_rows} prompts...\n")

        outputs = pipeline(
            prompts,
            return_full_text=PIPE_RETURN_FULL_TEXT,
            batch_size=args.batch_size,
        )

    # Process each output in the batch
    for output in outputs:
        if args.engine == "vllm":
            text = output.outputs[0].text
        else:
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
