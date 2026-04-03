"""
The main part of the thesis that produces the results of the study.
This acts as the starting point of the LLM hallucination detection pipeline.
This file starts a number of slurm jobs based on the hardcoded configs.
Current setup:

6 models: 2 Qwen (small, large), 2 llama (small, large), 2 Mistral (small, large-ish)
5 seeds: predetermined random seeds
11 run configurations:
    - Zero-shot, with instructions
    - One-shot, negative, without instructions
    - One-shot, positive, without instructions
    - One-shot, negative, with instructions
    - One-shot, positive, with instructions
    - Six demos, negative, without instructions
    - Six demos, negative, without instructions
    - Six demos, negative, without instructions
    - Six demos, negative, without instructions
    - Six demos, negative, without instructions
    - Six demos, negative, without instructions

In total 6 * 5 * 11 = 330 runs, worst case run time per reserved resources:
    165 * 2 + 165 * 3 = 825 hours ~ 34 days

The resource reservations are pretty generous, more likely runtime is around 1-2 days.
"""

import os
import sys
import subprocess
from time import sleep
import argparse
import json

os.chdir("/home/kaariaa3/mscthesis")
sys.path.append("./src/")  # Add module directory to path


# One set of resources for each model, reserve circa 10GB VRAM for KV cache
slurm_params = {
    "Qwen/Qwen3-8B": {
        "time": "02:00:00",
        "memory": "32GB",
        "vram": "32g",
        "batch_size": 1,
    },
    "Qwen/Qwen3-32B": {
        "time": "03:00:00",
        "memory": "32GB",
        "vram": "80g",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "time": "02:00:00",
        "memory": "32GB",
        "vram": "32g",
        "batch_size": 2,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "time": "03:00:00",
        "memory": "32GB",
        "vram": "140g",
        "batch_size": 1,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "time": "02:00:00",
        "memory": "32GB",
        "vram": "32g",
        "batch_size": 2,
    },
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
        "time": "02:00:00",
        "memory": "32GB",
        "vram": "80g",
        "max_number_of_sequences": 16,
    },
}

seeds = [1, 10, 42, 50, 100]
models = [  # 3 model families, big and small model
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Mistral-Small-3.2-24B-Instruct-2506",  # Use with vllm script
]

# number_of_demonstrations, type_of_demonstrations, use_instructions
runs = [
    (0, 0, 1),  # Zero-shot, with instructions
    (1, -1, 0),  # One-shot, negative, without instructions
    (1, 1, 0),  # One-shot, positive, without instructions
    (1, -1, 1),  # One-shot, negative, with instructions
    (1, 1, 1),  # One-shot, positive, with instructions
    (6, -1, 0),  # Six demos, negative, without instructions
    (6, 0, 0),  # Six demos, negative, without instructions
    (6, 1, 0),  # Six demos, negative, without instructions
    (6, -1, 1),  # Six demos, negative, without instructions
    (6, 0, 1),  # Six demos, negative, without instructions
    (6, 1, 1),  # Six demos, negative, without instructions
]


def construct_python_params(
    model,
    seed,
    n_demos,
    use_instruction,
    type_of_demo,
    version,
    debug,
    num_seqs,
    gpu_memory_utilization,
    batch_size,
):
    python_params = (
        f"--model {model} "
        + f"--seed {seed} "
        + f"--number_of_demonstrations {n_demos} "
        + f"--use_instructions {use_instruction} "
        + f"--type_of_demonstrations {type_of_demo} "
        + f"--version {version} "
    )

    if batch_size is not None:
        python_params += f"--batch_size {batch_size} "

    if num_seqs is not None:
        python_params += f"--max_number_of_sequences {num_seqs} "

    if gpu_memory_utilization is not None:
        python_params += f"--gpu_memory_utilization {gpu_memory_utilization} "

    python_params += "--type detect"

    if debug:
        python_params += " --n_rows 1"

    return python_params


def construct_slurm_params(model, version, debug):
    # Construct slurm params

    submit_script = "./src/submit.sh"
    model_args = slurm_params.get(model)

    if model_args is None:
        print(f"Slurm arguments not found for model {model}! Skipping...")
        return None

    slurm_args = [submit_script, "-m", model, "-w", version]
    for resource, amount in model_args.items():
        match resource:
            case "time":
                flag = "t"
                amount = "00:30:00" if debug else amount
            case "memory":
                flag = "r"
            case "vram":
                flag = "v"
            case _:
                continue

        slurm_args.append("-" + flag)
        slurm_args.append(amount)

    return slurm_args


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with reduced resources"
    )
    parser.add_argument("-c", "--config_file", type=str, default=None)
    parser.add_argument("-v", "--version", type=str, default="default")

    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file, "r", encoding="utf-8") as config_json:
            config = json.loads("\n".join(config_json.readlines()))

            slurm_args = construct_slurm_params(
                config["model"], args.version, args.debug
            )

            if slurm_args is None:
                return

            model = config["model"]
            params = slurm_params.get(model, {})

            python_params = construct_python_params(
                model,
                config["seed"],
                config["number_of_demonstrations"],
                config["use_instructions"],
                config["type_of_demonstrations"],
                config.get("version", args.version),
                args.debug,
                params.get("max_number_of_sequences"),
                params.get("gpu_memory_utilization"),
                params.get("batch_size"),
            )
            print(f"Args passed to python: {python_params}")
            subprocess.call(slurm_args + ["-p", python_params])
            return

    # Run each model once with these parameter configurations
    for model in models:

        slurm_args = construct_slurm_params(model, args.version, args.debug)

        params = slurm_params.get(model, {})

        # Construct python params
        for seed in seeds:
            for n_demos, type_of_demo, use_instruction in runs:
                python_params = construct_python_params(
                    model,
                    seed,
                    n_demos,
                    use_instruction,
                    type_of_demo,
                    args.version,
                    args.debug,
                    params.get("max_number_of_sequences"),
                    params.get("gpu_memory_utilization"),
                    params.get("batch_size"),
                )

                print(f"Called subprocess with args: {slurm_args}")
                print(f"Args passed to python: {python_params}")
                subprocess.call(slurm_args + ["-p", python_params])
                sleep(1)


if __name__ == "__main__":
    main()
