"""
Description of experiment:

1) several runs (effect of random seed) = 5 runs
2) model family (qwen vs llama) = 2 runs
3) model size (large vs small) = 2 runs
4) presence of instructions and number of demonstrations (zero shot with instructions, 1 shot with instructions, 1 shot no instructions, 6 shot with instructions, 6 shot no instructions) = 5 runs
5) type of demonstrations (only positive, mixed, only negative) -- mixed only for 6 shot = 3 runs

In total: 300 runs, max 300 hours of cluster time
"""

import os
import sys
import subprocess
from time import sleep
import argparse
import json

os.chdir("/home/kaariaa3/mscthesis")
sys.path.append("./src/")  # Add module directory to path


# One set of resources for each model
slurm_params = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "time": "02:00:00",
        "memory": "32GB",
        "vram": "20g",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "time": "02:00:00",
        "memory": "32GB",
        "vram": "40g",
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "time": "03:00:00",
        "memory": "32GB",
        "vram": "140g",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "time": "02:00:00",
        "memory": "32GB",
        "vram": "20g",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "time": "03:00:00",
        "memory": "32GB",
        "vram": "140g",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "time": "02:00:00",
        "memory": "32GB",
        "vram": "20g",
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "time": "03:00:00",
        "memory": "32GB",
        "vram": "120g",
    },
}

seeds = [1]  # , 10, 42, 50, 100]
models = [  # 3 model families, big vs small model (medium for mistral)
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
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


def construct_python_params(model, seed, n_demos, use_instruction, type_of_demo):
    return (
        f"--model {model} "
        + f"--seed {seed} "
        + f"--number_of_demonstrations {n_demos} "
        + f"--use_instructions {use_instruction} "
        + f"--type_of_demonstrations {type_of_demo} "
        + "--type detect"
    )


def construct_slurm_params(model):
    # Construct slurm params

    submit_script = "./src/submit.sh"
    model_args = slurm_params.get(model)

    if model_args is None:
        print(f"Slurm arguments not found for model {model}! Skipping...")
        return None

    slurm_args = [submit_script, "-m", model]
    for resource, amount in model_args.items():
        match resource:
            case "time":
                flag = "t"
            case "memory":
                flag = "r"
            case "vram":
                flag = "v"
            case _:
                raise ValueError("Resource not known!")

        slurm_args.append("-" + flag)
        slurm_args.append(amount)

    return slurm_args


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default=None)

    args = parser.parse_args()

    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as config_json:
            config = json.loads("\n".join(config_json.readlines()))

            slurm_args = construct_slurm_params(config["model"])

            if slurm_args is None:
                return

            python_params = construct_python_params(
                config["model"],
                config["seed"],
                config["number_of_demonstrations"],
                config["use_instructions"],
                config["type_of_demonstrations"],
            )
            print(f"Args passed to python: {python_params}")
            subprocess.call(slurm_args + ["-p", python_params])
            return

    # Run each model once with these parameter configurations
    for model in models:

        slurm_args = construct_slurm_params(model)

        # Construct python params
        for seed in seeds:
            for n_demos, type_of_demo, use_instruction in runs:
                python_params = construct_python_params(
                    model, seed, n_demos, use_instruction, type_of_demo
                )

                print(f"Called subprocess with args: {slurm_args}")
                print(f"Args passed to python: {python_params}")
                subprocess.call(slurm_args + ["-p", python_params])
                sleep(1)


if __name__ == "__main__":
    main()
