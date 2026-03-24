import os
import sys
import subprocess
from time import sleep

os.chdir("/home/kaariaa3/mscthesis")
sys.path.append("./src/")  # Add module directory to path


def main():
    submit_script = "./src/submit.sh"

    models = ["Qwen/Qwen2.5-14B-Instruct"]

    # One set of resources for each model, TODO: Params only needed for model scale, i.e. < 5B, [5, 15]B, [15, 30] etc
    slurm_params = {
        "Qwen/Qwen2.5-14B-Instruct": {
            "time": "01:00:00",
            "memory": "16GB",
            "vram": "40g",
        }
    }

    # Run each model once with these parameter configurations
    generate_params = [
        "-t d",  # zero-shot
        "-t d -rd 1",  # 1-shot with random sampling
        "-t d -rd 5",  # few-shot with 5 random samples
        "-t d -ufd True",  # few-shot with 6 hand-picked examples
    ]

    for model in models:

        # Construct slurm params
        slurm_args = slurm_params.get(model)

        if slurm_args is None:
            print(f"Slurm arguments not found for model {model}! Skipping...")
            continue

        args = [submit_script]
        for resource, amount in slurm_params.items():
            match resource:
                case "time":
                    flag = "t"
                case "memory":
                    flag = "r"
                case "vram":
                    flag = "v"
                case _:
                    raise ValueError("Resource not known!")

            args.append("-" + flag)
            args.append(amount)

        for python_params in generate_params:
            subprocess.call(args + ["-p", python_params])
            sleep(1)


if __name__ == "__main__":
    main()
