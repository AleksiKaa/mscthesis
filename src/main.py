import os
import sys
import subprocess
from time import sleep

os.chdir("/home/kaariaa3/mscthesis")
sys.path.append("./src/")  # Add module directory to path


def main():
    submit_script = "./src/submit.sh"

    runs = [
        "-t d",  # zero-shot
        "-t d -rd 1",  # 1-shot with random sampling
        "-t d -rd 5",  # few-shot with 5 random samples
        "-t d -ufd True",  # few-shot with 5 hand-picked examples
    ]

    for run in runs:
        subprocess.call([submit_script, "-p", run])
        sleep(1)


if __name__ == "__main__":
    main()
