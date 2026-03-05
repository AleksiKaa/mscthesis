import json

from .constants import ERROR_RESULT, DEFAULT_JUDGE_RESULT, DEFAULT_AUGMENT_RESULT
from .prompts import JUDGE_SYSTEM_PROMPT, AUGMENT_SYSTEM_PROMPT, JUDGE_TEMPLATE, AUGMENT_TEMPLATE


def get_system_prompt(task):
    match task:
        case "judge":
            return JUDGE_SYSTEM_PROMPT
        case "augment":
            return AUGMENT_SYSTEM_PROMPT


def get_task_type(tasktype):
    match tasktype:
        case "judge" | "j":
            task = "judge"
        case "augment" | "a":
            task = "augment"

    return task


def get_default_response(tasktype):
    match tasktype:
        case "judge":
            return DEFAULT_JUDGE_RESULT
        case "augment":
            return DEFAULT_AUGMENT_RESULT


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
        case "augment":
            return (
                AUGMENT_TEMPLATE.replace("$THEME$", row["theme"])
                .replace("$TOPIC$", row["topic"])
                .replace("$CONCEPT$", row["concept"])
                .replace("$TEXT$", row["problemDescription"])
                .replace("$CODE$", row["exampleSolution"])
            )
        case _:
            raise ValueError(f"Task type '{_}' not recognised as valid task type!")


def parse_output(text):
    # Try to find JSON
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return ERROR_RESULT

    try:
        data = json.loads(text[start : end + 1])

        # Is dict
        if not isinstance(data, dict):
            return ERROR_RESULT

        return data

    except Exception as e:
        print(e)
        return ERROR_RESULT
