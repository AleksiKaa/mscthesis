import json
import random

from .constants import (
    ERROR_RESULT,
    DEFAULT_DETECT_RESULT,
    DEFAULT_AUGMENT_RESULT,
    CONCEPT_TO_CHAPTER_MAPPING,
    THEME_TO_TOPICS_MAPPING,
)
from .prompts import (
    DETECT_SYSTEM_PROMPT,
    AUGMENT_SYSTEM_PROMPT,
    DETECT_TEMPLATE,
    AUGMENT_TEMPLATE,
)


def get_system_prompt(task):
    match task:
        case "detect":
            return DETECT_SYSTEM_PROMPT
        case "augment":
            return AUGMENT_SYSTEM_PROMPT


def get_task_type(tasktype):
    match tasktype:
        case "detect" | "d":
            task = "detect"
        case "augment" | "a":
            task = "augment"

    return task


def get_default_response(tasktype):
    match tasktype:
        case "detect":
            return DEFAULT_DETECT_RESULT
        case "augment":
            return DEFAULT_AUGMENT_RESULT


def make_prompt(row, task_type):
    match task_type:
        case "detect":
            concept = row["concept"]
            concept_chapter = CONCEPT_TO_CHAPTER_MAPPING.get(concept)
            allowed_concepts = ", ".join(  # Form comma separated string
                list(
                    map(
                        lambda t: t[0],  # Drop chapter number, use only concept value
                        filter(
                            lambda v: v[1]
                            <= concept_chapter,  # Only include concepts from current or earlier chapters
                            CONCEPT_TO_CHAPTER_MAPPING.items(),
                        ),
                    )
                )
            )

            return (
                DETECT_TEMPLATE.replace("$THEME$", row["theme"])
                .replace("$TOPIC$", row["topic"])
                .replace("$CONCEPTS$", allowed_concepts)
                .replace("$TEXT$", row["problemDescription"])
                .replace("$CODE$", row["exampleSolution"])
            )
        case "augment":
            new_theme = random.choice(list(THEME_TO_TOPICS_MAPPING.keys()))
            # Ensure topic is not the same
            new_topic = random.choice(
                list(
                    filter(
                        lambda x: x != row["topic"],
                        THEME_TO_TOPICS_MAPPING.get(new_theme),
                    )
                )
            )

            concept = row["concept"]
            concept_chapter = CONCEPT_TO_CHAPTER_MAPPING.get(concept)

            advanced_concept = random.choice(
                list(
                    filter(
                        lambda v: v[1] > concept_chapter,
                        CONCEPT_TO_CHAPTER_MAPPING.items(),
                    )
                )
            )[0]

            return (
                AUGMENT_TEMPLATE.replace("$THEME$", new_theme)
                .replace("$TOPIC$", new_topic)
                .replace("$CONCEPT$", advanced_concept)
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
