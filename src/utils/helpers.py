import json
import random
import numpy as np
import pandas as pd
from datasets import Dataset

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
    DEMONSTRATION_TEMPLATE,
)

from .plots import GT_COLS


def get_allowed_concepts(concept):
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

    return allowed_concepts


def get_disallowed_concepts(concept):
    concept_chapter = CONCEPT_TO_CHAPTER_MAPPING.get(concept)
    disallowed_concepts = ", ".join(  # Form comma separated string
        list(
            map(
                lambda t: t[0],  # Drop chapter number, use only concept value
                filter(
                    lambda v: v[1]
                    > concept_chapter,  # Only include concepts from current or earlier chapters
                    CONCEPT_TO_CHAPTER_MAPPING.items(),
                ),
            )
        )
    )

    return disallowed_concepts


def get_system_prompt(task, demonstrations=None, use_instructions=True):
    match task:
        case "detect":
            system_prompt = ""

            # Use demos
            if demonstrations is not None:
                system_prompt += "Use the demonstrations below as examples on how to answer the question.\n\n"
                system_prompt += make_demonstrations(demonstrations)

            if use_instructions:
                system_prompt += DETECT_SYSTEM_PROMPT

            return system_prompt
        case "augment":
            return AUGMENT_SYSTEM_PROMPT
        case _:
            raise ValueError(f"Task type not recognised as valid task type!")


def get_task_type(tasktype):
    match tasktype:
        case "detect" | "d":
            return "detect"
        case "augment" | "a":
            return "augment"
        case _:
            raise ValueError(f"Task type not recognised as valid task type!")


def get_default_response(tasktype):
    match tasktype:
        case "detect":
            return DEFAULT_DETECT_RESULT
        case "augment":
            return DEFAULT_AUGMENT_RESULT
        case _:
            raise ValueError(f"Task type not recognised as valid task type!")


def make_demonstrations(demonstrations):
    EVAL_COLS = [
        "The exercise description matched the selected theme (Yes/No)",
        "The exercise description matched the selected topic (Yes/No)",
        "Included concepts that were too advanced (Yes/No)",
    ]

    return "".join(
        [
            DEMONSTRATION_TEMPLATE.replace("$THEME$", row["theme"])
            .replace("$TOPIC$", row["topic"])
            .replace("$CONCEPTS$", get_allowed_concepts(row["concept"]))
            .replace("$DISALLOWED_CONCEPTS$", get_disallowed_concepts(row["concept"]))
            .replace("$TEXT$", row["problemDescription"])
            .replace("$CODE$", row["exampleSolution"])
            .replace("$THEMECORRECT$", row[EVAL_COLS[0]])
            .replace("$TOPICCORRECT$", row[EVAL_COLS[1]])
            .replace("$ADDITIONALCONCEPTS$", row[EVAL_COLS[2]])
            for row in demonstrations
        ]
    )


def make_prompt(row, task_type):
    match task_type:
        case "detect":
            return (
                DETECT_TEMPLATE.replace("$THEME$", row["theme"])
                .replace("$TOPIC$", row["topic"])
                .replace("$CONCEPTS$", get_allowed_concepts(row["concept"]))
                .replace(
                    "$DISALLOWED_CONCEPTS$", get_disallowed_concepts(row["concept"])
                )
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
            raise ValueError(f"Task type not recognised as valid task type!")


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


def sample_dataset(dataset, seed, num_random_demos, type_of_demonstrations):
    if num_random_demos is None or num_random_demos <= 0:
        return None

    labels_pos = ["yes", "yes", "no"]
    labels_neg = ["no", "no", "yes"]

    df = dataset.to_pandas()

    # Get ground-truth evaluations
    label_cols = df[GT_COLS]

    # Only consider rows where all labels match
    match_pos = (label_cols == labels_pos).all(axis=1)
    match_neg = (label_cols == labels_neg).all(axis=1)

    match type_of_demonstrations:
        case -1:
            demos = df[match_neg].sample(num_random_demos, random_state=seed)
        case 0:
            half = num_random_demos // 2
            demos = pd.concat(
                [
                    df[match_neg].sample(half, random_state=seed),
                    df[match_pos].sample(num_random_demos - half, random_state=seed),
                ],
                ignore_index=True,
            )
        case 1:
            demos = df[match_pos].sample(num_random_demos, random_state=seed)
        case _:
            raise ValueError("Unknown type of demonstration!")

    return Dataset.from_pandas(demos)
