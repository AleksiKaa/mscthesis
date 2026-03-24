import json
import random
import numpy as np

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


def get_system_prompt(task, demonstrations=None):
    match task:
        case "detect":
            system_prompt = ""

            # Use demos
            if demonstrations is not None:
                system_prompt += "Use the demonstrations below as examples on how to answer the question.\n\n"
                system_prompt += make_demonstrations(demonstrations)

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


def make_demonstrations(demonstrations, include_disallowed=False):
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


def sample_dataset_indices(rng, n_rows, size):
    if size is None or size <= 0:
        return None

    # Randomly select indices for demos, each row gets own demos, reproducible across runs
    return rng.integers(low=0, high=n_rows, size=(n_rows, size))


def create_demonstrations_set(dataset, idx_a=None, idx_b=None):
    if idx_a is None and idx_b is None:
        return None
    if idx_a is None:
        return dataset.select(idx_b)
    if idx_b is None:
        return dataset.select(idx_a)

    return dataset.select(np.unique(np.concatenate([idx_a, idx_b])))
