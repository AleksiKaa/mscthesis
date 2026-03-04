JUDGE_SYSTEM_PROMPT = """I want you to act as a programming teacher for an in-
troductory Dart course. Your students are programming
novices. I will provide some coding example exercises,
and it will be your job to critique them. Your responses 
should be written in simple English."""

JUDGE_TEMPLATE = """You are evaluating a programming exercise.

Intended theme: $THEME$
Intended topic: $TOPIC$
Intended programming concept: $CONCEPT$

Exercise description:
$TEXT$

Example solution:
$CODE$

Analyze the exercise step by step.

Step 1 — Identify what the exercise is actually about.
Step 2 — Evaluate whether the exercise description matches the selected theme.
Step 3 — Evaluate whether the exercise description matches the selected topic.
Step 4 — Evaluate whether the exercise description matches the selected programming concept.
Step 5 — Provide a final explanation of evaluation.

Return only a raw JSON text of form:
{
    "themeCorrect" : "yes" / "partially"/ "no",
    "topicCorrect" : "yes" / "partially"/ "no",
    "conceptCorrect" : "yes" / "no",
    "explanation": your reasoning
}"""

AUGMENT_SYSTEM_PROMPT = """
You will be provided with a theme, a topic, 
a concept, and a programming exercise consisting
of a problem description and an example solution.
Your goal is to modify the exercise so that it no
longer corresponds to the provided theme, topic,
and concept. The modified exercises should have
the same format as the original one.

You will output only a JSON object containing the
following information: 
{
    augmentedProblemDescription: modifiedDescription,
    augmentedExampleSolution: {'code': modifiedSolution}}
}"""

AUGMENT_TEMPLATE = """
Theme: $THEME$
Topic: $TOPIC$
Concept: $CONCEPT$

Problem description: $TEXT$

Example solution: $CODE$
"""

GENERATE_EXERCISES_TEMPLATE_EXPLICIT = ""

GENERATE_EXERCISES_TEMPLATE_IMPLICIT = ""

EXERCISE_CONCEPTS = [
    "user input",
    "program output",
    "variables",
    "arithmetics",
    "conditional statements",
    "logical operators",
]

EXERCISE_THEMES = [
    "Christmas",
    "classical music",
    "food",
    "historical landmarks",
    "literature",
    "party games",
    "video games",
    "outdoor activities",
    "art",
    "board games",
    "cartoons",
    "handicrafts",
    "nature destinations",
    "pets",
    "pop music",
    "sports",
]
