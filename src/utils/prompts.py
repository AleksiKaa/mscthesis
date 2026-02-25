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
    "ThemeCorrect" : "yes" / "partially"/ "no",
    "TopicCorrect" : "yes" / "partially"/ "no",
    "ConceptCorrect" : "yes" / "partially"/ "no",
    "Explanation": your reasoning
}"""

GENERATE_EXERCISES_SYSTEM_PROMPT = """I want you to act as a programming teacher for an
introductory Dart course. Your students are programming
novices. I will provide some coding example exercises,
and it will be your job to invent new ones. They should
contain the following name-value pairs in JSON: ti-
tle, problemDescription, exampleSolution, starterCode,
tests. Your responses should be written in simple English.
Do not cite music lyrics or books. Do not include any
greetings, be concise. Do not mention trigger words associated
with mental or physical disorders, for example,
weight loss or diet."""

GENERATE_EXERCISES_TEMPLATE_ZEROSHOT = """Please generate a short programming exercise in Dart
based on the example that I will provide. It should
be about $THEME$, specifically $TOPIC$. It should be at
the same difficulty level as the example /or It should
be slightly more complex than the example. It should
mainly cover $CONCEPT2$ but can also include $CONCEPT2$.
Please follow the structure of the example and
stay within its scope. You are allowed to include the
following concepts in the new exercise: $CONCEPTS$. Do
not use loops. Your response should be a JSON of form
{
    ???
}.
"""  # MODIFY!!

GENERATE_EXERCISES_TEMPLATE_FEWSHOT = """Please generate a short programming exercise in Dart
based on the example that I will provide. It should
be about $THEME$, specifically $TOPIC$. It should be at
the same difficulty level as the example /or It should
be slightly more complex than the example. It should
mainly cover $CONCEPT2$ but can also include $CONCEPT2$.
Please follow the structure of the example and
stay within its scope. You are allowed to include the
following concepts in the new exercise: $CONCEPTS$. Do
not use loops. Your response should be a JSON string.
Here is the example: $EXAMPLE_EXERCISE$
"""  # MODIFY!!

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
