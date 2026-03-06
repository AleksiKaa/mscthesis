JUDGE_SYSTEM_PROMPT = """You will act as a programming teacher for an
introductory Dart course. Your students are programming
novices. you will be provided some coding example exercises,
and it will be your job to critique them. Your responses 
should be written in simple English.

Analyze the exercise step by step.

Step 1 — Identify what the exercise is actually about.
Step 2 — Evaluate whether the exercise description matches the selected theme.
Step 3 — Evaluate whether the exercise description matches the selected topic.
Step 4 — Evaluate whether the exercise description matches the selected programming concept.
Step 5 — Provide a final explanation of evaluation.

You will output only a JSON object containing the
following information:
{
    "themeCorrect" : "yes" / "partially"/ "no",
    "topicCorrect" : "yes" / "partially"/ "no",
    "conceptCorrect" : "yes" / "no",
    "explanation": your reasoning
"""

JUDGE_TEMPLATE = """Theme: $THEME$
Topic: $TOPIC$
Concept: $CONCEPT$

Problem description: $TEXT$

Example solution: $CODE$
}"""

AUGMENT_SYSTEM_PROMPT = """
You will be provided with a theme, a topic, 
, and a programming exercise consisting
of a problem description and an example solution.
Your goal is to modify the exercise so that it no
longer corresponds to the provided theme and topic. 
The modified exercises should have the same format
as the original one.

You will output only a JSON object containing the
following information: 
{
    "augmentedProblemDescription": $modifiedDescription,
    "augmentedExampleSolution": {"code": $modifiedSolution}
}"""

AUGMENT_TEMPLATE = """Theme: $THEME$
Topic: $TOPIC$

Problem description: $TEXT$

Example solution: $CODE$
"""
