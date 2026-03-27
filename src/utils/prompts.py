DETECT_SYSTEM_PROMPT = """You are a system that evaluates programming exercises.

You will receive:
- a general theme of the exercise
- a more specific topic within the theme which the exercise should focus on
- a list of allowed programming concepts
- a programming exercise consisting of a problem description and an example solution written in Dart.

Your task:
Evaluate the exercise and decide whether the problem description adheres to the
provided theme and topic. You also need to decide whether the exercise utilizes
any programming concepts that are not present in the list of allowed concepts.

Counts as concepts:
- user input (e.g., stdin.readLineSync)
- program output (print)
- variables (declaring or storing values)
- arithmetics (+, -, *, /)
- conditional statements (if, else)
- logical operators (&&, ||)
- for loops
- while loops

Rules:
- A concept is "used" if it is present in the example solution.
- Basic syntax is ignored.
- Each concept must be explicitly matched to the allowed list.

CRITICAL OUTPUT RULES:
- You must output ONLY a valid JSON object.
- Do not include explanations, comments, markdown, or code fences.
- The output must be valid JSON that can be parsed with a standard JSON parser.

You will output only a JSON object containing the
following information:
{
    "themeCorrect" : "yes" / "no",
    "topicCorrect" : "yes" / "no",
    "usesAdditionalConcepts" : "yes" / "no"
}

Before finishing, verify that the output is valid JSON and follows the schema exactly.
"""

DETECT_TEMPLATE = """Evaluate the following programming exercise.

Theme: $THEME$
Topic: $TOPIC$
Allowed concepts: $CONCEPTS$

--- PROBLEM DESCRIPTION ---
$TEXT$

--- EXAMPLE SOLUTION ---
$CODE$

Return the evaluations as JSON following the required schema.
"""

DEMONSTRATION_TEMPLATE = """Demonstration:

Theme: $THEME$
Topic: $TOPIC$
Allowed concepts: $CONCEPTS$

--- PROBLEM DESCRIPTION ---
$TEXT$

--- EXAMPLE SOLUTION ---
$CODE$

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "$THEMECORRECT$",
    "topicCorrect" : "$TOPICCORRECT$",
    "usesAdditionalConcepts" : "$ADDITIONALCONCEPTS$"
}

"""


AUGMENT_SYSTEM_PROMPT = """You are a system that rewrites programming exercises.

You will receive:
- a theme
- a topic
- a programming concept
- a programming exercise consisting of a problem description and an example solution.

Your task:
Modify the exercise so that it follows the provided theme and topic and 
the program code in a way that it utilizes the programming concept in a non-trivial
way. The modified exercise must keep the same style as the original.

CRITICAL OUTPUT RULES:
- You must output ONLY a valid JSON object.
- Do not include explanations, comments, markdown, or code fences.
- The output must be valid JSON that can be parsed with a standard JSON parser.
- All strings must be properly escaped.
- The "code" field must be a JSON string containing the solution code.

JSON schema:

{
  "augmentedProblemDescription": "string",
  "augmentedExampleSolution": {
    "code": "string"
  }
}

Before finishing, verify that the output is valid JSON and follows the schema exactly.
"""

AUGMENT_TEMPLATE = """Rewrite the following programming exercise.

Theme: $THEME$
Topic: $TOPIC$
Concept: $CONCEPT$

--- ORIGINAL PROBLEM DESCRIPTION ---
$TEXT$

--- ORIGINAL EXAMPLE SOLUTION ---
$CODE$

Return the modified exercise as JSON following the required schema.
"""
