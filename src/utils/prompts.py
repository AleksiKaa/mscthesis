DETECT_SYSTEM_PROMPT = """You are a system that evaluates programming exercises.

You will receive:
- a general theme of the exercise
- a more specific topic within the theme which the exercise should focus on
- a list of programming concepts
- a programming exercise consisting of a problem description and an example solution written in Dart.

Your task:
Evaluate the exercise and decide whether the problem description adheres to the
provided theme and topic. You also need to decide whether the exercise utilizes
programming concepts that are not present in the list of provided concepts.

Count as concepts:
- user input (e.g., stdin.readLineSync)
- program output (print)
- variables (declaring or storing values)
- arithmetics (+, -, *, /, ~/)
- conditional statements (if, else)
- logical operators (&&, ||)
- for loops 
- while loops

Rules:
- A concept is "used" if it is necessary for solving the task.
- Basic syntax is ignored.
- Each concept must be explicitly matched to the allowed list.

CRITICAL OUTPUT RULES:
- You must output ONLY a valid JSON object.
- Do not include explanations, comments, markdown, or code fences.
- The output must be valid JSON that can be parsed with a standard JSON parser.
- All strings must be properly escaped.

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

FIXED_DEMONSTRATIONS = """Demonstration:

Theme: literature
Topic: Agatha Christie
Allowed concepts: user input, program output, variables, arithmetics, conditional statements, logical operators

--- PROBLEM DESCRIPTION ---
Vincent van Gogh, the renowned post-impressionist painter, has a unique way of describing his paintings based on their emotional impact. The emotional impact is represented as numbers and is accompanied by the following textual descriptions:
<table>
<tr>
<th>Impact</th>
<th>Description</th>
</tr>
<tr>
<th>5</th>
<th>Masterpiece</th>
</tr>
<tr>
<th>4</th>
<th>Stunning</th>
</tr>
<tr>
<th>3</th>
<th>Impressive</th>
</tr>
<tr>
<th>2</th>
<th>Noticeable</th>
</tr>
<tr>
<th>1</th>
<th>Faint</th>
</tr>
</table>
Write a program that asks the user for a number and prints the textual description related to that number. If the user enters any other number, the program should print the message <code>Invalid impact!</code>.

Below is an example of the expected operation of the program.

<pre>
What impact?
<b>&lt; 3</b>
Impressive
</pre>

Another example.

<pre>
What impact?
<b>&lt; 6</b>
Invalid impact!
</pre>

--- EXAMPLE SOLUTION ---
{"code": "import 'dart:io';main() {  print('What impact?');  var impact = int.parse(stdin.readLineSync()!);  while (true) {    if (impact == 5) {      print('Masterpiece');      break;    } else if (impact == 4) {      print('Stunning');      break;    } else if (impact == 3) {      print('Impressive');      break;    } else if (impact == 2) {      print('Noticeable');      break;    } else if (impact == 1) {      print('Faint');      break;    } else {      print('Invalid impact!');      break;    }  }}"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "no",
    "topicCorrect" : "no",
    "usesAdditionalConcepts" : "yes"
}

Demonstration:

Theme: outdoor activities
Topic: ice skating
Allowed concepts: user input, program output, variables, arithmetics, conditional statements, logical operators

--- PROBLEM DESCRIPTION ---
In a local ice skating competition, the scores are given in the range of 1 to 10 and are accompanied by the following textual descriptions:\n<table>\n<tr>\n<th>Score</th>\n<th>Assessment</th>\n</tr>\n<tr>\n<th>10</th>\n<th>Excellent</th>\n</tr>\n<tr>\n<th>9</th>\n<th>Very Good</th>\n</tr>\n<tr>\n<th>8</th>\n<th>Good</th>\n</tr>\n<tr>\n<th>7</th>\n<th>Fair</th>\n</tr>\n<tr>\n<th>6</th>\n<th>Satisfactory</th>\n</tr>\n<tr>\n<th>5</th>\n<th>Below Average</th>\n</tr>\n<tr>\n<th>4</th>\n<th>Poor</th>\n</tr>\n<tr>\n<th>3</th>\n<th>Very Poor</th>\n</tr>\n<tr>\n<th>2</th>\n<th>Bad</th>\n</tr>\n<tr>\n<th>1</th>\n<th>Terrible</th>\n</tr>\n</table>\nWrite a program that asks the user for a score and prints the textual description related to that score. If the user enters any other number, the program should print the message <code>Invalid Score!</code>.\n\nBelow is an example of the expected operation of the program.\n\n<pre>\nWhat is the score?\n<b>&lt; 7</b>\nFair\n</pre>\n\nAnother example.\n\n<pre>\nWhat is the score?\n<b>&lt; 11</b>\nInvalid Score!\n</pre>

--- EXAMPLE SOLUTION ---
{'code': "import 'dart:io';  main() {   print('What is the score?');    var score = int.parse(stdin.readLineSync()!);    if (score == 10) {     print('Excellent');   } else if (score == 9) {     print('Very Good');   } else if (score == 8) {     print('Good');   } else if (score == 7) {     print('Fair');   } else if (score == 6) {     print('Satisfactory');   } else if (score == 5) {     print('Below Average');   } else if (score == 4) {     print('Poor');   } else if (score == 3) {     print('Very Poor');   } else if (score == 2) {     print('Bad');   } else if (score == 1) {     print('Terrible');   } else {     print('Invalid Score!');   } }"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "yes",
    "topicCorrect" : "no",
    "usesAdditionalConcepts" : "no"
}

Demonstration:

Theme: literature
Topic: Agatha Christie
Allowed concepts: user input, program output, variables, arithmetics, conditional statements, logical operators

--- PROBLEM DESCRIPTION ---
Agatha Christie, the famous novelist, has a rating scale for her novels. The ratings are represented as numbers and are accompanied by the following textual descriptions:
<table>
<tr>
<th>Rating</th>
<th>Description</th>
</tr>
<tr>
<th>5</th>
<th>Masterpiece</th>
</tr>
<tr>
<th>4</th>
<th>Excellent</th>
</tr>
<tr>
<th>3</th>
<th>Good</th>
</tr>
<tr>
<th>2</th>
<th>Fair</th>
</tr>
<tr>
<th>1</th>
<th>Below Average</th>
</tr>
</table>
Write a program that asks the user for a number and prints the textual description related to that number. If the user enters any other number, the program should print the message <code>Invalid rating!</code>.

Below is an example of the expected operation of the program.

<pre>
What rating?
<b>&lt; 3</b>
Good
</pre>

Another example.

<pre>
What rating?
<b>&lt; 6</b>
Invalid rating!
</pre>

--- EXAMPLE SOLUTION ---
{'code': "import 'dart:io';  main() {   print('What rating?');    var rating = int.parse(stdin.readLineSync()!);    if (rating == 5) {     print('Masterpiece');   } else if (rating == 4) {     print('Excellent');   } else if (rating == 3) {     print('Good');   } else if (rating == 2) {     print('Fair');   } else if (rating == 1) {     print('Below Average');   } else {     print('Invalid rating!');   } }"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "yes",
    "topicCorrect" : "yes",
    "usesAdditionalConcepts" : "no"
}

Demonstration:

Theme: sports
Topic: skiing
Allowed concepts: user input, program output, variables, arithmetics

--- PROBLEM DESCRIPTION ---
Write a program that asks the user for the distance of the ski slope and the speed of the skier, then prints the time it would take for the skier to finish. If the user enters the distance as 100 meters and speed as 20 meters per second, the program should print the time as 5 seconds. The program should work as follows:

```
100
20
5
```

--- EXAMPLE SOLUTION ---
{'code': "import 'dart:io';  main() {   var distance = int.parse(stdin.readLineSync()!);   var speed = int.parse(stdin.readLineSync()!);   var time = distance ~/ speed;   print('$time'); }"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "yes",
    "topicCorrect" : "yes",
    "usesAdditionalConcepts" : "yes"
}

Demonstration:

Theme: outdoor activities
Topic: berry picking
Allowed concepts: user input, program output

--- PROBLEM DESCRIPTION ---
Write a program that asks the user for their favorite type of berry. After this, the program prints a message to the user ‘You like berryType!’, where berryType is the type of berry entered by the user. For example, with the input `Strawberry`, the program output is as follows:

```
What is your favorite berry?
Strawberry
You like Strawberry!
```

Similarly, if the user enters the name `Blueberry`, the program output is as follows:

```
What is your favorite berry?
Blueberry
You like Blueberry!
```

--- EXAMPLE SOLUTION ---
{'code': "import 'dart:io';main() {  print('What is your favorite berry?');  var berry = stdin.readLineSync();  print('You like $berry!');}"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "no",
    "topicCorrect" : "no",
    "usesAdditionalConcepts" : "no"
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
