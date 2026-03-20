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

Theme: handicrafts
Topic: pottery
Allowed concepts: user input, program output, variables, arithmetics, conditional statements, logical operators

--- PROBLEM DESCRIPTION ---
In the world of The Legend of Zelda, Link has discovered various treasures in a dungeon, each rated on a scale from 1 to 5, where each number corresponds to a specific treasure quality:
<table>
<tr>
<th>Rating</th>
<th>Description</th>
</tr>
<tr>
<th>5</th>
<th>Legendary</th>
</tr>
<tr>
<th>4</th>
<th>Rare</th>
</tr>
<tr>
<th>3</th>
<th>Uncommon</th>
</tr>
<tr>
<th>2</th>
<th>Common</th>
</tr>
<tr>
<th>1</th>
<th>Junk</th>
</tr>
</table>
Write a program that asks the user for a number and prints the treasure quality related to that number. If the user enters any other number, the program should print the message <code>Invalid rating!</code>.

Below is an example of the expected operation of the program.

<pre>
What rating?
<b>&lt; 3</b>
Uncommon
</pre>

Another example.

<pre>
What rating?
<b>&lt; 6</b>
Invalid rating!
</pre>

--- EXAMPLE SOLUTION ---
{"code": "import 'dart:io';main() {  print('What rating?');  var rating = int.parse(stdin.readLineSync()!);  String quality;  for (int i = 1; i <= 5; i++) {    if (rating == i) {      switch (i) {        case 5:          quality = 'Legendary';          break;        case 4:          quality = 'Rare';          break;        case 3:          quality = 'Uncommon';          break;        case 2:          quality = 'Common';          break;        case 1:          quality = 'Junk';          break;      }      print(quality);      return;    }  }  print('Invalid rating!');}"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "no",
    "topicCorrect" : "no",
    "usesAdditionalConcepts" : "yes"
}

Demonstration:

Theme: pop music
Topic: Dua Lipa
Allowed concepts: user input, program output, variables

--- PROBLEM DESCRIPTION ---
Write a program that asks the user to enter the start and end times of a song in seconds, and then prints the duration of the song. If the user enters the start time as 0 and the end time as 210, the program should print the number 210. Similarly, if the user enters the start time as 30 and the end time as 240, the program should print the number 210.

The program should work as follows:

```
0
210
210
```

```
30
240
210
```

--- EXAMPLE SOLUTION ---
{'code': "import 'dart:io';  main() {   var start = int.parse(stdin.readLineSync()!);   var end = int.parse(stdin.readLineSync()!);   var duration = end - start;   print('$duration'); }"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "yes",
    "topicCorrect" : "no",
    "usesAdditionalConcepts" : "no"
}

Demonstration:

Theme: handicrafts
Topic: pottery
Allowed concepts: user input, program output, variables, arithmetics, conditional statements, logical operators

--- PROBLEM DESCRIPTION ---
In a pottery workshop, each pottery item is rated on a scale from 1 to 5, where each number corresponds to a specific quality description:
<table>
<tr>
<th>Rating</th>
<th>Description</th>
</tr>
<tr>
<th>5</th>
<th>Excellent</th>
</tr>
<tr>
<th>4</th>
<th>Good</th>
</tr>
<tr>
<th>3</th>
<th>Average</th>
</tr>
<tr>
<th>2</th>
<th>Below Average</th>
</tr>
<tr>
<th>1</th>
<th>Poor</th>
</tr>
</table>
Write a program that asks the user for a number and prints the quality description related to that number. If the user enters any other number, the program should print the message <code>Invalid rating!</code>.

Below is an example of the expected operation of the program.

<pre>
What rating?
<b>&lt; 3</b>
Average
</pre>

Another example.

<pre>
What rating?
<b>&lt; 6</b>
Invalid rating!
</pre>

--- EXAMPLE SOLUTION ---
{'code': "import 'dart:io';  main() {   print('What rating?');    var rating = int.parse(stdin.readLineSync()!);    if (rating == 5) {     print('Excellent');   } else if (rating == 4) {     print('Good');   } else if (rating == 3) {     print('Average');   } else if (rating == 2) {     print('Below Average');   } else if (rating == 1) {     print('Poor');   } else {     print('Invalid rating!');   } }"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "yes",
    "topicCorrect" : "yes",
    "usesAdditionalConcepts" : "no"
}

Demonstration:

Theme: nature destinations
Topic: Pyhä-Luosto National Park
Allowed concepts: user input, program output, variables

--- PROBLEM DESCRIPTION ---
Write a program that asks the user for the coordinates of two points in Pyhä-Luosto National Park, and then prints the distance between them. The distance is calculated by subtracting the x coordinate of the first point from the x coordinate of the second point and the y coordinate of the first point from the y coordinate of the second point. The distances should be absolute values.

The program should work as follows:

```
1
2
3
4
2
2
```

--- EXAMPLE SOLUTION ---
{'code': "import 'dart:io';  main() {   var x1 = int.parse(stdin.readLineSync()!);   var y1 = int.parse(stdin.readLineSync()!);   var x2 = int.parse(stdin.readLineSync()!);   var y2 = int.parse(stdin.readLineSync()!);   var distanceX = (x1 - x2).abs();   var distanceY = (y1 - y2).abs();   print('Distance in x: $distanceX');   print('Distance in y: $distanceY'); }"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "yes",
    "topicCorrect" : "yes",
    "usesAdditionalConcepts" : "yes"
}

Demonstration:

Theme: party games
Topic: Pass the Parcel
Allowed concepts: user input, program output, variables

--- PROBLEM DESCRIPTION ---
Write a program that asks the user for their name and their favorite color. After this, the program prints a message to the user ‘Hello name, your favorite color is color!’, where name is the name entered by the user and color is the user's favorite color. For example, with the input `Eliel` and `Blue`, the program output is as follows:

```
What is your name?
Eliel
What is your favorite color?
Blue
Hello Eliel, your favorite color is Blue!
```

Similarly, if the user enters the name `Lilja` and `Red`, the program output is as follows:

```
What is your name?
Lilja
What is your favorite color?
Red
Hello Lilja, your favorite color is Red!
```

--- EXAMPLE SOLUTION ---
{'code': "import 'dart:io';main() {  print('What is your name?');  var name = stdin.readLineSync();  print('What is your favorite color?');  var color = stdin.readLineSync();  print('Hello $name, your favorite color is $color!');}"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "no",
    "topicCorrect" : "no",
    "usesAdditionalConcepts" : "no"
}

Demonstration:

Theme: classical music
Topic: Johann Sebastian Bach
Allowed concepts: user input, program output, variables

--- PROBLEM DESCRIPTION ---
Write a program that asks the user for their favorite composition by Giovanni Pierluigi da Palestrina. After this, the program prints a message to the user 'Your favorite Palestrina composition is composition!', where composition is the composition entered by the user. For example, with the input `Missa Papae Marcelli`, the program output is as follows:

```
What is your favorite Palestrina composition?
Missa Papae Marcelli
Your favorite Palestrina composition is Missa Papae Marcelli!
```

Similarly, if the user enters the composition `Stabat Mater`, the program output is as follows:

```
What is your favorite Palestrina composition?
Stabat Mater
Your favorite Palestrina composition is Stabat Mater!
```

--- EXAMPLE SOLUTION ---
{"code": "import 'dart:io';main() {  print('What is your favorite Palestrina composition?');  var composition = stdin.readLineSync();  print('Your favorite Palestrina composition is $composition!');}"}

--- GROUND-TRUTH EVALUATION ---
{
    "themeCorrect" : "yes",
    "topicCorrect" : "no",
    "usesAdditionalConcepts" : "yes"
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
