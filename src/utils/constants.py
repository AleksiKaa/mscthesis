DEFAULT_DATA = "/home/kaariaa3/mscthesis/data/final_dataset.csv"

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

PIPE_RETURN_FULL_TEXT = False

PIPE_MAX_NEW_TOKENS = 250

MAX_GENERATED_TOKENS = 64

MODEL_TEMPERATURE = 0.7

GT_COLS = [
    "The exercise description matched the selected theme (Yes/No)",
    "The exercise description matched the selected topic (Yes/No)",
    "Included concepts that were too advanced (Yes/No)",
]

PRED_COLS = ["themeCorrect", "topicCorrect", "usesAdditionalConcepts"]

LABELS = ["yes", "no"]

POS_LABELS = ["yes", "yes", "no"]

DEFAULT_DETECT_RESULT = {
    "themeCorrect": "PARSE ERROR",
    "topicCorrect": "PARSE ERROR",
    "usesAdditionalConcepts": "PARSE ERROR",
}

DEFAULT_AUGMENT_RESULT = {
    "augmentedProblemDescription": "PARSE ERROR",
    "augmentedExampleSolution": "PARSE ERROR",
}

ERROR_RESULT = {"Error": "PARSE ERROR"}

CONCEPT_TO_CHAPTER_MAPPING = {
    "user input": 1,
    "program output": 1,
    "variables": 1,
    "arithmetics": 2,
    "conditional statements": 3,
    "logical operators": 3,
    "for loops": 4,
    "while loops": 4,
}

THEME_TO_TOPICS_MAPPING = {
    "literature": {
        "Agatha Christie",
        "Chimamanda Ngozi Adichie",
        "Dan Brown",
        "George RR Martin",
        "Haruki Murakami",
        "JK Rowling",
        "John Grisham",
        "Margaret Atwood",
        "Paulo Coelho",
        "Stephen King",
    },
    "handicrafts": {
        "beading",
        "candle making",
        "embroidery",
        "knitting",
        "macrame",
        "origami",
        "painting",
        "paper quilling",
        "pottery",
        "soap making",
    },
    "food": {
        "Cinnamon buns",
        "Cloudberry jam",
        "Karelian pasties",
        "Lingonberry sauce",
        "Mushroom soup",
        "Rye bread",
        "Salmon gravlax",
        "Salmon soup",
    },
    "classical music": {
        "Antonio Vivaldi",
        "Claudio Monteverdi",
        "Franz Joseph Haydn",
        "Giovanni Pierluigi da Palestrina",
        "Henry Purcell",
        "Jean-Baptiste Lully",
        "Johann Sebastian Bach",
        "Ludwig van Beethoven",
        "Wolfgang Amadeus Mozart",
    },
    "board games": {
        "7 Wonders",
        "Carcassonne",
        "Catan",
        "Codenames",
        "Dixit",
        "Dominion",
        "Monopoly",
        "Scrabble",
    },
    "pop music": {
        "Adele",
        "Ariana Grande",
        "Bruno Mars",
        "Calvin Harris",
        "David Guetta",
        "Dua Lipa",
        "Ed Sheeran",
        "Justin Bieber",
        "Rihanna",
        "Taylor Swift",
    },
    "sports": {
        "badminton",
        "biathlon",
        "cross-country skiing",
        "football",
        "golf",
        "handball",
        "ice hockey",
        "skiing",
    },
    "art": {
        "Claude Monet",
        "Henri Matisse",
        "Johannes Vermeer",
        "Leonardo da Vinci",
        "Michelangelo",
        "Pablo Picasso",
        "Rembrandt",
        "Salvador Dali",
        "Vincent van Gogh",
    },
    "pets": {
        "birds",
        "cats",
        "dogs",
        "fish",
        "guinea pigs",
        "hamsters",
        "lizards",
        "rabbits",
        "snakes",
        "turtles",
    },
    "historical landmarks": {
        "Acropolis of Athens",
        "Charles Bridge",
        "Colosseum",
        "Grand Canal in Venice",
        "Sagrada Familia",
        "Tower of London",
        "Vatican City",
    },
    "cartoons": {
        "Bugs Bunny",
        "Looney Tunes",
        "Mickey Mouse",
        "Popeye",
        "SpongeBob SquarePants",
        "The Jetsons",
        "The Powerpuff Girls",
        "Tom and Jerry",
    },
    "video games": {
        "Assassin's Creed",
        "Call of Duty",
        "FIFA",
        "Fortnite",
        "Grand Theft Auto",
        "Minecraft",
        "Overwatch",
        "Pokémon",
        "Super Mario",
        "The Legend of Zelda",
    },
    "outdoor activities": {
        "berry picking",
        "camping",
        "canoeing",
        "fishing",
        "hiking",
        "ice skating",
        "skiing",
        "snowboarding",
        "wildlife spotting",
    },
    "party games": {
        "Charades",
        "Duck Duck Goose",
        "Freeze Dance",
        "Limbo",
        "Pass the Parcel",
        "Pictionary",
        "Pin the Tail on the Donkey",
        "Simon Says",
        "Treasure Hunt",
    },
    "nature destinations": {
        "Hossa National Park",
        "Koli National Park",
        "Nuuksio National Park",
        "Oulanka National Park",
        "Pallas-Yllästunturi National Park",
        "Pyhä-Luosto National Park",
        "Repovesi National Park",
        "Urho Kekkonen National Park",
    },
    "Christmas": {
        "attending holiday parades",
        "baking cookies",
        "decorating the tree",
        "exchanging gifts",
        "hanging stockings",
        "lighting advent candles",
        "making gingerbread houses",
        "singing carols",
        "watching holiday movies",
        "writing letters to Santa",
    },
}

BATCH_SIZE = 8
