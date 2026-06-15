# =========================================================
# 🎓 Personalized AI Tutor System (Adaptive Learning Edition)
#
# A Streamlit app that:
#   1) Lets a student type ANY topic they want to study
#   2) Generates a 9-question quiz (Memorization / Understanding / Application)
#   3) Scores the first attempt
#   4) Uses a small TensorFlow model to detect the student's WEAK area
#   5) Shows a personalized teaching module for that weak area
#   6) Generates a NEW adaptive quiz (focused on the weak area)
#   7) Compares both attempts and shows improvement
#   8) Displays everything on a friendly, colorful dashboard
#
# 100% TensorFlow, no external APIs, single file, beginner-friendly.
# =========================================================

import os
import warnings

# Hide unnecessary TensorFlow / Python warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# =========================================================
# PAGE CONFIGURATION + SIMPLE CUSTOM STYLING
# =========================================================
st.set_page_config(
    page_title="Personalized AI Tutor",
    page_icon="🎓",
    layout="centered",
)

# A little bit of custom CSS to make the app feel more "modern"
st.markdown(
    """
    <style>
    /* Make metric boxes look like little cards */
    div[data-testid="stMetric"] {
        background-color: rgba(120, 120, 255, 0.08);
        border: 1px solid rgba(120, 120, 255, 0.25);
        border-radius: 12px;
        padding: 12px 8px;
    }
    /* Slightly rounder buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 1) KNOWLEDGE BASE
# Each topic contains everything we need to:
#  - write quiz questions (Memorization / Understanding / Application)
#  - teach the student if they are weak in a category
# =========================================================
KNOWLEDGE_BASE = {

    "stack": {
        "name": "Stack (Data Structure)",
        "definition": "A stack is a linear data structure that stores elements in LIFO (Last In, First Out) order, meaning the last element added is the first one removed.",
        "key_terms": ["LIFO (Last In, First Out)", "Push operation", "Pop operation", "Peek/Top operation", "Stack Overflow & Underflow"],
        "explanation": "Think of a stack like a pile of plates. You can only add (push) or remove (pop) a plate from the top. The 'push' operation places a new item on top, and 'pop' removes the item from the top. You cannot access items in the middle without removing the ones above them first.",
        "examples": ["The Undo feature in a text editor", "The Back button history in a web browser", "The function call stack used by programming languages"],
        "scenarios": [
            {
                "situation": "You press the 'Undo' button three times in a row in a document editor that uses a stack to store your actions.",
                "correct": "The three most recent actions are reversed in last-to-first order.",
                "wrongs": ["The very first action you ever performed is reversed first.", "All actions in the document are reversed at once.", "Nothing happens because undo does not use a stack."],
            },
            {
                "situation": "A stack has a fixed size of 5 and currently holds 5 items. You try to push one more item onto it.",
                "correct": "A stack overflow occurs because the stack is full.",
                "wrongs": ["The oldest item is automatically removed to make space.", "The new item is stored in a separate temporary stack.", "The stack automatically doubles its size."],
            },
            {
                "situation": "You call function A, which calls function B, which calls function C, using the call stack.",
                "correct": "C finishes first, then B, then A.",
                "wrongs": ["A finishes first, then B, then C.", "All three finish at exactly the same time.", "B finishes first, then A, then C."],
            },
        ],
    },

    "queue": {
        "name": "Queue (Data Structure)",
        "definition": "A queue is a linear data structure that stores elements in FIFO (First In, First Out) order, meaning the first element added is the first one removed.",
        "key_terms": ["FIFO (First In, First Out)", "Enqueue operation", "Dequeue operation", "Front and Rear pointers", "Circular Queue"],
        "explanation": "Think of a queue like a line of people waiting at a ticket counter. The person who joins first is served first. New people join at the back (enqueue) and are removed from the front (dequeue).",
        "examples": ["People waiting in a checkout line", "Print jobs waiting to be printed", "Tasks waiting in a CPU scheduling queue"],
        "scenarios": [
            {
                "situation": "Three customers, A, B, and C, join a queue in that order at a bank counter.",
                "correct": "Customer A is served first, followed by B, then C.",
                "wrongs": ["Customer C is served first because they joined last.", "All three customers are served at the same time.", "The order does not matter for a queue."],
            },
            {
                "situation": "A printer uses a queue to manage print jobs. Job 1 is sent, then Job 2, then Job 3.",
                "correct": "Job 1 will be printed first, then Job 2, then Job 3.",
                "wrongs": ["Job 3 will be printed first because it was added last.", "All jobs print at exactly the same time.", "The printer chooses a random job to print first."],
            },
            {
                "situation": "A queue is completely full and a new element needs to be added.",
                "correct": "A queue overflow occurs and the new element cannot be added.",
                "wrongs": ["The oldest element is automatically replaced.", "The queue turns into a stack temporarily.", "The new element is added to the front instead."],
            },
        ],
    },

    "linked list": {
        "name": "Linked List (Data Structure)",
        "definition": "A linked list is a linear data structure where each element (node) contains data and a pointer/reference to the next node in the sequence.",
        "key_terms": ["Node", "Pointer / Reference", "Head", "Tail", "Traversal"],
        "explanation": "Imagine a treasure hunt where each clue tells you the location of the next clue. A linked list works the same way: each node holds some data and points to the next node, so you must follow the chain from the 'head' to reach any element.",
        "examples": ["A playlist where each song points to the next song", "A train where each car is connected to the next", "Browser history stored as a chain of pages"],
        "scenarios": [
            {
                "situation": "You want to insert a new node at the very beginning of a linked list.",
                "correct": "Point the new node to the current head, then make the new node the new head.",
                "wrongs": ["Search through the entire list to find an empty slot.", "Shift every existing element one position in memory.", "It is impossible to add a node at the beginning."],
            },
            {
                "situation": "You need to find the 5th element in a linked list of 100 elements.",
                "correct": "You must traverse from the head, following pointers one by one until you reach the 5th node.",
                "wrongs": ["You can jump directly to position 5 like an array index.", "The list automatically sorts itself so element 5 comes first.", "Linked lists cannot find elements by position at all."],
            },
            {
                "situation": "The 'next' pointer of the last node in a linked list is set to null.",
                "correct": "This indicates the end of the list has been reached.",
                "wrongs": ["This means the list is empty.", "This causes the list to loop back to the head.", "This means the list has an error and will crash."],
            },
        ],
    },

    "binary search": {
        "name": "Binary Search (Algorithm)",
        "definition": "Binary search is an algorithm that finds the position of a target value within a sorted array by repeatedly dividing the search range in half.",
        "key_terms": ["Sorted array", "Midpoint", "Low and High pointers", "Divide and Conquer", "Time Complexity O(log n)"],
        "explanation": "Binary search works like guessing a number between 1 and 100. Instead of guessing one by one, you guess 50 first. If the target is higher, you search only the upper half; if lower, the lower half. You keep cutting the search space in half until you find it.",
        "examples": ["Searching for a word in a dictionary by opening to the middle first", "Finding a contact in an alphabetically sorted phone book", "Searching for a value in a sorted database index"],
        "scenarios": [
            {
                "situation": "You run binary search on an array that is NOT sorted.",
                "correct": "The algorithm may give incorrect or unreliable results.",
                "wrongs": ["Binary search will automatically sort the array first.", "Binary search works exactly the same as on a sorted array.", "The algorithm will simply run faster."],
            },
            {
                "situation": "You are searching for the number 42 in a sorted array, and the middle element is 60.",
                "correct": "You should now search only the left half of the array (elements smaller than 60).",
                "wrongs": ["You should now search only the right half of the array.", "You should restart the search from the beginning.", "You should stop searching because 42 does not exist."],
            },
            {
                "situation": "An array has 1,000,000 sorted elements and you compare linear search with binary search for one element.",
                "correct": "Binary search will need far fewer comparisons than linear search.",
                "wrongs": ["Linear search will always be faster regardless of array size.", "Both algorithms need exactly the same number of comparisons.", "Binary search cannot be used on large arrays."],
            },
        ],
    },

    "photosynthesis": {
        "name": "Photosynthesis",
        "definition": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose (food) and oxygen.",
        "key_terms": ["Chlorophyll", "Sunlight (light energy)", "Carbon dioxide (CO2)", "Glucose (food)", "Oxygen (O2) release"],
        "explanation": "Plants act like tiny solar-powered factories. Their leaves contain a green pigment called chlorophyll, which captures sunlight. Using this light energy, the plant combines water (from its roots) and carbon dioxide (from the air) to make glucose for energy, releasing oxygen as a byproduct.",
        "examples": ["A houseplant growing faster near a sunny window", "Algae in a pond producing oxygen bubbles in sunlight", "Crops growing taller during long, sunny summer days"],
        "scenarios": [
            {
                "situation": "A plant is kept in a completely dark room for two weeks.",
                "correct": "The plant will turn yellow and become weak because it cannot photosynthesize without light.",
                "wrongs": ["The plant will grow faster because it conserves energy.", "The plant will produce more oxygen than usual.", "Nothing will change because plants do not need light."],
            },
            {
                "situation": "You cover one leaf of a plant with foil for a week, blocking sunlight from that leaf, while the rest of the plant stays in sunlight.",
                "correct": "The covered leaf will produce little to no glucose, while the rest of the plant continues photosynthesis normally.",
                "wrongs": ["The entire plant will stop photosynthesis completely.", "The covered leaf will produce more glucose than the others.", "Covering a leaf has no effect on photosynthesis."],
            },
            {
                "situation": "The amount of carbon dioxide in a sealed greenhouse is increased while light and water stay constant.",
                "correct": "Photosynthesis may increase, since CO2 is one of the raw materials plants use to make glucose.",
                "wrongs": ["Photosynthesis will stop completely.", "The plants will start releasing carbon dioxide instead of oxygen.", "Increasing CO2 has no effect on photosynthesis at all."],
            },
        ],
    },

    "newtons laws of motion": {
        "name": "Newton's Laws of Motion",
        "definition": "Newton's Laws of Motion are three fundamental principles that describe the relationship between a body's motion and the forces acting on it.",
        "key_terms": ["Inertia (1st Law)", "F = m x a (2nd Law)", "Action-Reaction pairs (3rd Law)", "Net force", "Equilibrium"],
        "explanation": "The first law says an object stays at rest or moves at constant velocity unless a force acts on it (inertia). The second law says acceleration depends on the net force and mass (F = m x a). The third law says every action has an equal and opposite reaction.",
        "examples": ["A ball rolling on a frictionless surface forever (1st law)", "Pushing a shopping cart harder makes it accelerate faster (2nd law)", "A rocket launches upward as gas is pushed downward (3rd law)"],
        "scenarios": [
            {
                "situation": "A car suddenly brakes while moving forward, and the passengers are not wearing seatbelts.",
                "correct": "The passengers continue moving forward due to inertia (Newton's First Law).",
                "wrongs": ["The passengers are pushed backward into their seats.", "The passengers immediately stop moving along with the car.", "The passengers float upward toward the ceiling."],
            },
            {
                "situation": "You push a heavy box and a light box with the same amount of force.",
                "correct": "The light box will accelerate faster than the heavy box (Newton's Second Law).",
                "wrongs": ["Both boxes will accelerate at exactly the same rate.", "The heavy box will accelerate faster than the light box.", "Neither box will move at all."],
            },
            {
                "situation": "A swimmer pushes water backward with their arms in order to move forward.",
                "correct": "The water pushes the swimmer forward with an equal and opposite force (Newton's Third Law).",
                "wrongs": ["The swimmer moves backward instead of forward.", "The water has no effect on the swimmer's motion.", "The swimmer's motion is unrelated to pushing the water."],
            },
        ],
    },

    "cell": {
        "name": "The Cell (Biology)",
        "definition": "A cell is the basic structural and functional unit of all living organisms, capable of carrying out life processes such as energy production, growth, and reproduction.",
        "key_terms": ["Cell membrane", "Nucleus", "Mitochondria", "Cytoplasm", "Cell wall (in plant cells)"],
        "explanation": "A cell is like a tiny city. The cell membrane is the city wall, controlling what goes in and out. The nucleus is city hall, controlling activities through DNA. Mitochondria are power plants producing energy. The cytoplasm is the space where everything floats and reactions happen.",
        "examples": ["A red blood cell carrying oxygen through the body", "A plant cell with a rigid cell wall for structure", "A nerve cell transmitting electrical signals"],
        "scenarios": [
            {
                "situation": "The mitochondria in a muscle cell stop functioning properly.",
                "correct": "The cell will produce less energy, making the muscle weak and tired more quickly.",
                "wrongs": ["The cell will produce more energy than before.", "The cell will immediately turn into a plant cell.", "There will be no effect on the cell's energy levels."],
            },
            {
                "situation": "A plant cell and an animal cell are placed side by side under a microscope.",
                "correct": "The plant cell will have a visible cell wall and chloroplasts, while the animal cell will not.",
                "wrongs": ["Both cells will look exactly identical in every way.", "The animal cell will have a cell wall but the plant cell will not.", "Neither cell will have a nucleus."],
            },
            {
                "situation": "A harmful substance damages the cell membrane, making it unable to control what enters or exits.",
                "correct": "Unwanted substances may enter the cell and essential materials may leak out, harming the cell.",
                "wrongs": ["The cell will function completely normally with no issues.", "The cell will become stronger and more protected.", "The nucleus will instantly take over the membrane's job."],
            },
        ],
    },

    "object oriented programming": {
        "name": "Object-Oriented Programming (OOP)",
        "definition": "Object-Oriented Programming (OOP) is a programming approach based on the concept of 'objects', which contain data (attributes) and code (methods) that operate on that data.",
        "key_terms": ["Class and Object", "Encapsulation", "Inheritance", "Polymorphism", "Abstraction"],
        "explanation": "Think of a 'class' as a blueprint for a house, and an 'object' as an actual house built from that blueprint. Encapsulation keeps data safe inside the object. Inheritance lets a new class reuse features of an existing class, like a child inheriting traits from a parent. Polymorphism allows the same action to behave differently for different objects.",
        "examples": ["A 'Car' class used to create many car objects with different colors and models", "An 'Animal' class with 'Dog' and 'Cat' classes inheriting from it", "A 'speak()' method that behaves differently for a Dog (barks) and a Cat (meows)"],
        "scenarios": [
            {
                "situation": "You create a 'Vehicle' class with common properties, then create 'Car' and 'Bike' classes that reuse those properties.",
                "correct": "This is an example of inheritance, where Car and Bike inherit from Vehicle.",
                "wrongs": ["This is an example of encapsulation only.", "This means Car and Bike are completely unrelated to Vehicle.", "This is not possible in object-oriented programming."],
            },
            {
                "situation": "Both a 'Dog' object and a 'Cat' object have a method called 'makeSound()', but each produces a different sound when called.",
                "correct": "This demonstrates polymorphism, where the same method behaves differently for different objects.",
                "wrongs": ["This is an error because method names must be unique.", "This demonstrates that Dog and Cat are the same class.", "This has nothing to do with object-oriented programming."],
            },
            {
                "situation": "A class hides its internal data and only allows access through specific methods (getters and setters).",
                "correct": "This demonstrates encapsulation, protecting the internal state of the object.",
                "wrongs": ["This demonstrates inheritance from a parent class.", "This means the data is publicly accessible to everyone.", "This is an example of polymorphism."],
            },
        ],
    },
}


def generic_topic_data(topic):
    """
    Fallback content for topics that are NOT in our knowledge base.
    It still produces a complete, usable set of quiz material,
    just less specific than the curated topics above.
    """
    name = topic.strip().title()
    return {
        "name": name,
        "definition": f"{name} is a topic from your study material that includes important definitions, core ideas, and real-world applications you need to learn.",
        "key_terms": [
            f"Definition of {name}",
            f"Core principles of {name}",
            f"Important examples of {name}",
            f"Common applications of {name}",
            f"Key vocabulary related to {name}",
        ],
        "explanation": f"To understand {name}, start with its basic definition, then learn how and why it works the way it does, and finally practice applying it to real situations. Breaking the topic into small steps makes it much easier to learn.",
        "examples": [
            f"A textbook section explaining {name} in detail",
            f"A real-life situation where {name} is used or observed",
            f"A practice problem based on {name}",
        ],
        "scenarios": [
            {
                "situation": f"While studying {name}, you come across a term you don't fully understand.",
                "correct": "Look up the definition, write it in your own words, and connect it to something you already know.",
                "wrongs": ["Skip it completely and hope it won't matter.", "Memorize the term without understanding its meaning.", "Assume it is unimportant and move on."],
            },
            {
                "situation": f"You are asked to apply your knowledge of {name} to a brand-new problem you haven't seen before.",
                "correct": f"Break the problem into smaller parts and apply what you know about {name} step by step.",
                "wrongs": ["Give up immediately because the problem is new.", "Use a completely unrelated idea to solve it.", "Wait for someone else to solve it for you."],
            },
            {
                "situation": f"You've studied {name} for a while but still feel unsure how it connects to real life.",
                "correct": "Search for real-world examples and try teaching the concept to someone else to test your understanding.",
                "wrongs": [f"Conclude that {name} has no real-world use.", "Stop studying the topic altogether.", "Avoid practicing any related questions."],
            },
        ],
    }


def find_topic(user_input):
    """
    Tries to match the user's typed topic to a topic in our
    KNOWLEDGE_BASE. If no good match is found, it falls back to
    generic_topic_data() so the app still works for ANY topic.
    """
    normalized = user_input.strip().lower()

    # Try direct substring matches first (in either direction)
    for key, data in KNOWLEDGE_BASE.items():
        if key in normalized or normalized in key:
            return key, data
        if data["name"].lower() in normalized or normalized in data["name"].lower():
            return key, data

    # Try matching on shared words (e.g. "stacks" matches "stack")
    input_words = set(normalized.replace("'", "").split())
    for key, data in KNOWLEDGE_BASE.items():
        key_words = set(key.split())
        if key_words & input_words:
            return key, data

    # No match found -> build generic content for this topic
    return None, generic_topic_data(user_input)


# =========================================================
# 2) QUESTION GENERATION HELPERS
# =========================================================

def get_distractor_pool(field, exclude_key=None):
    """
    Collects values of a given field (e.g. 'definition', 'key_terms',
    'examples') from ALL OTHER topics in the knowledge base. These are
    used as "wrong answer" options for multiple-choice questions.
    """
    items = []
    for key, data in KNOWLEDGE_BASE.items():
        if key == exclude_key:
            continue
        value = data.get(field)
        if isinstance(value, list):
            items.extend(value)
        else:
            items.append(value)
    return items


def build_memorization_pool(topic_key, topic_data):
    """Builds a pool of 'Memorization' (fact / definition recall) questions."""
    pool = []

    # --- Question type 1: match the correct definition ---
    other_defs = get_distractor_pool("definition", topic_key)
    distractors = random.sample(other_defs, min(3, len(other_defs)))
    while len(distractors) < 3:
        distractors.append("This is not related to the current topic.")
    options = [topic_data["definition"]] + distractors
    random.shuffle(options)
    pool.append({
        "category": "Memorization",
        "question": f"Which of the following correctly defines '{topic_data['name']}'?",
        "options": options,
        "answer": topic_data["definition"],
    })

    # --- Question type 2: one question per key term ---
    other_terms = get_distractor_pool("key_terms", topic_key)
    for term in topic_data["key_terms"]:
        distractor_terms = random.sample(other_terms, min(3, len(other_terms)))
        while len(distractor_terms) < 3:
            distractor_terms.append("Unrelated concept")
        options = [term] + distractor_terms
        random.shuffle(options)
        pool.append({
            "category": "Memorization",
            "question": f"Which of these is a key term/concept related to '{topic_data['name']}'?",
            "options": options,
            "answer": term,
        })

    # --- Question type 3: how many key terms exist? ---
    correct_count = len(topic_data["key_terms"])
    wrong_candidates = set()
    for delta in [-2, -1, 1, 2, 3]:
        val = correct_count + delta
        if val > 0 and val != correct_count:
            wrong_candidates.add(val)
    wrong_counts = random.sample(list(wrong_candidates), min(3, len(wrong_candidates)))
    options = [str(correct_count)] + [str(w) for w in wrong_counts]
    random.shuffle(options)
    pool.append({
        "category": "Memorization",
        "question": f"How many key terms/concepts are highlighted in your study notes for '{topic_data['name']}'?",
        "options": options,
        "answer": str(correct_count),
    })

    return pool


def build_understanding_pool(topic_key, topic_data):
    """Builds a pool of 'Understanding' (concept explanation) questions."""
    pool = []
    other_explanations = get_distractor_pool("explanation", topic_key)
    other_examples = get_distractor_pool("examples", topic_key)

    # --- Question type 1: match the correct explanation (generated twice for variety) ---
    for _ in range(2):
        distractors = random.sample(other_explanations, min(3, len(other_explanations)))
        while len(distractors) < 3:
            distractors.append("This explanation does not relate to the topic at all.")
        options = [topic_data["explanation"]] + distractors
        random.shuffle(options)
        pool.append({
            "category": "Understanding",
            "question": f"Which statement best explains the idea behind '{topic_data['name']}'?",
            "options": options,
            "answer": topic_data["explanation"],
        })

    # --- Question type 2: match a real-life example ---
    for example in topic_data["examples"]:
        distractor_examples = random.sample(other_examples, min(3, len(other_examples)))
        while len(distractor_examples) < 3:
            distractor_examples.append("An unrelated everyday activity")
        options = [example] + distractor_examples
        random.shuffle(options)
        pool.append({
            "category": "Understanding",
            "question": f"Which of these is a good real-life example of '{topic_data['name']}'?",
            "options": options,
            "answer": example,
        })

    return pool


def build_application_pool(topic_data):
    """Builds a pool of 'Application' (scenario-based) questions."""
    pool = []
    for scenario in topic_data["scenarios"]:
        # Build 2 shuffled variants of the same scenario for more variety
        for _ in range(2):
            options = [scenario["correct"]] + list(scenario["wrongs"])
            random.shuffle(options)
            pool.append({
                "category": "Application",
                "question": scenario["situation"] + " What is the most likely outcome or best response?",
                "options": options,
                "answer": scenario["correct"],
            })
    return pool


def generate_quiz(topic_key, topic_data, distribution, used_questions):
    """
    Generates a full quiz based on a distribution dict, e.g.
    {"Memorization": 3, "Understanding": 3, "Application": 3}.
    Tries to avoid repeating questions already asked (used_questions).
    """
    pools = {
        "Memorization": build_memorization_pool(topic_key, topic_data),
        "Understanding": build_understanding_pool(topic_key, topic_data),
        "Application": build_application_pool(topic_data),
    }

    selected = []
    for category, count in distribution.items():
        pool = pools[category]
        random.shuffle(pool)

        # Prefer questions that haven't been asked before
        fresh = [q for q in pool if q["question"] not in used_questions]
        chosen = fresh[:count] if len(fresh) >= count else pool[:count]

        # If still not enough unique questions, allow repeats from the pool
        i = 0
        while len(chosen) < count:
            chosen.append(pool[i % len(pool)])
            i += 1

        selected.extend(chosen[:count])

    random.shuffle(selected)
    return selected


def get_requiz_distribution(weak_category):
    """
    Returns a question distribution that focuses ~70% of the
    re-quiz on the student's weakest category.
    """
    categories = ["Memorization", "Understanding", "Application"]
    if weak_category not in categories:
        # Balanced -> keep an even 3/3/3 split
        return {"Memorization": 3, "Understanding": 3, "Application": 3}

    distribution = {c: 1 for c in categories}
    distribution[weak_category] += 5          # weak area gets 6 questions total
    others = [c for c in categories if c != weak_category]
    distribution[others[0]] += 1               # one other gets 2 questions
    # the remaining category keeps just 1 question
    return distribution


def compute_scores(questions, answers):
    """Returns (scores, totals) dictionaries per category."""
    scores = {"Memorization": 0, "Understanding": 0, "Application": 0}
    totals = {"Memorization": 0, "Understanding": 0, "Application": 0}
    for q, a in zip(questions, answers):
        totals[q["category"]] += 1
        if a == q["answer"]:
            scores[q["category"]] += 1
    return scores, totals


# =========================================================
# 3) TENSORFLOW MODEL: AI-BASED WEAKNESS DETECTION
# =========================================================

WEAKNESS_LABELS = {
    0: "Memorization",
    1: "Understanding",
    2: "Application",
    3: "Balanced",
}


@st.cache_resource(show_spinner="🧠 Training AI weakness-detection model...")
def build_weakness_model():
    """
    Builds and trains a small TensorFlow Dense neural network.

    Input:  3 numbers (Memorization score %, Understanding score %, Application score %),
            each scaled between 0 and 1.
    Output: which area the student is WEAKEST in (or 'Balanced' if all are close).

    The model is trained on a small synthetic dataset, generated using a
    simple rule: the category with the lowest score is the "weak" one,
    unless all three scores are close together (then it's "Balanced").
    This is cached so training only happens once per app session.
    """
    np.random.seed(7)
    tf.random.set_seed(7)

    X = []
    y = []

    # Generate 600 random score combinations for training
    for _ in range(600):
        scores = np.random.rand(3)  # 3 random fractions between 0 and 1
        spread = scores.max() - scores.min()

        if spread < 0.2:
            label = 3  # Balanced: all scores are close together
        else:
            label = int(np.argmin(scores))  # weakest category index (0, 1, or 2)

        X.append(scores)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # A small, lightweight Sequential neural network
    model = Sequential([
        Dense(16, activation="relu", input_shape=(3,)),
        Dense(16, activation="relu"),
        Dense(4, activation="softmax"),  # 4 classes: Mem / Under / App / Balanced
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(X, y, epochs=40, verbose=0)
    return model


def predict_weakness(mem_frac, under_frac, app_frac, model):
    """
    Takes the student's scores (as fractions 0-1 for each category)
    and returns (weak_label_string, probability_array).
    """
    input_data = np.array([[mem_frac, under_frac, app_frac]], dtype=np.float32)
    prediction = model.predict(input_data, verbose=0)
    label_index = int(np.argmax(prediction))
    return WEAKNESS_LABELS[label_index], prediction[0]


# =========================================================
# 4) PERSONALIZED TEACHING MODULE
# =========================================================

def show_teaching_module(weak_label, topic_data):
    """Displays a personalized mini-lesson based on the student's weak area."""
    st.markdown(f"### 📌 Topic: {topic_data['name']}")

    if weak_label == "Memorization":
        st.markdown("#### 🧠 Focus Area: Key Facts & Definitions")
        st.success(f"**Definition:** {topic_data['definition']}")
        st.markdown("**🔑 Key Terms to Remember:**")
        for term in topic_data["key_terms"]:
            st.markdown(f"- {term}")
        st.caption("💡 Tip: Try covering the list above and writing the terms from memory!")

    elif weak_label == "Understanding":
        st.markdown("#### 🔍 Focus Area: Concept Explanation")
        st.info(topic_data["explanation"])
        st.markdown("**🌍 Real-life Examples:**")
        for ex in topic_data["examples"]:
            st.markdown(f"- {ex}")
        st.caption("💡 Tip: Try explaining this topic in your own words to a friend!")

    elif weak_label == "Application":
        st.markdown("#### 🧩 Focus Area: Applying Knowledge to Real Scenarios")
        for i, sc in enumerate(topic_data["scenarios"], start=1):
            with st.container():
                st.markdown(f"**Scenario {i}:** {sc['situation']}")
                st.success(f"✅ What happens / best response: {sc['correct']}")
                st.markdown("---")
        st.caption("💡 Tip: Try to think of one more real-life situation like these!")

    else:  # Balanced
        st.markdown("#### 🌟 You're Balanced! Here's a Quick Recap of Everything:")
        st.success(f"**Definition:** {topic_data['definition']}")
        st.info(topic_data["explanation"])
        first_scenario = topic_data["scenarios"][0]
        with st.container():
            st.markdown(f"**Example Scenario:** {first_scenario['situation']}")
            st.markdown(f"✅ {first_scenario['correct']}")
        st.caption("💡 Tip: Keep reviewing all areas to stay sharp across the board!")


def get_feedback_message(weak_label, fractions):
    """Returns a personalized feedback message based on AI prediction."""
    if weak_label == "Memorization":
        return "📚 Your recall of key facts and terms needs a little boost. Review the notes below, then try the re-quiz!"
    elif weak_label == "Understanding":
        return "🔍 You know some facts, but the 'why' and 'how' behind them need more attention. Read the explanation below carefully."
    elif weak_label == "Application":
        return "🧩 You understand the theory, but applying it to new situations needs more practice. Study the scenarios below."
    else:
        return "🌟 Great job! Your performance is balanced across all areas. Keep practicing to stay sharp!"


def get_improvement_message(improvement_percent):
    """Returns a motivational message based on the improvement percentage."""
    if improvement_percent > 15:
        return "🚀 Fantastic improvement! Your focused practice really paid off — keep going!"
    elif improvement_percent > 0:
        return "👍 You're improving! Keep going, you're on the right track!"
    elif improvement_percent == 0:
        return "➡️ Your score stayed the same. Review the material once more and try again whenever you're ready."
    else:
        return "💪 Don't worry — learning takes time. Review the concepts again and keep practicing!"


# =========================================================
# 5) SESSION STATE SETUP
# =========================================================

DEFAULT_STATE = {
    "step": 1,
    "topic_text": "",
    "topic_key": None,
    "topic_data": None,
    "quiz1_questions": [],
    "quiz1_scores": {},
    "quiz1_totals": {},
    "weak_label": None,
    "weak_probs": None,
    "quiz2_questions": [],
    "quiz2_scores": {},
    "quiz2_totals": {},
    "used_questions": set(),
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        # Use a fresh copy for mutable defaults (lists/dicts/sets)
        st.session_state[key] = value.copy() if isinstance(value, (list, dict, set)) else value


def reset_app():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value.copy() if isinstance(value, (list, dict, set)) else value


# =========================================================
# 6) REUSABLE QUIZ FORM RENDERER
# =========================================================

def render_quiz_form(questions, form_key):
    """Renders a quiz as a Streamlit form and returns (submitted, answers)."""
    total = len(questions)
    with st.form(form_key):
        answers = []
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i + 1}. ({q['category']})** {q['question']}")
            ans = st.radio(
                "Choose your answer:",
                q["options"],
                key=f"{form_key}_{i}",
                index=None,
            )
            answers.append(ans)
            st.progress((i + 1) / total)
            st.markdown("---")
        submitted = st.form_submit_button("✅ Submit Answers")
    return submitted, answers


# =========================================================
# 7) SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🎓 AI Tutor")
page = st.sidebar.radio("📍 Navigate", ["🏠 Home", "🧠 Quiz", "📊 Dashboard"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Your Journey")
step_labels = {
    1: "1️⃣ Enter Topic",
    2: "2️⃣ Take Quiz",
    3: "3️⃣ View Result",
    4: "4️⃣ Learn",
    5: "5️⃣ Re-Quiz",
    6: "6️⃣ Dashboard",
}
current_step = st.session_state.step
for s, label in step_labels.items():
    if s < current_step:
        st.sidebar.markdown(f"✅ {label}")
    elif s == current_step:
        st.sidebar.markdown(f"➡️ **{label}**")
    else:
        st.sidebar.markdown(f"⬜ {label}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Start Over (New Topic)"):
    reset_app()
    st.rerun()

st.sidebar.caption("Built with Streamlit + TensorFlow 🐍")


# =========================================================
# 8) HOME PAGE
# =========================================================

if page == "🏠 Home":
    st.title("🎓 Personalized AI Tutor System")
    st.write(
        "Welcome! This app helps you study **any topic** using an "
        "**adaptive learning loop** powered by a small TensorFlow model. 🔥"
    )
    st.divider()

    st.subheader("🧭 How It Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Step 1 — 📝 Enter Topic**")
        st.write("Type any topic you're studying.")
        st.markdown("**Step 2 — 🧠 Take Quiz**")
        st.write("Answer 9 questions across Memorization, Understanding, and Application.")
        st.markdown("**Step 3 — 📊 View Result**")
        st.write("Our AI model analyzes your scores and finds your weakest area.")
    with col2:
        st.markdown("**Step 4 — 📚 Learn**")
        st.write("Get a personalized mini-lesson focused on your weak area.")
        st.markdown("**Step 5 — 🔄 Re-Quiz**")
        st.write("Take a new quiz with ~70% of questions focused on your weak area.")
        st.markdown("**Step 6 — 🏆 Dashboard**")
        st.write("See your improvement and get final feedback!")

    st.divider()
    st.success("👉 Click **'🧠 Quiz'** in the sidebar to begin your learning journey!")
    st.info("💡 Try topics like: Stack, Queue, Linked List, Binary Search, Photosynthesis, "
            "Newton's Laws of Motion, Cell, or Object-Oriented Programming for the richest content. "
            "Any other topic also works!")


# =========================================================
# 9) QUIZ PAGE (THE MAIN WIZARD)
# =========================================================

elif page == "🧠 Quiz":

    # ---------------------------------------------------
    # STEP 1: Enter Topic
    # ---------------------------------------------------
    if st.session_state.step == 1:
        st.header("Step 1️⃣: Enter a Topic to Study 📝")
        st.write("Type any topic below and we'll generate a quiz for it!")

        topic_text = st.text_input(
            "Enter a topic (e.g., 'Stack in Data Structures', 'Photosynthesis')",
            value=st.session_state.topic_text,
            placeholder="e.g. Stack in Data Structures",
        )
        st.caption("Tip: Stack, Queue, Linked List, Binary Search, Photosynthesis, "
                   "Newton's Laws, Cell, and Object-Oriented Programming have the richest content!")

        if st.button("🚀 Generate Quiz"):
            if topic_text.strip() == "":
                st.warning("⚠️ Please enter a topic first!")
            else:
                st.session_state.topic_text = topic_text
                topic_key, topic_data = find_topic(topic_text)
                st.session_state.topic_key = topic_key
                st.session_state.topic_data = topic_data

                distribution = {"Memorization": 3, "Understanding": 3, "Application": 3}
                st.session_state.quiz1_questions = generate_quiz(
                    topic_key, topic_data, distribution, st.session_state.used_questions
                )
                st.session_state.step = 2
                st.rerun()

    # ---------------------------------------------------
    # STEP 2: Take First Quiz
    # ---------------------------------------------------
    elif st.session_state.step == 2:
        st.header(f"Step 2️⃣: Quiz on '{st.session_state.topic_data['name']}' 🧠")
        st.info("Answer all 9 questions to the best of your ability. This is your **first attempt**!")

        submitted, answers = render_quiz_form(st.session_state.quiz1_questions, "quiz1")

        if submitted:
            if None in answers:
                st.warning("⚠️ Please answer all questions before submitting.")
            else:
                scores, totals = compute_scores(st.session_state.quiz1_questions, answers)
                st.session_state.quiz1_scores = scores
                st.session_state.quiz1_totals = totals
                st.session_state.used_questions.update(
                    q["question"] for q in st.session_state.quiz1_questions
                )
                st.session_state.step = 3
                st.rerun()

    # ---------------------------------------------------
    # STEP 3: View Result + AI Weakness Detection
    # ---------------------------------------------------
    elif st.session_state.step == 3:
        st.header("Step 3️⃣: Your Results & AI Analysis 📊")

        scores = st.session_state.quiz1_scores
        totals = st.session_state.quiz1_totals

        col1, col2, col3 = st.columns(3)
        col1.metric("🧠 Memorization", f"{scores['Memorization']}/{totals['Memorization']}")
        col2.metric("🔍 Understanding", f"{scores['Understanding']}/{totals['Understanding']}")
        col3.metric("🧩 Application", f"{scores['Application']}/{totals['Application']}")

        chart_df = pd.DataFrame({
            "Category": list(scores.keys()),
            "Score": list(scores.values()),
        })
        st.bar_chart(chart_df.set_index("Category"))

        # --- AI prediction using TensorFlow ---
        st.subheader("🤖 AI Weakness Detection")
        model = build_weakness_model()
        fractions = {cat: scores[cat] / totals[cat] for cat in scores}
        weak_label, probs = predict_weakness(
            fractions["Memorization"], fractions["Understanding"], fractions["Application"], model
        )
        st.session_state.weak_label = weak_label
        st.session_state.weak_probs = probs

        if weak_label == "Balanced":
            st.success("✅ AI Analysis: Your performance is **Balanced** across all areas! 🌟")
        else:
            st.warning(f"🤖 AI Analysis: You appear to be **Weak in {weak_label}**. Let's work on that!")

        st.info(get_feedback_message(weak_label, fractions))

        if st.button("➡️ Continue to Personalized Learning"):
            st.session_state.step = 4
            st.rerun()

    # ---------------------------------------------------
    # STEP 4: Personalized Teaching Module
    # ---------------------------------------------------
    elif st.session_state.step == 4:
        st.header("Step 4️⃣: Personalized Learning Module 📚")

        weak_label = st.session_state.weak_label
        topic_data = st.session_state.topic_data
        show_teaching_module(weak_label, topic_data)

        st.divider()
        if st.button("➡️ I'm Ready for the Re-Quiz! 🔥"):
            distribution = get_requiz_distribution(weak_label)
            st.session_state.quiz2_questions = generate_quiz(
                st.session_state.topic_key, topic_data, distribution, st.session_state.used_questions
            )
            st.session_state.step = 5
            st.rerun()

    # ---------------------------------------------------
    # STEP 5: Adaptive Re-Quiz
    # ---------------------------------------------------
    elif st.session_state.step == 5:
        st.header("Step 5️⃣: Re-Quiz — Focused Practice 🔥")

        weak_label = st.session_state.weak_label
        if weak_label in ["Memorization", "Understanding", "Application"]:
            st.info(f"This quiz has **more questions on {weak_label}** to help you improve. Keep going! 💪")
        else:
            st.info("Here's another balanced quiz to test your overall progress. You're improving! 🌟")

        submitted, answers = render_quiz_form(st.session_state.quiz2_questions, "quiz2")

        if submitted:
            if None in answers:
                st.warning("⚠️ Please answer all questions before submitting.")
            else:
                scores, totals = compute_scores(st.session_state.quiz2_questions, answers)
                st.session_state.quiz2_scores = scores
                st.session_state.quiz2_totals = totals
                st.session_state.step = 6
                st.rerun()

    # ---------------------------------------------------
    # STEP 6: Final Comparison + Improvement
    # ---------------------------------------------------
    elif st.session_state.step == 6:
        st.header("Step 6️⃣: Final Results & Improvement 🏆")

        s1, t1 = st.session_state.quiz1_scores, st.session_state.quiz1_totals
        s2, t2 = st.session_state.quiz2_scores, st.session_state.quiz2_totals

        pct1 = sum(s1.values()) / sum(t1.values()) * 100
        pct2 = sum(s2.values()) / sum(t2.values()) * 100
        improvement = pct2 - pct1

        col1, col2, col3 = st.columns(3)
        col1.metric("📝 First Quiz Score", f"{pct1:.1f}%")
        col2.metric("🔄 Re-Quiz Score", f"{pct2:.1f}%", delta=f"{improvement:.1f}%")
        col3.metric("🎯 Weak Area Targeted", st.session_state.weak_label)

        st.subheader("📊 Score Comparison by Category")
        compare_df = pd.DataFrame({
            "Category": list(s1.keys()),
            "First Attempt (%)": [round(s1[c] / t1[c] * 100, 1) for c in s1],
            "Re-Quiz (%)": [round(s2[c] / t2[c] * 100, 1) for c in s2],
        })
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.bar_chart(compare_df.set_index("Category"))

        st.subheader("💬 Feedback")
        st.info(get_improvement_message(improvement))

        st.divider()
        if st.button("🔁 Try a New Topic"):
            reset_app()
            st.rerun()

        st.caption("📊 You can also check the 'Dashboard' tab anytime for a full summary!")


# =========================================================
# 10) DASHBOARD PAGE
# =========================================================

elif page == "📊 Dashboard":
    st.title("📊 Student Dashboard")

    if st.session_state.topic_data is None:
        st.info("👉 Go to the '🧠 Quiz' tab and enter a topic to get started!")
    else:
        st.subheader(f"📚 Topic: {st.session_state.topic_data['name']}")

        # --- Initial scores ---
        if st.session_state.quiz1_scores:
            st.markdown("### 📝 Initial Quiz Results")
            s1, t1 = st.session_state.quiz1_scores, st.session_state.quiz1_totals
            col1, col2, col3 = st.columns(3)
            col1.metric("🧠 Memorization", f"{s1['Memorization']}/{t1['Memorization']}")
            col2.metric("🔍 Understanding", f"{s1['Understanding']}/{t1['Understanding']}")
            col3.metric("🧩 Application", f"{s1['Application']}/{t1['Application']}")
        else:
            st.warning("⚠️ You haven't taken the first quiz yet. Go to '🧠 Quiz' to start!")

        # --- Weakness detected ---
        if st.session_state.weak_label:
            st.markdown("### 🤖 AI-Detected Weakness")
            if st.session_state.weak_label == "Balanced":
                st.success("🌟 Balanced performance across all areas!")
            else:
                st.warning(f"🎯 Weak in: **{st.session_state.weak_label}**")

        # --- After-training scores + improvement ---
        if st.session_state.quiz2_scores:
            st.markdown("### 🔄 Re-Quiz Results")
            s2, t2 = st.session_state.quiz2_scores, st.session_state.quiz2_totals
            col1, col2, col3 = st.columns(3)
            col1.metric("🧠 Memorization", f"{s2['Memorization']}/{t2['Memorization']}")
            col2.metric("🔍 Understanding", f"{s2['Understanding']}/{t2['Understanding']}")
            col3.metric("🧩 Application", f"{s2['Application']}/{t2['Application']}")

            st.markdown("### 📈 Improvement Overview")
            s1, t1 = st.session_state.quiz1_scores, st.session_state.quiz1_totals
            pct1 = sum(s1.values()) / sum(t1.values()) * 100
            pct2 = sum(s2.values()) / sum(t2.values()) * 100
            improvement = pct2 - pct1

            colA, colB = st.columns(2)
            colA.metric("First Attempt", f"{pct1:.1f}%")
            colB.metric("Re-Quiz", f"{pct2:.1f}%", delta=f"{improvement:.1f}%")

            compare_df = pd.DataFrame({
                "Category": list(s1.keys()),
                "First Attempt (%)": [round(s1[c] / t1[c] * 100, 1) for c in s1],
                "Re-Quiz (%)": [round(s2[c] / t2[c] * 100, 1) for c in s2],
            })
            st.bar_chart(compare_df.set_index("Category"))

            st.info(get_improvement_message(improvement))
        else:
            st.info("ℹ️ Complete the personalized learning module and re-quiz to see your improvement here!")
