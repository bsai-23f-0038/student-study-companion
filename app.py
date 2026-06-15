# =========================================================
# 📚 Student Study Companion App
# A simple Streamlit app with 4 beginner-friendly features:
#   1) Mood Detection      -> Small TensorFlow text classifier
#   2) Study Planner       -> Simple priority-based timetable
#   3) Notes Simplifier    -> Rule-based text simplifier
#   4) Focus Timer         -> Basic Pomodoro timer
#
# No paid APIs, no Hugging Face, no PyTorch.
# Works on Python 3.11 and below.
# =========================================================

import os
import warnings

# Hide unnecessary TensorFlow / Python warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense


# ---------------------------------------------------------
# Basic page configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Student Study Companion", page_icon="📚", layout="centered")

# Set seeds so the model behaves consistently each run
np.random.seed(42)
tf.random.set_seed(42)


# =========================================================
# 1) MOOD DETECTION MODEL (TensorFlow)
# =========================================================
# We train a very small text classification model on a tiny,
# hand-written dataset. This is NOT a production model — it's
# just enough to show a working TensorFlow model in action.

MOOD_LABELS = {0: "Happy", 1: "Sad", 2: "Stressed"}

# Small training dataset (sentence -> mood label)
TRAIN_TEXTS = [
    # Happy (label 0)
    "I feel great today",
    "I am so happy right now",
    "Today was an amazing day",
    "I feel excited and full of energy",
    "Everything is going well for me",
    "I am proud of my achievements",
    "I had a wonderful time with my friends",
    "I feel relaxed and joyful",
    "I am grateful for everything I have",
    "Life feels good today",

    # Sad (label 1)
    "I feel really sad today",
    "I am feeling down and lonely",
    "Nothing seems to be going right",
    "I feel like crying",
    "I am disappointed in myself",
    "I miss my family a lot",
    "I feel empty inside",
    "I am upset about my results",
    "I feel hopeless about everything",
    "I had a bad day and feel low",

    # Stressed (label 2)
    "I have so many assignments due",
    "I am worried about my exams",
    "I feel overwhelmed with work",
    "There is too much pressure on me",
    "I cannot focus because of stress",
    "I am anxious about my deadlines",
    "I feel like I am running out of time",
    "My workload is too much to handle",
    "I am nervous about the upcoming test",
    "I feel tense and unable to relax",
]

TRAIN_LABELS = [0] * 10 + [1] * 10 + [2] * 10

MAX_LEN = 10      # max number of words per sentence (input length)
VOCAB_SIZE = 200  # maximum vocabulary size for the tokenizer


@st.cache_resource(show_spinner="Training mood detection model...")
def build_mood_model():
    """
    Builds, trains, and returns a small TensorFlow text
    classification model + the tokenizer used to prepare text.
    Cached so this only runs ONCE when the app starts.
    """
    # Step 1: Tokenizer converts words into numbers
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(TRAIN_TEXTS)

    # Step 2: Convert sentences to number sequences, then pad them
    sequences = tokenizer.texts_to_sequences(TRAIN_TEXTS)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    # Step 3: Build a very small neural network
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=16, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")   # 3 moods: Happy, Sad, Stressed
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Step 4: Train the model on the small dataset
    labels = np.array(TRAIN_LABELS)
    model.fit(padded, labels, epochs=60, verbose=0)

    return model, tokenizer


def predict_mood(text, model, tokenizer):
    """
    Takes raw user text, converts it using the tokenizer,
    and returns the predicted mood label as a string.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    mood_index = int(np.argmax(prediction))
    return MOOD_LABELS[mood_index]


def mood_message(mood):
    """Returns a friendly message based on the detected mood."""
    if mood == "Happy":
        return "😊 That's great to hear! Keep up the positive energy and enjoy your study session."
    elif mood == "Sad":
        return "💙 It's okay to feel sad sometimes. Take a short break, talk to a friend, then come back refreshed."
    else:  # Stressed
        return "😮‍💨 Feeling stressed is normal during studies. Try the Focus Timer to break your work into small chunks."


# =========================================================
# 2) STUDY PLANNER (simple logic, no AI)
# =========================================================

def generate_timetable(subjects, priorities, total_hours):
    """
    Distributes total available study hours among subjects
    based on their priority (1 to 5). Higher priority gets
    more hours.
    """
    total_priority = sum(priorities)
    timetable = []

    for subject, priority in zip(subjects, priorities):
        allocated = (priority / total_priority) * total_hours
        timetable.append({
            "Subject": subject,
            "Priority (1-5)": priority,
            "Allocated Hours": round(allocated, 2)
        })

    return pd.DataFrame(timetable)


# =========================================================
# 3) NOTES SIMPLIFIER (rule-based, no AI)
# =========================================================

# Dictionary: complex word -> simple word
SIMPLE_WORDS = {
    "utilize": "use",
    "utilizing": "using",
    "demonstrate": "show",
    "approximately": "about",
    "subsequently": "after that",
    "numerous": "many",
    "facilitate": "help",
    "obtain": "get",
    "consequently": "so",
    "additionally": "also",
    "commence": "start",
    "terminate": "end",
    "purchase": "buy",
    "sufficient": "enough",
    "endeavor": "try",
    "assistance": "help",
    "regarding": "about",
    "however": "but",
    "therefore": "so",
}

# Phrases that just add length without much meaning
FILLER_PHRASES = [
    "it is important to note that",
    "it should be noted that",
    "in other words",
    "as a matter of fact",
    "due to the fact that",
    "for the purpose of",
    "in order to",
    "at this point in time",
    "basically",
    "actually",
    "in conclusion",
]


def cap_first(text):
    """Capitalizes only the first letter, keeping rest of the text as-is."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def simplify_notes(text):
    """
    Very simple rule-based text simplifier:
    1. Removes common filler phrases.
    2. Replaces complex words with simpler ones.
    3. Splits very long sentences into shorter ones.
    """
    simplified = text

    # Step 1: Remove filler phrases (case-insensitive)
    for phrase in FILLER_PHRASES:
        simplified = re.sub(phrase, "", simplified, flags=re.IGNORECASE)

    # Step 2: Replace complex words with simpler ones (whole-word match)
    for hard, easy in SIMPLE_WORDS.items():
        simplified = re.sub(r"\b" + hard + r"\b", easy, simplified, flags=re.IGNORECASE)

    # Step 3: Split sentences, clean them, and break up long ones
    sentences = re.split(r'(?<=[.!?]) +', simplified)
    final_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        # Remove leftover punctuation/spaces caused by phrase removal
        sentence = re.sub(r'^[,;:\s]+', '', sentence)

        if not sentence:
            continue

        words = sentence.split()

        if len(words) > 18:
            # Split long sentence roughly in half
            mid = len(words) // 2
            part1 = " ".join(words[:mid]).strip()
            part2 = " ".join(words[mid:]).strip()

            if part1:
                final_sentences.append(cap_first(part1) + ".")
            if part2:
                final_sentences.append(cap_first(part2.rstrip(".!?")) + ".")
        else:
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            final_sentences.append(cap_first(sentence))

    result = " ".join(final_sentences)

    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# =========================================================
# STREAMLIT APP LAYOUT
# =========================================================

st.title("📚 Student Study Companion")
st.write("A simple all-in-one app to help you study smarter!")

# Sidebar navigation
page = st.sidebar.radio(
    "Choose a feature:",
    ["🏠 Home", "😊 Mood Detector", "🗓️ Study Planner", "📝 Notes Simplifier", "⏱️ Focus Timer"]
)


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "🏠 Home":
    st.header("Welcome!")
    st.write("""
    This app has 4 simple tools to help students:

    - **Mood Detector** – Tell the app how you feel, and a small TensorFlow model will guess your mood.
    - **Study Planner** – Enter your subjects and available hours to get a simple study schedule.
    - **Notes Simplifier** – Paste a paragraph and get a simpler version of it.
    - **Focus Timer** – A basic Pomodoro-style timer to help you focus.

    👉 Use the sidebar on the left to switch between tools.
    """)


# ---------------------------------------------------------
# MOOD DETECTOR PAGE
# ---------------------------------------------------------
elif page == "😊 Mood Detector":
    st.header("😊 Mood Detector")
    st.write("Type a sentence describing how you feel, and the app will guess your mood.")
    st.caption("Note: This is a small demo model trained on a tiny dataset, so it won't always be 100% accurate.")

    user_text = st.text_area("How are you feeling today?", placeholder="e.g. I am worried about my exam")

    if st.button("Detect Mood"):
        if user_text.strip() == "":
            st.warning("Please type something first!")
        else:
            model, tokenizer = build_mood_model()
            mood = predict_mood(user_text, model, tokenizer)
            st.success(f"Detected Mood: **{mood}**")
            st.info(mood_message(mood))


# ---------------------------------------------------------
# STUDY PLANNER PAGE
# ---------------------------------------------------------
elif page == "🗓️ Study Planner":
    st.header("🗓️ Simple Study Planner")
    st.write("Enter your subjects, set a priority for each, and your total study hours.")

    num_subjects = st.number_input("How many subjects?", min_value=1, max_value=8, value=3, step=1)

    subjects = []
    priorities = []

    for i in range(int(num_subjects)):
        col1, col2 = st.columns([2, 1])
        with col1:
            subject_name = st.text_input(f"Subject {i + 1} name", key=f"subject_{i}")
        with col2:
            priority = st.slider(f"Priority {i + 1}", 1, 5, 3, key=f"priority_{i}")

        if subject_name.strip() != "":
            subjects.append(subject_name)
            priorities.append(priority)

    total_hours = st.number_input(
        "Total study hours available today",
        min_value=1.0, max_value=24.0, value=4.0, step=0.5
    )

    if st.button("Generate Timetable"):
        if len(subjects) == 0:
            st.warning("Please enter at least one subject name.")
        else:
            timetable_df = generate_timetable(subjects, priorities, total_hours)
            st.subheader("Your Study Plan")
            st.dataframe(timetable_df, use_container_width=True)
            st.caption("Higher priority subjects get more study hours.")


# ---------------------------------------------------------
# NOTES SIMPLIFIER PAGE
# ---------------------------------------------------------
elif page == "📝 Notes Simplifier":
    st.header("📝 Notes Simplifier")
    st.write("Paste a paragraph from your notes, and get a simpler version (rule-based, no AI).")

    notes_input = st.text_area("Paste your notes here:", height=150)

    if st.button("Simplify Notes"):
        if notes_input.strip() == "":
            st.warning("Please paste some text first!")
        else:
            simplified = simplify_notes(notes_input)
            st.subheader("Simplified Version")
            st.write(simplified)


# ---------------------------------------------------------
# FOCUS TIMER PAGE (Pomodoro)
# ---------------------------------------------------------
elif page == "⏱️ Focus Timer":
    st.header("⏱️ Focus Timer (Pomodoro)")
    st.write("Use this simple timer to study in focused sessions.")

    # Let the user choose session length
    minutes = st.selectbox("Select session length (minutes)", [5, 15, 25, 30, 45], index=2)

    # Initialize session state variables (only runs once)
    if "timer_running" not in st.session_state:
        st.session_state.timer_running = False
    if "remaining_seconds" not in st.session_state:
        st.session_state.remaining_seconds = minutes * 60
    if "selected_minutes" not in st.session_state:
        st.session_state.selected_minutes = minutes

    # If user changes the dropdown while timer is NOT running, reset the time
    if not st.session_state.timer_running and st.session_state.selected_minutes != minutes:
        st.session_state.selected_minutes = minutes
        st.session_state.remaining_seconds = minutes * 60

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start"):
            st.session_state.timer_running = True

    with col2:
        if st.button("⏸️ Stop"):
            st.session_state.timer_running = False

    with col3:
        if st.button("🔄 Reset"):
            st.session_state.timer_running = False
            st.session_state.remaining_seconds = minutes * 60

    # Display the remaining time
    mins, secs = divmod(st.session_state.remaining_seconds, 60)
    st.markdown(f"## ⏳ {mins:02d}:{secs:02d}")

    # If timer is running and time is left, wait 1 second and refresh
    if st.session_state.timer_running and st.session_state.remaining_seconds > 0:
        time.sleep(1)
        st.session_state.remaining_seconds -= 1
        st.rerun()

    # If time has just run out, show a completion message
    if st.session_state.remaining_seconds == 0 and st.session_state.timer_running:
        st.session_state.timer_running = False
        st.success("✅ Time's up! Take a short break.")
