import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Page configuration
st.set_page_config(
    page_title="LSTM Quote Generator",
    page_icon="🧠",
    layout="centered"
)


# Load saved files
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()


# Predict next N words
def predict_next_n_words(text, num_words):
    generated_text = text.strip()

    for _ in range(num_words):

        # Convert the current text into integer tokens
        sequence = tokenizer.texts_to_sequences(
            [generated_text]
        )[0]

        # Stop if the input contains no known words
        if not sequence:
            break

        # Prepare the sequence for the model
        sequence = pad_sequences(
            [sequence],
            maxlen=max_len - 1,
            padding="pre",
            truncating="pre"
        )

        # Predict the next word
        predictions = model.predict(
            sequence,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        # Convert the predicted index back to a word
        predicted_word = tokenizer.index_word.get(
            predicted_index
        )

        # Stop if the predicted index has no matching word
        if predicted_word is None:
            break

        # Append the predicted word to the text
        generated_text += " " + predicted_word

    return generated_text


# Streamlit UI
st.title("🧠 LSTM Quote Generator")

st.write(
    "Enter a starting sentence, choose how many words "
    "to generate, and let your LSTM model continue the text."
)

user_input = st.text_input(
    "✍️ Enter your starting text",
    placeholder="Example: The best way to"
)

num_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

if st.button("✨ Generate Text", type="primary"):

    if not user_input.strip():
        st.warning("Please enter some text first.")

    else:
        with st.spinner("Generating text..."):

            result = predict_next_n_words(
                user_input,
                num_words
            )

        st.subheader("Generated Text")

        st.write(result)

        # Show only the newly generated continuation
        original_words = user_input.strip()
        continuation = result[len(original_words):].strip()

        st.subheader("Predicted Continuation")
        st.write(continuation if continuation else
                 "No additional words could be generated.")

        st.caption(
            f"Requested up to {num_words} new words."
        )


# Footer
st.markdown("---")

st.caption(
    "LSTM-based next-word generation using Streamlit"
)
