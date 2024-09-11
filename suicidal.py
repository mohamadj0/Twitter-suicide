# !pip install googletrans==4.0.0-rc1
from googletrans import Translator

from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

import streamlit as st


# from google.colab import drive
# data = drive.mount('/content/drive')
# /content/drive/MyDrive/


# Load the best models
model_lstm2 = load_model(
    './lastm-2-layer-best_model.h5')

# Load tokenizer object
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_suicide_post(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=60, dtype='int32')

    # Predictions
    pred_lstm2 = model_lstm2.predict(padded_sequences, batch_size=1, verbose=0)

    # Choose the class with the highest probability
    result = {
        'LSTM-2 Layer': 'Potential Suicide Post' if pred_lstm2[0][0] > pred_lstm2[0][1] else 'Not Suicide Post',
    }

    return result


def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text


def main():
    user_input = ''
    user_input = st.text_area(label='enter something...')
    # user_input = input("Enter a sentence: ")

    if user_input.isascii():
        # If the input is in English, continue with the prediction
        prediction_result = predict_suicide_post(user_input)
    else:
        # If the input is in Persian, translate it to English
        translated_input = translate_to_english(user_input)
        # print(f"Translated input: {translated_input}")
        prediction_result = predict_suicide_post(translated_input)

    # Display the result
    # print("\nPrediction Results:")
    st.write("\nPrediction Results:")
    for model, result in prediction_result.items():
        # print(f"{model}: {result}")
        st.write(f"{model}: {result}")


if __name__ == "__main__":
    main()
