import pandas as pd
from bs4 import BeautifulSoup
import nltk
nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.text import Text
import streamlit as st
def clean_text(text):
    if not isinstance(text, str):
        return ' '
    soup = BeautifulSoup(text, 'html.parser')
    cleaned = soup.get_text(separator=' ')
    return " ".join(cleaned.split())

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df

@st.cache_data
def process_text_column(df, column_name):
    df['clean_text'] = df[column_name].apply(clean_text)
    combined_cleaned_text = ' '.join(df['clean_text'])
    tokens = word_tokenize(combined_cleaned_text)
    return tokens

def find_concordances(tokens, search_term, window=5):
    """
    Find occurrences of search_term in tokens and return a DataFrame
    with left context, the match, and right context.
    window = number of tokens before/after the match to display.
    """
    results = []
    search_term_lower = search_term.lower()
    for i, token in enumerate(tokens):
        if token.lower() == search_term_lower:
            start = max(i - window, 0)
            end = min(i + window + 1, len(tokens))
            left_context = " ".join(tokens[start:i])
            match = tokens[i]
            right_context = " ".join(tokens[i+1:end])
            results.append([left_context, match, right_context])
    return pd.DataFrame(results, columns=["Left Context", "Matched Term", "Right Context"])


st.title("Concordancer Web App")


df = load_data(st.secrets['path'])

st.write("First 5 rows of your data:")
st.dataframe(df.head(5))

text_column = st.selectbox("Select the text column for analysis:", df.columns)
tokens = process_text_column(df, text_column)

search_term = st.text_input("Enter a search term to find its concordance:")
window_size = st.slider("Context window size (words before and after the match)", 1, 20, 5)
search_button = st.button("Find Concordance")

if search_button and search_term.strip():
    results_df = find_concordances(tokens, search_term, window=window_size)
    if not results_df.empty:
        st.dataframe(results_df)
    else:
        st.warning(f"No concordance found for '{search_term}'.")

  
