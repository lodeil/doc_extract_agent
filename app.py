# Imports : 

import os
import json
import re
import io
import streamlit as st
# import pytesseract
from functions import *
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
import seaborn as sns
from utils import download_button


# Config :
# To launch : streamlit run app.py
# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'D:\\softwares\\Tesseract-OCR\\tesseract.exe'

MODEL = 'all-mpnet-base-v2'
TOP_N = 15
min_Ngrams = 1
max_Ngrams = 2
use_MMR = False # high/low diversity of results
Diversity = 0.5
MAX_WORDS = 1000
StopWordsCheckbox = True or False
if StopWordsCheckbox:
    StopWords = "english"
else:
    StopWords = None
mmr = use_MMR
if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams > max_Ngrams , shouldnt be")
    st.stop()

# Informations : 
st.set_page_config(
    page_title="Extract from documents",
    page_icon="⛏️",
)
st.header("")
st.title("🌳 Extract important element from your 📃 documents 📃. 🌳")
st.header("")

with st.expander("ℹ️ - Information on this app", expanded=True):

    st.write(
        """     
-   The *Extract from documents* is an easy-to-use interface for extracting the most relevent data from your documents
-   It build to be easy to use with nominal performance 
-   Document types :  csv , xlsx , xls , pdf , txt , xml , jpg , jpeg
    """
    )

    st.markdown("")

# Select files : 
st.header("")
st.markdown("🌌 Select your files ")
st.header("")

with st.form(key="my_form"):

    c29, c30, c31 = st.columns([0.08, 6, 0.18])

    with c30:


        files = st.file_uploader(
            "",
            key="1",
            help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
            accept_multiple_files=True
        )

        st.info(
                f"""
                    👆 Upload a file first. Here is a sample to try: [card.jpg](https://upload.wikimedia.org/wikipedia/commons/2/23/French_Identity_card_1988_-_1994.jpg)
                    """
            )

        submit_button = st.form_submit_button(label="🧾 Parse it all !")


if not submit_button:
    st.stop()

# Parsing results : 
st.markdown("🍂 Your files as a text 🍂")

text_of_files = files_to_text(files=files)

st.text(
    text_of_files
    )

st.markdown("🌼 The end of text extraction 🌼")

# Extraction launch :
st.markdown(" ⏰ Launch of the nlp model ⏰")

@st.cache(allow_output_mutation=True)
def load_model():
    return KeyBERT(MODEL)

kw_model = load_model()

doc = text_of_files

res = len(re.findall(r"\w+", doc))
if res > MAX_WORDS:
    st.warning(
        "⚠️ Your text contains "
        + str(res)
        + " words."
        + f" Only the first {MAX_WORDS} words will be reviewed."
    )

    doc = doc[:MAX_WORDS]

st.markdown("🛠️ Model loaded and ready to process the data 🛠️")

keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=TOP_N,
    diversity=Diversity,
)

keywords_as_dataframe = DataFrame(keywords, columns=["Keyelements", "Relevancy"])

# Extraction results :
st.markdown("## ⚗️ Your results are ready ⚗️")

st.header("")

@st.cache
def convert_df_csv(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False,sep=";").encode('utf-8')

@st.cache
def convert_df_json(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_json().encode('utf-8')

@st.cache
def convert_df_txt(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_string(header=True, index=True).encode('utf-8')

csv = convert_df_csv(keywords_as_dataframe)
json = convert_df_json(keywords_as_dataframe)
txt = convert_df_txt(keywords_as_dataframe)

cs, c1, c2, c3, cLast = st.columns([0.5, 2, 2, 2, 0.5])
# # cs, c1 , cLast = st.columns([2,1, 2])

with c1:
    Button = download_button(csv, "Results_extraction.csv", "📩 Download data as CSV")

with c2:
    Button = download_button(txt, "Results_extraction.txt", "📩 Download data as TXT")
with c3:
    Button = download_button(json, "Results_extraction.json", "📩 Download data as JSON")

# Visualization :
st.header("")

df = (
    DataFrame(keywords, columns=["Keyelements", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 5, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)

st.markdown("## 😄 Thanks 😄")