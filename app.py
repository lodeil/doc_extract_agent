import os
import json
import io
import streamlit as st



# Config :
# To launch : streamlit run app.py

st.set_page_config(
    page_title="Project :",
    page_icon="🌠",
)

st.title("🌳 Step 1 ")
st.header("")

with st.expander("ℹ️ - Information on this app", expanded=True):

    st.write(
        """     
-   It build to be easy to use with nominal performance 
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("🌌 Step 2 ")

with st.form(key="my_form"):

    c29, c30, c31 = st.columns([0.08, 6, 0.18])

    with c30:

        files = st.file_uploader(
            "",
            key="1",
            help="",
            accept_multiple_files=True
        )

        if files is None:
            st.info(
                f"""
                    👆 Upload a file file first. Sample to try: [trees.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/trees.csv)
                    """
            )

        submit_button = st.form_submit_button(label="🧾 Get me the data!")


if not submit_button:
    st.stop()

st.markdown("🌲 Step 3 ")

st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

st.markdown("🌼 Step 4 ")

st.metric(label="Temperature", value="70 °F", delta="1.2 °F")