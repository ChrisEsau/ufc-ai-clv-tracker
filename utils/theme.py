import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        /* paste your main CSS block here */
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        /* paste your selectbox CSS block here */
        </style>
        """,
        unsafe_allow_html=True,
    )
