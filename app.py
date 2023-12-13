# app.py
import streamlit as st
from sentiment_analysis_page import main1
from trend_analysis_page import main2


def main():
    st.sidebar.title("Navigation")
    app_selection = st.sidebar.radio("Go to", ["Sentiment Analysis", "Trend Analysis"])

    if app_selection == "Sentiment Analysis":
        main1()
    elif app_selection == "Trend Analysis":
        main2()


if __name__ == "__main__":
    main()
