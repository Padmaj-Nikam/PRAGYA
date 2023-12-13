import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataframe from the pickle file
df_loaded = pd.read_pickle(
    "/home/paddie/Documents/pragya1/PRAGYA/sentiment-analysis/dataframe.pkl"
)

# Load the correlation matrix from the pickle file
with open(
    "/home/paddie/Documents/pragya1/PRAGYA/sentiment-analysis/correlation_matrix.pkl",
    "rb",
) as file:
    corr_matrix_loaded = pickle.load(file)


# Streamlit App
def main1():
    st.title("Data Analysis Web App")

    # Display the loaded dataframe
    st.subheader("Loaded DataFrame:")
    st.dataframe(df_loaded)

    # Display the correlation matrix
    st.subheader("Correlation Matrix:")
    st.write("Heatmap of the Correlation Matrix:")

    # Customize the heatmap style using seaborn
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix_loaded, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    st.pyplot()


if __name__ == "__main__":
    main1()
