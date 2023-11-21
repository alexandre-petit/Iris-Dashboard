#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = px.data.iris()
data.drop('species_id', axis=1, inplace=True)

st.title("Iris Dashboard")
def data_scaler(data: pd.DataFrame) -> pd.DataFrame:

    scaler = StandardScaler().set_output(transform='pandas')

    y = data[['species']].copy()
    X = data.drop('species', axis=1)

    X_scaled = scaler.fit_transform(X)
    X_scaled['species'] = y

    return X_scaled

info = "This dashboard was created to explore the Iris dataset and predict the class of a sample"
info += "\n\nOriginally made with Dash, and adapted to work in Streamlit"
st.info(info)


left_col, main_frame = st.columns([0.3, 0.7])

with st.sidebar:
    st.subheader("Filter Data")
    normalize = st.checkbox("Normalize data")
    outliers = st.checkbox("Remove Outliers")
    if outliers:
        z_score = st.slider("select z-score",
                            min_value=2.0, max_value=3.0, step=0.25)
    else:
        st.text("\n")
        st.text("\n")

    title_pred = st.subheader("Input your Sample")
    slider_1 = st.slider("Select Sepal Length",
                         min_value=2.0, max_value=8.0, step=0.1)
    slider_2 = st.slider("Select Sepal Width",
                         min_value=1.0, max_value=6.0, step=0.1)
    slider_3 = st.slider("Select Petal Length",
                         min_value=2.0, max_value=6.0, step=0.1)
    slider_4 = st.slider("Select Petal Width",
                         min_value=0.0, max_value=4.0, step=0.1)
    st.divider()

    slider_5 = st.slider("Test Size",
                         min_value=0.2, max_value=0.5, step=0.05)

    do_predict = st.button("Predict Sample Class")

if normalize:
    data = data_scaler(data)

graph_data = data.copy()

column = st.selectbox("Select a column to display",
                      tuple(graph_data.columns))

if outliers and column != 'species':
    for iris_specie in graph_data['species'].unique():
        sub_df = graph_data[graph_data['species'] == iris_specie].copy()
        left_df = graph_data[graph_data['species'] != iris_specie].copy()

        mean = sub_df[column].mean()
        std = sub_df[column].std()

        mask_low = (sub_df[column] >= mean - std * z_score)
        mask_high = (sub_df[column] <= mean + std * z_score)

        sub_df = sub_df[mask_low & mask_high].copy()

        graph_data = pd.concat([sub_df, left_df], axis=0)
        graph_data.sort_index(inplace=True)

fig = px.histogram(graph_data,
                 x=column,
                 color='species',
                 barmode='group')
st.plotly_chart(fig)



if do_predict:

    y = data['species'].copy()
    X = data.drop('species', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=slider_5,
                                                        stratify=y,
                                                        random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    sample = pd.DataFrame(columns=list(data.columns[:-1]))

    sample.loc[0] = {
            "sepal_length": slider_1,
            "sepal_width": slider_2,
            "petal_length": slider_3,
            "petal_width": slider_4
        }
    prediction = model.predict(sample)[0]
    
    st.info(f"The sample seems to be an Iris {prediction}")    
