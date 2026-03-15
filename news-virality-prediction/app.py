import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model.pkl')
feature_cols = joblib.load('feature_cols.pkl')

st.set_page_config(page_title="News Virality Predictor", layout="centered")
st.title("News Virality Predictor")
st.write("Adjust the sliders below to describe your article and see if it will go viral.")
st.divider()

st.subheader("Article Content")
col1, col2 = st.columns(2)

with col1:
    n_tokens_title = st.slider("Title word count", 2, 20, 10)
    n_tokens_content = st.slider("Article word count", 100, 5000, 500)
    num_imgs = st.slider("Number of images", 0, 20, 2)
    num_videos = st.slider("Number of videos", 0, 10, 0)

with col2:
    num_hrefs = st.slider("Number of links", 0, 100, 10)
    kw_avg_avg = st.slider("Avg keyword popularity", 0, 10000, 3000)
    global_sentiment_polarity = st.slider("Sentiment polarity", -0.5, 0.5, 0.0)
    global_subjectivity = st.slider("Subjectivity (0=objective, 1=opinionated)", 0.0, 1.0, 0.5)

st.divider()
st.subheader("Publication Details")
col3, col4 = st.columns(2)

with col3:
    channel = st.selectbox("News channel", [
        "lifestyle", "entertainment", "bus", "socmed", "tech", "world"
    ])

with col4:
    day = st.selectbox("Day published", [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ])

st.divider()
if st.button("Predict Virality", type="primary", use_container_width=True):

    row = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

    row['n_tokens_title']            = n_tokens_title
    row['n_tokens_content']          = n_tokens_content
    row['num_imgs']                  = num_imgs
    row['num_videos']                = num_videos
    row['num_hrefs']                 = num_hrefs
    row['kw_avg_avg']                = kw_avg_avg
    row['global_sentiment_polarity'] = global_sentiment_polarity
    row['global_subjectivity']       = global_subjectivity
    row['strong_sentiment']          = int(abs(global_sentiment_polarity) > 0.2)

    for c in ['lifestyle', 'entertainment', 'bus', 'socmed', 'tech', 'world']:
        col_name = 'data_channel_is_' + c
        if col_name in feature_cols:
            row[col_name] = 1 if c == channel else 0

    day_map = {
        'Monday':    'weekday_is_monday',
        'Tuesday':   'weekday_is_tuesday',
        'Wednesday': 'weekday_is_wednesday',
        'Thursday':  'weekday_is_thursday',
        'Friday':    'weekday_is_friday',
        'Saturday':  'weekday_is_saturday',
        'Sunday':    'weekday_is_sunday'
    }
    for d, col_name in day_map.items():
        if col_name in feature_cols:
            row[col_name] = 1 if d == day else 0
    if day in ['Saturday', 'Sunday'] and 'is_weekend' in feature_cols:
        row['is_weekend'] = 1

    prob = model.predict_proba(row)[0][1]
    percent = round(prob * 100, 1)

    st.divider()
    if prob >= 0.6:
        st.success("VIRAL — " + str(percent) + "% virality probability")
        st.balloons()
    elif prob >= 0.4:
        st.warning("BORDERLINE — " + str(percent) + "% virality probability")
    else:
        st.error("NOT VIRAL — " + str(percent) + "% virality probability")

    st.progress(float(prob), text="Virality score: " + str(percent) + "%")

    st.divider()
    st.subheader("Tips to improve virality")
    tips = []
    if n_tokens_title < 8:
        tips.append("Write a longer headline (aim for 8-12 words)")
    if kw_avg_avg < 3000:
        tips.append("Target more popular keywords in your article")
    if num_imgs == 0:
        tips.append("Add at least 1-2 images to boost engagement")
    if global_sentiment_polarity == 0.0:
        tips.append("Add stronger emotional tone to your writing")
    if channel == 'world':
        tips.append("World news tends to go viral less — consider a tech or social angle")

    if tips:
        for tip in tips:
            st.write("- " + tip)
    else:
        st.write("Your article looks well optimised for virality!")