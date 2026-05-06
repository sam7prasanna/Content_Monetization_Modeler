import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# LOAD MODEL & DATA
# =========================

@st.cache_resource
def load_model():
    return joblib.load("best_youtube_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("youtube_ad_revenue_dataset_featured.csv")


model = load_model()
df = load_data()


# =========================
# DROPDOWN OPTIONS
# =========================

category_options = sorted(df["category"].dropna().unique())
device_options = sorted(df["device"].dropna().unique())
country_options = sorted(df["country"].dropna().unique())


# =========================
# APP TITLE
# =========================

st.title("📺 YouTube Content Monetization Modeler")

st.write("""
This application predicts YouTube ad revenue based on:
- Views
- Engagement
- Watch Time
- Subscribers
- Category
- Device
- Country
""")


# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("🔢 Enter Video Metrics")

views = st.sidebar.number_input(
    "Views",
    min_value=0,
    value=1000,
    step=100
)

likes = st.sidebar.number_input(
    "Likes",
    min_value=0,
    value=100,
    step=10
)

comments = st.sidebar.number_input(
    "Comments",
    min_value=0,
    value=20,
    step=5
)

watch_time_minutes = st.sidebar.number_input(
    "Watch Time (Minutes)",
    min_value=0.0,
    value=4000.0,
    step=100.0
)

video_length_minutes = st.sidebar.number_input(
    "Video Length (Minutes)",
    min_value=0.1,
    value=10.0,
    step=0.5
)

subscribers = st.sidebar.number_input(
    "Subscribers",
    min_value=0,
    value=10000,
    step=100
)

category = st.sidebar.selectbox(
    "Category",
    category_options
)

device = st.sidebar.selectbox(
    "Device",
    device_options
)

country = st.sidebar.selectbox(
    "Country",
    country_options
)


# =========================
# FEATURE ENGINEERING
# =========================

def compute_features(
    views,
    likes,
    comments,
    watch_time_minutes,
    video_length_minutes
):

    engagement_rate = (
        (likes + comments) / views
        if views > 0 else 0
    )

    watch_per_view = (
        watch_time_minutes / views
        if views > 0 else 0
    )

    views_per_minute = (
        views / video_length_minutes
        if video_length_minutes > 0 else 0
    )

    return (
        engagement_rate,
        watch_per_view,
        views_per_minute
    )


engagement_rate, watch_per_view, views_per_minute = compute_features(
    views,
    likes,
    comments,
    watch_time_minutes,
    video_length_minutes
)


# =========================
# INPUT DATAFRAME
# =========================

input_df = pd.DataFrame({
    "views": [views],
    "likes": [likes],
    "comments": [comments],
    "watch_time_minutes": [watch_time_minutes],
    "video_length_minutes": [video_length_minutes],
    "subscribers": [subscribers],
    "engagement_rate": [engagement_rate],
    "watch_per_view": [watch_per_view],
    "views_per_minute": [views_per_minute],
    "category": [category],
    "device": [device],
    "country": [country]
})


# =========================
# PREDICTION SECTION
# =========================

st.subheader("📊 Prediction")

st.write("Input Values")
st.write(input_df)

if st.button("Predict Ad Revenue"):

    try:
        prediction = model.predict(input_df)[0]

        st.success(
            f"💰 Predicted Ad Revenue: ${prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# =========================
# REVENUE VS VIEWS
# =========================

st.subheader("📈 Revenue vs Views")

sample_df = df.sample(2000)

fig, ax = plt.subplots(figsize=(10, 5))

sns.scatterplot(
    data=sample_df,
    x="views",
    y="ad_revenue_usd",
    ax=ax
)

ax.set_title("Revenue vs Views")

st.pyplot(fig)


# =========================
# CORRELATION HEATMAP
# =========================

st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)


# =========================
# REVENUE BY CATEGORY
# =========================

st.subheader("📊 Average Revenue by Category")

avg_cat = (
    df.groupby("category")["ad_revenue_usd"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(avg_cat)


# =========================
# REVENUE BY DEVICE
# =========================

st.subheader("📱 Average Revenue by Device")

avg_device = (
    df.groupby("device")["ad_revenue_usd"]
    .mean()
)

st.bar_chart(avg_device)


# =========================
# FEATURE IMPORTANCE
# =========================

importance_df = pd.DataFrame({
    "Feature": [
        "watch_time_minutes",
        "engagement_rate",
        "likes",
        "views",
        "watch_per_view"
    ],
    "Importance": [
        0.977806,
        0.021250,
        0.000442,
        0.000304,
        0.000101
    ]
})

st.subheader("🧠 Top Important Features")

st.bar_chart(
    importance_df.set_index("Feature")
)


# =========================
# KEY INSIGHTS
# =========================

st.subheader("📌 Key Insights")

st.markdown("""
- Watch time is the strongest predictor of ad revenue.
- Higher engagement improves monetization potential.
- Videos with better retention generate more revenue.
- Content category influences monetization performance.
- The model achieved over 95% R² score.
""")


# =========================
# FOOTER
# =========================

st.markdown("---")

st.write(
    "Built using Streamlit, Scikit-learn, Pandas, and Machine Learning."
)