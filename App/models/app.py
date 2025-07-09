# CORE PKGS
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import joblib
import re

# Load Model
try:
    pipe_lr = joblib.load(open("emotion_classifier_pipe_lr_07_july_2025.pkl", "rb"))
except FileNotFoundError:
    pipe_lr = None
    st.error("Model file not found in the same directory as app.py")

# Prediction Functions
def predict_emotions(docx):
    if pipe_lr:
        results = pipe_lr.predict([docx])
        return results[0]
    return "neutral"

def get_prediction_proba(docx):
    if pipe_lr:
        results = pipe_lr.predict_proba([docx])
        return results
    return np.array([[0.0]])

# Emoji Mapping
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🨮", "fear": "😨😱", "happy": "𫷦",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Page Configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    layout="centered",
)

# Custom CSS (Focus on fonts, layout, no color overrides)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* App container padding */
.stApp {
    padding: 2rem;
}

/* Headings */
h1, h2, h3, h4, label, p, div {
    font-weight: 600;
}

/* Inputs */
textarea, .stTextInput > div > div > input {
    border-radius: 12px;
    padding: 10px;
    font-size: 1rem;
}

/* Submit button styling */
[data-testid="stFormSubmitButton"] button {
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.3s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: scale(1.05);
    cursor: pointer;
}

/* Success message styling */
.stSuccess {
    border-radius: 10px;
    padding: 1rem;
    font-weight: 600;
    text-align: center;
}

/* Container styling */
.css-1aumxhk, .stColumn {
    border-radius: 12px;
    padding: 1rem !important;
    margin-bottom: 1rem;
    border: 1px solid;
    box-shadow: 0 0 20px;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    border-radius: 10px;
}

/* Sidebar adjustments */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .stApp {
        padding: 1rem;
    }
    .css-1aumxhk, .stColumn {
        padding: 0.75rem !important;
    }
}

</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
        <h1 style="text-align:center;font-size:42px;">
        <span style="background: linear-gradient(90deg, #a64bf4, #5e17eb); -webkit-background-clip: text; color: transparent;">
         AI Sentiment Analyzer
        </span>
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;'>Detect emotions in any text using an AI-powered ML model.</p>", unsafe_allow_html=True)

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        with st.container():
            st.subheader("Analyze Text")
            with st.form(key='myform'):
                raw_text = st.text_area("Type or paste your text here...")
                submit_text = st.form_submit_button(label='Analyze')

            if submit_text:
                cleaned_text = raw_text.strip()
                if not cleaned_text or len(re.findall(r'[a-zA-Z]', cleaned_text)) < 2:
                    st.warning("⚠️ Please enter valid text (not just symbols or dots).")
                elif pipe_lr is None:
                    st.error("🚫 Sentiment analysis model not loaded. Please check the model file path.")
                else:
                    prediction = predict_emotions(raw_text)
                    probability = get_prediction_proba(raw_text)
                    emoji_icon = emotions_emoji_dict.get(prediction, "")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 📝 Original Text")
                        st.write(raw_text)

                        st.markdown("#### 🎯 Prediction")
                        st.success(f"{prediction.capitalize()} {emoji_icon}")
                        if probability is not None and probability.size > 0:
                            st.markdown(f"**Confidence:** {np.max(probability):.2f}")
                        else:
                            st.markdown("**Confidence:** N/A")

                    with col2:
                        st.markdown("#### 📊 Prediction Probability")

                        valid_probabilities = (
                            probability is not None and
                            isinstance(probability, np.ndarray) and
                            probability.size > 0 and
                            not np.isnan(probability).any() and
                            np.isfinite(probability).all() and
                            np.max(probability) > 0
                        )

                        if valid_probabilities and pipe_lr.classes_ is not None and len(pipe_lr.classes_) > 0:
                            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                            proba_df_clean = proba_df.T.reset_index()
                            proba_df_clean.columns = ["emotions", "probability"]

                            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                                x=alt.X('emotions', sort='-y'),
                                y='probability',
                                color=alt.Color('emotions', legend=None)
                            ).properties(
                                width=300,
                                height=300
                            ).interactive()

                            st.altair_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ Could not generate chart due to invalid or zero probabilities, or missing model classes.")

    #elif choice == "Monitor":
    #    st.subheader("📡 Monitor (Coming Soon)")
     #   st.markdown("This section will show app usage statistics and logs.")

    else:
        st.subheader("About")
        st.markdown("""
        This AI-powered sentiment analysis tool is designed by Rabia Noor and Taiba Asif.

        Built using:
        - Streamlit for frontend
        - scikit-learn and joblib for ML modeling
        - Altair for interactive visualizations
        """)

if __name__ == '__main__':
    main()
