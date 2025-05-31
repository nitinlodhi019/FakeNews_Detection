import joblib
import streamlit as st

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Set page config
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

st.title("🧠 Fake News Detector")
st.markdown("Paste any news article text below to check whether it's **Fake** or **Real**.")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'news_input' not in st.session_state:
    st.session_state.news_input = ""
if 'clear_input_flag' not in st.session_state:
    st.session_state.clear_input_flag = False

# Handle Clear Buttons before rendering widgets
if st.session_state.clear_input_flag:
    st.session_state.news_input = ""
    st.session_state.clear_input_flag = False
    st.success("🧾 Input text cleared!")

# Main Form
with st.form("news_form", clear_on_submit=False):
    st.markdown("### 📝 Enter News Article Text")
    news_input = st.text_area(
        label="",
        height=250,
        key="news_input"  # Binds directly to session_state.news_input
    )

    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        submit = st.form_submit_button("🔍 Predict")
    with col2:
        clear_input = st.form_submit_button("🧾 Clear Text")
    with col3:
        clear_history = st.form_submit_button("📜 Clear History")

# Handle Clear Buttons After Form Submission
if clear_input:
    st.session_state.clear_input_flag = True
    st.rerun()

if clear_history:
    st.session_state.history = []
    st.success("📜 Prediction history cleared!")

# Handle Prediction
if submit:
    if news_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Predict
        transformed = vectorizer.transform([news_input])
        prediction = model.predict(transformed)[0]
        prob = model.predict_proba(transformed)[0]

        # Show result
        if prediction == 0:
            result = f"🟥 FAKE NEWS (Confidence: {prob[0] * 100:.2f}%)"
            st.error(f"📢 Prediction: {result}")
        else:
            result = f"🟩 REAL NEWS (Confidence: {prob[1] * 100:.2f}%)"
            st.success(f"📢 Prediction: {result}")

        # Add to history
        st.session_state.history.append({
            "news": news_input[:80] + ("..." if len(news_input) > 80 else ""),
            "result": result
        })

# Show History
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Prediction History")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**{i}.** *{item['news']}* → **{item['result']}**")
