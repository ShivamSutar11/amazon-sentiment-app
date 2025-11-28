import streamlit as st
from transformers import pipeline

# Path to your local fine-tuned model folder
MODEL_PATH = "amazon_sentiment_finetuned"

# Map raw labels to human-readable ones
LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE",
}

@st.cache_resource
def load_model():
    clf = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH
    )
    return clf

clf = load_model()

st.set_page_config(page_title="Amazon Review Sentiment", layout="centered")
st.title("🛒 Amazon Review Sentiment Classifier")
st.write("Type an Amazon-style product review and see if your model thinks it's **positive** or **negative**.")

# Input area
review = st.text_area(
    "Enter a review:",
    placeholder="Example: The product quality is amazing for the price, totally worth it!",
    height=150
)

if st.button("Analyze sentiment"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Thinking..."):
            out = clf(review)[0]
            raw_label = out["label"]
            sentiment = LABEL_MAP.get(raw_label, raw_label)
            score = float(out["score"])

        # Display result
        st.subheader("Result")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sentiment:** {sentiment}")
        with col2:
            st.markdown(f"**Confidence:** {score:.3f}")

        # Show raw details if you want
        with st.expander("See raw model output"):
            st.json(out)

st.markdown("---")
st.caption("Powered by your fine-tuned DistilBERT model.")
