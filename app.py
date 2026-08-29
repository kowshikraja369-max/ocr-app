import streamlit as st
from google import genai
from PIL import Image

st.title("AI Medical & Prescription Analyzer")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Gemini API Key not found in Streamlit Secrets!")
else:
    client = genai.Client(api_key=api_key)

    uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        if st.button("Analyze Medicine Details"):
            with st.spinner("Analyzing..."):
                try:
                    prompt = "Analyze this medical image, identify the medicine/prescription details, and explain what it is used for clearly."
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=[image, prompt]
                    )
                    st.write(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
