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
            with st.spinner("Analyzing comprehensive medical details..."):
                try:
                    prompt = """
                    Analyze this medical prescription/image in detail and provide the output clearly structured with these exact sections:
                    1. **Medicine Name**: 
                    2. **Purpose / What it is used for**: 
                    3. **Side-Effects**: (List common or potential side-effects)
                    4. **Dosage / How to take it**: 
                    5. **Timing (When to eat)**: 
                    6. **Frequency (How many times per day)**: 
                    7. **Suitable Age Group**: 
                    8. **Medicine Rating (out of 5)**: 
                    9. **Safety Color Indicator**: (Output strictly as 🟢 Green, 🟡 Yellow, or 🔴 Red)
                    """
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=[image, prompt]
                    )
                    st.write(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
