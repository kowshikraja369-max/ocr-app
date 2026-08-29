import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import easyocr
import numpy as np
import streamlit as st
from google import genai
from PIL import Image

GEMINI_API_KEY = "AQ.Ab8RN6JdMJM8kqinfPwD3bLuKG1lE1Bh5-SEAms_gLZT79ZeRg"

st.set_page_config(page_title="AI Medical & Text Reader", layout="centered")

st.title("💊 Smart Medical & Prescription AI")
st.write(
    "Upload a prescription/medicine image to extract text and analyze details."
)


@st.cache_resource
def load_ocr():
  return easyocr.Reader(["en"])


reader = load_ocr()

uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image", use_container_width=True)

  if st.button("🔍 Analyze Medicine Details"):
    with st.spinner("Reading image and analyzing with AI..."):
      image_np = np.array(image)
      results = reader.readtext(image_np)
      raw_text = " ".join([d[1] for d in results])

      if not raw_text.strip():
        st.warning("No text detected in this image.")
      else:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
                You are a medical text analyzer. Below is noisy OCR text extracted from a prescription or medicine image:
                "{raw_text}"

                Tasks:
                1. Identify potential medicine names from the noisy text and correct typos (e.g., "Betaloc IOO" -> "Betaloc 100", "Dorzolmilm" -> "Dorzolamide").
                2. For each identified medicine, provide:
                   - **Correct Medicine Name**
                   - **Purpose / What it is used for**
                   - **Common Side Effects**
                   - **Health & Safety Status**: Mark as 🟢 **[GREEN] Safe/Standard**, 🟡 **[YELLOW] Caution / Prescription Required**, or 🔴 **[RED] Warning / High Risk**.
                3. Add a clear general medical disclaimer at the bottom.

                Format the output cleanly in Markdown with bold headers and bullet points.
                """

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )

        st.subheader("📋 Medical AI Report")
        st.markdown(response.text)
