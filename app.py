import streamlit as st
import pandas as pd
import plotly.express as px
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import google.generativeai as genai

# Streamlit configuration
st.set_page_config(page_title="AI Personal Finance Dashboard", layout="wide")
st.title("💰 AI-Powered Personal Finance Dashboard")

# Sidebar inputs
google_api_key = st.sidebar.text_input("Enter your Google API Key (Gemini):", type="password")
uploaded_file = st.sidebar.file_uploader("Upload your transactions CSV", type=["csv"])

# Helper: Generate insights with GenAI
def generate_genai_insights(df, api_key):
    genai.configure(api_key=api_key)
    prompt = (
        "You are a personal finance assistant. "
        "Analyze the following transactions and generate a concise summary with insights, "
        "budget recommendations, and tips for saving money:\n\n"
    )
    prompt += df.to_string(index=False)
    try:
        # Use a supported Gemini model (2.5 or 3.0)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return None

# Create PDF report
def create_pdf(df, insights):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(72, 800, "Personal Finance Report")
    text = c.beginText(40, 780)
    text.setFont("Helvetica", 12)
    
    text.textLine("Transactions Summary:")
    for line in df.to_string(index=False).split('\n'):
        text.textLine(line)
    
    text.textLine("\nAI Insights:")
    for line in insights.split('\n'):
        text.textLine(line)
    
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# Main logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Transactions Data")
    st.dataframe(df)

    # Visual: Spending per category
    if 'Category' in df.columns and 'Amount' in df.columns:
        fig = px.bar(
            df.groupby('Category')['Amount'].sum().reset_index(),
            x='Category', y='Amount', title="Spending by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Generate AI insights
    if google_api_key and st.button("Generate AI Insights"):
        insights = generate_genai_insights(df, google_api_key)
        if insights:
            st.subheader("📝 AI Insights")
            st.write(insights)

            # PDF download
            pdf_bytes = create_pdf(df, insights)
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="Finance_Report.pdf",
                mime="application/pdf"
            )
