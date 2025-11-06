import streamlit as st
import pandas as pd
import plotly.express as px
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import google.generativeai as genai

st.set_page_config(page_title="AI Personal Finance Assistant", layout="wide")
st.title("💰 AI Personal Finance Assistant")

# Sidebar Inputs
google_api_key = st.sidebar.text_input("Enter your Google API Key (Gemini):", type="password")
uploaded_file = st.sidebar.file_uploader("Upload your transactions CSV", type=["csv"])
qa_question = st.sidebar.text_input("Ask a question about your transactions:")

# Configure GenAI
def configure_genai(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("models/gemini-2.5-pro")

# Auto-categorize uncategorized transactions
def categorize_transactions(df, model):
    uncategorized = df[df['Category'].isnull() | (df['Category'] == '')]
    if len(uncategorized) == 0:
        return df
    prompt = "Categorize these transactions into appropriate categories:\n"
    prompt += uncategorized.to_string(index=False)
    response = model.generate_content(prompt)
    categories = [line.split(":")[1].strip() if ":" in line else "Other" for line in response.text.split("\n") if line.strip()]
    df.loc[uncategorized.index, 'Category'] = categories
    return df

# Generate AI insights
def generate_insights(df, model):
    prompt = "You are a personal finance assistant. Analyze the following transactions and generate a concise summary with insights, trends, and money-saving tips:\n\n"
    prompt += df.to_string(index=False)
    response = model.generate_content(prompt)
    return response.text

# Answer Q&A
def answer_question(df, question, model):
    prompt = f"You are a personal finance assistant. Based on these transactions:\n{df.to_string(index=False)}\nAnswer the question: {question}"
    response = model.generate_content(prompt)
    return response.text

# PDF report
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
if uploaded_file and google_api_key:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Transactions Data")
    st.dataframe(df)

    model = configure_genai(google_api_key)

    # Auto-categorize
    df = categorize_transactions(df, model)
    
    # Trend Charts
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Month'] = df['Date'].dt.to_period('M').astype(str)

        # Weekly Spending
        weekly = df.groupby('Week')['Amount'].sum().reset_index()
        fig_week = px.line(weekly, x='Week', y='Amount', title="Weekly Spending")
        st.plotly_chart(fig_week, use_container_width=True)

        # Monthly Spending
        monthly = df.groupby('Month')['Amount'].sum().reset_index()
        fig_month = px.line(monthly, x='Month', y='Amount', title="Monthly Spending")
        st.plotly_chart(fig_month, use_container_width=True)

    # Category Spending
    if 'Category' in df.columns:
        cat_fig = px.bar(df.groupby('Category')['Amount'].sum().reset_index(), x='Category', y='Amount', title="Spending by Category")
        st.plotly_chart(cat_fig, use_container_width=True)

    # Generate AI Insights
    if st.button("Generate AI Insights"):
        insights = generate_insights(df, model)
        st.subheader("📝 AI Insights")
        st.write(insights)

        # PDF download
        pdf_bytes = create_pdf(df, insights)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="Finance_Report.pdf", mime="application/pdf")

    # Q&A
    if qa_question:
        answer = answer_question(df, qa_question, model)
        st.subheader("❓ Answer")
        st.write(answer)
