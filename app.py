import streamlit as st
import pandas as pd
import plotly.express as px
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import openai

# Streamlit page config
st.set_page_config(page_title="AI Personal Finance Assistant", layout="wide")
st.title("💰 AI Personal Finance Assistant")

# Sidebar Inputs
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
uploaded_file = st.sidebar.file_uploader("Upload your transactions CSV", type=["csv"])
qa_question = st.sidebar.text_input("Ask a question about your transactions:")

# Configure OpenAI API
def configure_openai(api_key):
    openai.api_key = api_key

# Helper function to call GPT
def call_gpt(prompt, max_tokens=300):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

# Auto-categorize transactions
def categorize_transactions(df):
    if 'Category' not in df.columns:
        df['Category'] = ""

    uncategorized = df[df['Category'].isnull() | (df['Category'] == '')]
    if len(uncategorized) == 0:
        return df

    prompt = "Categorize these transactions into appropriate categories:\n"
    prompt += uncategorized.to_string(index=False)

    categories = call_gpt(prompt, max_tokens=500)
    if categories:
        categories_list = [line.strip() for line in categories.split("\n") if line.strip()]
        if len(categories_list) != len(uncategorized):
            categories_list = ["Other"] * len(uncategorized)
        df.loc[uncategorized.index, 'Category'] = categories_list
    else:
        df.loc[uncategorized.index, 'Category'] = "Other"

    return df

# Generate AI insights
def generate_insights(df):
    prompt = "Analyze the following transactions and generate a concise summary with insights, trends, and money-saving tips:\n"
    prompt += df.to_string(index=False)
    insights = call_gpt(prompt, max_tokens=500)
    return insights if insights else "No insights generated."

# Answer user Q&A
def answer_question(df, question):
    prompt = f"Based on these transactions:\n{df.to_string(index=False)}\nAnswer the question: {question}"
    answer = call_gpt(prompt, max_tokens=300)
    return answer if answer else "No answer generated."

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
if uploaded_file and openai_api_key:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Transactions Data")
    st.dataframe(df)

    configure_openai(openai_api_key)

    # Auto-categorize
    df = categorize_transactions(df)

    # Ensure proper columns
    if 'Amount' not in df.columns:
        st.error("CSV must contain 'Amount' column")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Amount'])
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Month'] = df['Date'].dt.to_period('M').astype(str)

        # Weekly Spending Chart
        weekly = df.groupby('Week')['Amount'].sum().reset_index()
        if not weekly.empty:
            fig_week = px.line(weekly, x='Week', y='Amount', title="Weekly Spending")
            st.plotly_chart(fig_week, use_container_width=True)

        # Monthly Spending Chart
        monthly = df.groupby('Month')['Amount'].sum().reset_index()
        if not monthly.empty:
            fig_month = px.line(monthly, x='Month', y='Amount', title="Monthly Spending")
            st.plotly_chart(fig_month, use_container_width=True)

    # Category Spending Chart
    if 'Category' not in df.columns:
        df['Category'] = "Other"
    if 'Amount' in df.columns:
        cat_df = df.groupby('Category')['Amount'].sum().reset_index()
        if not cat_df.empty:
            cat_fig = px.bar(cat_df, x='Category', y='Amount', title="Spending by Category")
            st.plotly_chart(cat_fig, use_container_width=True)

    # Generate AI Insights
    if st.button("Generate AI Insights"):
        insights = generate_insights(df)
        st.subheader("📝 AI Insights")
        st.write(insights)

        # PDF download
        pdf_bytes = create_pdf(df, insights)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="Finance_Report.pdf", mime="application/pdf")

    # Q&A Section
    if qa_question:
        answer = answer_question(df, qa_question)
        st.subheader("❓ Answer")
        st.write(answer)

else:
    st.info("Upload a CSV and enter your OpenAI API Key to start.")
