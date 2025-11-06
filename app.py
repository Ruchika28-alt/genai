import streamlit as st
import pandas as pd
import plotly.express as px
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import pipeline

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="AI Personal Finance Assistant", layout="wide")
st.title("💰 AI Personal Finance Assistant")

# Sidebar Inputs
uploaded_file = st.sidebar.file_uploader("Upload your transactions CSV", type=["csv"])
qa_question = st.sidebar.text_input("Ask a question about your transactions:")

# -----------------------------
# Initialize Hugging Face model
# -----------------------------
@st.cache_resource
def get_hf_model():
    # Small CPU-friendly instruction-following model
    return pipeline("text-generation", model="google/flan-t5-small")

hf_model = get_hf_model()

def call_hf(prompt, max_tokens=200):
    try:
        result = hf_model(prompt, max_new_tokens=max_tokens)
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Hugging Face error: {e}")
        return None

# -----------------------------
# Auto-categorize transactions
# -----------------------------
def categorize_transactions(df):
    if 'Category' not in df.columns:
        df['Category'] = ""
    uncategorized = df[df['Category'].isnull() | (df['Category'] == '')]
    if len(uncategorized) == 0:
        return df

    table_text = uncategorized.to_string(index=False)
    prompt = (
        "You are a helpful personal finance assistant. "
        "Categorize the following transactions into appropriate categories:\n"
        f"{table_text}\n"
        "Provide only the category names in the same order, one per line."
    )

    categories = call_hf(prompt, max_tokens=300)
    if categories:
        categories_list = [line.strip() for line in categories.split("\n") if line.strip()]
        if len(categories_list) != len(uncategorized):
            categories_list = ["Other"] * len(uncategorized)
        df.loc[uncategorized.index, 'Category'] = categories_list
    else:
        df.loc[uncategorized.index, 'Category'] = "Other"
    return df

# -----------------------------
# Generate AI insights
# -----------------------------
def generate_insights(df):
    preview_df = df.head(20)  # limit rows for small model
    table_text = preview_df.to_string(index=False)
    prompt = (
        "You are a helpful personal finance assistant. "
        "Analyze the following transactions and provide a concise summary including:\n"
        "- Main spending categories\n"
        "- Notable trends\n"
        "- Suggestions to save money\n\n"
        f"Transactions:\n{table_text}\n\n"
        "Provide your insights in simple, clear sentences."
    )
    insights = call_hf(prompt, max_tokens=300)
    return insights if insights else "No insights generated."

# -----------------------------
# Answer Q&A
# -----------------------------
def answer_question(df, question):
    preview_df = df.head(20)
    table_text = preview_df.to_string(index=False)
    prompt = (
        f"You are a personal finance assistant. "
        f"Based on these transactions:\n{table_text}\nAnswer this question: {question}"
    )
    answer = call_hf(prompt, max_tokens=150)
    return answer if answer else "No answer generated."

# -----------------------------
# Create PDF report
# -----------------------------
def create_pdf(df, insights):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(72, 800, "Personal Finance Report")
    text = c.beginText(40, 780)
    text.setFont("Helvetica", 12)

    text.textLine("Transactions Summary:")
    for line in df.to_string(index=False).split("\n"):
        text.textLine(line)

    text.textLine("\nAI Insights:")
    for line in insights.split("\n"):
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# -----------------------------
# Main logic
# -----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Transactions Data")
    st.dataframe(df)

    # Categorize transactions
    df = categorize_transactions(df)

    # Trend Charts
    if 'Amount' in df.columns:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Week'] = df['Date'].dt.isocalendar().week
            df['Month'] = df['Date'].dt.to_period('M').astype(str)
            weekly = df.groupby('Week')['Amount'].sum().reset_index()
            fig_week = px.line(weekly, x='Week', y='Amount', title="Weekly Spending")
            st.plotly_chart(fig_week, use_container_width=True)
            monthly = df.groupby('Month')['Amount'].sum().reset_index()
            fig_month = px.line(monthly, x='Month', y='Amount', title="Monthly Spending")
            st.plotly_chart(fig_month, use_container_width=True)

        if 'Category' not in df.columns:
            df['Category'] = "Other"
        cat_df = df.groupby('Category')['Amount'].sum().reset_index()
        cat_fig = px.bar(cat_df, x='Category', y='Amount', title="Spending by Category")
        st.plotly_chart(cat_fig, use_container_width=True)

    # Generate AI Insights
    if st.button("Generate AI Insights"):
        insights = generate_insights(df)
        st.subheader("📝 AI Insights")
        st.write(insights)
        pdf_bytes = create_pdf(df, insights)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="Finance_Report.pdf", mime="application/pdf")

    # Q&A Section
    if qa_question:
        answer = answer_question(df, qa_question)
        st.subheader("❓ Answer")
        st.write(answer)
else:
    st.info("Upload a CSV to start.")
