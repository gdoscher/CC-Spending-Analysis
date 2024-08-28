import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
from langchain_community.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatAnthropic
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from serpapi import GoogleSearch

# Initialize the Anthropic client
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]

# Initialize SerpAPI client
serpapi_api_key = st.secrets["SERP_API_KEY"]

# Initialize global variables
if 'df' not in st.session_state:
    st.session_state.df = None

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def web_search(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_api_key,
        "num": 3
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" in results:
        return "\n".join([f"Title: {r['title']}\nSnippet: {r['snippet']}\nLink: {r['link']}\n" 
                          for r in results["organic_results"][:3]])
    return "No results found."

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))
            process_data()
            return "File uploaded and processed successfully. You can now ask questions about your spending."
        except Exception as e:
            return f"Error loading CSV: {str(e)}"
    return "No file uploaded."

def process_data():
    df = st.session_state.df
    
    # Determine if it's AMEX or Chase data
    if 'Transaction Date' in df.columns:
        date_col = 'Transaction Date'
    elif 'Date' in df.columns:
        date_col = 'Date'
    else:
        st.error("Unable to identify date column. Please ensure your CSV has a 'Date' or 'Transaction Date' column.")
        return

    # Standardize column names
    column_mapping = {
        date_col: 'Date',
        'Description': 'Description',
        'Category': 'Category',
        'Amount': 'Amount',
    }
    df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required_columns = ['Date', 'Amount', 'Category']
    if not all(col in df.columns for col in required_columns):
        st.error("CSV must contain 'Date', 'Amount', and 'Category' columns")
        return None

    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure Amount is numeric and expenses are positive
    df['Amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    df['Amount'] = df['Amount'].abs()

    # Remove rows where Category is empty (likely payments or credits)
    df = df[df['Category'] != '']

    st.session_state.df = df

def analyze_categories(*args, **kwargs):
    if st.session_state.df is None:
        return "No data has been uploaded yet. Please upload a CSV file first."

    category_summary = st.session_state.df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    total_spending = category_summary.sum()
    category_percentages = (category_summary / total_spending * 100).round(2)

    summary = "Spending by category:\n"
    for category, amount in category_summary.items():
        percentage = category_percentages[category]
        summary += f"{category}: ${amount:.2f} ({percentage}%)\n"

    return summary

def create_pie_chart(*args, **kwargs):
    if st.session_state.df is None:
        return "No data has been uploaded yet. Please upload a CSV file first."

    spending_by_category = st.session_state.df.groupby('Category')['Amount'].sum().abs()
    fig = px.pie(values=spending_by_category.values, names=spending_by_category.index, title='Spending by Category')
    return fig

def get_df_info():
    if st.session_state.df is None:
        return "No data has been uploaded yet. Please upload a CSV file first."

    info = f"DataFrame Shape: {st.session_state.df.shape}\n"
    info += f"Column Names: {', '.join(st.session_state.df.columns)}\n"
    info += f"Data Types:\n{st.session_state.df.dtypes}\n"
    info += f"Total transactions: {len(st.session_state.df)}\n"
    info += f"Date range: {st.session_state.df['Date'].min()} to {st.session_state.df['Date'].max()}\n"
    info += f"Total spending: ${st.session_state.df['Amount'].sum():.2f}\n"
    return info

# Define tools
tools = [
    Tool(
        name="Web Search",
        func=web_search,
        description="Useful for searching the web for information about credit cards, rewards programs, or transaction categories."
    ),
    Tool(
        name="Analyze Categories",
        func=analyze_categories,
        description="Analyzes the spending categories in the uploaded CSV data."
    ),
    Tool(
        name="Create Pie Chart",
        func=create_pie_chart,
        description="Creates a pie chart of spending by category."
    ),
    Tool(
        name="Get DataFrame Info",
        func=get_df_info,
        description="Get information about the uploaded DataFrame."
    ),
    Tool(
        name="Python REPL",
        func=PythonREPL().run,
        description="Useful for performing complex calculations or data manipulations on the DataFrame 'st.session_state.df'."
    )
]

# Initialize the agent
llm = ChatAnthropic(temperature=0, anthropic_api_key=anthropic_api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)

# Streamlit UI
st.title("AI-Powered Spending Analysis and Credit Card Recommendation")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    result = load_data(uploaded_file)
    st.success(result)

st.subheader("Chat with AI Assistant")
user_input = st.text_input("Ask a question about your spending or credit card recommendations:")

if user_input:
    if st.session_state.df is None:
        st.warning("Please upload a CSV file before asking questions.")
    else:
        st_callback = StreamlitCallbackHandler(st.container())
        # Provide context about the DataFrame to the agent
        df_info = get_df_info()
        full_input = f"DataFrame Info: {df_info}\n\nUser Question: {user_input}"
        response = agent.run(full_input, callbacks=[st_callback])
        st.write(response)

        # Handle the response
        if isinstance(response, str):
            st.write(response)

            # Check if the response mentions creating a pie chart
            if "pie chart" in response.lower():
                st.write("Generating pie chart...")
                pie_chart = create_pie_chart()
                st.plotly_chart(pie_chart)
        elif hasattr(response, 'data'):  # Check if it's a plotly figure
            st.plotly_chart(response)
        else:
            st.write("Unexpected response type. Please try a different question.")