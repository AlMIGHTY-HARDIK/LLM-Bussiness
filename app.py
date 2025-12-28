import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import re
import warnings
import traceback

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="AI Analyst: Enterprise Edition", layout="wide", page_icon="🧠")
warnings.filterwarnings("ignore")

# API Setup
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("🚨 Groq API Key missing! Check .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)
MODEL_ID = "llama-3.3-70b-versatile" 

# ==========================================
# 🧠 LAYER 1: ADAPTIVE DATA ENGINE
# ==========================================

@st.cache_data
def load_and_adapt_data(file):
    """Ingests data and automatically adapts types."""
    try:
        file.seek(0)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)

        # 1. Normalize Headers
        df.columns = [
            str(c).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_').replace('-', '_') 
            for c in df.columns
        ]

        # 2. Adaptive Type Conversion
        date_col = next((c for c in df.columns if 'date' in c or 'invoice' in c), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], format='%d-%b-%y', errors='coerce')
            if df[date_col].isnull().mean() > 0.5: 
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            df['cleaned_year'] = df[date_col].dt.year

        sales_col = next((c for c in df.columns if c in ['gross_amount', 'net_amount_inr', 'net_amount', 'total_amount', 'sales']), None)
        if sales_col:
            if df[sales_col].dtype == 'object':
                 df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce')
            df['cleaned_sales'] = df[sales_col]

        # 3. String Normalization
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        return df
    except Exception as e:
        return None

# ==========================================
# 👁️ LAYER 2: DATA HEALTH SCANNER
# ==========================================

def get_data_health_report(df):
    """Scans for empty columns to prevent hallucinations."""
    report = []
    report.append(f"Total Rows: {len(df)}")
    
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0.5:
            report.append(f"⚠️ CRITICAL WARNING: Column '{col}' is {null_pct:.1%} empty. DO NOT USE IT. Find a valid alternative (e.g., 'shade_no').")
            
    return "\n".join(report)

# ==========================================
# 🛡️ LAYER 3: CODE SANITIZER (NEW)
# ==========================================

def clean_llm_response(text):
    """
    Extracts pure Python code from the LLM's response.
    Removes conversational fluff like 'Here is the code...'
    """
    # Pattern 1: Look for markdown code blocks
    code_block_pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
    else:
        # Pattern 2: If no markdown, use the whole text but strip generic text lines
        # This is a fallback. Ideally, the LLM adheres to instructions.
        code = text.strip()
    
    # Remove any line that doesn't look like code (basic heuristic)
    lines = code.split('\n')
    clean_lines = [line for line in lines if not line.lower().startswith(('here is', 'to address', 'note:', 'i have'))]
    
    return "\n".join(clean_lines)

# ==========================================
# 🛠️ LAYER 4: SELF-HEALING LOGIC
# ==========================================

# ==========================================
# 🛠️ LAYER 4: SELF-HEALING LOGIC (UPDATED)
# ==========================================

def generate_code_prompt(query, df, history_context, error_context=None):
    health_report = get_data_health_report(df)
    
    # 1. SEMANTIC HINTS (The Fix for State vs City)
    # We explicitly tell the AI how to find locations to prevent the specific bug you faced.
    hints = """
    ### 💡 SMART SEARCH HINTS:
    - **Location Logic:** 'Maharashtra' is a STATE. Look for columns like `agent_state`, `state`, or `region`. 
    - **Fallback:** If State columns are missing/empty, filter by known Cities: ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik'].
    - **Shade Logic:** If `shade_name` is empty, ALWAYS use `shade_no`.
    """

    prompt = f"""
    You are an Autonomous Python Data Analyst.
    
    ### DATA PROFILE:
    - Columns: {list(df.columns)}
    
    ### 🛡️ HEALTH WARNINGS:
    {health_report}
    
    {hints}
    
    ### 🧠 CODING RULES:
    1. **Fuzzy Matching:** Use `df[col].str.contains('val', case=False, na=False)` for text.
    2. **Empty Data Handling:** If a filter returns empty data, the code is WRONG. Try a different column.
    3. **Output:** Return ONLY valid Python code. Assign result to `result_df`.
    """
    
    if error_context:
        prompt += f"\n\n### 🚨 PREVIOUS ATTEMPT FAILED:"
        prompt += f"\nError: {error_context}"
        prompt += f"\nFIX: The previous logic returned no data or crashed. Try a different column or approach."
    
    prompt += f"\n\n### QUERY: {query}"
    
    if history_context:
        prompt += f"\n\n### CONTEXT: {history_context}"
        
    return prompt

def execute_with_self_correction(query, df, history_context, max_retries=2):
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # 1. Generate Code
            prompt = generate_code_prompt(query, df, history_context, last_error)
            
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1 # Slight creativity allowed for problem solving
            )
            raw_response = resp.choices[0].message.content
            code = clean_llm_response(raw_response)
            
            # 2. Execute Code
            code = re.sub(r"pd\.read_csv\(.*?\)", "df.copy()", code) 
            env = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns}
            exec(code, env)
            
            result_df = env.get("result_df")
            
            # 3. VALIDATION (Crucial Step)
            if not isinstance(result_df, pd.DataFrame):
                raise ValueError("Result is not a DataFrame.")
                
            # NEW: Treat "No Data" as an Error to trigger a retry!
            if result_df.empty:
                raise ValueError("Query returned 0 rows. The filter logic (e.g., City vs State) might be wrong.")
            
            # If we get here, success!
            if attempt > 0:
                st.toast(f"✅ Auto-corrected logic in attempt {attempt+1}", icon="🧠")
            return result_df, code, None
            
        except Exception as e:
            last_error = str(e)
            # If we have retries left, loop again. The 'last_error' will be fed back to the AI.
            if attempt < max_retries:
                st.toast(f"⚠️ Attempt {attempt+1} failed: {str(e)}. Retrying...", icon="🔧")
                continue
            else:
                # If all retries fail, return the empty result or error
                return None, code, last_error
            
# ==========================================
# 🗣️ LAYER 5: NARRATIVE ENGINE
# ==========================================

def generate_narrative(query, result_df):
    if result_df.empty: return "No data found."
    
    data_str = result_df.head(15).to_markdown(index=False)
    
    prompt = f"""
    You are a Senior Analyst. Query: "{query}"
    
    Data:
    {data_str}
    
    Format:
    1. Markdown Tables for lists.
    2. '₹' for currency.
    3. Direct, professional insights.
    """
    
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content

# ==========================================
# 🖥️ UI IMPLEMENTATION
# ==========================================

st.title("🧠 AI Analyst: Enterprise Edition")

with st.sidebar:
    file = st.file_uploader("Upload Data", type=["csv", "xlsx"])
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "data" in msg:
            with st.expander("View Data"):
                st.dataframe(msg["data"])

if file:
    if "df" not in st.session_state:
        st.session_state.df = load_and_adapt_data(file)
        st.toast("Data Loaded Successfully")
    
    df = st.session_state.df
    query = st.chat_input("Ask a question...")
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                history = "\n".join([m["content"] for m in st.session_state.messages[-3:]])
                
                result_df, code, error = execute_with_self_correction(query, df, history)
                
                if result_df is not None:
                    response = generate_narrative(query, result_df)
                    st.markdown(response)
                    with st.expander("Technical Details"):
                        st.code(code)
                        st.dataframe(result_df)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response, "data": result_df})
                else:
                    st.error("Analysis Failed.")
                    st.code(error)