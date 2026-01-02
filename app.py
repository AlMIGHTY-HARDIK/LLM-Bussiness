# # # import streamlit as st
# # # import pandas as pd
# # # import io
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # from groq import Groq
# # # import re
# # # import warnings
# # # import traceback

# # # # --- 1. CORE CONFIGURATION ---
# # # st.set_page_config(page_title="AI Analyst: Enterprise Edition", layout="wide", page_icon="🧠")
# # # warnings.filterwarnings("ignore")

# # # # API Setup
# # # api_key = st.secrets.get("GROQ_API_KEY")
# # # if not api_key:
# # #     st.error("🚨 Groq API Key missing! Check .streamlit/secrets.toml")
# # #     st.stop()

# # # client = Groq(api_key=api_key)
# # # MODEL_ID = "llama-3.3-70b-versatile" 

# # # # ==========================================
# # # # 🧠 LAYER 1: ADAPTIVE DATA ENGINE
# # # # ==========================================

# # # @st.cache_data
# # # def load_and_adapt_data(file):
# # #     """Ingests data and automatically adapts types."""
# # #     try:
# # #         file.seek(0)
# # #         if file.name.endswith('.csv'):
# # #             df = pd.read_csv(file, low_memory=False)
# # #         else:
# # #             df = pd.read_excel(file)

# # #         # 1. Normalize Headers
# # #         df.columns = [
# # #             str(c).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_').replace('-', '_') 
# # #             for c in df.columns
# # #         ]

# # #         # 2. Adaptive Type Conversion
# # #         date_col = next((c for c in df.columns if 'date' in c or 'invoice' in c), None)
# # #         if date_col:
# # #             df[date_col] = pd.to_datetime(df[date_col], format='%d-%b-%y', errors='coerce')
# # #             if df[date_col].isnull().mean() > 0.5: 
# # #                 df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
# # #             df['cleaned_year'] = df[date_col].dt.year

# # #         sales_col = next((c for c in df.columns if c in ['gross_amount', 'net_amount_inr', 'net_amount', 'total_amount', 'sales']), None)
# # #         if sales_col:
# # #             if df[sales_col].dtype == 'object':
# # #                  df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce')
# # #             df['cleaned_sales'] = df[sales_col]

# # #         # 3. String Normalization
# # #         for col in df.select_dtypes(include=['object']).columns:
# # #             df[col] = df[col].astype(str).str.strip()

# # #         return df
# # #     except Exception as e:
# # #         return None

# # # # ==========================================
# # # # 👁️ LAYER 2: DATA HEALTH SCANNER
# # # # ==========================================

# # # def get_data_health_report(df):
# # #     """Scans for empty columns to prevent hallucinations."""
# # #     report = []
# # #     report.append(f"Total Rows: {len(df)}")
    
# # #     for col in df.columns:
# # #         null_pct = df[col].isnull().mean()
# # #         if null_pct > 0.5:
# # #             report.append(f"⚠️ CRITICAL WARNING: Column '{col}' is {null_pct:.1%} empty. DO NOT USE IT. Find a valid alternative (e.g., 'shade_no').")
            
# # #     return "\n".join(report)

# # # # ==========================================
# # # # 🛡️ LAYER 3: CODE SANITIZER (NEW)
# # # # ==========================================

# # # def clean_llm_response(text):
# # #     """
# # #     Extracts pure Python code from the LLM's response.
# # #     Removes conversational fluff like 'Here is the code...'
# # #     """
# # #     # Pattern 1: Look for markdown code blocks
# # #     code_block_pattern = r"```(?:python)?\s*(.*?)```"
# # #     match = re.search(code_block_pattern, text, re.DOTALL)
    
# # #     if match:
# # #         code = match.group(1).strip()
# # #     else:
# # #         # Pattern 2: If no markdown, use the whole text but strip generic text lines
# # #         # This is a fallback. Ideally, the LLM adheres to instructions.
# # #         code = text.strip()
    
# # #     # Remove any line that doesn't look like code (basic heuristic)
# # #     lines = code.split('\n')
# # #     clean_lines = [line for line in lines if not line.lower().startswith(('here is', 'to address', 'note:', 'i have'))]
    
# # #     return "\n".join(clean_lines)

# # # # ==========================================
# # # # 🛠️ LAYER 4: SELF-HEALING LOGIC
# # # # ==========================================

# # # # ==========================================
# # # # 🛠️ LAYER 4: SELF-HEALING LOGIC (UPDATED)
# # # # ==========================================

# # # def generate_code_prompt(query, df, history_context, error_context=None):
# # #     health_report = get_data_health_report(df)
    
# # #     # 1. SEMANTIC HINTS (The Fix for State vs City)
# # #     # We explicitly tell the AI how to find locations to prevent the specific bug you faced.
# # #     hints = """
# # #     ### 💡 SMART SEARCH HINTS:
# # #     - **Location Logic:** 'Maharashtra' is a STATE. Look for columns like `agent_state`, `state`, or `region`. 
# # #     - **Fallback:** If State columns are missing/empty, filter by known Cities: ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik'].
# # #     - **Shade Logic:** If `shade_name` is empty, ALWAYS use `shade_no`.
# # #     """

# # #     prompt = f"""
# # #     You are an Autonomous Python Data Analyst.
    
# # #     ### DATA PROFILE:
# # #     - Columns: {list(df.columns)}
    
# # #     ### 🛡️ HEALTH WARNINGS:
# # #     {health_report}
    
# # #     {hints}
    
# # #     ### 🧠 CODING RULES:
# # #     1. **Fuzzy Matching:** Use `df[col].str.contains('val', case=False, na=False)` for text.
# # #     2. **Empty Data Handling:** If a filter returns empty data, the code is WRONG. Try a different column.
# # #     3. **Output:** Return ONLY valid Python code. Assign result to `result_df`.
# # #     """
    
# # #     if error_context:
# # #         prompt += f"\n\n### 🚨 PREVIOUS ATTEMPT FAILED:"
# # #         prompt += f"\nError: {error_context}"
# # #         prompt += f"\nFIX: The previous logic returned no data or crashed. Try a different column or approach."
    
# # #     prompt += f"\n\n### QUERY: {query}"
    
# # #     if history_context:
# # #         prompt += f"\n\n### CONTEXT: {history_context}"
        
# # #     return prompt

# # # def execute_with_self_correction(query, df, history_context, max_retries=2):
# # #     last_error = None
    
# # #     for attempt in range(max_retries + 1):
# # #         try:
# # #             # 1. Generate Code
# # #             prompt = generate_code_prompt(query, df, history_context, last_error)
            
# # #             resp = client.chat.completions.create(
# # #                 model=MODEL_ID,
# # #                 messages=[{"role": "system", "content": prompt}],
# # #                 temperature=0.1 # Slight creativity allowed for problem solving
# # #             )
# # #             raw_response = resp.choices[0].message.content
# # #             code = clean_llm_response(raw_response)
            
# # #             # 2. Execute Code
# # #             code = re.sub(r"pd\.read_csv\(.*?\)", "df.copy()", code) 
# # #             env = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns}
# # #             exec(code, env)
            
# # #             result_df = env.get("result_df")
            
# # #             # 3. VALIDATION (Crucial Step)
# # #             if not isinstance(result_df, pd.DataFrame):
# # #                 raise ValueError("Result is not a DataFrame.")
                
# # #             # NEW: Treat "No Data" as an Error to trigger a retry!
# # #             if result_df.empty:
# # #                 raise ValueError("Query returned 0 rows. The filter logic (e.g., City vs State) might be wrong.")
            
# # #             # If we get here, success!
# # #             if attempt > 0:
# # #                 st.toast(f"✅ Auto-corrected logic in attempt {attempt+1}", icon="🧠")
# # #             return result_df, code, None
            
# # #         except Exception as e:
# # #             last_error = str(e)
# # #             # If we have retries left, loop again. The 'last_error' will be fed back to the AI.
# # #             if attempt < max_retries:
# # #                 st.toast(f"⚠️ Attempt {attempt+1} failed: {str(e)}. Retrying...", icon="🔧")
# # #                 continue
# # #             else:
# # #                 # If all retries fail, return the empty result or error
# # #                 return None, code, last_error
            
# # # # ==========================================
# # # # 🗣️ LAYER 5: NARRATIVE ENGINE
# # # # ==========================================

# # # def generate_narrative(query, result_df):
# # #     if result_df.empty: return "No data found."
    
# # #     data_str = result_df.head(15).to_markdown(index=False)
    
# # #     prompt = f"""
# # #     You are a Senior Analyst. Query: "{query}"
    
# # #     Data:
# # #     {data_str}
    
# # #     Format:
# # #     1. Markdown Tables for lists.
# # #     2. '₹' for currency.
# # #     3. Direct, professional insights.
# # #     """
    
# # #     resp = client.chat.completions.create(
# # #         model=MODEL_ID,
# # #         messages=[{"role": "system", "content": prompt}],
# # #         temperature=0.7
# # #     )
# # #     return resp.choices[0].message.content

# # # # ==========================================
# # # # 🖥️ UI IMPLEMENTATION
# # # # ==========================================

# # # st.title("🧠 AI Analyst: Enterprise Edition")

# # # with st.sidebar:
# # #     file = st.file_uploader("Upload Data", type=["csv", "xlsx"])
# # #     if st.button("Clear History"):
# # #         st.session_state.messages = []
# # #         st.rerun()

# # # if "messages" not in st.session_state:
# # #     st.session_state.messages = []

# # # # Display History
# # # for msg in st.session_state.messages:
# # #     with st.chat_message(msg["role"]):
# # #         st.markdown(msg["content"])
# # #         if "data" in msg:
# # #             with st.expander("View Data"):
# # #                 st.dataframe(msg["data"])

# # # if file:
# # #     if "df" not in st.session_state:
# # #         st.session_state.df = load_and_adapt_data(file)
# # #         st.toast("Data Loaded Successfully")
    
# # #     df = st.session_state.df
# # #     query = st.chat_input("Ask a question...")
    
# # #     if query:
# # #         st.session_state.messages.append({"role": "user", "content": query})
# # #         with st.chat_message("user"):
# # #             st.write(query)
            
# # #         with st.chat_message("assistant"):
# # #             with st.spinner("Analyzing..."):
# # #                 history = "\n".join([m["content"] for m in st.session_state.messages[-3:]])
                
# # #                 result_df, code, error = execute_with_self_correction(query, df, history)
                
# # #                 if result_df is not None:
# # #                     response = generate_narrative(query, result_df)
# # #                     st.markdown(response)
# # #                     with st.expander("Technical Details"):
# # #                         st.code(code)
# # #                         st.dataframe(result_df)
                    
# # #                     st.session_state.messages.append({"role": "assistant", "content": response, "data": result_df})
# # #                 else:
# # #                     st.error("Analysis Failed.")
# # #                     st.code(error)





# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from groq import Groq, RateLimitError
# # import re
# # import warnings
# # import traceback

# # # --- 1. CONFIGURATION & SETUP ---
# # st.set_page_config(
# #     page_title="AI Analyst: Enterprise Edition", 
# #     layout="wide", 
# #     page_icon="🧠"
# # )
# # warnings.filterwarnings("ignore")

# # # API Setup
# # api_key = st.secrets.get("GROQ_API_KEY")
# # if not api_key:
# #     st.error("🚨 Groq API Key missing! Check .streamlit/secrets.toml")
# #     st.stop()

# # client = Groq(api_key=api_key)
# # MODEL_ID = "llama-3.3-70b-versatile" 

# # # ==========================================
# # # 🧠 LAYER 1: ADAPTIVE DATA ENGINE
# # # Capabilities: Robust Loading, Auto-Cleaning, Type Inference
# # # ==========================================

# # @st.cache_data
# # def load_and_adapt_data(file):
# #     """
# #     Ingests data, normalizes headers, and intelligently detects date/currency columns.
# #     """
# #     try:
# #         file.seek(0)
# #         if file.name.endswith('.csv'):
# #             df = pd.read_csv(file, low_memory=False)
# #         else:
# #             df = pd.read_excel(file)

# #         # 1. Standardize Headers (Snake Case)
# #         df.columns = [
# #             str(c).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_').replace('-', '_') 
# #             for c in df.columns
# #         ]

# #         # 2. Adaptive Date Parsing
# #         date_col = next((c for c in df.columns if 'date' in c or 'invoice' in c), None)
# #         if date_col:
# #             df[date_col] = pd.to_datetime(df[date_col], format='%d-%b-%y', errors='coerce')
# #             # Fallback for mixed formats
# #             if df[date_col].isnull().mean() > 0.5: 
# #                 df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
# #             df['cleaned_year'] = df[date_col].dt.year

# #         # 3. Adaptive Currency Cleaning
# #         # Finds columns like 'net_amount', 'sales', 'gross_amount'
# #         sales_col = next((c for c in df.columns if c in ['net_amount_inr', 'net_amount', 'gross_amount', 'total_amount', 'sales']), None)
# #         if sales_col:
# #             if df[sales_col].dtype == 'object':
# #                  df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce')
# #             df['cleaned_sales'] = df[sales_col]

# #         # 4. String Normalization (Crucial for fuzzy matching)
# #         for col in df.select_dtypes(include=['object']).columns:
# #             df[col] = df[col].astype(str).str.strip()

# #         return df
# #     except Exception as e:
# #         return None

# # # ==========================================
# # # 👁️ LAYER 2: DATA HEALTH SCANNER
# # # Capabilities: Prevents hallucinations by flagging empty columns
# # # ==========================================

# # def get_data_health_report(df):
# #     """
# #     Scans the dataset to warn the AI about empty or low-quality columns.
# #     """
# #     report = []
# #     for col in df.columns:
# #         null_pct = df[col].isnull().mean()
# #         if null_pct > 0.5:
# #             report.append(f"⚠️ CRITICAL WARNING: Column '{col}' is {null_pct:.1%} empty. DO NOT USE IT. Find a valid alternative (e.g., use 'shade_no' if 'shade_name' is empty).")
# #     return "\n".join(report)

# # # ==========================================
# # # 🛡️ LAYER 3: CODE SANITIZER
# # # Capabilities: Extracts pure Python from chatty LLM responses
# # # ==========================================

# # def clean_llm_response(text):
# #     """
# #     Strips conversational text (e.g., "Here is the code") to prevent SyntaxErrors.
# #     """
# #     code_block_pattern = r"```(?:python)?\s*(.*?)```"
# #     match = re.search(code_block_pattern, text, re.DOTALL)
    
# #     if match:
# #         code = match.group(1).strip()
# #     else:
# #         code = text.strip()
    
# #     # Remove lines that definitely aren't code
# #     lines = code.split('\n')
# #     clean_lines = [line for line in lines if not line.lower().startswith(('here is', 'to address', 'note:', 'i have', 'python'))]
# #     return "\n".join(clean_lines)

# # # ==========================================
# # # 🛠️ LAYER 4: SELF-HEALING LOGIC ENGINE
# # # Capabilities: Retries on error, checks for empty results, enforces hints
# # # ==========================================

# # def generate_code_prompt(query, df, history_context, error_context=None):
# #     health_report = get_data_health_report(df)
    
# #     # SEMANTIC HINTS (The "Brains" of the operation)
# #     hints = """
# #     ### 💡 ANALYST HINTS:
# #     - **Null Handling:** If ranking Top N, ALWAYS filter out NaNs first: `df.dropna(subset=['col'])`.
# #     - **Location Logic:** 'Maharashtra' is a STATE. Cities are 'Mumbai', 'Pune'. Use `agent_state` or `bill_to_party_city`.
# #     - **Fuzzy Match:** NEVER use `==` for strings. Use `df[col].str.contains('val', case=False, na=False)`.
# #     - **Date Logic:** Use `cleaned_year` for fiscal/calendar year queries.
# #     - **Visuals:** If the user asks for a trend, comparison, or distribution, generate a Plotly chart.
# #     - **Plotting Rule:** Create a figure object named `fig`. DO NOT use `fig.show()`. Streamlit will handle it.
# #     - **Example:**
# #       ```python
# #       import plotly.express as px
# #       fig = px.bar(df, x='city', y='sales', title='Sales by City')
# #       ```
# #     """

# #     prompt = f"""
# #     You are an Expert Python Data Analyst.
    
# #     ### DATA PROFILE:
# #     - Columns: {list(df.columns)}
    
# #     ### 🛡️ HEALTH WARNINGS:
# #     {health_report}
    
# #     {hints}
    
# #     ### 🧠 STRICT RULES:
# #     1. **Valid Code Only:** Output pure Python code inside ```python tags.
# #     2. **Result Variable:** Assign the final output dataframe to `result_df`.
# #     3. **No Empty Results:** If your filter logic returns an empty dataframe, the logic is wrong. Try a different column.
# #     """
    
# #     if error_context:
# #         prompt += f"\n\n### 🚨 PREVIOUS ATTEMPT FAILED:\nError: {error_context}\nFix: Rewrite the code to fix this specific error."
    
# #     prompt += f"\n\n### QUERY: {query}"
    
# #     if history_context:
# #         prompt += f"\n\n### CONTEXT: {history_context}"
        
# #     return prompt

# # def execute_with_self_correction(query, df, history_context, max_retries=2):
# #     last_error = None
    
# #     for attempt in range(max_retries + 1):
# #         try:
# #             # 1. Generate Code
# #             prompt = generate_code_prompt(query, df, history_context, last_error)
            
# #             resp = client.chat.completions.create(
# #                 model=MODEL_ID,
# #                 messages=[{"role": "system", "content": prompt}],
# #                 temperature=0.1 # Low temp for precision
# #             )
# #             raw_response = resp.choices[0].message.content
# #             code = clean_llm_response(raw_response)
            
# #             # 2. Execute Code in Sandbox
# #             # Safety: Replace read_csv with direct copy to prevent file access issues
# #             code = re.sub(r"pd\.read_csv\(.*?\)", "df.copy()", code) 
# #             env = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns, "px": px}
# #             exec(code, env)
            
# #             result_df = env.get("result_df")
# #             fig = env.get("fig")
            
# #             # 3. Validation Checks
# #             if not isinstance(result_df, pd.DataFrame):
# #                 raise ValueError("The code ran, but did not create a 'result_df' DataFrame.")
            
# #             # CRITICAL: Treat empty results as a failure to trigger a retry with better logic
# #             if result_df.empty:
# #                 raise ValueError("Query returned 0 rows. The filter logic (e.g., City vs State, or spelling) might be incorrect.")
            
# #             # Success!
# #             if attempt > 0:
# #                 st.toast(f"✅ Auto-corrected logic in attempt {attempt+1}", icon="🧠")
                
# #             return result_df, code, None, fig # Return fig too
            
# #         except Exception as e:
# #             last_error = str(e)
# #             if attempt < max_retries:
# #                 st.toast(f"⚠️ Retrying... (Error: {last_error})", icon="🔧")
# #                 continue
# #             else:
# #                 return None, code, last_error

# # # ==========================================
# # # 🗣️ LAYER 5: ELITE NARRATIVE ENGINE
# # # Capabilities: "McKinsey-style" formatting, Anti-Scientific Notation
# # # ==========================================

# # def generate_narrative(query, result_df):
# #     if result_df.empty: return "No data found."
    
# #     # --- STEP 1: FORCE-FORMAT NUMBERS ---
# #     # We convert numbers to strings (e.g., "1,234.56") so the LLM *cannot* hallucinate values.
# #     display_df = result_df.copy()
# #     for col in display_df.select_dtypes(include=['number']).columns:
# #         # Check if column is likely monetary or count
# #         if display_df[col].mean() > 1000:
# #              display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}")
    
# #     data_str = display_df.head(10).to_markdown(index=False)
    
# #     # --- STEP 2: ELITE PERSONA PROMPT ---
# #     prompt = f"""
# #     You are an Elite Strategy Consultant (McKinsey/BCG style).
    
# #     ### USER QUERY: "{query}"
    
# #     ### DATA EVIDENCE (Already Formatted):
# #     {data_str}
    
# #     ### NARRATIVE GUIDELINES:
# #     1. **Structure:**
# #        - **Executive Summary:** One clear, direct sentence answering the question.
# #        - **Key Insights:** 2-3 bullet points highlighting drivers, outliers, or trends.
# #        - **Strategic Note:** One brief recommendation or observation.
    
# #     2. **Tone & Style:**
# #        - Professional, concise, and high-impact.
# #        - Use **Bold** for key figures and names.
# #        - **NEVER** use scientific notation (e.g., 2.3e7). Use the formatted numbers provided (e.g., 23,000,000).
       
# #     3. **Data Integrity:**
# #        - If the data contains 'nan' or 'Unclassified', explicitly mention this as a "Data Quality Note".
# #     """
    
# #     try:
# #         resp = client.chat.completions.create(
# #             model=MODEL_ID,
# #             messages=[{"role": "system", "content": prompt}],
# #             temperature=0.7
# #         )
# #         return resp.choices[0].message.content
# #     except RateLimitError:
# #         return f"**⚠️ High Traffic:** I analyzed the data successfully, but cannot generate the narrative summary right now.\n\n**Here is your data:**\n{data_str}"
# #     except Exception as e:
# #         return f"Error generating summary. Here is the raw data:\n\n{data_str}"

# # # ==========================================
# # # 🖥️ UI IMPLEMENTATION
# # # ==========================================

# # st.title("🧠 AI Analyst: Enterprise Edition")
# # st.markdown("### Intelligent Data Analysis & Strategy")

# # with st.sidebar:
# #     st.header("Configuration")
# #     file = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"])
# #     if st.button("🧹 Clear Conversation"):
# #         st.session_state.messages = []
# #         st.rerun()

# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # # Display Chat History
# # for msg in st.session_state.messages:
# #     with st.chat_message(msg["role"]):
# #         st.markdown(msg["content"])
# #         if "data" in msg:
# #             with st.expander("📊 View Supporting Data"):
# #                 st.dataframe(msg["data"])

# # if file:
# #     # Load Data (Layer 1)
# #     if "df" not in st.session_state:
# #         with st.spinner("🚀 Indexing and optimizing data..."):
# #             st.session_state.df = load_and_adapt_data(file)
# #         st.success("Data successfully loaded!")
    
# #     df = st.session_state.df
# #     query = st.chat_input("Ask a question about your data...")
    
# #     if query:
# #         st.session_state.messages.append({"role": "user", "content": query})
# #         with st.chat_message("user"):
# #             st.write(query)
            
# #         with st.chat_message("assistant"):
# #             with st.spinner("⚡ analyzing..."):
# #                 # Get Context
# #                 history = "\n".join([m["content"] for m in st.session_state.messages[-3:]])
                
# #                 # Execute Analysis (Layers 3 & 4)
# #                 result_df, code, error = execute_with_self_correction(query, df, history), fig = execute_with_self_correction(query, df, history)
                
# #                 if result_df is not None:
# #                     # Generate Narrative (Layer 5)
# #                     response = generate_narrative(query, result_df)
# #                     st.markdown(response)
                    
# #                     # Show Tech Details
# #                     with st.expander("🔍 Analyst Logic (Code)"):
# #                         st.code(code, language='python')
# #                         st.dataframe(result_df)
                    
# #                     st.session_state.messages.append({
# #                         "role": "assistant", 
# #                         "content": response, 
# #                         "data": result_df
# #                     })
# #                 if fig:
# #                     st.plotly_chart(fig, use_container_width=True) # <--- DISPLAY IT
                    
# #                 else:
# #                     st.error("Unable to complete analysis.")
# #                     with st.expander("See Error Logs"):
# #                         st.code(error)


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# from groq import Groq, RateLimitError
# import re
# import warnings
# import traceback

# # --- 1. CONFIGURATION & SETUP ---
# st.set_page_config(
#     page_title="AI Analyst: Enterprise Edition", 
#     layout="wide", 
#     page_icon="🧠"
# )
# warnings.filterwarnings("ignore")

# # API Setup
# api_key = st.secrets.get("GROQ_API_KEY")
# if not api_key:
#     st.error("🚨 Groq API Key missing! Check .streamlit/secrets.toml")
#     st.stop()

# client = Groq(api_key=api_key)
# MODEL_ID = "llama-3.3-70b-versatile" 

# # ==========================================
# # 🧠 LAYER 1: ADAPTIVE DATA ENGINE
# # ==========================================

# @st.cache_data
# def load_and_adapt_data(file):
#     try:
#         file.seek(0)
#         if file.name.endswith('.csv'):
#             df = pd.read_csv(file, low_memory=False)
#         else:
#             df = pd.read_excel(file)

#         # 1. Standardize Headers
#         df.columns = [
#             str(c).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_').replace('-', '_') 
#             for c in df.columns
#         ]

#         # 2. Adaptive Date Parsing
#         date_col = next((c for c in df.columns if 'date' in c or 'invoice' in c), None)
#         if date_col:
#             df[date_col] = pd.to_datetime(df[date_col], format='%d-%b-%y', errors='coerce')
#             if df[date_col].isnull().mean() > 0.5: 
#                 df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
#             df['cleaned_year'] = df[date_col].dt.year

#         # 3. Adaptive Currency Cleaning
#         sales_col = next((c for c in df.columns if c in ['net_amount_inr', 'net_amount', 'gross_amount', 'total_amount', 'sales']), None)
#         if sales_col:
#             if df[sales_col].dtype == 'object':
#                  df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce')
#             df['cleaned_sales'] = df[sales_col]

#         # 4. String Normalization
#         for col in df.select_dtypes(include=['object']).columns:
#             df[col] = df[col].astype(str).str.strip()

#         return df
#     except Exception as e:
#         return None

# # ==========================================
# # 👁️ LAYER 2: DATA HEALTH SCANNER
# # ==========================================

# def get_data_health_report(df):
#     report = []
#     for col in df.columns:
#         null_pct = df[col].isnull().mean()
#         if null_pct > 0.5:
#             report.append(f"⚠️ CRITICAL WARNING: Column '{col}' is {null_pct:.1%} empty. DO NOT USE IT. Find a valid alternative.")
#     return "\n".join(report)

# # ==========================================
# # 🛡️ LAYER 3: CODE SANITIZER
# # ==========================================

# def clean_llm_response(text):
#     code_block_pattern = r"```(?:python)?\s*(.*?)```"
#     match = re.search(code_block_pattern, text, re.DOTALL)
    
#     if match:
#         code = match.group(1).strip()
#     else:
#         code = text.strip()
    
#     lines = code.split('\n')
#     clean_lines = [line for line in lines if not line.lower().startswith(('here is', 'to address', 'note:', 'i have', 'python'))]
#     return "\n".join(clean_lines)

# # ==========================================
# # 🛠️ LAYER 4: SELF-HEALING LOGIC ENGINE
# # ==========================================

# def generate_code_prompt(query, df, history_context, error_context=None):
#     health_report = get_data_health_report(df)
    
#     hints = """
#     ### 💡 ANALYST HINTS:
#     - **Null Handling:** If ranking Top N, ALWAYS filter out NaNs first: `df.dropna(subset=['col'])`.
#     - **Customer Logic:** ALWAYS use `bill_to_party` (Name) for analysis. NEVER use `bill_to_party_code` (Code).
#     - **Location Logic:** 'Maharashtra' is a STATE. Cities are 'Mumbai', 'Pune'.
#     - **Fuzzy Match:** NEVER use `==` for strings. Use `df[col].str.contains('val', case=False, na=False)`.
#     - **Visuals:** If the user asks for a trend, comparison, or distribution, generate a Plotly chart.
#     - **Plotting Rule:** Create a figure object named `fig`. DO NOT use `fig.show()`. Streamlit will handle it.
#     - **Example:**
#       ```python
#       import plotly.express as px
#       fig = px.bar(df, x='city', y='sales', title='Sales by City')
#       ```
#     """

#     prompt = f"""
#     You are an Expert Python Data Analyst.
    
#     ### DATA PROFILE:
#     - Columns: {list(df.columns)}
    
#     ### 🛡️ HEALTH WARNINGS:
#     {health_report}
    
#     {hints}
    
#     ### 🧠 STRICT RULES:
#     1. **Valid Code Only:** Output pure Python code inside ```python tags.
#     2. **Result Variable:** Assign the final output dataframe to `result_df`.
#     3. **No Empty Results:** If your filter logic returns an empty dataframe, raise a ValueError.
#     4. **Smart Columns:** If a column has a "_code" and a normal version (e.g., party_code vs party), USE THE NORMAL VERSION.
#     """
    
#     if error_context:
#         prompt += f"\n\n### 🚨 PREVIOUS ATTEMPT FAILED:\nError: {error_context}\nFix: Rewrite the code to fix this."
    
#     prompt += f"\n\n### QUERY: {query}"
    
#     if history_context:
#         prompt += f"\n\n### CONTEXT: {history_context}"
        
#     return prompt

# def execute_with_self_correction(query, df, history_context, max_retries=2):
#     last_error = None
    
#     for attempt in range(max_retries + 1):
#         try:
#             # 1. Generate Code
#             prompt = generate_code_prompt(query, df, history_context, last_error)
            
#             resp = client.chat.completions.create(
#                 model=MODEL_ID,
#                 messages=[{"role": "system", "content": prompt}],
#                 temperature=0.1 
#             )
#             raw_response = resp.choices[0].message.content
#             code = clean_llm_response(raw_response)
            
#             # 2. Execute Code
#             code = re.sub(r"pd\.read_csv\(.*?\)", "df.copy()", code) 
#             env = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns, "px": px}
#             exec(code, env)
            
#             result_df = env.get("result_df")
#             fig = env.get("fig")
            
#             # 3. Validation
#             if not isinstance(result_df, pd.DataFrame):
#                 raise ValueError("The code ran, but did not create a 'result_df' DataFrame.")
            
#             if result_df.empty:
#                 raise ValueError("Query returned 0 rows. The filter logic might be incorrect.")
            
#             if attempt > 0:
#                 st.toast(f"✅ Auto-corrected logic in attempt {attempt+1}", icon="🧠")
                
#             return result_df, code, None, fig 
            
#         except Exception as e:
#             last_error = str(e)
#             if attempt < max_retries:
#                 st.toast(f"⚠️ Retrying... (Error: {last_error})", icon="🔧")
#                 continue
#             else:
#                 return None, code, last_error, None # Ensuring 4 values are returned

# # ==========================================
# # 🗣️ LAYER 5: ELITE NARRATIVE ENGINE
# # ==========================================

# def generate_narrative(query, result_df):
#     if result_df.empty: return "No data found."
    
#     display_df = result_df.copy()
#     for col in display_df.select_dtypes(include=['number']).columns:
#         if display_df[col].mean() > 1000:
#              display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}")
    
#     data_str = display_df.head(10).to_markdown(index=False)
    
#     prompt = f"""
#     You are an Elite Strategy Consultant.
#     User Query: "{query}"
#     Data (Formatted):
#     {data_str}
    
#     Guidelines:
#     1. **Structure:** Executive Summary, Key Insights, Strategic Note.
#     2. **Formatting:** - Use **Bold** for Customer Names.
#        - Do NOT bold the numbers themselves if they are inside a table or list, to avoid formatting errors.
#        - NO scientific notation.
#     """
    
#     try:
#         resp = client.chat.completions.create(
#             model=MODEL_ID,
#             messages=[{"role": "system", "content": prompt}],
#             temperature=0.7
#         )
#         return resp.choices[0].message.content
#     except Exception as e:
#         return f"Error generating summary. Raw Data:\n{data_str}"

# # ==========================================
# # 🖥️ UI IMPLEMENTATION
# # ==========================================

# st.title("🧠 AI Analyst: Enterprise Edition")
# st.markdown("### Intelligent Data Analysis & Strategy")

# # --- SIDEBAR WITH FEATURE BUTTONS ---
# with st.sidebar:
#     st.header("⚙️ Data & Controls")
#     file = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"])
    
#     if file:
#         st.divider()
#         st.subheader("🚀 Quick Actions")
#         # Feature Buttons to trigger analysis
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("📊 Overview"):
#                 st.session_state.quick_query = "Show me a high-level overview of sales and key trends."
#         with col2:
#             if st.button("🏆 Top Cust."):
#                 st.session_state.quick_query = "Who are the top 5 customers by total revenue?"
        
#         st.divider()
        
#         st.subheader("🧠 Memory Manager")
#         # 1. View Memory
#         with st.expander("📜 View Active Memory"):
#             if "messages" in st.session_state and st.session_state.messages:
#                 for i, msg in enumerate(st.session_state.messages):
#                     st.text(f"{i+1}. {msg['role'].title()}: {msg['content'][:60]}...")
#             else:
#                 st.info("Memory is empty.")

#         # 2. Download Memory
#         if "messages" in st.session_state and st.session_state.messages:
#             chat_log = "\n".join([f"[{m['role'].upper()}] {m['content']}" for m in st.session_state.messages])
#             st.download_button("💾 Save Chat", chat_log, "chat_log.txt")

#         # 3. Clear Memory
#         if st.button("🧹 Clear Memory", type="primary"):
#             st.session_state.messages = []
#             st.rerun()

# # --- MAIN CHAT LOGIC ---

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display History
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if "data" in msg:
#             with st.expander("📊 View Data"):
#                 st.dataframe(msg["data"].style.format(precision=2))
#         if "fig" in msg and msg["fig"]:
#              st.plotly_chart(msg["fig"], use_container_width=True)

# if file:
#     if "df" not in st.session_state:
#         with st.spinner("🚀 Indexing..."):
#             st.session_state.df = load_and_adapt_data(file)
#         st.success("Data Ready!")
    
#     df = st.session_state.df
    
#     # Check for Quick Query trigger or Input
#     query = st.chat_input("Ask a question...")
#     if "quick_query" in st.session_state and st.session_state.quick_query:
#         query = st.session_state.quick_query
#         del st.session_state.quick_query # Reset
    
#     if query:
#         st.session_state.messages.append({"role": "user", "content": query})
#         with st.chat_message("user"):
#             st.write(query)
            
#         with st.chat_message("assistant"):
#             with st.spinner("⚡ analyzing..."):
#                 history = "\n".join([m["content"] for m in st.session_state.messages[-3:]])
                
#                 # ✅ THIS IS THE CORRECT LINE
#                 result_df, code, error, fig = execute_with_self_correction(query, df, history)
                
#                 if result_df is not None:
#                     response = generate_narrative(query, result_df)
#                     st.markdown(response)
                    
#                     if fig:
#                         st.plotly_chart(fig, use_container_width=True)
                    
#                     with st.expander("🔍 Analyst Logic"):
#                         st.code(code, language='python')
#                         st.dataframe(result_df)
                    
#                     st.session_state.messages.append({
#                         "role": "assistant", 
#                         "content": response, 
#                         "data": result_df,
#                         "fig": fig
#                     })
#                 else:
#                     st.error("Analysis Failed.")
#                     st.code(error) # Uses the 'error' variable to show what went wrong


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from groq import Groq, RateLimitError
import re
import warnings
import traceback

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="AI Analyst: Enterprise Edition", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")


# API Setup
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("🚨 Groq API Key missing! Check .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=api_key)
MODEL_ID = "llama-3.3-70b-versatile" 

# Initialize Token Counter
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# ==========================================
# 🧠 LAYER 1: ADAPTIVE DATA ENGINE
# ==========================================

@st.cache_data
def load_and_adapt_data(file):
    """
    Ingests data, normalizes headers, and intelligently detects date/currency columns.
    """
    
    try:
        file.seek(0)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)

        # 1. Standardize Headers
        df.columns = [
            str(c).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_').replace('-', '_') 
            for c in df.columns
        ]

        # 2. Adaptive Date Parsing
        date_col = next((c for c in df.columns if 'date' in c or 'invoice' in c), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], format='%d-%b-%y', errors='coerce')
            if df[date_col].isnull().mean() > 0.5: 
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            df['cleaned_year'] = df[date_col].dt.year

        # 3. Adaptive Currency Cleaning
        sales_col = next((c for c in df.columns if c in ['net_amount_inr', 'net_amount', 'gross_amount', 'total_amount', 'sales']), None)
        if sales_col:
            if df[sales_col].dtype == 'object':
                 df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(r'[$,₹]', '', regex=True), errors='coerce')
            df['cleaned_sales'] = df[sales_col]

        # 4. String Normalization
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        return df
    except Exception as e:
        return None

# ==========================================
# 👁️ LAYER 2: DATA HEALTH SCANNER
# ==========================================

def get_data_health_report(df):
    report = []
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0.5:
            report.append(f"⚠️ CRITICAL WARNING: Column '{col}' is {null_pct:.1%} empty. DO NOT USE IT. Find a valid alternative (e.g., use 'shade_no' if 'shade_name' is empty).")
    return "\n".join(report)

# ==========================================
# 🛡️ LAYER 3: CODE SANITIZER
# ==========================================

def clean_llm_response(text):
    """
    Strips conversational text (e.g., "Here is the code") to prevent SyntaxErrors.
    """
    code_block_pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
    else:
        code = text.strip()
    
    lines = code.split('\n')
    clean_lines = [line for line in lines if not line.lower().startswith(('here is', 'to address', 'note:', 'i have', 'python'))]
    return "\n".join(clean_lines)

# ==========================================
# 🛠️ LAYER 4: SELF-HEALING LOGIC ENGINE
# ==========================================

def generate_code_prompt(query, df, history_context, error_context=None):
    health_report = get_data_health_report(df)
    
    hints = """
    ### 💡 ANALYST HINTS & 💡 ANALYST LOGIC & RULES (STRICT):
    - **Growth Calculation Rule:** When calculating % growth, FIRST filter out any customers where the baseline (previous year) is <= 0. Growth from a negative number is invalid.
    - **Customer Logic:** ALWAYS use `bill_to_party` (Name).
    - **Null Handling:** If ranking Top N, ALWAYS filter out NaNs first: `df.dropna(subset=['col'])`.
    - **Customer Logic:** ALWAYS use `bill_to_party` (Name) for analysis. NEVER use `bill_to_party_code` (Code).
    - **Location Logic:** 'Maharashtra' is a STATE. Cities are 'Mumbai', 'Pune'. Use `agent_state` or `bill_to_party_city`.
    - **Fuzzy Match:** NEVER use `==` for strings. Use `df[col].str.contains('val', case=False, na=False)`.
    - **Date Logic:** Use `cleaned_year` for fiscal/calendar year queries.
    # - **Visuals:** If the user asks for a trend, comparison, or distribution, generate a Plotly chart.
    # - **Plotting Rule:** Create a figure object named `fig`. DO NOT use `fig.show()`. Streamlit will handle it.
    # - **Example:**
    #    ```python
    #    import plotly.express as px
    #    fig = px.bar(df, x='city', y='sales', title='Sales by City')

    1.  **📍 GEOGRAPHY RULES:**
        - **State:** If query asks for a STATE (e.g., 'Rajasthan', 'Gujarat'), filter by `agent_state`.
        - **City:** If query asks for a CITY (e.g., 'Mumbai', 'Surat'), filter by `bill_to_party_city`.
        - **Region:** Use `region_zone`.

    2.  **📈 FINANCIAL MATH RULES (CRITICAL):**
        - **Growth %:** `((Current - Previous) / Previous) * 100`
        - **Negative Baseline Filter:** BEFORE calculating growth, YOU MUST remove rows where the Previous Year Sales are <= 0. 
          - *Reason:* Calculating growth from a negative number is mathematically invalid and yields wrong results.
        - **Sales Column:** ALWAYS prefer `net_amount_inr` for revenue calculations.

    3.  **🔎 SEARCH & MATCHING:**
        - **Fuzzy Match:** NEVER use exact `==` for text filters. 
          - *Correct:* `df[df['col'].str.contains('Pattern', case=False, na=False)]`
        - **Entity Names:**
          - Customer -> `bill_to_party` (NOT `bill_to_party_code`)
          - Design -> `design` (NOT `ainocular_design`)
          - Item/Product -> `material`

    4.  **🗓️ DATE LOGIC:**
        - Use `fiscal_year` for yearly comparisons (2023 vs 2024).
        - Use `invoice_date` for daily/monthly trends.

    5.  **💰 PRICING vs. REVENUE (CRITICAL):**
        - **"Most Expensive" / "Highest Price":** Sort by `basic_price`, `unit_price`, or `rate`. 
          - *Do NOT* use `net_amount` (which is Price * Qty).
        - **"Top Selling" / "Highest Revenue":** Sort by `net_amount_inr`.
    
    6.  **🏷️ NAMING CONVENTION:**
        - **Products:** When listing products, ALWAYS show the `design`, `quality`, or `material_description` column alongside the ID.
        - **Customers:** Use `bill_to_party` (Name), not `bill_to_party_code`.

    6.  **📊 VISUALIZATION:**
        - Create a Plotly figure object named `fig`.
        - Example: `fig = px.bar(df, x='agent_state', y='net_amount_inr', title='...')`
        - Do NOT use `fig.show()`.
      ```
    """

    prompt = f"""
    You are a Lead Data Scientist. Write Python code to answer the user's query.
    
    ### DATA PROFILE:
    - Columns: {list(df.columns)}
    
    ### 🛡️ HEALTH WARNINGS:
    {health_report}
    
    {hints}
    
    ### 🧠 STRICT RULES:
    1. **Valid Code Only:** Output pure Python code inside ```python tags.
    2. **Result Variable:** Assign the final output dataframe to `result_df`.
    3. **Variable:** If a chart is created, assign it to `fig`.
    4. **No Empty Results:** If your filter logic returns an empty dataframe, the logic is wrong. Try a different column.
    5. **Smart Columns:** If a column has a "_code" and a normal version (e.g., party_code vs party), USE THE NORMAL VERSION.
    """
    
    if error_context:
        prompt += f"\n\n### 🚨 PREVIOUS ATTEMPT FAILED:\nError: {error_context}\nFix: Rewrite the code to fix this specific error."
    
    prompt += f"\n\n### QUERY: {query}"
    
    if history_context:
        prompt += f"\n\n### CONTEXT: {history_context}"
        
    return prompt

def execute_with_self_correction(query, df, history_context, max_retries=8):
    last_error = None
    tokens_used = 0
    code = ""  # <--- 🛠️ FIX: Initialize 'code' here so it never crashes
    
    for attempt in range(max_retries + 1):
        try:
            # 1. Generate Code
            prompt = generate_code_prompt(query, df, history_context, last_error)
            
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1 
            )
            # Track Tokens
            if resp.usage:
                tokens_used += resp.usage.total_tokens

            raw_response = resp.choices[0].message.content
            code = clean_llm_response(raw_response)
            
            # 2. Execute Code
            code = re.sub(r"pd\.read_csv\(.*?\)", "df.copy()", code) 
            env = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns, "px": px}
            exec(code, env)
            
            result_df = env.get("result_df")
            fig = env.get("fig")
            
            # 3. Validation
            if not isinstance(result_df, pd.DataFrame):
                raise ValueError("The code ran, but did not create a 'result_df' DataFrame.")
            
            if result_df.empty:
                raise ValueError("Query returned 0 rows. The filter logic (e.g., City vs State, or spelling) might be incorrect.")
            
            if attempt > 0:
                st.toast(f"✅ Auto-corrected logic in attempt {attempt+1}", icon="🧠")
                
            return result_df, code, None, fig, tokens_used
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                st.toast(f"⚠️ Retrying... (Error: {last_error})", icon="🔧")
                continue
            else:
                return None, code, last_error, None, tokens_used

# ==========================================
# 🗣️ LAYER 5: ELITE NARRATIVE ENGINE
# ==========================================

def generate_narrative(query, result_df):
    if result_df.empty: return "No data found.", 0
    
    display_df = result_df.copy()
    for col in display_df.select_dtypes(include=['number']).columns:
        if display_df[col].mean() > 1000:
             display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}")
    
    data_str = display_df.head(10).to_markdown(index=False)
    
    prompt = f"""
    # You are an Elite Strategy Consultant (McKinsey/BCG style).
    You are the Chief Strategy Officer (CSO) of a Fortune 500 company.
    User Query: "{query}"
    Data (Formatted):
    {data_str}
    
    Guidelines:
    1. **Structure:** Executive Summary, Table(when ever needed), Key Insights, Strategic Note.
    2. **Formatting:** - Use **Bold** for Customer Names.
       - Do NOT bold the numbers themselves if they are inside a table or list, to avoid formatting errors.
       - NO scientific notation.
    3. **Tone & Style:**
       - Professional, concise, and high-impact.
       - Use **Bold** for key figures and names.
       - **NEVER** use scientific notation (e.g., 2.3e7). Use the formatted numbers provided (e.g., 23,000,000).
    4. **Data Integrity:**
       - If the data contains 'nan' or 'Unclassified', explicitly mention this as a "Data Quality Note".

    ### 🧠 NARRATIVE FRAMEWORK:
    1.  **The "So What?":** Start with a single, high-impact sentence summarizing the answer (e.g., "Revenue grew by 15%, driven primarily by X."), table when ever needed. 
    2.  **Evidence-Based Insights:** - Provide 2-3 specific data points to back up your claim.
        - **Contextualize:** Don't just list numbers; explain *why* they matter (e.g., "X is 2x larger than Y").
    3.  **Strategic Implication:** Offer one forward-looking recommendation or observation.
    
    ### ⛔ CRITICAL RULES:
    - **TRUST THE DATA:** The numbers in the table above are 100% accurate. Do not round them differently or "hallucinate".
    - **FORMATTING:** - Use **Bold** for Key Entities (Customer Names, Regions).
        - **ABSOLUTELY NO SCIENTIFIC NOTATION** (e.g., 2.34e8). Use standard formatting (e.g., 234 Million).
    """
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7
        )
        tokens = resp.usage.total_tokens if resp.usage else 0
        return resp.choices[0].message.content, tokens
    except RateLimitError:
        return f"**⚠️ High Traffic:** I analyzed the data successfully, but cannot generate the narrative summary right now.\n\n**Here is your data:**\n{data_str}", 0
    except Exception as e:
        return f"Error generating summary. Raw Data:\n{data_str}", 0

# ==========================================
# 🖥️ UI IMPLEMENTATION
# ==========================================

st.title("🧠 AI Analyst: Enterprise Edition")
st.markdown("### ⚡ Intelligent Data Strategy & Insights")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data & Controls")
    
    # TOKEN TRACKER WIDGET
    st.markdown("### 🔋 System Status")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("🪙")
    with col2:
        st.metric("Tokens Used", f"{st.session_state.total_tokens:,}", help="Total AI tokens consumed in this session.")
    
    st.divider()
    
    file = st.file_uploader("Upload Data (CSV/Excel)", type=["csv", "xlsx"])
    
    if file:
        st.divider()
        st.subheader("🚀 Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Overview"):
                st.session_state.quick_query = "Show me a high-level overview of sales and key trends."
        with col2:
            if st.button("🏆 Top Cust."):
                st.session_state.quick_query = "Who are the top 5 customers by total revenue?"
        
        st.divider()
        st.subheader("🧠 Memory Core")
        
        use_memory = st.toggle("Active Memory", value=True, help="Enable for conversational context.")
        
        if st.button("🧹 Clear RAM", type="primary"):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.rerun()

# --- MAIN CHAT ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "data" in msg:
            with st.expander("📊 View Data Table"):
                st.dataframe(msg["data"].style.format(precision=2))
        if "fig" in msg and msg["fig"]:
             st.plotly_chart(msg["fig"], use_container_width=True)

if file:
    if "df" not in st.session_state:
        with st.spinner("🚀 Indexing..."):
            st.session_state.df = load_and_adapt_data(file)
        st.success("System Online. Data Indexed.")
    
    df = st.session_state.df
    
    # Input Handling
    query = st.chat_input("Ask a strategic question...")
    if "quick_query" in st.session_state and st.session_state.quick_query:
        query = st.session_state.quick_query
        del st.session_state.quick_query
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            with st.spinner("⚡ Processing Logic..."):
                
                # Context Logic
                history = ""
                if use_memory:
                    history = "\n".join([m["content"] for m in st.session_state.messages[-3:]])
                
                # Run Analysis
                result_df, code, error, fig, tokens_logic = execute_with_self_correction(query, df, history)
                st.session_state.total_tokens += tokens_logic # Update Counter
                
                if result_df is not None:
                    # Generate Narrative
                    response, tokens_narrative = generate_narrative(query, result_df)
                    st.session_state.total_tokens += tokens_narrative # Update Counter
                    
                    st.markdown(response)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("🔍 View Analyst Logic (Code)"):
                        st.code(code, language='python')
                        st.dataframe(result_df)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "data": result_df,
                        "fig": fig
                    })
                    
                    # Force a rerun to update the sidebar token counter immediately
                    st.rerun()
                    
                else:
                    st.error("Analysis Failed")
                    st.warning("⚠️ Logic Error: The AI could not process the request.")
                    with st.expander("See Error Details"):
                        st.code(error)
