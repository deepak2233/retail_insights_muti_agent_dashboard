"""
Response Generation Agent - Creates human-readable responses
"""
from typing import Dict, Any
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate
from agents.query_agent import AgentState
from utils.llm_utils import get_llm, create_prompt_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


class ResponseAgent:
    """Agent that generates natural language responses from query results"""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.3)  # Slightly higher for more natural responses
    
    def generate_response(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate natural language response from query results
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with final answer
        """
        try:
            # Check if validation passed
            if not state.get("validation_passed"):
                error = state.get("error", "Unknown error")
                return {
                    **state,
                    "final_answer": f"I encountered an issue processing your query: {error}"
                }
            
            query_result = state.get("query_result")
            query_intent = state.get("query_intent")
            question = state.get("question")
            
            if not query_result or not query_intent:
                # If we have report content, we can still try to answer
                if not state.get("report_content"):
                    return {
                        **state,
                        "final_answer": "I couldn't process your query. Please try rephrasing your question."
                    }
            
            # Format the data for the LLM
            data_summary = self._format_results(query_result)
            sql_query = query_intent.sql_query if query_intent else "No SQL query generated."
            
            # Debug: Print what we are sending to LLM
            print(f"\n--- DEBUG: DATA SENT TO LLM ---\n{data_summary}\n-------------------------------\n")
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", create_prompt_template(
                    "Retail Analytics Response Specialist",
                    """You provide clear, insightful answers to business questions about retail sales data.
                    
Your responses should:
1. Directly answer the user's question.
2. USE THE DATA PROVIDED. If the data shows numbers (revenue, quantities, counts), you MUST include them in your answer. Never say "not specified" if the data is present in the "Data Retrieved" section.
3. Highlight key insights and trends.
4. Show the SQL Query if the user specifically asked for it (e.g., if they said "show query" or "sql query").
5. Be concise but comprehensive.
6. Provide context and business implications when relevant.
7. Format data clearly (use bullet points, tables, or structured text).

If the data shows trends, explain what they mean for the business.
If comparing values, clearly state the differences and their significance.
"""
                )),
                ("user", """
Original Question: {question}

Query Explanation: {explanation}

SQL Query Used: 
{sql_query}

Data Retrieved from Database:
{data_summary}

Additional Report Context:
{report_content}

Please provide a clear, business-focused answer to the original question. 
- If the question is about the data retrieved, focus on that and include specific numbers.
- If the user asked for the SQL query, include it in your response in a code block.
- Synthesize information from both the Database and the Additional Report Context if both are relevant.
""")
            ])
            
            chain = prompt | self.llm
            
            response = chain.invoke({
                "question": question,
                "explanation": query_intent.explanation if query_intent else "N/A",
                "sql_query": sql_query,
                "data_summary": data_summary,
                "report_content": state.get("report_content", "No additional report context provided.")
            })
            
            final_answer = response.content
            
            print(f"✅ Response generated successfully")
            
            return {
                **state,
                "final_answer": final_answer,
                "chart_data": self._generate_chart(query_result, query_intent)
            }
            
        except Exception as e:
            error_msg = f"Response generation error: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                **state,
                "final_answer": f"I encountered an error generating the response: {error_msg}"
            }
    
    def _format_results(self, query_result: Dict[str, Any]) -> str:
        """Format query results for LLM consumption - optimized for data visibility"""
        df = query_result.get("dataframe")
        
        if df is None or df.empty:
            return "No data found matching the query criteria."
        
        # Determine which columns to show. 
        # For analytical queries, we usually have few columns, so show them all.
        # If there are too many columns (>15), prioritize important ones.
        if len(df.columns) <= 15:
            display_cols = df.columns.tolist()
        else:
            important_patterns = ['id', 'date', 'category', 'status', 'revenue', 'profit', 'amount', 'total', 'count', 'sum', 'avg']
            display_cols = [c for c in df.columns if any(p in c.lower() for p in important_patterns)]
            # Ensure we at least have some columns
            if not display_cols:
                display_cols = df.columns[:10].tolist()
        
        # Use subset of columns
        df_subset = df[display_cols]
        
        formatted = f"Results: {len(df)} records, {len(df.columns)} columns\n"
        formatted += f"Columns shown: {', '.join(display_cols)}\n\n"
        
        # If small dataset, show all rows
        if len(df) <= 20:
            formatted += df_subset.to_string(index=False)
        else:
            # Show summary statistics for numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                formatted += "Summary Statistics (All Records):\n"
                formatted += numeric_df.describe().loc[['mean', 'min', 'max', 'sum']].to_string()
                formatted += "\n\n"
            
            formatted += f"Sample Data (first 10 rows):\n"
            formatted += df_subset.head(10).to_string(index=False)
            formatted += f"\n\n(Showing 10 of {len(df)} total records)"
        
        return formatted
    
    def _detect_chart_type(self, df: pd.DataFrame, sql_query: str) -> Optional[str]:
        """Detect the most appropriate chart type based on query structure and results"""
        if df is None or df.empty or len(df) > 100:
            return None
        
        sql_lower = sql_query.lower()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # No numeric data to visualize
        if not num_cols:
            return None
        
        # Time series detection
        if any(col in df.columns for col in ['year', 'month', 'date', 'time_period']):
            return 'line'
        
        # Distribution/comparison detection
        if 'group by' in sql_lower:
            # Pie chart for small categorical distributions
            if len(df) <= 8 and len(num_cols) == 1:
                return 'pie'
            # Bar chart for comparisons
            elif len(df) <= 20:
                return 'bar'
        
        # Top N detection
        if 'limit' in sql_lower and 'order by' in sql_lower:
            return 'bar'
        
        return None
    
    def _generate_chart(self, query_result: Dict[str, Any], query_intent) -> Optional[Dict[str, Any]]:
        """Generate a Plotly chart based on query results"""
        try:
            df = query_result.get("dataframe")
            if df is None or df.empty:
                return None
            
            sql_query = query_intent.sql_query if query_intent else ""
            chart_type = self._detect_chart_type(df, sql_query)
            
            if not chart_type:
                return None
            
            # Get numeric and categorical columns
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            if not num_cols or not cat_cols:
                return None
            
            # Primary columns for visualization
            x_col = cat_cols[0]
            y_col = num_cols[0]
            
            # Color scheme
            color_scale = 'Viridis'
            
            # Generate chart based on type
            if chart_type == 'bar':
                fig = px.bar(
                    df, 
                    x=x_col, 
                    y=y_col,
                    color=y_col,
                    color_continuous_scale=color_scale,
                    title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    height=400,
                    xaxis_tickangle=-45
                )
            
            elif chart_type == 'pie':
                fig = px.pie(
                    df,
                    values=y_col,
                    names=x_col,
                    title=f"{y_col.replace('_', ' ').title()} Distribution",
                    hole=0.4
                )
                fig.update_layout(height=400)
            
            elif chart_type == 'line':
                # For line charts, try to sort by time-related columns
                if 'month' in df.columns:
                    df = df.sort_values('month')
                
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col.replace('_', ' ').title()} Trend",
                    markers=True
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
            
            else:
                return None
            
            return {
                "figure": fig,
                "type": chart_type
            }
        
        except Exception as e:
            print(f"Chart generation error: {e}")
            return None
