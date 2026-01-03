"""
Query Resolution Agent - Converts natural language to SQL
"""
from typing import Dict, Any, TypedDict
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate
try:
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:
    from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from utils.llm_utils import get_llm, create_prompt_template


class QueryIntent(BaseModel):
    """Structured output for query intent"""
    intent_type: str = Field(description="Type of query: 'summary', 'comparison', 'trend', 'filter', or 'aggregation'")
    entities: Dict[str, Any] = Field(description="Extracted entities (regions, categories, time periods, etc.)")
    sql_query: str = Field(description="Generated SQL query for DuckDB")
    explanation: str = Field(description="Brief explanation of what the query does")


class QueryResolutionAgent:
    """Agent that resolves natural language queries to SQL"""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.parser = PydanticOutputParser(pydantic_object=QueryIntent)
        
    def get_schema_context(self) -> str:
        """Return Amazon sales database schema information"""
        return """
Database Schema:
Table: sales (Amazon India e-commerce orders data)

Columns:
- order_id (VARCHAR): Unique order identifier
- date (DATE): Order date
- year (INTEGER): Year of order
- month (INTEGER): Month number (1-12)
- month_name (VARCHAR): Month name
- quarter (INTEGER): Quarter (1, 2, 3, 4)
- quarter_name (VARCHAR): Quarter name
- status (VARCHAR): Order status (Shipped, Cancelled, Pending, etc.)
- fulfilment (VARCHAR): Fulfillment type (Amazon, Merchant)
- sales_channel (VARCHAR): Sales channel (Amazon.in)
- service_level (VARCHAR): Shipping level (Standard, Expedited)
- style (VARCHAR): Product style
- sku (VARCHAR): Stock Keeping Unit
- category (VARCHAR): Product category (Set, Kurta, Western Dress, Top, etc.)
- size (VARCHAR): Product size
- asin (VARCHAR): Amazon Standard Identification Number
- courier_status (VARCHAR): Courier delivery status
- quantity (INTEGER): Quantity ordered
- currency (VARCHAR): Currency (INR)
- amount (DOUBLE): Order amount in INR
- city (VARCHAR): Shipping city
- state (VARCHAR): Shipping state/region
- postal_code (VARCHAR): Postal code
- country (VARCHAR): Shipping country
- is_b2b (BOOLEAN): Business-to-business flag
- order_value_category (VARCHAR): Order size (small/medium/large based on amount)
- is_cancelled (BOOLEAN): True if order was cancelled
- is_shipped (BOOLEAN): True if order was shipped
- revenue (DOUBLE): Amount for non-cancelled orders (0 if cancelled)
- estimated_profit (DOUBLE): Estimated 20% profit on revenue
- unit_price (DOUBLE): Price per unit
- customer (VARCHAR): Customer identifier
- data_source (VARCHAR): Data source identifier

Available aggregations: SUM, AVG, COUNT, MIN, MAX
Available filters: WHERE, GROUP BY, HAVING, ORDER BY
Date range: 2021-2022 orders (various months)

Important Notes:
- Use 'revenue' column for financial analysis (excludes cancelled orders)
- Use 'amount' for gross order values (includes cancelled)
- Check 'status' or 'is_cancelled' to filter out cancelled orders
- 'state' contains regional data for geographical analysis
"""
    
    def create_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for query resolution"""
        
        instructions = """
Your task is to convert natural language questions about retail sales data into SQL queries.

{schema}

Guidelines:
1. Analyze the user's question to understand their intent
2. Extract relevant entities (states, categories, time periods, metrics)
3. Generate a valid DuckDB SQL query for Amazon sales data
4. Use proper aggregations (SUM for revenue/profit, COUNT for orders, AVG for averages)
5. Apply appropriate filters and GROUP BY clauses
6. For growth comparisons, calculate percentages
7. For trends, order by time periods (year, month)
7. If the user mentions "AQL", they likely mean "SQL" for DuckDB.
8. Always include quantitative columns (revenue, amount, quantity, etc.) and their aggregations (SUM, AVG) in the results.
9. Always filter out cancelled orders when analyzing revenue (use 'revenue' column or WHERE status != 'Cancelled').

Examples:

Question: "What were total sales in March 2022?"
SQL: SELECT SUM(revenue) as total_revenue, COUNT(*) as orders FROM sales WHERE year = 2022 AND month = 3

Question: "Which state had the highest revenue?"
SQL: SELECT state, SUM(revenue) as total_revenue, COUNT(*) as orders FROM sales WHERE state IS NOT NULL GROUP BY state ORDER BY total_revenue DESC LIMIT 1

Question: "Top 5 product categories by profit?"
SQL: SELECT category, SUM(estimated_profit) as total_profit, SUM(revenue) as total_revenue FROM sales WHERE category IS NOT NULL GROUP BY category ORDER BY total_profit DESC LIMIT 5

Question: "What is the cancellation rate?"
SQL: SELECT 
    SUM(CASE WHEN is_cancelled THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as cancellation_rate,
    SUM(CASE WHEN is_cancelled THEN 1 ELSE 0 END) as cancelled_orders,
    COUNT(*) as total_orders
FROM sales

Question: "Monthly revenue trend in 2022"
SQL: SELECT year, month, SUM(revenue) as monthly_revenue, COUNT(*) as orders FROM sales WHERE year = 2022 GROUP BY year, month ORDER BY month

Question: "B2B vs B2C revenue comparison"
SQL: SELECT 
    CASE WHEN is_b2b THEN 'B2B' ELSE 'B2C' END as customer_type,
    SUM(revenue) as revenue,
    COUNT(*) as orders
FROM sales 
GROUP BY is_b2b

User Question: {question}

{format_instructions}
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", create_prompt_template(
                "Query Resolution Specialist",
                "You convert natural language questions into precise SQL queries for retail analytics."
            )),
            ("user", instructions)
        ])
        
        return prompt
    
    def resolve_query(self, question: str, context: str = None) -> QueryIntent:
        """
        Resolve natural language query to SQL
        
        Args:
            question: User's natural language question
            context: Optional conversation context
            
        Returns:
            QueryIntent with SQL query and metadata
        """
        try:
            prompt = self.create_prompt()
            
            # Include context if available
            full_question = question
            if context:
                full_question = f"{context}\n\nCurrent question: {question}"
            
            chain = prompt | self.llm | self.parser
            
            result = chain.invoke({
                "question": full_question,
                "schema": self.get_schema_context(),
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return result
            
        except Exception as e:
            print(f"❌ Query resolution error: {e}")
            # Return a safe default
            return QueryIntent(
                intent_type="error",
                entities={},
                sql_query="SELECT * FROM sales LIMIT 10",
                explanation=f"Error parsing query: {str(e)}. Showing sample data."
            )
    
    def resolve_with_retry(self, question: str, max_retries: int = 3, 
                           context: str = None, error_history: list = None) -> QueryIntent:
        """
        Resolve query with retry logic and error learning
        
        Args:
            question: User's natural language question
            max_retries: Maximum number of retry attempts
            context: Optional conversation context
            error_history: List of previous errors for learning
            
        Returns:
            QueryIntent with SQL query and metadata
        """
        errors = error_history or []
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Build error context from previous attempts
                error_context = ""
                if errors:
                    error_context = "\n\nPrevious attempts failed with these errors:\n"
                    for i, err in enumerate(errors[-3:], 1):  # Last 3 errors
                        error_context += f"{i}. {err}\n"
                    error_context += "\nPlease avoid these mistakes in your SQL query."
                
                # Get the prompt with error context
                prompt = self.create_prompt()
                
                full_question = question
                if context:
                    full_question = f"{context}\n\nCurrent question: {question}"
                if error_context:
                    full_question += error_context
                
                chain = prompt | self.llm | self.parser
                
                result = chain.invoke({
                    "question": full_question,
                    "schema": self.get_schema_context(),
                    "format_instructions": self.parser.get_format_instructions()
                })
                
                # Validate the generated SQL syntax
                validation_result = self._validate_sql_syntax(result.sql_query)
                if not validation_result["valid"]:
                    errors.append(validation_result["error"])
                    last_error = validation_result["error"]
                    print(f"⚠️  Attempt {attempt + 1}: SQL validation failed - {last_error}")
                    continue
                
                print(f"✅ Query resolved successfully on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                error_msg = str(e)
                errors.append(error_msg)
                last_error = error_msg
                print(f"⚠️  Attempt {attempt + 1} failed: {error_msg}")
        
        # All retries exhausted
        print(f"❌ All {max_retries} attempts failed. Returning safe fallback.")
        return QueryIntent(
            intent_type="error",
            entities={},
            sql_query="SELECT category, COUNT(*) as count, SUM(revenue) as revenue FROM sales GROUP BY category ORDER BY revenue DESC LIMIT 10",
            explanation=f"Query generation failed after {max_retries} attempts. Showing top categories by revenue as fallback. Last error: {last_error}"
        )
    
    def _validate_sql_syntax(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL syntax before execution
        
        Args:
            sql: SQL query string
            
        Returns:
            Dict with valid flag and optional error message
        """
        if not sql or not sql.strip():
            return {"valid": False, "error": "Empty SQL query"}
        
        sql_upper = sql.upper().strip()
        
        # Must be a SELECT query
        if not sql_upper.startswith("SELECT"):
            return {"valid": False, "error": "Query must start with SELECT"}
        
        # Must reference the sales table
        if "FROM SALES" not in sql_upper and "FROM `SALES`" not in sql_upper:
            return {"valid": False, "error": "Query must reference the 'sales' table"}
        
        # Check for dangerous operations
        dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return {"valid": False, "error": f"Dangerous keyword '{keyword}' not allowed"}
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            return {"valid": False, "error": "Unbalanced parentheses in query"}
        
        # Check for common SQL syntax issues
        if sql_upper.count("SELECT") > sql_upper.count("FROM"):
            return {"valid": False, "error": "SELECT count doesn't match FROM count (subquery issue)"}
        
        # Basic keyword presence check
        if "FROM" not in sql_upper:
            return {"valid": False, "error": "Query missing FROM clause"}
        
        return {"valid": True, "error": None}


class AgentState(TypedDict):
    """State for agent graph"""
    question: str
    query_intent: QueryIntent | None
    query_result: Any
    validation_passed: bool
    final_answer: str
    error: str | None
    # New fields for enhanced state
    confidence_scores: Dict[str, float] | None
    conversation_context: str | None
    edge_case_handled: bool | None
    facts: list | None
    report_content: str | None
