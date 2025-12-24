"""
Data layer for efficient querying using DuckDB
Handles real e-commerce sales data with robust error handling
"""
import duckdb
import pandas as pd
from typing import Optional, List, Dict, Any
import os
import logging
from config import settings

logger = logging.getLogger(__name__)


class DataLayer:
    """Efficient data querying using DuckDB with production-grade error handling"""
    
    def __init__(self, csv_path: Optional[str] = None, db_path: Optional[str] = None):
        """
        Initialize data layer
        
        Args:
            csv_path: Path to CSV file (processed data)
            db_path: Path to DuckDB database
        """
        # Use processed data if available, fallback to original
        self.csv_path = csv_path or "data/processed_sales_data.csv"
        if not os.path.exists(self.csv_path):
            self.csv_path = settings.data_path
        
        self.db_path = db_path or settings.duckdb_path
        self.conn = None
        self.schema_info = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize DuckDB connection and load data with error handling"""
        try:
            # Use in-memory database to avoid file conflicts on Streamlit Cloud
            if self.db_path == ":memory:" or not self.db_path:
                self.conn = duckdb.connect(":memory:")
                logger.info("📂 Connected to in-memory DuckDB")
            else:
                self.conn = duckdb.connect(self.db_path)
                logger.info(f"📂 Connected to DuckDB: {self.db_path}")
            
            # Check if table already exists (for cached connections)
            try:
                existing = self.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
                if existing > 0:
                    logger.info(f"✅ Using existing table with {existing:,} records")
                    self.schema_info = self._get_schema()
                    return
            except:
                pass  # Table doesn't exist, continue loading
            
            # Load CSV into DuckDB if exists
            if os.path.exists(self.csv_path):
                logger.info(f"📊 Loading data from {self.csv_path}...")
                
                try:
                    # Create table from CSV with error handling
                    self.conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS sales AS 
                        SELECT * FROM read_csv_auto('{self.csv_path}', 
                            ignore_errors=true,
                            null_padding=true
                        )
                    """)
                    
                    # Create indexes for better performance
                    self._create_indexes()
                    
                    # Store schema info
                    self.schema_info = self._get_schema()
                    
                    # Get row count
                    row_count = self.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
                    logger.info(f"✅ Loaded {row_count:,} records into DuckDB")
                    print(f"✅ Loaded {row_count:,} records into DuckDB")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading CSV: {e}")
                    print(f"❌ Error loading CSV: {e}")
                    # Try fallback: load with pandas then insert
                    self._fallback_load()
                    
            else:
                logger.warning(f"⚠️  CSV file not found: {self.csv_path}")
                print(f"⚠️  CSV file not found: {self.csv_path}")
                print(f"💡 Run data ingestion first: python utils/data_ingestion.py")
                
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
            print(f"❌ Initialization failed: {e}")
            # Don't raise - allow app to continue with limited functionality
    
    def _create_indexes(self):
        """Create indexes on key columns"""
        try:
            # Get available columns
            cols = self.conn.execute("DESCRIBE sales").fetchdf()['column_name'].tolist()
            
            # Create indexes on common query columns
            index_columns = [
                'date', 'year', 'month', 'quarter', 
                'category', 'state', 'status', 'sku'
            ]
            
            for col in index_columns:
                if col in cols:
                    try:
                        self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON sales({col})")
                    except:
                        pass  # Index might already exist or column might not support indexing
                        
            logger.info("✅ Created indexes on key columns")
        except Exception as e:
            logger.warning(f"⚠️  Could not create all indexes: {e}")
    
    def _fallback_load(self):
        """Fallback method to load data using pandas"""
        try:
            logger.info("🔄 Attempting fallback load with pandas...")
            df = pd.read_csv(self.csv_path, low_memory=False)
            
            # Check if table exists first
            try:
                existing = self.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
                if existing > 0:
                    logger.info(f"✅ Table already exists with {existing:,} records")
                    return
            except:
                pass
            
            # Create table from dataframe
            self.conn.execute("CREATE TABLE IF NOT EXISTS sales AS SELECT * FROM df")
            
            row_count = self.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
            logger.info(f"✅ Fallback load successful: {row_count:,} records")
            print(f"✅ Fallback load successful: {row_count:,} records")
            
        except Exception as e:
            logger.error(f"❌ Fallback load also failed: {e}")
            print(f"❌ Fallback load also failed: {e}")
    
    def _get_schema(self) -> pd.DataFrame:
        """Get current table schema"""
        try:
            return self.conn.execute("DESCRIBE sales").fetchdf()
        except:
            return pd.DataFrame()
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        try:
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            print(f"❌ Query execution error: {e}")
            raise
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the real Amazon sales dataset"""
        try:
            stats = {}
            
            # Overall stats
            overall = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    SUM(revenue) as total_revenue,
                    SUM(estimated_profit) as total_profit,
                    AVG(amount) as avg_order_value,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    SUM(CASE WHEN is_cancelled THEN 1 ELSE 0 END) as cancelled_orders
                FROM sales
            """).fetchone()
            
            stats["overall"] = {
                "total_orders": overall[0],
                "total_revenue": round(overall[1] or 0, 2),
                "total_profit": round(overall[2] or 0, 2),
                "avg_order_value": round(overall[3] or 0, 2),
                "date_range": f"{overall[4]} to {overall[5]}",
                "cancelled_orders": overall[6]
            }
            
            # By state
            states = self.conn.execute("""
                SELECT 
                    state,
                    SUM(revenue) as revenue,
                    COUNT(*) as orders
                FROM sales
                WHERE state IS NOT NULL
                GROUP BY state
                ORDER BY revenue DESC
                LIMIT 10
            """).fetchdf()
            stats["top_states"] = states.to_dict('records')
            
            # By category
            categories = self.conn.execute("""
                SELECT 
                    category,
                    SUM(revenue) as revenue,
                    COUNT(*) as orders
                FROM sales
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY revenue DESC
            """).fetchdf()
            stats["by_category"] = categories.to_dict('records')
            
            # Monthly trends
            monthly = self.conn.execute("""
                SELECT 
                    year,
                    month,
                    SUM(revenue) as revenue,
                    SUM(estimated_profit) as profit,
                    COUNT(*) as orders
                FROM sales
                GROUP BY year, month
                ORDER BY year, month
            """).fetchdf()
            stats["monthly_trend"] = monthly.to_dict('records')
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting summary stats: {e}")
            return {}
    
    def get_schema_context(self) -> str:
        """Get schema context for LLM with real Amazon sales data structure"""
        if self.schema_info is not None and len(self.schema_info) > 0:
            schema_str = "Database: 'sales' table\nColumns:\n"
            for _, row in self.schema_info.iterrows():
                schema_str += f"- {row['column_name']}: {row['column_type']}\n"
            return schema_str
        
        # Fallback schema documentation for Amazon sales data
        return """
Database: 'sales' table
Columns:
- order_id: VARCHAR (unique order identifier)
- date: DATE (order date)
- status: VARCHAR (order status: Shipped, Cancelled, etc.)
- fulfilment: VARCHAR (Amazon, Merchant)
- sales_channel: VARCHAR (Amazon.in)
- service_level: VARCHAR (Standard, Expedited)
- style: VARCHAR (product style)
- sku: VARCHAR (stock keeping unit)
- category: VARCHAR (product category)
- size: VARCHAR (product size)
- asin: VARCHAR (Amazon Standard Identification Number)
- courier_status: VARCHAR (shipping status)
- quantity: INTEGER (quantity ordered)
- currency: VARCHAR (INR)
- amount: DOUBLE (order amount)
- city: VARCHAR (shipping city)
- state: VARCHAR (shipping state)
- postal_code: VARCHAR (postal code)
- country: VARCHAR (country)
- is_b2b: BOOLEAN (business-to-business flag)
- year: INTEGER (extracted year)
- month: INTEGER (extracted month)
- month_name: VARCHAR (month name)
- quarter: INTEGER (extracted quarter)
- quarter_name: VARCHAR (quarter name)
- order_value_category: VARCHAR (small/medium/large based on amount)
- is_cancelled: BOOLEAN (true if status is Cancelled)
- is_shipped: BOOLEAN (true if order is shipped)
- revenue: DOUBLE (amount for non-cancelled orders)
- estimated_profit: DOUBLE (estimated 20% profit on revenue)
- unit_price: DOUBLE (price per unit)

Common query patterns:
- Revenue analysis: SELECT SUM(revenue) FROM sales WHERE status != 'Cancelled'
- Top categories: SELECT category, SUM(revenue) as total FROM sales GROUP BY category ORDER BY total DESC
- Monthly trends: SELECT year, month, SUM(revenue) FROM sales GROUP BY year, month ORDER BY year, month
- Regional performance: SELECT state, COUNT(*) as orders, SUM(revenue) FROM sales GROUP BY state
- Cancellation rate: SELECT is_cancelled, COUNT(*) FROM sales GROUP BY is_cancelled
"""
    
    def validate_query_result(self, df: pd.DataFrame) -> bool:
        """
        Validate query results
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            return False
        return True
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Singleton instance
_data_layer_instance = None


def get_data_layer() -> DataLayer:
    """Get singleton instance of DataLayer"""
    global _data_layer_instance
    if _data_layer_instance is None:
        _data_layer_instance = DataLayer()
    return _data_layer_instance
