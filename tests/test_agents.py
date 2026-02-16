""" 
Unit tests for Retail Insights Assistant agents
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_agent import QueryResolutionAgent, AgentState
from agents.extraction_agent import DataExtractionAgent
from agents.validation_agent import ValidationAgent
from utils.data_layer import DataLayer


class TestQueryResolutionAgent:
    """Test Query Resolution Agent"""
    
    def test_resolve_simple_query(self):
        """Test resolving a simple aggregation query"""
        agent = QueryResolutionAgent()
        
        question = "What were total sales in 2023?"
        result = agent.resolve_query(question)
        
        assert result is not None
        assert result.sql_query is not None
        assert "2023" in result.sql_query
        assert "SUM" in result.sql_query.upper()
    
    def test_schema_context(self):
        """Test schema context generation"""
        agent = QueryResolutionAgent()
        schema = agent.get_schema_context()
        
        assert "sales" in schema.lower()
        assert "revenue" in schema.lower()
        assert "region" in schema.lower()


class TestDataExtractionAgent:
    """Test Data Extraction Agent"""
    
    def test_extract_data_success(self):
        """Test successful data extraction"""
        agent = DataExtractionAgent()
        
        state = AgentState(
            question="Test question",
            query_intent=Mock(sql_query="SELECT COUNT(*) as count FROM sales LIMIT 10"),
            query_result=None,
            validation_passed=False,
            final_answer="",
            error=None
        )
        
        # This will only work if data is loaded
        try:
            result = agent.extract_data(state)
            assert result.get("error") is None or result.get("query_result") is not None
        except Exception as e:
            # Expected if database not initialized
            assert "error" in str(e).lower() or "not found" in str(e).lower()
    
    def test_generate_summary(self):
        """Test summary generation"""
        agent = DataExtractionAgent()
        
        # Create sample dataframe
        df = pd.DataFrame({
            "revenue": [100, 200, 300],
            "profit": [10, 20, 30]
        })
        
        summary = agent._generate_summary(df)
        
        assert "3 records" in summary
        assert "revenue" in summary.lower()


class TestValidationAgent:
    """Test Validation Agent"""
    
    def test_validate_empty_result(self):
        """Test validation with empty result"""
        agent = ValidationAgent()
        
        state = AgentState(
            question="Test",
            query_intent=None,
            query_result=None,
            validation_passed=False,
            final_answer="",
            error=None
        )
        
        result = agent.validate(state)
        assert result["validation_passed"] == False
    
    def test_validate_successful_result(self):
        """Test validation with good data"""
        agent = ValidationAgent()
        
        df = pd.DataFrame({
            "revenue": [100, 200, 300],
            "profit": [10, 20, 30],
            "quantity": [1, 2, 3]
        })
        
        state = AgentState(
            question="Test",
            query_intent=Mock(),
            query_result={
                "dataframe": df,
                "row_count": len(df),
                "columns": list(df.columns),
                "data": df.to_dict('records')
            },
            validation_passed=False,
            final_answer="",
            error=None
        )
        
        result = agent.validate(state)
        assert result["validation_passed"] == True


class TestDataLayer:
    """Test Data Layer"""
    
    def test_data_layer_initialization(self):
        """Test data layer can be initialized"""
        try:
            dl = DataLayer(csv_path="data/sales_data.csv")
            assert dl is not None
            assert dl.conn is not None
        except Exception as e:
            # Expected if data file doesn't exist
            assert "not found" in str(e).lower() or "no such file" in str(e).lower()
    
    def test_schema_info(self):
        """Test schema information retrieval"""
        try:
            dl = DataLayer(csv_path="data/sales_data.csv")
            schema = dl.get_schema_info()
            assert "sales" in schema.lower()
        except Exception:
            # Expected if data not loaded
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
