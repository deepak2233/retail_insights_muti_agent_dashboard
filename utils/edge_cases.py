"""
Edge Case Handler
Handles ambiguous queries, out-of-scope questions, date ranges, typos, and other edge cases
"""
from typing import Dict, List, Optional, Tuple, Any
import re
from difflib import SequenceMatcher
from dataclasses import dataclass


@dataclass
class EdgeCaseResult:
    """Result of edge case handling"""
    is_edge_case: bool
    edge_case_type: Optional[str]
    message: Optional[str]
    suggestions: List[str]
    modified_question: Optional[str]
    requires_clarification: bool


class EdgeCaseHandler:
    """
    Handles various edge cases in user queries
    """
    
    def __init__(self, data_layer=None):
        """
        Initialize edge case handler
        
        Args:
            data_layer: DataLayer instance for data context
        """
        self.data_layer = data_layer
        
        # Known categories (will be populated from data)
        self._categories = None
        self._states = None
        self._date_range = None
        
        # Out of scope keywords
        self.out_of_scope_patterns = [
            r'\b(weather|news|sports|politics|recipes?|movies?|music)\b',
            r'\b(who is|what is the capital|tell me about)\b',
            r'\b(joke|story|poem|song)\b',
            r'\b(calculate|compute|solve)\b(?!.*\b(sales|revenue|profit|orders?)\b)',
        ]
        
        # Retail domain keywords
        self.retail_keywords = [
            'sales', 'revenue', 'profit', 'order', 'orders', 'customer', 'customers',
            'product', 'products', 'category', 'categories', 'region', 'state', 'states',
            'month', 'quarter', 'year', 'trend', 'growth', 'decline', 'top', 'bottom',
            'best', 'worst', 'average', 'total', 'count', 'sum', 'compare', 'comparison',
            'shipped', 'cancelled', 'pending', 'b2b', 'b2c', 'amazon', 'sku', 'asin'
        ]
    
    def handle(self, question: str) -> EdgeCaseResult:
        """
        Main entry point: handle edge cases in a question
        
        Args:
            question: User's question
            
        Returns:
            EdgeCaseResult with handling details
        """
        question_lower = question.lower().strip()
        
        # Check for empty or too short
        if len(question_lower) < 3:
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="too_short",
                message="Please provide a more detailed question about your retail data.",
                suggestions=["Show me total revenue", "What are the top selling categories?"],
                modified_question=None,
                requires_clarification=True
            )
        
        # Check for out of scope
        out_of_scope = self._check_out_of_scope(question_lower)
        if out_of_scope:
            return out_of_scope
        
        # Check for ambiguity
        ambiguity = self._check_ambiguity(question_lower)
        if ambiguity:
            return ambiguity
        
        # Check for typos in known entities
        typo_result = self._check_typos(question)
        if typo_result:
            return typo_result
        
        # Check for date range issues
        date_result = self._check_date_range(question_lower)
        if date_result:
            return date_result
        
        # Check for overly complex queries
        complexity = self._check_complexity(question_lower)
        if complexity:
            return complexity
        
        # No edge case detected
        return EdgeCaseResult(
            is_edge_case=False,
            edge_case_type=None,
            message=None,
            suggestions=[],
            modified_question=question,
            requires_clarification=False
        )
    
    def _check_out_of_scope(self, question: str) -> Optional[EdgeCaseResult]:
        """Check if question is out of scope for retail analytics"""
        
        # Check for out of scope patterns
        for pattern in self.out_of_scope_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return EdgeCaseResult(
                    is_edge_case=True,
                    edge_case_type="out_of_scope",
                    message="I specialize in retail analytics. I can help you with questions about sales, revenue, products, orders, and customer data.",
                    suggestions=[
                        "What was the total revenue last month?",
                        "Which categories have the highest sales?",
                        "Show me the order trends by state",
                        "What is the cancellation rate?"
                    ],
                    modified_question=None,
                    requires_clarification=True
                )
        
        # Check if any retail keyword is present
        has_retail_context = any(kw in question for kw in self.retail_keywords)
        
        if not has_retail_context and len(question) > 20:
            # Longer question with no retail context
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="unclear_intent",
                message="I'm not sure how this relates to your retail data. Could you rephrase your question to focus on sales, orders, or products?",
                suggestions=[
                    "Show total sales",
                    "Compare revenue by category",
                    "What are the top states by orders?"
                ],
                modified_question=None,
                requires_clarification=True
            )
        
        return None
    
    def _check_ambiguity(self, question: str) -> Optional[EdgeCaseResult]:
        """Check for ambiguous queries that need clarification"""
        
        ambiguous_patterns = [
            (r'^(show|get|display|list)(\s+me)?(\s+the)?\s*(data|info|information)?\s*$', 
             "What specific data would you like to see?",
             ["Total revenue by month", "Top 10 products by sales", "Orders by state"]),
            
            (r'^(how much|what)\??\s*$',
             "Could you specify what metric you're interested in?",
             ["How much revenue in total?", "What is the average order value?"]),
            
            (r'^(compare|comparison)\s*$',
             "What would you like to compare?",
             ["Compare B2B vs B2C sales", "Compare revenue by quarter", "Compare top categories"]),
            
            (r'\b(it|they|them|this|that|these|those)\b(?!.*\b(category|product|state|order)\b)',
             None,  # This is handled by memory resolution
             []),
        ]
        
        for pattern, message, suggestions in ambiguous_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                if message:  # Skip if it's a reference that memory should handle
                    return EdgeCaseResult(
                        is_edge_case=True,
                        edge_case_type="ambiguous",
                        message=message,
                        suggestions=suggestions,
                        modified_question=None,
                        requires_clarification=True
                    )
        
        return None
    
    def _check_typos(self, question: str) -> Optional[EdgeCaseResult]:
        """Check for typos in known entities and suggest corrections"""
        
        # Get known entities
        categories = self._get_categories()
        states = self._get_states()
        
        # Extract potential entity mentions from question
        words = re.findall(r'\b[A-Za-z]{3,}\b', question)
        
        corrections = []
        modified = question
        
        for word in words:
            word_lower = word.lower()
            
            # Skip common words
            common_words = {'the', 'and', 'for', 'with', 'what', 'how', 'show', 'total', 
                           'revenue', 'sales', 'orders', 'top', 'best', 'worst', 'from'}
            if word_lower in common_words:
                continue
            
            # Check against categories
            for cat in categories:
                similarity = self._similarity(word_lower, cat.lower())
                if 0.6 < similarity < 1.0:  # Similar but not exact
                    corrections.append((word, cat, "category"))
                    modified = modified.replace(word, cat)
                    break
            
            # Check against states
            for state in states:
                similarity = self._similarity(word_lower, state.lower())
                if 0.6 < similarity < 1.0:
                    corrections.append((word, state, "state"))
                    modified = modified.replace(word, state)
                    break
        
        if corrections:
            correction_msg = ", ".join([f"'{c[0]}' → '{c[1]}'" for c in corrections])
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="typo_correction",
                message=f"I corrected some terms: {correction_msg}",
                suggestions=[],
                modified_question=modified,
                requires_clarification=False  # Auto-corrected
            )
        
        return None
    
    def _check_date_range(self, question: str) -> Optional[EdgeCaseResult]:
        """Check if requested date range is available in data"""
        
        # Extract year mentions
        year_pattern = re.compile(r'\b(20\d{2})\b')
        years = [int(y) for y in year_pattern.findall(question)]
        
        # Get available date range
        available_range = self._get_date_range()
        if not available_range:
            return None
        
        min_year, max_year = available_range
        
        for year in years:
            if year < min_year or year > max_year:
                return EdgeCaseResult(
                    is_edge_case=True,
                    edge_case_type="date_out_of_range",
                    message=f"I have data from {min_year} to {max_year}. Year {year} is not available.",
                    suggestions=[
                        f"Show revenue for {max_year}",
                        f"Compare {min_year} and {max_year}",
                        f"Monthly trend in {max_year}"
                    ],
                    modified_question=None,
                    requires_clarification=True
                )
        
        # Check for future dates
        future_patterns = [
            r'\b(next|upcoming|future|will be|forecast|predict)\b',
            r'\b(2026|2027|2028|2029|2030)\b'
        ]
        
        for pattern in future_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return EdgeCaseResult(
                    is_edge_case=True,
                    edge_case_type="future_prediction",
                    message="I can only analyze historical data, not make future predictions. I can show you trends from past data that might inform your planning.",
                    suggestions=[
                        "Show me the revenue trend over time",
                        "What is the month-over-month growth rate?",
                        "Which categories are growing fastest?"
                    ],
                    modified_question=None,
                    requires_clarification=True
                )
        
        return None
    
    def _check_complexity(self, question: str) -> Optional[EdgeCaseResult]:
        """Check for overly complex queries that should be broken down"""
        
        # Count conjunctions and conditions
        conjunction_count = len(re.findall(r'\b(and|or|but|also|as well as|together with)\b', question))
        
        # Count different metrics requested
        metric_keywords = ['revenue', 'profit', 'sales', 'orders', 'quantity', 'average', 'count', 'rate']
        metric_count = sum(1 for m in metric_keywords if m in question)
        
        # Count different dimensions
        dimension_keywords = ['state', 'category', 'month', 'quarter', 'year', 'product', 'customer']
        dimension_count = sum(1 for d in dimension_keywords if d in question)
        
        if conjunction_count >= 3 or (metric_count >= 3 and dimension_count >= 2):
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="complex_query",
                message="This is a complex query. Let me break it down into simpler parts for more accurate results.",
                suggestions=[
                    "Try asking one question at a time",
                    "Start with: What is the total revenue?",
                    "Then: How does it break down by category?"
                ],
                modified_question=question,  # Still attempt it
                requires_clarification=False
            )
        
        return None
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings"""
        return SequenceMatcher(None, a, b).ratio()
    
    def _get_categories(self) -> List[str]:
        """Get available categories from data layer"""
        if self._categories is not None:
            return self._categories
        
        if self.data_layer:
            try:
                result = self.data_layer.execute_query(
                    "SELECT DISTINCT category FROM sales WHERE category IS NOT NULL LIMIT 50"
                )
                self._categories = result['category'].tolist()
                return self._categories
            except:
                pass
        
        # Default categories from schema
        self._categories = ['Set', 'Kurta', 'Western Dress', 'Top', 'Ethnic Dress', 'Blouse', 'Saree']
        return self._categories
    
    def _get_states(self) -> List[str]:
        """Get available states from data layer"""
        if self._states is not None:
            return self._states
        
        if self.data_layer:
            try:
                result = self.data_layer.execute_query(
                    "SELECT DISTINCT state FROM sales WHERE state IS NOT NULL LIMIT 50"
                )
                self._states = result['state'].tolist()
                return self._states
            except:
                pass
        
        # Default states
        self._states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Gujarat', 
                       'Uttar Pradesh', 'West Bengal', 'Telangana', 'Rajasthan', 'Kerala']
        return self._states
    
    def _get_date_range(self) -> Optional[Tuple[int, int]]:
        """Get available date range from data layer"""
        if self._date_range is not None:
            return self._date_range
        
        if self.data_layer:
            try:
                result = self.data_layer.execute_query(
                    "SELECT MIN(year) as min_year, MAX(year) as max_year FROM sales"
                )
                if not result.empty:
                    self._date_range = (int(result['min_year'].iloc[0]), 
                                       int(result['max_year'].iloc[0]))
                    return self._date_range
            except:
                pass
        
        # Default range based on schema
        self._date_range = (2021, 2022)
        return self._date_range
    
    def refresh_cache(self):
        """Refresh cached entity data"""
        self._categories = None
        self._states = None
        self._date_range = None


def get_edge_case_handler(data_layer=None) -> EdgeCaseHandler:
    """Factory function to create edge case handler"""
    return EdgeCaseHandler(data_layer)
