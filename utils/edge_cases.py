"""
Edge Case Handler — Production-grade edge case detection and handling
with data layer integration, rate limiting, multi-language detection,
and empty result prediction.
"""
from typing import Dict, List, Optional, Tuple, Any
import re
import time
import logging
from difflib import SequenceMatcher
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EdgeCaseResult:
    """Result of edge case handling"""
    is_edge_case: bool
    edge_case_type: Optional[str]
    message: Optional[str]
    suggestions: List[str]
    modified_question: Optional[str]
    requires_clarification: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int = 15, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: List[float] = []

    def check(self) -> bool:
        """Returns True if rate limit is exceeded."""
        now = time.time()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        self._timestamps.append(now)
        return len(self._timestamps) > self.max_requests

    @property
    def current_rate(self) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        return sum(1 for t in self._timestamps if t > cutoff)


class EdgeCaseHandler:
    """
    Production-grade edge case handler with:
    - Data layer integration for real-time entity validation
    - Rate limiting
    - Multi-language fragment detection
    - Empty result prediction
    - Enhanced typo correction with DB-backed vocabulary
    """

    def __init__(self, data_layer=None):
        self.data_layer = data_layer
        self.rate_limiter = RateLimiter(max_requests=15, window_seconds=60)

        # Cached data from DB
        self._categories: Optional[List[str]] = None
        self._states: Optional[List[str]] = None
        self._date_range: Optional[Tuple[int, int]] = None

        # Out of scope patterns
        self.out_of_scope_patterns = [
            r'\b(weather|news|sports|politics|recipes?|movies?|music|games?)\b',
            r'\b(who is|what is the capital|tell me about|biography)\b',
            r'\b(joke|story|poem|song|lyrics)\b',
            r'\b(calculate|compute|solve)(?!.*\b(sales|revenue|profit|orders?)\b)\b',
            r'\b(code|program|debug|compile|script)\b',
            r'\b(medical|health|diagnosis|symptoms)\b',
            r'\b(stock\s*market|crypto|bitcoin|invest)\b',
        ]

        # Retail domain keywords
        self.retail_keywords = [
            'sales', 'revenue', 'profit', 'order', 'orders', 'customer', 'customers',
            'product', 'products', 'category', 'categories', 'region', 'state', 'states',
            'month', 'quarter', 'year', 'trend', 'growth', 'decline', 'top', 'bottom',
            'best', 'worst', 'average', 'total', 'count', 'sum', 'compare', 'comparison',
            'shipped', 'cancelled', 'pending', 'b2b', 'b2c', 'amazon', 'sku', 'asin',
            'kurta', 'saree', 'blouse', 'western', 'dress', 'ethnic', 'set',
            'fulfillment', 'fulfilment', 'merchant', 'expedited', 'standard',
            'amount', 'quantity', 'price', 'unit',
        ]

        # Non-English detection (common non-ASCII scripts)
        self._non_english_re = re.compile(
            r'[\u0900-\u097F]|'  # Devanagari
            r'[\u0600-\u06FF]|'  # Arabic
            r'[\u4E00-\u9FFF]|'  # CJK
            r'[\u3040-\u309F]|'  # Hiragana
            r'[\uAC00-\uD7AF]',  # Korean
        )

    def handle(self, question: str) -> EdgeCaseResult:
        """Main entry point: handle edge cases in a question."""
        question_lower = question.lower().strip()

        # ── Rate limiting ──
        if self.rate_limiter.check():
            logger.warning("Rate limit exceeded: %d requests in window", self.rate_limiter.current_rate)
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="rate_limited",
                message="You're sending queries quite fast. Please wait a moment before your next question to ensure accurate results.",
                suggestions=["Take a moment, then try your question again"],
                modified_question=None,
                requires_clarification=True,
                metadata={"rate": self.rate_limiter.current_rate},
            )

        # ── Empty / whitespace-only ──
        if not question_lower or not question_lower.strip():
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="empty_query",
                message="It looks like you submitted an empty question. What would you like to know about your retail data?",
                suggestions=[
                    "What is the total revenue?",
                    "Show me the top 5 categories by sales",
                    "What is the cancellation rate?",
                ],
                modified_question=None,
                requires_clarification=True,
            )

        # ── Too short ──
        if len(question_lower) < 3:
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="too_short",
                message="Please provide a more detailed question about your retail data.",
                suggestions=["Show me total revenue", "What are the top selling categories?"],
                modified_question=None,
                requires_clarification=True,
            )

        # ── Single-word query ──
        words = question_lower.split()
        if len(words) == 1 and len(question_lower) >= 3:
            word = words[0].strip("?!.")
            suggestions_map = {
                "revenue": "What is the total revenue?",
                "sales": "What are total sales by category?",
                "profit": "What is the total profit?",
                "orders": "How many orders were placed in total?",
                "categories": "What are the top categories by revenue?",
                "states": "Which states have the most orders?",
                "cancellations": "What is the cancellation rate?",
            }
            suggestion = suggestions_map.get(word, f"What is the total {word}?")
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="single_word",
                message=f"Could you elaborate? A single word isn't enough for accurate analysis.",
                suggestions=[
                    suggestion,
                    "Show me total revenue by category",
                    "What are the top 5 states by sales?",
                ],
                modified_question=None,
                requires_clarification=True,
            )

        # ── Excessively long query (>500 chars) ──
        if len(question) > 500:
            truncated = question[:500].rsplit(' ', 1)[0] + "..."
            logger.warning("Query truncated from %d to 500 chars", len(question))
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="too_long",
                message="Your question is very long. I've focused on the main part. For best results, keep questions concise.",
                suggestions=[],
                modified_question=truncated,
                requires_clarification=False,
                metadata={"original_length": len(question)},
            )

        # ── Repeated punctuation abuse ──
        if re.search(r'[!?]{4,}|\.{4,}', question):
            cleaned = re.sub(r'([!?.])\1{2,}', r'\1', question)
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="punctuation_normalized",
                message=None,
                suggestions=[],
                modified_question=cleaned,
                requires_clarification=False,
            )

        # ── Multi-language detection ──
        if self._non_english_re.search(question):
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="non_english",
                message="I work best with English queries. Please rephrase your question in English so I can provide accurate analytics.",
                suggestions=[
                    "What is the total revenue?",
                    "Show top categories by sales",
                    "Compare B2B vs B2C orders",
                ],
                modified_question=None,
                requires_clarification=True,
            )

        # ── Out of scope ──
        out_of_scope = self._check_out_of_scope(question_lower)
        if out_of_scope:
            return out_of_scope

        # ── Ambiguity ──
        ambiguity = self._check_ambiguity(question_lower)
        if ambiguity:
            return ambiguity

        # ── Typos (with DB-backed correction) ──
        typo_result = self._check_typos(question)
        if typo_result:
            return typo_result

        # ── Entity validation (predict empty results) ──
        entity_result = self._validate_entities(question)
        if entity_result:
            return entity_result

        # ── Date range ──
        date_result = self._check_date_range(question_lower)
        if date_result:
            return date_result

        # ── Complexity ──
        complexity = self._check_complexity(question_lower)
        if complexity:
            return complexity

        # ── No edge case ──
        return EdgeCaseResult(
            is_edge_case=False,
            edge_case_type=None,
            message=None,
            suggestions=[],
            modified_question=question,
            requires_clarification=False,
        )

    # ──────────────────────────────────────────────
    # Check methods
    # ──────────────────────────────────────────────

    def _check_out_of_scope(self, question: str) -> Optional[EdgeCaseResult]:
        """Check if question is out of scope for retail analytics."""
        for pattern in self.out_of_scope_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return EdgeCaseResult(
                    is_edge_case=True,
                    edge_case_type="out_of_scope",
                    message="I specialize in retail analytics and can help with sales, revenue, product, order, and customer data analysis.",
                    suggestions=[
                        "What was the total revenue?",
                        "Which categories have the highest sales?",
                        "Show me the order trends by state",
                        "What is the cancellation rate?",
                    ],
                    modified_question=None,
                    requires_clarification=True,
                )

        # Check if any retail keyword is present
        has_retail_context = any(kw in question for kw in self.retail_keywords)
        if not has_retail_context and len(question) > 25:
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="unclear_intent",
                message="I'm not sure how this relates to retail data. Could you rephrase to focus on sales, orders, or products?",
                suggestions=[
                    "Show total sales",
                    "Compare revenue by category",
                    "What are the top states by orders?",
                ],
                modified_question=None,
                requires_clarification=True,
            )

        return None

    def _check_ambiguity(self, question: str) -> Optional[EdgeCaseResult]:
        """Check for ambiguous queries that need clarification."""
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
        ]

        for pattern, message, suggestions in ambiguous_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return EdgeCaseResult(
                    is_edge_case=True,
                    edge_case_type="ambiguous",
                    message=message,
                    suggestions=suggestions,
                    modified_question=None,
                    requires_clarification=True,
                )

        return None

    def _check_typos(self, question: str) -> Optional[EdgeCaseResult]:
        """Check for typos in known entities with DB-backed vocabulary."""
        categories = self._get_categories()
        states = self._get_states()

        words = re.findall(r'\b[A-Za-z]{3,}\b', question)
        corrections = []
        modified = question

        common_words = {
            'the', 'and', 'for', 'with', 'what', 'how', 'show', 'total',
            'revenue', 'sales', 'orders', 'top', 'best', 'worst', 'from',
            'compare', 'between', 'give', 'tell', 'display', 'list', 'get',
            'average', 'count', 'sum', 'which', 'where', 'when', 'many',
            'much', 'all', 'most', 'least', 'that', 'this', 'category',
            'state', 'month', 'year', 'quarter', 'profit', 'amount',
        }

        for word in words:
            word_lower = word.lower()
            if word_lower in common_words:
                continue

            # Check against categories
            for cat in categories:
                similarity = SequenceMatcher(None, word_lower, cat.lower()).ratio()
                if 0.65 < similarity < 1.0:
                    corrections.append((word, cat, "category"))
                    modified = modified.replace(word, cat)
                    break

            # Check against states
            for state in states:
                similarity = SequenceMatcher(None, word_lower, state.lower()).ratio()
                if 0.65 < similarity < 1.0:
                    corrections.append((word, state, "state"))
                    modified = modified.replace(word, state)
                    break

        if corrections:
            correction_msg = ", ".join([f"'{c[0]}' -> '{c[1]}'" for c in corrections])
            logger.info("Typo corrections applied: %s", correction_msg)
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="typo_correction",
                message=f"I corrected some terms: {correction_msg}",
                suggestions=[],
                modified_question=modified,
                requires_clarification=False,
            )

        return None

    def _validate_entities(self, question: str) -> Optional[EdgeCaseResult]:
        """
        Predict empty results by checking if mentioned entities exist in DB.
        """
        if not self.data_layer:
            return None

        categories = self._get_categories()
        states = self._get_states()

        # Look for quoted entity names or capitalized words
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
        mentioned_entities = [q[0] or q[1] for q in quoted]

        # Also check capitalized words (potential entity names)
        capitalized = re.findall(r'\b([A-Z][a-z]{2,})\b', question)
        mentioned_entities.extend(capitalized)

        for entity in mentioned_entities:
            entity_lower = entity.lower()

            # Check if it looks like a category but doesn't exist
            cat_matches = [c for c in categories if c.lower() == entity_lower]
            state_matches = [s for s in states if s.lower() == entity_lower]

            if not cat_matches and not state_matches:
                # Find closest matches to suggest
                cat_suggestions = self._find_closest(entity_lower, categories, n=3)
                state_suggestions = self._find_closest(entity_lower, states, n=3)

                all_suggestions = cat_suggestions + state_suggestions
                if all_suggestions:
                    return EdgeCaseResult(
                        is_edge_case=True,
                        edge_case_type="entity_not_found",
                        message=f"'{entity}' was not found in the data. Did you mean one of these?",
                        suggestions=all_suggestions,
                        modified_question=None,
                        requires_clarification=True,
                        metadata={"missing_entity": entity},
                    )

        return None

    def _check_date_range(self, question: str) -> Optional[EdgeCaseResult]:
        """Check if requested date range is available in data."""
        year_pattern = re.compile(r'\b(20\d{2})\b')
        years = [int(y) for y in year_pattern.findall(question)]

        available_range = self._get_date_range()
        if not available_range:
            return None

        min_year, max_year = available_range

        for year in years:
            if year < min_year or year > max_year:
                return EdgeCaseResult(
                    is_edge_case=True,
                    edge_case_type="date_out_of_range",
                    message=f"Data is available from {min_year} to {max_year}. Year {year} is outside this range.",
                    suggestions=[
                        f"Show revenue for {max_year}",
                        f"Compare {min_year} and {max_year}",
                        f"Monthly trend in {max_year}",
                    ],
                    modified_question=None,
                    requires_clarification=True,
                )

        # future date patterns
        future_patterns = [
            r'\b(next|upcoming|future|will be|forecast|predict)\b',
            r'\b(202[6-9]|20[3-9]\d)\b',
        ]
        for pattern in future_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return EdgeCaseResult(
                    is_edge_case=True,
                    edge_case_type="future_prediction",
                    message="I analyze historical data only, not future predictions. I can show trends to support your planning.",
                    suggestions=[
                        "Show me the revenue trend over time",
                        "What is the month-over-month growth rate?",
                        "Which categories are growing fastest?",
                    ],
                    modified_question=None,
                    requires_clarification=True,
                )

        return None

    def _check_complexity(self, question: str) -> Optional[EdgeCaseResult]:
        """Check for overly complex queries that should be broken down."""
        conjunction_count = len(re.findall(r'\b(and|or|but|also|as well as|together with)\b', question))
        metric_keywords = ['revenue', 'profit', 'sales', 'orders', 'quantity', 'average', 'count', 'rate']
        metric_count = sum(1 for m in metric_keywords if m in question)
        dimension_keywords = ['state', 'category', 'month', 'quarter', 'year', 'product', 'customer']
        dimension_count = sum(1 for d in dimension_keywords if d in question)

        if conjunction_count >= 3 or (metric_count >= 3 and dimension_count >= 2):
            return EdgeCaseResult(
                is_edge_case=True,
                edge_case_type="complex_query",
                message="This is a complex query. Let me break it down for more accurate results.",
                suggestions=[
                    "Try asking one question at a time",
                    "Start with: What is the total revenue?",
                    "Then: How does it break down by category?",
                ],
                modified_question=question,  # Still attempt it
                requires_clarification=False,
            )

        return None

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _find_closest(self, word: str, candidates: List[str], n: int = 3) -> List[str]:
        """Find the N closest matches from a list of candidates."""
        scored = []
        for candidate in candidates:
            score = SequenceMatcher(None, word.lower(), candidate.lower()).ratio()
            if score > 0.4:
                scored.append((candidate, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:n]]

    def _get_categories(self) -> List[str]:
        """Get categories from data layer or defaults."""
        if self._categories is not None:
            return self._categories

        if self.data_layer:
            try:
                result = self.data_layer.execute_query(
                    "SELECT DISTINCT category FROM sales WHERE category IS NOT NULL LIMIT 50"
                )
                self._categories = result['category'].tolist()
                return self._categories
            except Exception:
                pass

        self._categories = ['Set', 'Kurta', 'Western Dress', 'Top', 'Ethnic Dress', 'Blouse', 'Saree']
        return self._categories

    def _get_states(self) -> List[str]:
        """Get states from data layer or defaults."""
        if self._states is not None:
            return self._states

        if self.data_layer:
            try:
                result = self.data_layer.execute_query(
                    "SELECT DISTINCT state FROM sales WHERE state IS NOT NULL LIMIT 50"
                )
                self._states = result['state'].tolist()
                return self._states
            except Exception:
                pass

        self._states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Gujarat',
                        'Uttar Pradesh', 'West Bengal', 'Telangana', 'Rajasthan', 'Kerala']
        return self._states

    def _get_date_range(self) -> Optional[Tuple[int, int]]:
        """Get date range from data layer or defaults."""
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
            except Exception:
                pass

        self._date_range = (2021, 2022)
        return self._date_range

    def refresh_cache(self):
        """Refresh cached entity data."""
        self._categories = None
        self._states = None
        self._date_range = None


def get_edge_case_handler(data_layer=None) -> EdgeCaseHandler:
    """Factory function to create edge case handler."""
    return EdgeCaseHandler(data_layer)
