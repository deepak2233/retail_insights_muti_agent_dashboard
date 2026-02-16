"""
Query Preprocessor — Production-grade query preprocessing pipeline
Handles spell correction, normalization, synonym expansion, and input sanitization.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
import html
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Result of query preprocessing"""
    original: str
    normalized: str
    corrections: List[Tuple[str, str]]  # (original_word, corrected_word)
    expansions: List[Tuple[str, str]]   # (abbreviation, expanded)
    was_modified: bool
    sanitized: bool
    flags: Dict[str, bool] = field(default_factory=dict)


class QueryPreprocessor:
    """
    Production-grade query preprocessing pipeline.
    
    Pipeline order:
    1. Sanitize (strip HTML, limit length, detect injection)
    2. Normalize (lowercase, trim whitespace, expand abbreviations)
    3. Spell-correct (fuzzy-match against retail vocabulary)
    4. Synonym expansion (map user terms to schema columns)
    """

    # --- Abbreviation map (user shorthand → full term) ---
    ABBREVIATIONS = {
        'rev': 'revenue', 'qty': 'quantity', 'cat': 'category',
        'prod': 'product', 'cust': 'customer', 'avg': 'average',
        'mo': 'month', 'yr': 'year', 'qtr': 'quarter',
        'mh': 'maharashtra', 'ka': 'karnataka', 'tn': 'tamil nadu',
        'dl': 'delhi', 'gj': 'gujarat', 'up': 'uttar pradesh',
        'wb': 'west bengal', 'ts': 'telangana', 'rj': 'rajasthan',
        'kl': 'kerala', 'ap': 'andhra pradesh',
        'canc': 'cancelled', 'amt': 'amount',
        'b2b': 'B2B', 'b2c': 'B2C',
        'yoy': 'year over year', 'mom': 'month over month',
        'aov': 'average order value',
    }

    # --- Synonym map (user-friendly term → schema / SQL concept) ---
    SYNONYMS = {
        # Metric synonyms
        'sales': 'revenue',
        'income': 'revenue',
        'earnings': 'revenue',
        'turnover': 'revenue',
        'profit margin': 'estimated_profit',
        'profit': 'estimated_profit',
        'margins': 'estimated_profit',
        'order value': 'amount',
        'price': 'unit_price',
        # Aggregation synonyms
        'how many': 'count of',
        'number of': 'count of',
        'total number': 'total count',
        'break down by': 'group by',
        'broken down by': 'group by',
        'split by': 'group by',
        'per': 'group by',
        'breakdown': 'group by',
        # Status synonyms
        'returned': 'cancelled',
        'refunded': 'cancelled',
        'delivered': 'shipped',
        'fulfilled': 'shipped',
        'pending orders': 'orders where status is pending',
        # Channel synonyms
        'business orders': 'B2B orders',
        'consumer orders': 'B2C orders',
        'retail orders': 'B2C orders',
        'wholesale': 'B2B',
    }

    # --- Query pattern normalizers (intent detection helpers) ---
    PATTERN_NORMALIZERS = [
        (r"\bwhat(?:'s| is| are) the\b", ''),
        (r"\bcan you (?:show|tell|give|get)\b", 'show'),
        (r"\bi (?:want|need|would like) to (?:see|know|find)\b", 'show'),
        (r"\bplease\b", ''),
        (r"\bcould you\b", ''),
        (r"\bshow me\b", 'show'),
        (r"\bgive me\b", 'show'),
        (r"\btell me\b", 'show'),
        (r"\blist (?:all |the )?", 'show '),
        (r"\bdisplay\b", 'show'),
    ]

    # --- Retail vocabulary for spell correction ---
    RETAIL_VOCABULARY = [
        # Metrics
        'revenue', 'sales', 'profit', 'orders', 'quantity', 'amount',
        'average', 'total', 'count', 'growth', 'rate', 'trend',
        'comparison', 'percentage', 'margin', 'value',
        # Dimensions
        'category', 'state', 'region', 'city', 'month', 'quarter',
        'year', 'product', 'customer', 'channel', 'status',
        # Categories (from schema)
        'kurta', 'saree', 'blouse', 'western', 'dress', 'ethnic', 'top', 'set',
        # Statuses
        'shipped', 'cancelled', 'pending', 'delivered',
        # Channels
        'amazon', 'merchant', 'expedited', 'standard',
        # States
        'maharashtra', 'karnataka', 'tamil', 'nadu', 'delhi', 'gujarat',
        'uttar', 'pradesh', 'bengal', 'telangana', 'rajasthan', 'kerala',
        'andhra', 'bihar', 'madhya', 'odisha', 'punjab', 'haryana',
        # Common analytics words
        'highest', 'lowest', 'best', 'worst', 'compare', 'versus',
        'between', 'distribution', 'fulfillment', 'fulfilment',
        'cancellation', 'performance', 'analytics', 'insights',
    ]

    # --- Injection / attack patterns ---
    INJECTION_PATTERNS = [
        r'(?:DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE)\s+',
        r';\s*--',
        r'UNION\s+SELECT',
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
        r'ignore (?:your |all |previous )?instructions',
        r'forget (?:your |all |previous )?instructions',
        r'you are now',
        r'act as',
        r'pretend to be',
        r'new persona',
        r'system prompt',
        r'reveal (?:your )?(?:prompt|instructions)',
    ]

    MAX_QUERY_LENGTH = 500

    def __init__(self, data_layer=None):
        self.data_layer = data_layer
        self._vocab_set = set(self.RETAIL_VOCABULARY)
        self._injection_re = re.compile(
            '|'.join(self.INJECTION_PATTERNS), re.IGNORECASE
        )
        # Cache for dynamic vocabulary from DB
        self._db_categories: Optional[List[str]] = None
        self._db_states: Optional[List[str]] = None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def preprocess(self, query: str) -> PreprocessResult:
        """
        Run the full preprocessing pipeline.

        Returns a PreprocessResult with the cleaned query and metadata
        about what was changed.
        """
        original = query
        corrections = []
        expansions = []
        flags: Dict[str, bool] = {}

        # Step 1 — Sanitize
        query, sanitized = self._sanitize(query)
        flags['sanitized'] = sanitized
        flags['injection_blocked'] = False

        if self._injection_re.search(query):
            logger.warning("Potential injection detected: %s", original[:80])
            flags['injection_blocked'] = True
            return PreprocessResult(
                original=original,
                normalized=original,  # Return untouched
                corrections=[],
                expansions=[],
                was_modified=False,
                sanitized=sanitized,
                flags=flags,
            )

        # Step 2 — Normalize
        query, abbr_expansions = self._normalize(query)
        expansions.extend(abbr_expansions)

        # Step 3 — Spell correction
        query, spell_corrections = self._spell_correct(query)
        corrections.extend(spell_corrections)

        # Step 4 — Synonym expansion
        query, syn_expansions = self._expand_synonyms(query)
        expansions.extend(syn_expansions)

        # Step 5 — Pattern normalization (strip filler words)
        query = self._normalize_patterns(query)

        # Final cleanup
        query = re.sub(r'\s+', ' ', query).strip()

        was_modified = query.lower() != original.lower()

        return PreprocessResult(
            original=original,
            normalized=query,
            corrections=corrections,
            expansions=expansions,
            was_modified=was_modified,
            sanitized=sanitized,
            flags=flags,
        )

    # ──────────────────────────────────────────────
    # Pipeline steps
    # ──────────────────────────────────────────────

    def _sanitize(self, query: str) -> Tuple[str, bool]:
        """Strip HTML, limit length, basic cleanup."""
        sanitized = False

        # Strip HTML entities and tags
        cleaned = html.unescape(query)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        if cleaned != query:
            sanitized = True

        # Limit length
        if len(cleaned) > self.MAX_QUERY_LENGTH:
            cleaned = cleaned[:self.MAX_QUERY_LENGTH]
            sanitized = True

        # Strip control characters
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)

        return cleaned, sanitized

    def _normalize(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Lowercase, strip whitespace, expand abbreviations."""
        normalized = query.strip()
        expansions = []

        # Expand abbreviations (word-boundary match)
        words = normalized.split()
        new_words = []
        for word in words:
            word_lower = word.lower().strip('?!.,;:')
            if word_lower in self.ABBREVIATIONS:
                expanded = self.ABBREVIATIONS[word_lower]
                expansions.append((word, expanded))
                new_words.append(expanded)
            else:
                new_words.append(word)

        normalized = ' '.join(new_words)
        return normalized, expansions

    def _spell_correct(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Fuzzy-match words against retail vocabulary."""
        corrections = []

        # Build extended vocabulary from DB if available
        vocab = self._get_extended_vocabulary()

        words = query.split()
        corrected_words = []

        # Words to skip (common English + already in vocabulary)
        skip_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'can',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'and', 'or', 'not', 'no', 'but', 'if', 'then', 'than',
            'that', 'this', 'these', 'those', 'it', 'its',
            'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'only', 'same', 'so', 'just',
            'show', 'me', 'my', 'give', 'get', 'see', 'find',
            'top', 'bottom', 'best', 'worst', 'last', 'first',
            'vs', 'versus', 'between', 'compare',
            'b2b', 'b2c', 'group', 'by', 'per',
        }

        for word in words:
            clean_word = re.sub(r'[?!.,;:]', '', word).lower()

            # Skip short words, numbers, and known words
            if (len(clean_word) < 3 or
                    clean_word in skip_words or
                    clean_word in vocab or
                    clean_word.isdigit()):
                corrected_words.append(word)
                continue

            # Find best match in vocabulary
            best_match, score = self._find_best_match(clean_word, vocab)

            if best_match and 0.70 <= score < 1.0:
                corrections.append((word, best_match))
                corrected_words.append(best_match)
                logger.info("Spell corrected: '%s' -> '%s' (%.0f%%)", word, best_match, score * 100)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words), corrections

    def _expand_synonyms(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Replace user-friendly terms with schema-aligned terms."""
        expanded = query
        expansions = []

        # Sort by length (longest first) to avoid partial replacements
        sorted_synonyms = sorted(self.SYNONYMS.items(), key=lambda x: -len(x[0]))

        for user_term, schema_term in sorted_synonyms:
            pattern = r'\b' + re.escape(user_term) + r'\b'
            if re.search(pattern, expanded, re.IGNORECASE):
                expanded = re.sub(pattern, schema_term, expanded, flags=re.IGNORECASE)
                expansions.append((user_term, schema_term))

        return expanded, expansions

    def _normalize_patterns(self, query: str) -> str:
        """Strip filler words and normalize common question patterns."""
        result = query
        for pattern, replacement in self.PATTERN_NORMALIZERS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _find_best_match(self, word: str, vocab: set) -> Tuple[Optional[str], float]:
        """Find the closest vocabulary word using SequenceMatcher."""
        best_match = None
        best_score = 0.0

        for v_word in vocab:
            # Quick length filter — skip if lengths differ too much
            if abs(len(word) - len(v_word)) > 3:
                continue

            score = SequenceMatcher(None, word, v_word).ratio()
            if score > best_score:
                best_score = score
                best_match = v_word

        return best_match, best_score

    def _get_extended_vocabulary(self) -> set:
        """Get vocabulary extended with actual DB values."""
        vocab = set(self._vocab_set)

        # Add categories from DB
        if self._db_categories is None and self.data_layer:
            try:
                result = self.data_layer.execute_query(
                    "SELECT DISTINCT category FROM sales WHERE category IS NOT NULL"
                )
                self._db_categories = [c.lower() for c in result['category'].tolist()]
            except Exception:
                self._db_categories = []

        if self._db_categories:
            vocab.update(self._db_categories)

        # Add states from DB
        if self._db_states is None and self.data_layer:
            try:
                result = self.data_layer.execute_query(
                    "SELECT DISTINCT state FROM sales WHERE state IS NOT NULL"
                )
                self._db_states = [s.lower() for s in result['state'].tolist()]
            except Exception:
                self._db_states = []

        if self._db_states:
            vocab.update(self._db_states)

        return vocab

    def refresh_vocabulary(self):
        """Force-refresh the vocabulary cache from the database."""
        self._db_categories = None
        self._db_states = None


# Module-level singleton
_preprocessor_instance: Optional[QueryPreprocessor] = None


def get_preprocessor(data_layer=None) -> QueryPreprocessor:
    """Get singleton preprocessor instance."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = QueryPreprocessor(data_layer=data_layer)
    return _preprocessor_instance
