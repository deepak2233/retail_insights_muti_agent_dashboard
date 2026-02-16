"""
Conversation Memory System — Production-grade multi-turn dialogue memory
with LRU response caching, semantic duplicate detection, topic tracking,
entity memory with decay, and session analytics.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque, OrderedDict
import json
import re
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    LRU cache for query responses.
    Avoids re-executing identical (normalized) queries.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        """Create a normalized cache key from a query string."""
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        normalized = ' '.join(sorted(normalized.split()))
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Look up a cached response. Returns None on miss."""
        key = self._make_key(query)
        if key in self._cache:
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry['timestamp'] < self.ttl_seconds:
                self._cache.move_to_end(key)
                self._hits += 1
                logger.info("Cache HIT for query: %s", query[:60])
                return entry['response']
            else:
                # Expired
                del self._cache[key]

        self._misses += 1
        return None

    def put(self, query: str, response: Dict[str, Any]):
        """Store a response in cache."""
        key = self._make_key(query)
        self._cache[key] = {
            'query': query,
            'response': response,
            'timestamp': time.time(),
        }
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def invalidate(self):
        """Clear the entire cache."""
        self._cache.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0.0,
        }


class ConversationMemory:
    """
    Production-grade conversation memory with:
    - Short-term memory (recent turns with deque)
    - Response cache (LRU with TTL)
    - Semantic duplicate detection
    - Topic tracking for drift detection
    - Entity memory with decay
    - Session analytics (latency, errors, cache stats)
    """

    def __init__(self, max_turns: int = 20, cache_ttl: int = 300):
        self.max_turns = max_turns
        self.short_term: deque = deque(maxlen=max_turns)
        self.entity_memory: Dict[str, Any] = {}
        self.topic_history: List[str] = []
        self.session_start = datetime.now()

        # Response cache
        self.response_cache = ResponseCache(max_size=100, ttl_seconds=cache_ttl)

        # Session analytics
        self._analytics = {
            'total_queries': 0,
            'errors': 0,
            'total_latency_ms': 0.0,
            'query_latencies': [],
        }

    # ──────────────────────────────────────────────
    # Core: Turn Management
    # ──────────────────────────────────────────────

    def add_turn(self, question: str, answer: str, sql: Optional[str] = None,
                 entities: Optional[Dict] = None, metadata: Optional[Dict] = None,
                 intent: Optional[str] = None):
        """Add a conversation turn to memory."""
        turn = {
            "turn_id": self._analytics['total_queries'] + 1,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:800],
            "sql": sql,
            "entities": entities or {},
            "metadata": metadata or {},
            "intent": intent,
        }

        self.short_term.append(turn)
        self._analytics['total_queries'] += 1

        # Update entity memory
        if entities:
            self._update_entity_memory(entities)

        # Track topic
        if intent:
            self.topic_history.append(intent)
            if len(self.topic_history) > 20:
                self.topic_history = self.topic_history[-20:]

        # Cache the response for this query
        if sql and answer:
            self.response_cache.put(question, {
                'answer': answer,
                'sql': sql,
                'entities': entities,
            })

    # ──────────────────────────────────────────────
    # Duplicate Detection (Semantic)
    # ──────────────────────────────────────────────

    def is_duplicate(self, question: str, threshold: float = 0.90) -> bool:
        """
        Semantic duplicate detection — checks against all recent turns,
        not just the last one. Uses normalized word-set comparison
        combined with SequenceMatcher for near-misses.
        """
        if not self.short_term:
            return False

        normalized_new = self._normalize_for_comparison(question)

        for turn in reversed(list(self.short_term)[-5:]):
            normalized_old = self._normalize_for_comparison(turn["question"])

            # Exact match after normalization
            if normalized_new == normalized_old:
                return True

            # Fuzzy match
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, normalized_new, normalized_old).ratio()
            if ratio >= threshold:
                return True

        return False

    def get_cached_response(self, question: str) -> Optional[Dict[str, Any]]:
        """Check if we have a cached response for this query."""
        return self.response_cache.get(question)

    # ──────────────────────────────────────────────
    # Context Retrieval
    # ──────────────────────────────────────────────

    def get_recent_context(self, n_turns: int = 3, max_chars: int = 2000) -> str:
        """Get recent conversation context with character limit optimization."""
        if not self.short_term:
            return "No previous conversation context."

        recent = list(self.short_term)[-n_turns:]
        context_parts = ["Recent conversation:"]
        current_len = 0

        for turn in reversed(recent):
            user_msg = f"User: {turn['question']}"
            short_answer = turn['answer'][:300] + "..." if len(turn['answer']) > 300 else turn['answer']
            ai_msg = f"Assistant: {short_answer}"

            turn_text = f"{user_msg}\n{ai_msg}"
            if current_len + len(turn_text) > max_chars:
                break

            context_parts.insert(1, turn_text)
            current_len += len(turn_text)

        return "\n".join(context_parts)

    def get_entity_context(self) -> Dict[str, Any]:
        """Get all remembered entities with recency weighting."""
        now = datetime.now()
        weighted_entities = {}

        for key, info in self.entity_memory.items():
            last_mentioned = datetime.fromisoformat(info['last_mentioned'])
            age_minutes = (now - last_mentioned).total_seconds() / 60

            # Decay: entities older than 30 min get lower priority
            weight = max(0.1, 1.0 - (age_minutes / 60.0))
            weighted_entities[key] = {
                'value': info['value'],
                'weight': round(weight, 2),
                'mention_count': info.get('mention_count', 1),
            }

        return weighted_entities

    # ──────────────────────────────────────────────
    # Topic Tracking
    # ──────────────────────────────────────────────

    def get_topic_history(self) -> List[str]:
        """Get the recent topic/intent history for drift detection."""
        return list(self.topic_history)

    def get_current_topic(self) -> Optional[str]:
        """Get the most recent topic."""
        return self.topic_history[-1] if self.topic_history else None

    # ──────────────────────────────────────────────
    # Reference Resolution
    # ──────────────────────────────────────────────

    def resolve_reference(self, question: str) -> str:
        """Resolve pronouns and references in follow-up questions."""
        resolved = question

        reference_patterns = [
            (r'\b(it|that|this)\b', self._get_last_topic),
            (r'\b(them|those|these)\b', self._get_last_plural_topic),
            (r'\b(same|again)\b', self._get_last_query_context),
            (r'\bjust the top (\d+)\b', self._add_limit_context),
            (r'\b(more details?|elaborate)\b', self._get_expansion_context),
        ]

        for pattern, resolver in reference_patterns:
            if re.search(pattern, question.lower()):
                context = resolver()
                if context:
                    resolved = f"{question} [Context: {context}]"
                    break

        return resolved

    # ──────────────────────────────────────────────
    # Session Analytics
    # ──────────────────────────────────────────────

    def record_query_latency(self, latency_ms: float):
        """Record a query's latency."""
        self._analytics['total_latency_ms'] += latency_ms
        self._analytics['query_latencies'].append(latency_ms)
        # Keep last 100
        if len(self._analytics['query_latencies']) > 100:
            self._analytics['query_latencies'] = self._analytics['query_latencies'][-100:]

    def record_error(self):
        """Record an error."""
        self._analytics['errors'] += 1

    def get_session_analytics(self) -> Dict[str, Any]:
        """Get comprehensive session analytics."""
        duration = datetime.now() - self.session_start
        latencies = self._analytics['query_latencies']
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            'total_queries': self._analytics['total_queries'],
            'errors': self._analytics['errors'],
            'error_rate': (
                self._analytics['errors'] / self._analytics['total_queries']
                if self._analytics['total_queries'] > 0 else 0.0
            ),
            'avg_latency_ms': round(avg_latency, 1),
            'p95_latency_ms': round(
                sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1
            ),
            'cache': self.response_cache.stats,
            'session_duration_min': round(duration.total_seconds() / 60, 1),
            'turns_in_memory': len(self.short_term),
            'entities_tracked': len(self.entity_memory),
        }

    # ──────────────────────────────────────────────
    # Conversation Summary & Compaction
    # ──────────────────────────────────────────────

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation session."""
        if not self.short_term:
            return {"turns": 0, "topics": [], "duration": "0 minutes"}

        turns = list(self.short_term)
        topics = set()
        for turn in turns:
            entities = turn.get("entities", {})
            for key, value in entities.items():
                if value:
                    topics.add(f"{key}: {value}")

        duration = datetime.now() - self.session_start
        minutes = int(duration.total_seconds() / 60)

        return {
            "turns": len(turns),
            "topics": list(topics),
            "duration": f"{minutes} minutes",
            "entities_remembered": list(self.entity_memory.keys()),
            "analytics": self.get_session_analytics(),
        }

    def get_compact_context(self, max_turns: int = 5) -> str:
        """
        Get a compacted context string from older turns.
        Summarizes old turns instead of dropping them entirely.
        """
        if len(self.short_term) <= max_turns:
            return self.get_recent_context(n_turns=max_turns)

        turns = list(self.short_term)
        old_turns = turns[:-max_turns]
        recent_turns = turns[-max_turns:]

        # Compact old turns into a summary
        old_topics = set()
        old_entities = {}
        for turn in old_turns:
            entities = turn.get("entities", {})
            for k, v in entities.items():
                if v:
                    old_topics.add(turn.get("question", "")[:50])
                    old_entities[k] = v

        summary = f"Earlier in this conversation, we discussed: {', '.join(list(old_topics)[:5])}"
        if old_entities:
            summary += f"\nEntities mentioned: {old_entities}"

        # Add recent turns in full
        recent_context = []
        for turn in recent_turns:
            short_answer = turn['answer'][:200] + "..." if len(turn['answer']) > 200 else turn['answer']
            recent_context.append(f"User: {turn['question']}\nAssistant: {short_answer}")

        return f"{summary}\n\nRecent conversation:\n" + "\n".join(recent_context)

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for duplicate comparison."""
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = sorted(text.split())
        return ' '.join(words)

    def _update_entity_memory(self, entities: Dict):
        """Update remembered entities with mention counting."""
        for key, value in entities.items():
            if value:
                existing = self.entity_memory.get(key, {})
                self.entity_memory[key] = {
                    "value": value,
                    "last_mentioned": datetime.now().isoformat(),
                    "mention_count": existing.get("mention_count", 0) + 1,
                }

    def _get_last_topic(self) -> Optional[str]:
        if not self.short_term:
            return None
        last_turn = self.short_term[-1]
        entities = last_turn.get("entities", {})
        for key in ["category", "state", "product", "metric"]:
            if key in entities and entities[key]:
                return f"{key}={entities[key]}"
        return None

    def _get_last_plural_topic(self) -> Optional[str]:
        if not self.short_term:
            return None
        last_turn = self.short_term[-1]
        entities = last_turn.get("entities", {})
        for key in ["categories", "states", "products"]:
            if key in entities:
                return f"{key}={entities[key]}"
        sql = last_turn.get("sql", "")
        if sql and "GROUP BY" in sql.upper():
            match = re.search(r'GROUP BY\s+(\w+)', sql, re.IGNORECASE)
            if match:
                return f"grouped by {match.group(1)}"
        return None

    def _get_last_query_context(self) -> Optional[str]:
        if not self.short_term:
            return None
        return self.short_term[-1].get("sql")

    def _add_limit_context(self) -> Optional[str]:
        if not self.short_term:
            return None
        sql = self.short_term[-1].get("sql", "")
        return f"Previous query: {sql}"

    def _get_expansion_context(self) -> Optional[str]:
        if not self.short_term:
            return None
        return f"Expand on: {self.short_term[-1]['question']}"

    # ──────────────────────────────────────────────
    # Serialization & Lifecycle
    # ──────────────────────────────────────────────

    def clear(self):
        """Clear all memory."""
        self.short_term.clear()
        self.entity_memory.clear()
        self.topic_history.clear()
        self.response_cache.invalidate()
        self._analytics = {
            'total_queries': 0,
            'errors': 0,
            'total_latency_ms': 0.0,
            'query_latencies': [],
        }
        self.session_start = datetime.now()

    def to_dict(self) -> Dict:
        """Serialize memory to dictionary."""
        return {
            "short_term": list(self.short_term),
            "entity_memory": self.entity_memory,
            "topic_history": self.topic_history,
            "session_start": self.session_start.isoformat(),
            "analytics": self._analytics,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationMemory":
        """Deserialize memory from dictionary."""
        memory = cls()
        for turn in data.get("short_term", []):
            memory.short_term.append(turn)
        memory.entity_memory = data.get("entity_memory", {})
        memory.topic_history = data.get("topic_history", [])
        memory.session_start = datetime.fromisoformat(
            data.get("session_start", datetime.now().isoformat())
        )
        return memory


# Singleton for session persistence
_memory_instance: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Get singleton memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance


def reset_memory():
    """Reset memory instance."""
    global _memory_instance
    if _memory_instance:
        _memory_instance.clear()
    _memory_instance = ConversationMemory()
