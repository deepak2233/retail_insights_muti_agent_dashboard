"""
Conversation Memory System for Multi-Turn Dialogue
Enables context-aware follow-up questions and personalization
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import json
import re


class ConversationMemory:
    """
    Manages conversation history for context-aware responses.
    Supports short-term (session) and summarized context.
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize memory system
        
        Args:
            max_turns: Maximum conversation turns to keep in short-term memory
        """
        self.max_turns = max_turns
        self.short_term: deque = deque(maxlen=max_turns)
        self.context_cache: Dict[str, Any] = {}
        self.entity_memory: Dict[str, Any] = {}  # Remembered entities
        self.session_start = datetime.now()
        
    def add_turn(self, question: str, answer: str, sql: Optional[str] = None,
                 entities: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """
        Add a conversation turn to memory
        
        Args:
            question: User's question
            answer: System's response
            sql: Generated SQL query (if any)
            entities: Extracted entities (regions, categories, etc.)
            metadata: Additional metadata (timing, confidence, etc.)
        """
        turn = {
            "turn_id": len(self.short_term) + 1,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:500],  # Truncate long answers
            "sql": sql,
            "entities": entities or {},
            "metadata": metadata or {}
        }
        
        self.short_term.append(turn)
        
        # Update entity memory with any new entities
        if entities:
            self._update_entity_memory(entities)
    
    def _update_entity_memory(self, entities: Dict):
        """Update remembered entities from latest query"""
        for key, value in entities.items():
            if value:  # Only store non-empty values
                self.entity_memory[key] = {
                    "value": value,
                    "last_mentioned": datetime.now().isoformat()
                }
    
    def get_recent_context(self, n_turns: int = 3) -> str:
        """
        Get recent conversation context as formatted string
        
        Args:
            n_turns: Number of recent turns to include
            
        Returns:
            Formatted conversation history
        """
        if not self.short_term:
            return "No previous conversation context."
        
        recent = list(self.short_term)[-n_turns:]
        
        context_parts = ["Recent conversation:"]
        for turn in recent:
            context_parts.append(f"User: {turn['question']}")
            # Truncate answer for context
            short_answer = turn['answer'][:200] + "..." if len(turn['answer']) > 200 else turn['answer']
            context_parts.append(f"Assistant: {short_answer}")
        
        return "\n".join(context_parts)
    
    def get_referenced_entities(self) -> Dict[str, Any]:
        """Get all entities mentioned in the conversation"""
        return self.entity_memory.copy()
    
    def resolve_reference(self, question: str) -> str:
        """
        Resolve pronouns and references in follow-up questions
        
        Args:
            question: User's question with potential references
            
        Returns:
            Question with resolved references
        """
        resolved = question
        
        # Common reference patterns
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
    
    def _get_last_topic(self) -> Optional[str]:
        """Get the main topic from the last query"""
        if not self.short_term:
            return None
        
        last_turn = self.short_term[-1]
        entities = last_turn.get("entities", {})
        
        # Priority order for topic resolution
        for key in ["category", "state", "product", "metric"]:
            if key in entities and entities[key]:
                return f"{key}={entities[key]}"
        
        return None
    
    def _get_last_plural_topic(self) -> Optional[str]:
        """Get plural topic from last query"""
        if not self.short_term:
            return None
        
        last_turn = self.short_term[-1]
        entities = last_turn.get("entities", {})
        
        for key in ["categories", "states", "products"]:
            if key in entities:
                return f"{key}={entities[key]}"
        
        # Check if last query was about multiple items
        sql = last_turn.get("sql", "")
        if sql and "GROUP BY" in sql.upper():
            # Extract grouped column
            match = re.search(r'GROUP BY\s+(\w+)', sql, re.IGNORECASE)
            if match:
                return f"grouped by {match.group(1)}"
        
        return None
    
    def _get_last_query_context(self) -> Optional[str]:
        """Get context from the last SQL query"""
        if not self.short_term:
            return None
        
        last_turn = self.short_term[-1]
        return last_turn.get("sql")
    
    def _add_limit_context(self) -> Optional[str]:
        """Add limit context from previous query"""
        if not self.short_term:
            return None
        
        last_turn = self.short_term[-1]
        sql = last_turn.get("sql", "")
        
        # Remove existing LIMIT and add new one based on context
        return f"Previous query: {sql}"
    
    def _get_expansion_context(self) -> Optional[str]:
        """Get context for expanding previous answer"""
        if not self.short_term:
            return None
        
        last_turn = self.short_term[-1]
        return f"Expand on: {last_turn['question']}"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation session"""
        if not self.short_term:
            return {"turns": 0, "topics": [], "duration": "0 minutes"}
        
        turns = list(self.short_term)
        
        # Extract topics discussed
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
            "entities_remembered": list(self.entity_memory.keys())
        }
    
    def clear(self):
        """Clear all memory"""
        self.short_term.clear()
        self.context_cache.clear()
        self.entity_memory.clear()
        self.session_start = datetime.now()
    
    def to_dict(self) -> Dict:
        """Serialize memory to dictionary"""
        return {
            "short_term": list(self.short_term),
            "entity_memory": self.entity_memory,
            "session_start": self.session_start.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationMemory":
        """Deserialize memory from dictionary"""
        memory = cls()
        for turn in data.get("short_term", []):
            memory.short_term.append(turn)
        memory.entity_memory = data.get("entity_memory", {})
        memory.session_start = datetime.fromisoformat(data.get("session_start", datetime.now().isoformat()))
        return memory


# Singleton for session persistence
_memory_instance: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Get singleton memory instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance


def reset_memory():
    """Reset memory instance"""
    global _memory_instance
    if _memory_instance:
        _memory_instance.clear()
    _memory_instance = ConversationMemory()
