"""
Agent Orchestrator — Production-grade LangGraph pipeline
with query preprocessing, response caching, circuit breaker,
structured logging, per-node timing, and graceful degradation.
"""
from typing import Dict, Any, Optional
import time
import logging

from langgraph.graph import StateGraph, END
from agents.query_agent import QueryResolutionAgent, AgentState
from agents.extraction_agent import DataExtractionAgent
from agents.validation_agent import ValidationAgent
from agents.response_agent import ResponseAgent
from agents.router_agent import RouterAgent

# Enhancement modules — imported with graceful fallback
from utils.memory import get_memory, ConversationMemory
from utils.edge_cases import get_edge_case_handler, EdgeCaseHandler
from utils.hallucination_prevention import FactExtractor, GroundedResponseGenerator
from utils.evaluation import get_evaluation_framework, EvaluationFramework

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# Circuit Breaker
# ────────────────────────────────────────────────

class CircuitBreaker:
    """
    Simple circuit breaker for LLM calls.
    After `threshold` consecutive failures the circuit "opens" and
    all calls are fast-failed for `cooldown_seconds`.
    """

    def __init__(self, threshold: int = 3, cooldown_seconds: int = 60):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._open_since: Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self._open_since is None:
            return False
        if time.time() - self._open_since >= self.cooldown_seconds:
            # half-open → allow one attempt
            self._reset()
            return False
        return True

    def record_success(self):
        self._consecutive_failures = 0
        self._open_since = None

    def record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.threshold:
            self._open_since = time.time()
            logger.warning(
                "Circuit breaker OPEN after %d consecutive failures — "
                "bypassing LLM for %ds",
                self.threshold, self.cooldown_seconds,
            )

    def _reset(self):
        self._consecutive_failures = 0
        self._open_since = None
        logger.info("Circuit breaker reset (cooldown elapsed)")


# ────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Production-grade orchestrator with:
    - Query preprocessor as first pipeline step
    - Response cache check (skip SQL on cache hit)
    - Topic-shift handling via memory
    - Circuit breaker for resilient LLM calls
    - Structured logging (logging module, no print())
    - Per-node timing instrumentation
    - Graceful degradation for optional modules
    - Pipeline flag tracking for UI visibility
    """

    def __init__(self, enable_memory: bool = True, enable_evaluation: bool = True):
        # Core agents
        self.query_agent = QueryResolutionAgent()
        self.extraction_agent = DataExtractionAgent()
        self.validation_agent = ValidationAgent()
        self.response_agent = ResponseAgent()
        self.router_agent = RouterAgent()

        # Enhancement modules (with graceful degradation)
        self.memory: Optional[ConversationMemory] = self._safe_init(
            "memory", lambda: get_memory() if enable_memory else None
        )
        self.edge_case_handler: Optional[EdgeCaseHandler] = self._safe_init(
            "edge_cases", get_edge_case_handler
        )
        self.fact_extractor = self._safe_init("fact_extractor", FactExtractor)
        self.grounded_generator = self._safe_init(
            "grounded_generator", GroundedResponseGenerator
        )
        self.evaluation: Optional[EvaluationFramework] = self._safe_init(
            "evaluation",
            lambda: get_evaluation_framework() if enable_evaluation else None,
        )

        # Preprocessor (new)
        self.preprocessor = self._safe_init(
            "query_preprocessor", self._init_preprocessor
        )

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(threshold=3, cooldown_seconds=60)

        # Configuration
        self.enable_memory = enable_memory
        self.enable_evaluation = enable_evaluation
        self.retry_on_error = True
        self.max_retries = 3

        # Per-node timing accumulator (surfaced in evaluation dashboard)
        self._node_timings: Dict[str, list] = {}

        # Pipeline flags — reset per query, surfaced in UI
        self._pipeline_flags: Dict[str, Any] = {}

        # Build the graph
        self.graph = self._build_graph()

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _safe_init(name: str, factory):
        """Initialise a module; return None on failure."""
        try:
            return factory()
        except Exception as exc:
            logger.warning("Optional module '%s' failed to initialise: %s", name, exc)
            return None

    @staticmethod
    def _init_preprocessor():
        from utils.query_preprocessor import get_preprocessor
        return get_preprocessor()

    def _timed(self, node_name: str, func, *args, **kwargs):
        """Execute *func* and record wall-clock time under *node_name*."""
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - t0) * 1000
        self._node_timings.setdefault(node_name, []).append(elapsed_ms)
        logger.info("[%s] completed in %.1f ms", node_name, elapsed_ms)
        return result

    # ──────────────────────────────────────────
    # Graph Construction
    # ──────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """Build the enhanced agent workflow graph."""

        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("route_intent", self._route_intent_node)
        workflow.add_node("resolve_query", self._resolve_query_node)
        workflow.add_node("extract_data", self._extract_data_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("extract_facts", self._extract_facts_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("postprocess", self._postprocess_node)
        workflow.add_node("handle_error", self._handle_error_node)
        workflow.add_node("handle_edge_case", self._handle_edge_case_node)

        # Entry
        workflow.set_entry_point("preprocess")

        # Edges
        workflow.add_conditional_edges(
            "preprocess",
            self._route_after_preprocess,
            {
                "resolve": "route_intent",
                "edge_case": "handle_edge_case",
                "cached": "postprocess",
            },
        )

        workflow.add_conditional_edges(
            "route_intent",
            self._route_based_on_intent,
            {
                "analytics": "resolve_query",
                "direct_response": "postprocess",
            },
        )

        workflow.add_edge("resolve_query", "extract_data")

        workflow.add_conditional_edges(
            "extract_data",
            self._should_validate,
            {
                "validate": "validate",
                "retry": "resolve_query",
                "error": "handle_error",
            },
        )

        workflow.add_conditional_edges(
            "validate",
            self._should_extract_facts,
            {
                "extract_facts": "extract_facts",
                "error": "handle_error",
            },
        )

        workflow.add_edge("extract_facts", "generate_response")
        workflow.add_edge("generate_response", "postprocess")

        # Terminal edges
        workflow.add_edge("postprocess", END)
        workflow.add_edge("handle_error", END)
        workflow.add_edge("handle_edge_case", END)

        return workflow.compile()

    # ──────────────────────────────────────────
    # Pipeline Nodes
    # ──────────────────────────────────────────

    def _preprocess_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 0 — Preprocessing pipeline:
        1. Query preprocessor (spell-correct, normalize, synonyms, sanitize)
        2. Reference resolution from memory
        3. Cache check (skip full pipeline on hit)
        4. Edge-case detection
        """
        def _inner(state):
            question = state["question"]
            original_question = question
            logger.info("Preprocessing: %s", question[:120])

            # Reset pipeline flags for this query
            self._pipeline_flags = {
                "cache_hit": False,
                "spell_corrected": False,
                "original_query": original_question,
                "normalized_query": question,
                "guardrail_blocked": False,
                "guardrail_type": None,
                "topic_shift": False,
                "duplicate_detected": False,
                "circuit_breaker_open": False,
                "intent": "analytics",
                "confidence": 0.0,
                "entity_context_used": False,
            }

            # ── 1. Query preprocessor ──
            if self.preprocessor:
                try:
                    preprocess_result = self.preprocessor.preprocess(question)
                    if preprocess_result.was_modified:
                        logger.info(
                            "Preprocessor modified query: %s → %s",
                            question[:60], preprocess_result.normalized[:60],
                        )
                        self._pipeline_flags["spell_corrected"] = True
                    question = preprocess_result.normalized
                    self._pipeline_flags["normalized_query"] = question

                    # Check injection flag from preprocessor
                    if preprocess_result.flags.get("injection_detected"):
                        logger.warning("Injection blocked by preprocessor")
                        self._pipeline_flags["guardrail_blocked"] = True
                        self._pipeline_flags["guardrail_type"] = "injection"
                        return {
                            **state,
                            "edge_case_handled": True,
                            "final_answer": (
                                "I'm your Retail Insights Assistant, focused on "
                                "helping you analyze sales data. How can I help "
                                "with your retail analytics?"
                            ),
                            "error": None,
                        }
                except Exception as exc:
                    logger.warning("Preprocessor failed (non-fatal): %s", exc)

            # ── 2. Memory-based reference resolution ──
            context = None
            if self.memory:
                try:
                    context = self.memory.get_recent_context(n_turns=3)
                    resolved = self.memory.resolve_reference(question)
                    if resolved != question:
                        logger.info("Resolved reference: %s", resolved[:80])
                        question = resolved
                except Exception as exc:
                    logger.warning("Memory context retrieval failed: %s", exc)

            # ── 3. Cache check ──
            if self.memory:
                try:
                    cached = self.memory.get_cached_response(question)
                    if cached:
                        logger.info("Cache HIT — skipping full pipeline")
                        self._pipeline_flags["cache_hit"] = True
                        return {
                            **state,
                            "question": question,
                            "conversation_context": context,
                            "edge_case_handled": False,
                            "final_answer": cached.get("answer", ""),
                            "_cache_hit": True,
                            "report_content": state.get("report_content"),
                        }
                except Exception as exc:
                    logger.warning("Cache lookup failed: %s", exc)

            # ── 4. Edge-case detection ──
            if self.edge_case_handler:
                try:
                    edge_result = self.edge_case_handler.handle(question)
                    if edge_result.is_edge_case:
                        logger.info("Edge case: %s", edge_result.edge_case_type)
                        if edge_result.requires_clarification:
                            self._pipeline_flags["guardrail_blocked"] = True
                            self._pipeline_flags["guardrail_type"] = edge_result.edge_case_type
                            return {
                                **state,
                                "edge_case_handled": True,
                                "final_answer": self._format_edge_case_response(edge_result),
                                "error": None,
                            }
                        if edge_result.modified_question:
                            question = edge_result.modified_question
                except Exception as exc:
                    logger.warning("Edge-case handler failed: %s", exc)

            return {
                **state,
                "question": question,
                "conversation_context": context,
                "edge_case_handled": False,
                "report_content": state.get("report_content"),
            }

        return self._timed("preprocess", _inner, state)

    def _route_intent_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 1 — Intent classification with duplicate detection."""
        def _inner(state):
            question = state["question"]
            logger.info("Agent 0: Intent Routing — %s", question[:80])

            # Duplicate detection
            if self.memory and self.memory.is_duplicate(question):
                logger.info("Duplicate query — reusing previous answer")
                self._pipeline_flags["duplicate_detected"] = True
                self._pipeline_flags["intent"] = "duplicate"
                last_turn = self.memory.short_term[-1]
                return {
                    **state,
                    "intent": "duplicate",
                    "final_answer": last_turn["answer"],
                }

            # Classify via router (pass memory for topic-shift awareness)
            result = self.router_agent.classify(
                question,
                report_content=state.get("report_content"),
                memory=self.memory,
            )

            intent = result.get("intent", "analytics")
            confidence = result.get("confidence", 0.0)
            logger.info(
                "Intent: %s  confidence: %.2f  reasoning: %s",
                intent, confidence, result.get("reasoning", "")[:80],
            )

            # Update pipeline flags
            self._pipeline_flags["intent"] = intent
            self._pipeline_flags["confidence"] = confidence
            if result.get("topic_shift"):
                self._pipeline_flags["topic_shift"] = True
            if intent != "analytics":
                self._pipeline_flags["guardrail_blocked"] = True
                self._pipeline_flags["guardrail_type"] = intent

            return {
                **state,
                "intent": intent,
                "final_answer": result.get("response_if_not_analytics", ""),
            }

        return self._timed("route_intent", _inner, state)

    def _route_based_on_intent(self, state: AgentState) -> str:
        intent = state.get("intent", "analytics")
        return "analytics" if intent == "analytics" else "direct_response"

    def _resolve_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 2 — NL-to-SQL with circuit breaker protection."""
        def _inner(state):
            question = state["question"]
            context = state.get("conversation_context")
            error_history = state.get("_error_history", [])
            logger.info("Agent 1: Query Resolution")

            # Circuit breaker check
            if self.circuit_breaker.is_open:
                logger.warning("Circuit breaker OPEN — returning fallback SQL")
                self._pipeline_flags["circuit_breaker_open"] = True
                from agents.query_agent import QueryIntent
                return {
                    **state,
                    "query_intent": QueryIntent(
                        intent_type="fallback",
                        entities={},
                        sql_query=(
                            "SELECT category, COUNT(*) as count, "
                            "SUM(revenue) as revenue FROM sales "
                            "GROUP BY category ORDER BY revenue DESC LIMIT 10"
                        ),
                        explanation=(
                            "LLM is temporarily unavailable. Showing top "
                            "categories by revenue as a fallback."
                        ),
                    ),
                }

            try:
                # Entity context from memory for context-aware SQL
                entity_context = ""
                if self.memory:
                    try:
                        entities = self.memory.get_entity_context()
                        if entities:
                            parts = [f"{k}={v['value']}" for k, v in entities.items()
                                     if v.get("weight", 0) > 0.3]
                            if parts:
                                entity_context = (
                                    "\nActive context from conversation: "
                                    + ", ".join(parts)
                                )
                                self._pipeline_flags["entity_context_used"] = True
                    except Exception:
                        pass

                full_context = (context or "") + entity_context

                if self.retry_on_error and error_history:
                    query_intent = self.query_agent.resolve_with_retry(
                        question,
                        max_retries=self.max_retries,
                        context=full_context if full_context.strip() else None,
                        error_history=error_history,
                    )
                else:
                    query_intent = self.query_agent.resolve_query(
                        question,
                        full_context if full_context.strip() else None,
                    )

                self.circuit_breaker.record_success()
                return {**state, "query_intent": query_intent}

            except Exception as exc:
                self.circuit_breaker.record_failure()
                logger.error("Query resolution failed: %s", exc)
                return {**state, "error": str(exc)}

        return self._timed("resolve_query", _inner, state)

    def _extract_data_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 3 — Execute SQL via extraction agent."""
        def _inner(state):
            logger.info("Agent 2: Data Extraction")
            result = self.extraction_agent.extract_data(state)
            if result.get("error"):
                error_history = state.get("_error_history", [])
                error_history.append(result["error"])
                result["_error_history"] = error_history
                result["_retry_count"] = state.get("_retry_count", 0) + 1
                logger.warning("Extraction error (retry %d): %s",
                               result["_retry_count"], result["error"][:120])
            return result

        return self._timed("extract_data", _inner, state)

    def _validate_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 4 — Result validation & confidence scoring."""
        def _inner(state):
            logger.info("Agent 3: Validation")
            result = self.validation_agent.validate(state)
            confidence = result.get("confidence_scores", {})
            if confidence:
                logger.info("Confidence: %.1f%%", confidence.get("overall", 0) * 100)
            return result

        return self._timed("validate", _inner, state)

    def _extract_facts_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 5 — Extract verifiable facts for hallucination prevention."""
        def _inner(state):
            logger.info("Fact Extraction")
            query_result = state.get("query_result", {})
            df = query_result.get("dataframe")
            query_intent = state.get("query_intent")

            facts = []
            if df is not None and not df.empty and self.fact_extractor:
                try:
                    facts = self.fact_extractor.extract_facts(df, query_intent)
                    logger.info("Extracted %d verifiable facts", len(facts))
                except Exception as exc:
                    logger.warning("Fact extraction failed (non-fatal): %s", exc)

            return {**state, "facts": facts}

        return self._timed("extract_facts", _inner, state)

    def _generate_response_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 6 — Generate grounded NL response."""
        def _inner(state):
            logger.info("Agent 4: Response Generation")
            result = self.response_agent.generate_response(state)

            # Validate against facts
            facts = state.get("facts", [])
            if facts and result.get("final_answer") and self.grounded_generator:
                try:
                    validation_result = self.grounded_generator.validate_and_annotate(
                        result["final_answer"], facts,
                    )
                    if validation_result["issues"]:
                        logger.warning(
                            "Grounding issues: %d", len(validation_result["issues"]),
                        )
                    if validation_result["confidence"] < 0.8:
                        result["final_answer"] += (
                            f"\n\nNote: Some insights may require verification "
                            f"(confidence: {validation_result['confidence']*100:.0f}%)"
                        )
                except Exception as exc:
                    logger.warning("Grounded validation failed (non-fatal): %s", exc)

            return result

        return self._timed("generate_response", _inner, state)

    def _postprocess_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 7 — Update memory, run evaluation, record analytics."""
        def _inner(state):
            logger.info("Postprocessing")

            # ── Update memory ──
            if self.memory and state.get("final_answer"):
                try:
                    entities = {}
                    intent_str = state.get("intent")
                    if state.get("query_intent"):
                        qi = state["query_intent"]
                        entities = qi.entities if hasattr(qi, "entities") else {}

                    self.memory.add_turn(
                        question=state["question"],
                        answer=state["final_answer"],
                        sql=(
                            state["query_intent"].sql_query
                            if state.get("query_intent")
                            else None
                        ),
                        entities=entities,
                        metadata={
                            "confidence": state.get("confidence_scores", {}).get("overall", 0),
                        },
                        intent=intent_str,
                    )
                    logger.info("Saved to conversation memory")
                except Exception as exc:
                    logger.warning("Memory update failed: %s", exc)

            # ── Run evaluation ──
            if self.evaluation and state.get("final_answer"):
                try:
                    query_result = state.get("query_result", {})
                    df = query_result.get("dataframe")
                    if df is not None:
                        eval_result = self.evaluation.evaluate_response(
                            question=state["question"],
                            sql=(
                                state["query_intent"].sql_query
                                if state.get("query_intent")
                                else ""
                            ),
                            result_df=df,
                            response=state["final_answer"],
                            facts=state.get("facts", []),
                        )
                        logger.info(
                            "Evaluation: accuracy=%.2f  faithfulness=%.2f  overall=%.2f",
                            eval_result.accuracy_score,
                            eval_result.faithfulness_score,
                            eval_result.overall_score,
                        )
                except Exception as exc:
                    logger.warning("Evaluation failed (non-fatal): %s", exc)

            return state

        return self._timed("postprocess", _inner, state)

    def _handle_error_node(self, state: AgentState) -> Dict[str, Any]:
        """Terminal node — graceful error response."""
        error = state.get("error", "Unknown error occurred")
        logger.error("Error handler invoked: %s", error[:200])

        # Record error in memory analytics
        if self.memory:
            try:
                self.memory.record_error()
            except Exception:
                pass

        suggestions = [
            "Try rephrasing your question",
            "Ask about specific metrics like revenue, orders, or profit",
            "Specify a time period or region",
            "Ask 'What data is available?' to see options",
        ]

        response = (
            "I apologize, but I encountered an issue processing your request.\n\n"
            f"**Error:** {error}\n\n"
            "**Suggestions:**\n"
            + "\n".join(f"* {s}" for s in suggestions)
            + "\n\n**Example questions you can ask:**\n"
            "* What is the total revenue?\n"
            "* Show me the top 5 categories by sales\n"
            "* Compare B2B and B2C orders\n"
        )

        return {**state, "final_answer": response}

    def _handle_edge_case_node(self, state: AgentState) -> Dict[str, Any]:
        """Terminal node — edge-case response was already set in preprocess."""
        logger.info("Edge-case handler (response pre-set)")
        return state

    # ──────────────────────────────────────────
    # Routing helpers
    # ──────────────────────────────────────────

    def _format_edge_case_response(self, edge_result) -> str:
        response = f"**{edge_result.message}**\n\n"
        if edge_result.suggestions:
            response += "**Try one of these instead:**\n"
            for s in edge_result.suggestions[:5]:
                response += f"* {s}\n"
        return response

    def _route_after_preprocess(self, state: AgentState) -> str:
        if state.get("edge_case_handled"):
            return "edge_case"
        if state.get("_cache_hit"):
            return "cached"
        return "resolve"

    def _should_validate(self, state: AgentState) -> str:
        if state.get("error"):
            retry_count = state.get("_retry_count", 0)
            if self.retry_on_error and retry_count < self.max_retries:
                logger.info("Retrying... (attempt %d/%d)", retry_count + 1, self.max_retries)
                return "retry"
            return "error"
        return "validate"

    def _should_extract_facts(self, state: AgentState) -> str:
        return "extract_facts" if state.get("validation_passed") else "error"

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def process_query(self, question: str, report_content: Optional[str] = None) -> str:
        """
        Process a user question through the full agent pipeline.
        Returns just the answer string (backward compatible).
        """
        result = self.process_query_with_metadata(question, report_content)
        return result["answer"]

    def process_query_with_metadata(
        self, question: str, report_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user question and return rich metadata for the UI.

        Returns:
            Dict with keys: answer, pipeline_flags, timings, sql_query, chart_data
        """
        t0 = time.time()
        try:
            logger.info("=" * 80)
            logger.info("Question: %s", question)
            logger.info("=" * 80)

            initial_state = AgentState(
                question=question,
                query_intent=None,
                query_result=None,
                validation_passed=False,
                final_answer="",
                error=None,
                confidence_scores=None,
                conversation_context=None,
                edge_case_handled=None,
                facts=None,
                report_content=report_content,
            )

            final_state = self.graph.invoke(initial_state)

            elapsed_ms = (time.time() - t0) * 1000
            logger.info("Processing complete in %.0f ms", elapsed_ms)

            # Record latency in memory analytics
            if self.memory:
                try:
                    self.memory.record_query_latency(elapsed_ms)
                except Exception:
                    pass

            # Build per-node timing snapshot for this query
            node_names = ["preprocess", "route_intent", "resolve_query",
                          "extract_data", "validate", "extract_facts",
                          "generate_response", "postprocess"]
            timings = {"total_ms": round(elapsed_ms, 1)}
            for n in node_names:
                vals = self._node_timings.get(n, [])
                timings[f"{n}_ms"] = round(vals[-1], 1) if vals else 0

            # Extract SQL from query intent
            sql_query = None
            if final_state.get("query_intent"):
                qi = final_state["query_intent"]
                sql_query = qi.sql_query if hasattr(qi, "sql_query") else None

            return {
                "answer": final_state.get("final_answer", "I couldn't generate a response."),
                "pipeline_flags": dict(self._pipeline_flags),
                "timings": timings,
                "sql_query": sql_query,
                "chart_data": final_state.get("chart_data"),
            }

        except Exception as exc:
            elapsed_ms = (time.time() - t0) * 1000
            logger.error("Orchestrator error after %.0f ms: %s", elapsed_ms, exc)
            if self.memory:
                try:
                    self.memory.record_error()
                except Exception:
                    pass
            return {
                "answer": f"I encountered an unexpected error: {str(exc)}",
                "pipeline_flags": dict(self._pipeline_flags) if self._pipeline_flags else {},
                "timings": {"total_ms": round(elapsed_ms, 1)},
                "sql_query": None,
                "chart_data": None,
            }

    # ──────────────────────────────────────────
    # Analytics & Lifecycle
    # ──────────────────────────────────────────

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation session."""
        if self.memory:
            return self.memory.get_conversation_summary()
        return {"message": "Memory not enabled"}

    def get_evaluation_summary(self) -> Dict[str, float]:
        """Get average evaluation scores."""
        if self.evaluation:
            return self.evaluation.get_average_scores()
        return {"message": "Evaluation not enabled"}

    def get_node_timings(self) -> Dict[str, Dict[str, float]]:
        """Return per-node timing stats (avg, p95, count)."""
        stats = {}
        for node, timings in self._node_timings.items():
            if not timings:
                continue
            sorted_t = sorted(timings)
            stats[node] = {
                "count": len(timings),
                "avg_ms": round(sum(timings) / len(timings), 1),
                "p95_ms": round(sorted_t[int(len(sorted_t) * 0.95)], 1),
                "max_ms": round(sorted_t[-1], 1),
            }
        return stats

    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            logger.info("Memory cleared")
        if hasattr(self.router_agent, "reset_streaks"):
            self.router_agent.reset_streaks()

    def generate_summary(self) -> str:
        """Generate a comprehensive summary of the retail data."""
        try:
            logger.info("Generating Data Summary...")

            summary_questions = [
                "What is the total revenue?",
                "Which state has the highest revenue?",
                "What are the top 3 categories by revenue?",
                "What is the cancellation rate?",
            ]

            summaries = []
            for question in summary_questions:
                answer = self.process_query(question)
                summaries.append(f"**{question}**\n{answer}\n")

            full_summary = "\n".join(summaries)

            # Session stats
            session_stats = ""
            if self.memory:
                conv = self.get_conversation_summary()
                session_stats = (
                    f"\n---\n*Session: {conv.get('turns', 0)} queries processed*"
                )

            if self.evaluation:
                eval_summary = self.get_evaluation_summary()
                if "overall" in eval_summary:
                    session_stats += (
                        f"\n*Quality Score: {eval_summary['overall']*100:.1f}%*"
                    )

            return (
                "# Retail Insights Summary Report\n\n"
                f"{full_summary}\n{session_stats}\n"
                "---\n*Generated by Retail Insights Assistant (Enhanced)*\n"
            )

        except Exception as exc:
            return f"Error generating summary: {str(exc)}"


# ────────────────────────────────────────────────
# Singleton
# ────────────────────────────────────────────────

_orchestrator_instance = None


def get_orchestrator() -> AgentOrchestrator:
    """Get singleton instance of AgentOrchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgentOrchestrator()
    return _orchestrator_instance


def reset_orchestrator():
    """Reset the orchestrator instance."""
    global _orchestrator_instance
    _orchestrator_instance = None
