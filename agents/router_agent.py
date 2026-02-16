"""
Router Agent — Advanced intent classification with topic-shift detection,
multi-turn guardrails, prompt-injection defense, and confidence scoring.
"""
from typing import Dict, Any, List, Optional
import re
import logging
from utils.llm_utils import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    intent: str = Field(description="One of: analytics, greeting, out_of_scope, appreciation, clarification_needed")
    confidence: float = Field(description="Confidence score 0.0–1.0")
    reasoning: str = Field(description="Brief reason for classification")
    response_if_not_analytics: str = Field(description="Polite response if not analytics, otherwise empty")


class RouterAgent:
    """
    Production-grade router with:
    - Fast regex paths for obvious intents
    - LLM-based classification with confidence scoring
    - Topic-shift detection (tracks domain drift across turns)
    - Multi-turn guardrails (escalating off-topic responses)
    - Prompt injection / jailbreak defense
    """

    # --- Fast-path regex patterns (skip LLM for obvious intents) ---
    GREETING_PATTERNS = re.compile(
        r'^(hi|hello|hey|hola|good\s*(morning|afternoon|evening)|howdy|greetings|yo)\s*[!?.]*$',
        re.IGNORECASE
    )
    CAPABILITIES_PATTERNS = re.compile(
        r'^(what\s+(can|do)\s+you\s+do|what\s+are\s+you|'
        r'who\s+are\s+you|what\s+can\s+you\s+(help|do)\s*(with|for)|'
        r'help\s*me|what\s+is\s+this|tell\s+me\s+about\s+(yourself|you)|'
        r'what\s+do\s+you\s+know|what\s+are\s+your\s+(capabilities|features|skills)|'
        r'what\s+kind\s+of\s+questions|show\s+me\s+what\s+you\s+can|'
        r'how\s+can\s+you\s+help|what\s+you\s+can\s+do\s+for\s+me|'
        r'what\s+can\s+you\s+do\s+for\s+me)'
        r'\s*[?!.]*$',
        re.IGNORECASE
    )
    APPRECIATION_PATTERNS = re.compile(
        r'^(thanks?(\s+you)?|thank\s+you|great(\s+job)?|awesome|good\s+job|'
        r'well\s+done|nice|perfect|excellent|wonderful|amazing|helpful|'
        r'you\'?re?\s+(the\s+)?best|much\s+appreciated|cheers)\s*[!?.]*$',
        re.IGNORECASE
    )
    ANALYTICS_FAST_PATTERNS = re.compile(
        r'\b(revenue|sales|profit|orders?|quantity|amount|top\s+\d+|'
        r'total|average|avg|compare|comparison|trend|growth|rate|'
        r'cancel\w*|ship\w*|b2b|b2c|categor\w*|state|region|'
        r'best\s+sell|worst\s+sell|highest|lowest|monthly|quarterly|'
        r'year\w*|how\s+much|how\s+many|count|sum|distribution)\b',
        re.IGNORECASE
    )

    # --- Prompt injection / jailbreak patterns ---
    INJECTION_PATTERNS = re.compile(
        r'(ignore\s+(your|all|previous)\s+instructions|'
        r'forget\s+(your|all|previous)\s+instructions|'
        r'you\s+are\s+now|act\s+as\s+(?!a\s+retail)|'
        r'pretend\s+to\s+be|new\s+persona|'
        r'system\s+prompt|reveal\s+(your\s+)?prompt|'
        r'what\s+are\s+your\s+instructions|'
        r'bypass\s+(?:your\s+)?(?:rules|filters|guardrails)|'
        r'DAN\s+mode|jailbreak)',
        re.IGNORECASE
    )

    # --- Off-topic escalation responses ---
    ESCALATION_RESPONSES = [
        # Level 1 — gentle redirect
        "That's an interesting topic! While it's outside my specialty, I'm great at retail analytics. "
        "I can help you explore revenue trends, product performance, regional sales, and much more. "
        "What would you like to know about your data?",
        # Level 2 — firmer redirect
        "I appreciate your curiosity, but my expertise is specifically in retail data analysis. "
        "I can answer questions about sales, orders, categories, regional performance, and business metrics. "
        "Try asking something like 'What are the top 5 categories by revenue?'",
        # Level 3 — firm boundary
        "I'm designed exclusively for retail analytics. I can help you with:\n"
        "- Revenue and profit analysis\n- Order trends and status\n- Category and product performance\n"
        "- Regional breakdowns\n- B2B vs B2C comparisons\n\n"
        "Please ask a question about your retail data and I'll provide detailed insights.",
    ]

    def __init__(self):
        self.llm = get_llm(temperature=0)
        self.parser = JsonOutputParser(pydantic_object=IntentClassification)
        self._off_topic_streak = 0
        self._topic_history: List[str] = []  # last N intents

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized router for a Retail Insights Assistant.
Classify the user's input into one of these categories:
1. analytics: Questions about sales, revenue, products, orders, customers, categories, states, or data analysis.
2. greeting: Simple hellos, greetings.
3. appreciation: Thanks, compliments.
4. out_of_scope: Unrelated topics (weather, recipes, news, personal advice, coding help).
5. clarification_needed: Vague or ambiguous queries that need more detail to process.

Provide a confidence score (0.0–1.0) indicating how certain you are.

Information about available context:
- Database: Amazon India e-commerce sales data (orders, revenue, categories, states, B2B/B2C)
- Report Context: {report_info}

Response guidelines:
- For greetings: Warm professional welcome, mention retail analytics capability.
- For appreciation: Humble response, offer further analytics assistance.
- For out_of_scope: Acknowledge politely, redirect to retail insights.
- For clarification_needed: Ask what specific metric/dimension they want.

Return ONLY a JSON object."""),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm | self.parser

    def classify(self, question: str, report_content: str = None,
                 memory=None) -> Dict[str, Any]:
        """
        Classify user intent with fast paths, guardrails, and confidence.
        
        Args:
            question: User's query
            report_content: Optional report context
            memory: Optional ConversationMemory for topic-shift detection
        """
        try:
            q_stripped = question.strip()
            q_lower = q_stripped.lower()

            # ── 0. Injection defense ──
            if self.INJECTION_PATTERNS.search(q_stripped):
                logger.warning("Prompt injection attempt blocked: %s", q_stripped[:80])
                return {
                    "intent": "out_of_scope",
                    "confidence": 1.0,
                    "reasoning": "Prompt injection attempt detected",
                    "response_if_not_analytics": (
                        "I'm your Retail Insights Assistant, focused on helping you "
                        "analyze sales data. How can I help with your retail analytics?"
                    ),
                }

            # ── 1. Fast-path: greetings ──
            if self.GREETING_PATTERNS.match(q_stripped):
                self._record_intent("greeting")
                return {
                    "intent": "greeting",
                    "confidence": 1.0,
                    "reasoning": "Greeting detected via pattern match",
                    "response_if_not_analytics": (
                        "Hello! I'm your **Retail Insights Assistant**. Here's what I can help with:\n\n"
                        "- **Revenue & profit** analysis by category, state, or time period\n"
                        "- **Order trends** and cancellation patterns\n"
                        "- **B2B vs B2C** comparisons\n"
                        "- **Top/bottom performers** across products and regions\n"
                        "- **Custom SQL queries** on your sales data\n\n"
                        "Try asking: *'What is the total revenue by category?'*"
                    ),
                }

            # ── 1b. Fast-path: capability questions ──
            if self.CAPABILITIES_PATTERNS.match(q_stripped):
                self._record_intent("greeting")
                return {
                    "intent": "greeting",
                    "confidence": 1.0,
                    "reasoning": "Capability question detected via pattern match",
                    "response_if_not_analytics": (
                        "I'm your **Retail Insights Assistant**, built to analyze your e-commerce sales data. "
                        "Here's what I can do:\n\n"
                        "- **Revenue & profit** — totals, breakdowns by category/state/time\n"
                        "- **Order analysis** — status, fulfillment methods, cancellation rates\n"
                        "- **B2B vs B2C** — channel comparisons\n"
                        "- **Top/bottom performers** — best & worst products, regions\n"
                        "- **Trends** — monthly/quarterly growth, seasonal patterns\n"
                        "- **Custom queries** — ask anything about your sales data\n\n"
                        "Try asking: *'What is the total revenue by category?'*"
                    ),
                }

            # ── 2. Fast-path: appreciation ──
            if self.APPRECIATION_PATTERNS.match(q_stripped):
                self._record_intent("appreciation")
                return {
                    "intent": "appreciation",
                    "confidence": 1.0,
                    "reasoning": "Appreciation detected via pattern match",
                    "response_if_not_analytics": (
                        "You're welcome! I'm here whenever you need retail insights. "
                        "Feel free to ask about revenue, orders, categories, or any other metric."
                    ),
                }

            # ── 3. Fast-path: obvious analytics ──
            if self.ANALYTICS_FAST_PATTERNS.search(q_stripped):
                self._record_intent("analytics")
                self._off_topic_streak = 0
                return {
                    "intent": "analytics",
                    "confidence": 0.95,
                    "reasoning": "Analytics keywords detected via fast path",
                    "response_if_not_analytics": "",
                }

            # ── 4. Topic-shift detection ──
            topic_shift_msg = self._detect_topic_shift(memory)

            # ── 5. LLM classification ──
            report_info = (
                "A summarized report is available as additional context."
                if report_content else
                "No additional reports loaded."
            )

            result = self.chain.invoke({
                "input": question,
                "report_info": report_info,
            })

            intent = result.get("intent", "analytics")
            confidence = result.get("confidence", 0.5)

            # ── 6. Low-confidence handling ──
            if intent == "analytics" and confidence < 0.6:
                intent = "clarification_needed"
                result["intent"] = intent
                result["response_if_not_analytics"] = (
                    "I'm not quite sure what you're looking for. Could you be more specific? "
                    "For example:\n"
                    "- 'What is the total revenue?'\n"
                    "- 'Show top 5 categories by sales'\n"
                    "- 'Compare B2B vs B2C orders'"
                )

            # ── 7. Off-topic streak tracking ──
            if intent in ("out_of_scope", "clarification_needed"):
                self._off_topic_streak += 1
                # Escalate response based on streak
                escalation_level = min(self._off_topic_streak - 1, len(self.ESCALATION_RESPONSES) - 1)
                if intent == "out_of_scope" and self._off_topic_streak >= 1:
                    result["response_if_not_analytics"] = self.ESCALATION_RESPONSES[escalation_level]
                    if topic_shift_msg:
                        result["response_if_not_analytics"] = topic_shift_msg + "\n\n" + result["response_if_not_analytics"]
            else:
                self._off_topic_streak = 0

            self._record_intent(intent)
            return result

        except Exception as e:
            logger.error("Router classification error: %s", e)
            # Fallback: assume analytics
            return {
                "intent": "analytics",
                "confidence": 0.5,
                "reasoning": f"LLM classification failed ({e}), defaulting to analytics",
                "response_if_not_analytics": "",
            }

    def _detect_topic_shift(self, memory=None) -> Optional[str]:
        """
        Detect if user has shifted away from retail analytics
        and is now on a different topic.
        """
        if not memory or not hasattr(memory, 'get_topic_history'):
            return None

        topic_history = memory.get_topic_history()
        if len(topic_history) < 2:
            return None

        # Check if user was doing analytics and suddenly went off-topic
        recent = topic_history[-3:]
        analytics_count = sum(1 for t in recent if t == "analytics")
        off_topic_count = sum(1 for t in recent if t in ("out_of_scope", "clarification_needed"))

        if analytics_count >= 1 and off_topic_count >= 1:
            return (
                "I noticed we were discussing your retail data earlier. "
                "Would you like to continue that analysis?"
            )
        return None

    def _record_intent(self, intent: str):
        """Record intent for topic-shift tracking."""
        self._topic_history.append(intent)
        # Keep last 10
        if len(self._topic_history) > 10:
            self._topic_history = self._topic_history[-10:]

    def get_topic_history(self) -> List[str]:
        """Get the recent intent history."""
        return list(self._topic_history)

    def reset_streaks(self):
        """Reset off-topic streak (e.g., on memory clear)."""
        self._off_topic_streak = 0
        self._topic_history.clear()
