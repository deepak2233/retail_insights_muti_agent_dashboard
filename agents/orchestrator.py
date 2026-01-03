"""
Agent Orchestrator using LangGraph
Coordinates multiple agents with enhanced memory, edge case handling, 
hallucination prevention, and evaluation
"""
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agents.query_agent import QueryResolutionAgent, AgentState
from agents.extraction_agent import DataExtractionAgent
from agents.validation_agent import ValidationAgent
from agents.response_agent import ResponseAgent
from agents.router_agent import RouterAgent

# Import new enhancement modules
from utils.memory import get_memory, ConversationMemory
from utils.edge_cases import get_edge_case_handler, EdgeCaseHandler
from utils.hallucination_prevention import FactExtractor, GroundedResponseGenerator
from utils.evaluation import get_evaluation_framework, EvaluationFramework


class AgentOrchestrator:
    """
    Enhanced orchestrator with:
    - Conversation memory for multi-turn dialogue
    - Edge case handling for robustness
    - Hallucination prevention for accuracy
    - Evaluation framework for quality tracking
    """
    
    def __init__(self, enable_memory: bool = True, enable_evaluation: bool = True):
        # Initialize core agents
        self.query_agent = QueryResolutionAgent()
        self.extraction_agent = DataExtractionAgent()
        self.validation_agent = ValidationAgent()
        self.response_agent = ResponseAgent()
        self.router_agent = RouterAgent()
        
        # Initialize enhancement modules
        self.memory: ConversationMemory = get_memory() if enable_memory else None
        self.edge_case_handler: EdgeCaseHandler = get_edge_case_handler()
        self.fact_extractor = FactExtractor()
        self.grounded_generator = GroundedResponseGenerator()
        self.evaluation: EvaluationFramework = get_evaluation_framework() if enable_evaluation else None
        
        # Configuration
        self.enable_memory = enable_memory
        self.enable_evaluation = enable_evaluation
        self.retry_on_error = True
        self.max_retries = 3
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the enhanced agent workflow graph"""
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents) with enhanced processing
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
        
        # Set entry point
        workflow.set_entry_point("preprocess")
        
        # Add edges with conditional routing
        workflow.add_conditional_edges(
            "preprocess",
            self._route_after_preprocess,
            {
                "resolve": "route_intent",
                "edge_case": "handle_edge_case"
            }
        )
        
        workflow.add_conditional_edges(
            "route_intent",
            self._route_based_on_intent,
            {
                "analytics": "resolve_query",
                "direct_response": "postprocess"
            }
        )
        
        workflow.add_edge("resolve_query", "extract_data")
        
        # Conditional edge after extraction
        workflow.add_conditional_edges(
            "extract_data",
            self._should_validate,
            {
                "validate": "validate",
                "retry": "resolve_query",
                "error": "handle_error"
            }
        )
        
        # Edge after validation
        workflow.add_conditional_edges(
            "validate",
            self._should_extract_facts,
            {
                "extract_facts": "extract_facts",
                "error": "handle_error"
            }
        )
        
        # After fact extraction
        workflow.add_edge("extract_facts", "generate_response")
        
        # After response generation
        workflow.add_edge("generate_response", "postprocess")
        
        # End edges
        workflow.add_edge("postprocess", END)
        workflow.add_edge("handle_error", END)
        workflow.add_edge("handle_edge_case", END)
        
        return workflow.compile()
    
    def _preprocess_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Preprocess question with memory and edge case handling"""
        print("\n[INFO] Preprocessing...")
        question = state["question"]
        
        # Get conversation context from memory
        context = None
        if self.memory:
            context = self.memory.get_recent_context(n_turns=3)
            # Resolve references (e.g., "it", "them")
            resolved_question = self.memory.resolve_reference(question)
            if resolved_question != question:
                print(f"   [INFO] Resolved reference: {resolved_question}")
                question = resolved_question
        
        # Check for edge cases
        edge_result = self.edge_case_handler.handle(question)
        
        if edge_result.is_edge_case:
            print(f"   [EDGE] Edge case detected: {edge_result.edge_case_type}")
            
            if edge_result.requires_clarification:
                return {
                    **state,
                    "edge_case_handled": True,
                    "final_answer": self._format_edge_case_response(edge_result),
                    "error": None
                }
            
            # Use modified question if available
            if edge_result.modified_question:
                question = edge_result.modified_question
        
        return {
            **state,
            "question": question,
            "conversation_context": context,
            "edge_case_handled": False,
            "report_content": state.get("report_content")
        }
    
    def _route_intent_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Classify intent and decide if we need more processing"""
        print("\n[INFO] Agent 0: Intent Routing")
        question = state["question"]
        
        # Check for duplicates in memory first
        if self.memory and self.memory.is_duplicate(question):
            print("   [INFO] Duplicate query detected, using previous result")
            last_turn = self.memory.short_term[-1]
            return {
                **state,
                "intent": "duplicate",
                "final_answer": last_turn["answer"]
            }
            
        result = self.router_agent.classify(question, report_content=state.get("report_content"))
        print(f"   [INTENT] Intent: {result.get('intent', 'analytics')}")
        
        return {
            **state,
            "intent": result.get("intent", "analytics"),
            "final_answer": result.get("response_if_not_analytics", "")
        }
        
    def _route_based_on_intent(self, state: AgentState) -> str:
        """Route based on classified intent"""
        intent = state.get("intent", "analytics")
        if intent == "analytics":
            return "analytics"
        return "direct_response"
    
    def _resolve_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Resolve natural language query to SQL with retry"""
        print("\n[INFO] Agent 1: Query Resolution")
        question = state["question"]
        context = state.get("conversation_context")
        error_history = state.get("_error_history", [])
        
        # Use retry-enabled resolution
        if self.retry_on_error and error_history:
            query_intent = self.query_agent.resolve_with_retry(
                question, 
                max_retries=self.max_retries,
                context=context,
                error_history=error_history
            )
        else:
            query_intent = self.query_agent.resolve_query(question, context)
        
        return {
            **state,
            "query_intent": query_intent
        }
    
    def _extract_data_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Extract data using SQL query"""
        print("\n🔄 Agent 2: Data Extraction")
        result = self.extraction_agent.extract_data(state)
        
        # Track errors for retry logic
        if result.get("error"):
            error_history = state.get("_error_history", [])
            error_history.append(result["error"])
            result["_error_history"] = error_history
            result["_retry_count"] = state.get("_retry_count", 0) + 1
        
        return result
    
    def _validate_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Validate query results with confidence scoring"""
        print("\n🔄 Agent 3: Validation")
        result = self.validation_agent.validate(state)
        
        # Log confidence scores
        confidence = result.get("confidence_scores", {})
        if confidence:
            overall = confidence.get("overall", 0) * 100
            print(f"   📊 Confidence Score: {overall:.1f}%")
        
        return result
    
    def _extract_facts_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Extract verifiable facts from data for grounded response"""
        print("\n[INFO] Fact Extraction")
        
        query_result = state.get("query_result", {})
        df = query_result.get("dataframe")
        query_intent = state.get("query_intent")
        
        facts = []
        if df is not None and not df.empty:
            facts = self.fact_extractor.extract_facts(df, query_intent)
            print(f"   [INFO] Extracted {len(facts)} verifiable facts")
        
        return {
            **state,
            "facts": facts
        }
    
    def _generate_response_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Generate grounded natural language response"""
        print("\n[INFO] Agent 4: Response Generation")
        
        # Generate response using standard agent
        result = self.response_agent.generate_response(state)
        
        # Validate response against facts
        facts = state.get("facts", [])
        if facts and result.get("final_answer"):
            validation_result = self.grounded_generator.validate_and_annotate(
                result["final_answer"], 
                facts
            )
            
            if validation_result["issues"]:
                print(f"   [WARN] Found {len(validation_result['issues'])} potential issues in response")
            
            # Add confidence indicator to response
            if validation_result["confidence"] < 0.8:
                result["final_answer"] += f"\n\nNote: Some insights may require verification (confidence: {validation_result['confidence']*100:.0f}%)"
        
        return result
    
    def _postprocess_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Post-process response, update memory, run evaluation"""
        print("\n📝 Postprocessing...")
        
        # Update conversation memory
        if self.memory and state.get("final_answer"):
            entities = {}
            if state.get("query_intent"):
                entities = state["query_intent"].entities if hasattr(state["query_intent"], 'entities') else {}
            
            self.memory.add_turn(
                question=state["question"],
                answer=state["final_answer"],
                sql=state["query_intent"].sql_query if state.get("query_intent") else None,
                entities=entities,
                metadata={
                    "confidence": state.get("confidence_scores", {}).get("overall", 0)
                }
            )
            print(f"   [INFO] Saved to conversation memory")
        
        # Run evaluation
        if self.evaluation and state.get("final_answer"):
            query_result = state.get("query_result", {})
            df = query_result.get("dataframe")
            
            if df is not None:
                eval_result = self.evaluation.evaluate_response(
                    question=state["question"],
                    sql=state["query_intent"].sql_query if state.get("query_intent") else "",
                    result_df=df,
                    response=state["final_answer"],
                    facts=state.get("facts", [])
                )
                print(f"   [EVAL] Evaluation: accuracy={eval_result.accuracy_score:.2f}, faithfulness={eval_result.faithfulness_score:.2f}, overall={eval_result.overall_score:.2f}")
        
        return state
    
    def _handle_error_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Handle errors gracefully with helpful suggestions"""
        print("\n[ERROR] Error Handler")
        error = state.get("error", "Unknown error occurred")
        
        # Provide more helpful error messages
        suggestions = [
            "Try rephrasing your question",
            "Ask about specific metrics like revenue, orders, or profit",
            "Specify a time period or region",
            "Ask 'What data is available?' to see options"
        ]
        
        response = f"""I apologize, but I encountered an issue processing your request.

**Error:** {error}

**Suggestions:**
{chr(10).join(f'• {s}' for s in suggestions)}

**Example questions you can ask:**
• What is the total revenue?
• Show me the top 5 categories by sales
• Compare B2B and B2C orders
"""
        
        return {
            **state,
            "final_answer": response
        }
    
    def _handle_edge_case_node(self, state: AgentState) -> Dict[str, Any]:
        """Node: Handle edge cases that were flagged during preprocessing"""
        print("\n[INFO] Edge Case Handler")
        # Response was already set in preprocess
        return state
    
    def _format_edge_case_response(self, edge_result) -> str:
        """Format edge case response with suggestions"""
        response = f"**{edge_result.message}**\n\n"
        
        if edge_result.suggestions:
            response += "**Try one of these instead:**\n"
            for suggestion in edge_result.suggestions[:5]:
                response += f"• {suggestion}\n"
        
        return response
    
    def _route_after_preprocess(self, state: AgentState) -> str:
        """Route after preprocessing based on edge case detection"""
        if state.get("edge_case_handled"):
            return "edge_case"
        return "resolve"
    
    def _should_validate(self, state: AgentState) -> str:
        """Decide whether to validate, retry, or handle error"""
        if state.get("error"):
            retry_count = state.get("_retry_count", 0)
            if self.retry_on_error and retry_count < self.max_retries:
                print(f"   [INFO] Retrying... (attempt {retry_count + 1}/{self.max_retries})")
                return "retry"
            return "error"
        return "validate"
    
    def _should_extract_facts(self, state: AgentState) -> str:
        """Decide whether to extract facts or handle error"""
        if state.get("validation_passed"):
            return "extract_facts"
        return "error"
    
    def process_query(self, question: str, report_content: Optional[str] = None) -> str:
        """
        Process a user question through the enhanced agent workflow
        
        Args:
            question: User's natural language question
            
        Returns:
            Final answer string
        """
        try:
            print(f"\n{'='*80}")
            print(f"INFO: Question: {question}")
            print(f"{'='*80}")
            
            # Initialize state with new fields
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
                report_content=report_content
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            print(f"\n{'='*80}")
            print("INFO: Processing Complete")
            print(f"{'='*80}\n")
            
            return final_state["final_answer"]
            
        except Exception as e:
            error_msg = f"Orchestrator error: {str(e)}"
            print(f"❌ {error_msg}")
            return f"I encountered an unexpected error: {error_msg}"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation session"""
        if self.memory:
            return self.memory.get_conversation_summary()
        return {"message": "Memory not enabled"}
    
    def get_evaluation_summary(self) -> Dict[str, float]:
        """Get average evaluation scores"""
        if self.evaluation:
            return self.evaluation.get_average_scores()
        return {"message": "Evaluation not enabled"}
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            print("[INFO] Memory cleared")
    
    def generate_summary(self) -> str:
        """
        Generate a comprehensive summary of the retail data
        
        Returns:
            Summary report
        """
        try:
            print("\n[INFO] Generating Data Summary...")
            
            summary_questions = [
                "What is the total revenue?",
                "Which state has the highest revenue?",
                "What are the top 3 categories by revenue?",
                "What is the cancellation rate?"
            ]
            
            summaries = []
            for question in summary_questions:
                answer = self.process_query(question)
                summaries.append(f"**{question}**\n{answer}\n")
            
            full_summary = "\n".join(summaries)
            
            # Include session stats
            session_stats = ""
            if self.memory:
                conv_summary = self.get_conversation_summary()
                session_stats = f"\n---\n*Session: {conv_summary.get('turns', 0)} queries processed*"
            
            if self.evaluation:
                eval_summary = self.get_evaluation_summary()
                if 'overall' in eval_summary:
                    session_stats += f"\n*Quality Score: {eval_summary['overall']*100:.1f}%*"
            
            return f"""# Retail Insights Summary Report

{full_summary}
{session_stats}
---
*Generated by Retail Insights Assistant (Enhanced)*
"""
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"


# Singleton instance
_orchestrator_instance = None


def get_orchestrator() -> AgentOrchestrator:
    """Get singleton instance of AgentOrchestrator"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgentOrchestrator()
    return _orchestrator_instance


def reset_orchestrator():
    """Reset the orchestrator instance"""
    global _orchestrator_instance
    _orchestrator_instance = None
