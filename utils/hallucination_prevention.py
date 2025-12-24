"""
Hallucination Prevention System
Ensures responses are grounded in actual data with fact extraction and validation
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import pandas as pd
from dataclasses import dataclass


@dataclass
class VerifiedFact:
    """A fact that has been verified against source data"""
    claim: str
    source_column: str
    source_value: Any
    confidence: float  # 0.0 to 1.0
    is_verified: bool


class FactExtractor:
    """Extracts verifiable facts from query results"""
    
    def extract_facts(self, df: pd.DataFrame, query_intent: Any) -> List[Dict[str, Any]]:
        """
        Extract verifiable facts from DataFrame
        
        Args:
            df: Query result DataFrame
            query_intent: The query intent with context
            
        Returns:
            List of extractable facts with their sources
        """
        if df is None or df.empty:
            return [{"type": "empty", "claim": "No data found", "verified": True}]
        
        facts = []
        
        # Extract numeric aggregates
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            facts.extend(self._extract_numeric_facts(df, col))
        
        # Extract categorical facts
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            facts.extend(self._extract_categorical_facts(df, col))
        
        # Extract ranking facts if multiple rows
        if len(df) > 1:
            facts.extend(self._extract_ranking_facts(df, numeric_cols))
        
        # Extract comparison facts
        if len(df) >= 2:
            facts.extend(self._extract_comparison_facts(df, numeric_cols))
        
        return facts
    
    def _extract_numeric_facts(self, df: pd.DataFrame, col: str) -> List[Dict]:
        """Extract facts about numeric columns"""
        facts = []
        
        col_data = df[col].dropna()
        if len(col_data) == 0:
            return facts
        
        # Single value fact
        if len(col_data) == 1:
            facts.append({
                "type": "single_value",
                "column": col,
                "value": float(col_data.iloc[0]),
                "claim": f"{col} is {col_data.iloc[0]:,.2f}",
                "verified": True
            })
        else:
            # Aggregate facts
            facts.append({
                "type": "sum",
                "column": col,
                "value": float(col_data.sum()),
                "claim": f"Total {col} is {col_data.sum():,.2f}",
                "verified": True
            })
            
            facts.append({
                "type": "average",
                "column": col,
                "value": float(col_data.mean()),
                "claim": f"Average {col} is {col_data.mean():,.2f}",
                "verified": True
            })
            
            facts.append({
                "type": "max",
                "column": col,
                "value": float(col_data.max()),
                "claim": f"Maximum {col} is {col_data.max():,.2f}",
                "verified": True
            })
            
            facts.append({
                "type": "min",
                "column": col,
                "value": float(col_data.min()),
                "claim": f"Minimum {col} is {col_data.min():,.2f}",
                "verified": True
            })
        
        return facts
    
    def _extract_categorical_facts(self, df: pd.DataFrame, col: str) -> List[Dict]:
        """Extract facts about categorical columns"""
        facts = []
        
        unique_values = df[col].dropna().unique()
        
        facts.append({
            "type": "unique_count",
            "column": col,
            "value": len(unique_values),
            "claim": f"There are {len(unique_values)} unique {col} values",
            "verified": True
        })
        
        if len(unique_values) <= 10:
            facts.append({
                "type": "categories",
                "column": col,
                "value": list(unique_values),
                "claim": f"{col} includes: {', '.join(map(str, unique_values[:5]))}",
                "verified": True
            })
        
        return facts
    
    def _extract_ranking_facts(self, df: pd.DataFrame, numeric_cols) -> List[Dict]:
        """Extract ranking facts (top/bottom)"""
        facts = []
        
        for col in numeric_cols:
            if col in df.columns:
                # Find the identifier column (first non-numeric column)
                id_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(id_cols) > 0:
                    id_col = id_cols[0]
                    
                    # Top performer
                    top_idx = df[col].idxmax()
                    if pd.notna(top_idx):
                        top_row = df.loc[top_idx]
                        facts.append({
                            "type": "ranking_top",
                            "column": col,
                            "identifier": str(top_row[id_col]),
                            "value": float(top_row[col]),
                            "claim": f"{top_row[id_col]} has the highest {col} at {top_row[col]:,.2f}",
                            "verified": True
                        })
                    
                    # Bottom performer
                    bottom_idx = df[col].idxmin()
                    if pd.notna(bottom_idx):
                        bottom_row = df.loc[bottom_idx]
                        facts.append({
                            "type": "ranking_bottom",
                            "column": col,
                            "identifier": str(bottom_row[id_col]),
                            "value": float(bottom_row[col]),
                            "claim": f"{bottom_row[id_col]} has the lowest {col} at {bottom_row[col]:,.2f}",
                            "verified": True
                        })
        
        return facts
    
    def _extract_comparison_facts(self, df: pd.DataFrame, numeric_cols) -> List[Dict]:
        """Extract comparison facts between rows"""
        facts = []
        
        for col in numeric_cols:
            if col in df.columns and len(df) >= 2:
                values = df[col].dropna()
                if len(values) >= 2:
                    max_val = values.max()
                    min_val = values.min()
                    
                    if min_val > 0:
                        ratio = max_val / min_val
                        facts.append({
                            "type": "ratio",
                            "column": col,
                            "value": float(ratio),
                            "claim": f"The highest {col} is {ratio:.1f}x the lowest",
                            "verified": True
                        })
                    
                    difference = max_val - min_val
                    facts.append({
                        "type": "difference",
                        "column": col,
                        "value": float(difference),
                        "claim": f"The range of {col} is {difference:,.2f}",
                        "verified": True
                    })
        
        return facts


class ResponseValidator:
    """Validates LLM responses against extracted facts"""
    
    def __init__(self):
        self.number_pattern = re.compile(r'[\d,]+\.?\d*')
    
    def validate_response(self, response: str, facts: List[Dict], 
                          tolerance: float = 0.05) -> Tuple[str, List[Dict]]:
        """
        Validate response claims against facts
        
        Args:
            response: LLM generated response
            facts: List of verified facts
            tolerance: Acceptable numeric tolerance (5% default)
            
        Returns:
            Tuple of (validated_response, list of issues found)
        """
        issues = []
        
        # Extract numbers from response
        response_numbers = self._extract_numbers(response)
        
        # Extract numbers from facts
        fact_numbers = {}
        for fact in facts:
            if "value" in fact and isinstance(fact["value"], (int, float)):
                fact_numbers[fact["value"]] = fact
        
        # Check each number in response
        for num in response_numbers:
            is_valid = self._validate_number(num, fact_numbers, tolerance)
            if not is_valid:
                issues.append({
                    "type": "unverified_number",
                    "value": num,
                    "message": f"Number {num:,.2f} not found in source data"
                })
        
        # Check for percentage claims
        percentage_issues = self._validate_percentages(response, facts)
        issues.extend(percentage_issues)
        
        return response, issues
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        numbers = []
        matches = self.number_pattern.findall(text)
        
        for match in matches:
            try:
                # Remove commas and convert
                num = float(match.replace(',', ''))
                if num > 0:  # Ignore zeros and negatives for validation
                    numbers.append(num)
            except ValueError:
                continue
        
        return numbers
    
    def _validate_number(self, num: float, fact_numbers: Dict, 
                         tolerance: float) -> bool:
        """Check if a number is close to any fact number"""
        for fact_num in fact_numbers.keys():
            if fact_num == 0:
                continue
            
            # Check within tolerance
            diff_ratio = abs(num - fact_num) / abs(fact_num)
            if diff_ratio <= tolerance:
                return True
        
        # Also check for common transformations (thousands, millions)
        for fact_num in fact_numbers.keys():
            if fact_num == 0:
                continue
            
            # Check if it's a scaled version
            for scale in [1000, 1000000, 100]:  # K, M, percentage
                scaled = fact_num / scale
                if scaled > 0:
                    diff_ratio = abs(num - scaled) / scaled
                    if diff_ratio <= tolerance:
                        return True
        
        return False
    
    def _validate_percentages(self, response: str, facts: List[Dict]) -> List[Dict]:
        """Validate percentage claims in response"""
        issues = []
        
        # Find percentage mentions
        pct_pattern = re.compile(r'(\d+\.?\d*)\s*%')
        matches = pct_pattern.findall(response)
        
        for match in matches:
            pct = float(match)
            
            # Check if this percentage is derivable from facts
            is_valid = False
            for fact in facts:
                if fact.get("type") == "ratio":
                    # Ratio could be expressed as percentage
                    fact_pct = (fact["value"] - 1) * 100
                    if abs(pct - fact_pct) < 5:  # 5% tolerance
                        is_valid = True
                        break
            
            if not is_valid and pct not in [0, 100]:
                issues.append({
                    "type": "unverified_percentage",
                    "value": pct,
                    "message": f"Percentage {pct}% may not be supported by data"
                })
        
        return issues


class GroundedResponseGenerator:
    """Generates responses grounded in verified facts"""
    
    def __init__(self):
        self.fact_extractor = FactExtractor()
        self.validator = ResponseValidator()
    
    def create_grounded_prompt(self, question: str, facts: List[Dict], 
                               raw_data_summary: str) -> str:
        """
        Create a prompt that encourages grounded responses
        
        Args:
            question: User's question
            facts: List of verified facts
            raw_data_summary: Summary of raw data
            
        Returns:
            Prompt string with grounding instructions
        """
        facts_text = "\n".join([f"- {f['claim']}" for f in facts if f.get('verified')])
        
        prompt = f"""Answer the following question using ONLY the verified facts provided below.

RULES:
1. Only use numbers that appear in the VERIFIED FACTS section
2. Do not interpolate or estimate values not explicitly provided
3. If you're unsure about a claim, say "based on the available data"
4. Do not make predictions or assumptions beyond the data
5. If the data doesn't answer the question, say so clearly

VERIFIED FACTS (from source data):
{facts_text}

RAW DATA SUMMARY:
{raw_data_summary}

USER QUESTION: {question}

Provide a clear, factual response using only the verified information above:"""
        
        return prompt
    
    def validate_and_annotate(self, response: str, facts: List[Dict]) -> Dict[str, Any]:
        """
        Validate response and add annotations
        
        Args:
            response: Generated response
            facts: Verified facts
            
        Returns:
            Dict with response and validation info
        """
        validated_response, issues = self.validator.validate_response(response, facts)
        
        confidence = 1.0 - (len(issues) * 0.1)  # Reduce confidence per issue
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "response": validated_response,
            "confidence": confidence,
            "issues": issues,
            "facts_used": len([f for f in facts if f.get('verified')]),
            "grounded": len(issues) == 0
        }


def create_grounded_response_system():
    """Factory function to create grounded response system"""
    return GroundedResponseGenerator()
