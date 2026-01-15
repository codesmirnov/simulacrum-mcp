"""
Simple equation parser for mathematical expressions.
"""

import re
from typing import List, Any, Dict
from .interfaces import IEquationParser


class SimpleEquationParser(IEquationParser):
    """
    Simple equation parser for basic mathematical expressions.

    Supports basic arithmetic operations and variable references.
    This is a placeholder implementation - a full parser would use
    proper expression parsing libraries.
    """

    def __init__(self):
        """Initialize the equation parser."""
        pass

    def parse(self, expression: str) -> Any:
        """
        Parse mathematical expression into executable form.

        Args:
            expression: Mathematical expression string

        Returns:
            Parsed expression (placeholder)
        """
        # Basic validation
        if not expression or not expression.strip():
            raise ValueError("Expression cannot be empty")

        # Simple syntax check - balanced parentheses
        if expression.count('(') != expression.count(')'):
            raise ValueError("Unbalanced parentheses in expression")

        # For now, just return the expression as-is
        # A full implementation would parse into an AST
        return expression

    def evaluate(self, parsed_expression: Any, variables: Dict[str, float]) -> float:
        """
        Evaluate parsed expression with given variable values.

        Args:
            parsed_expression: Parsed expression
            variables: Variable values

        Returns:
            Expression result
        """
        # Very simple evaluation for basic expressions
        # This is a placeholder - real implementation would evaluate AST
        expression = str(parsed_expression)

        # Replace variable names with values
        for var_name, var_value in variables.items():
            expression = re.sub(rf'\b{re.escape(var_name)}\b', str(var_value), expression)

        # Simple evaluation using eval (not safe for production!)
        # In production, use a proper expression evaluator
        try:
            # Only allow safe operations
            allowed_names = {
                "abs": abs,
                "max": max,
                "min": min,
                "__builtins__": {}
            }
            result = eval(expression, allowed_names)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {str(e)}")

    def get_dependencies(self, expression: str) -> List[str]:
        """
        Extract variable dependencies from expression.

        Args:
            expression: Mathematical expression

        Returns:
            List of variable names used in the expression
        """
        # Simple regex to find potential variable names
        # This matches word characters that are not numbers and not common function names
        function_names = {
            'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs',
            'max', 'min', 'if', 'and', 'or', 'not'
        }

        # Find all word tokens
        tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)

        # Filter out function names and keywords
        variables = []
        for token in tokens:
            if token not in function_names and not token.isdigit():
                variables.append(token)

        return list(set(variables))  # Remove duplicates
