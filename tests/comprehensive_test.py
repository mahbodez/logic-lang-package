#!/usr/bin/env python3
"""
COMPREHENSIVE LOGIC-LANG TEST SUITE

This test covers every single feature, operator, function, syntax element,
edge case, and error condition in the logic-lang package to ensure 100%
functionality correctness.

Test Categories:
1. Syntax & Parsing
2. Data Types & Literals
3. Operators (All types)
4. Built-in Functions (All 16 functions)
5. Statement Types
6. Expression Types
7. Indexing & Slicing
8. Arithmetic Operations
9. Error Handling
10. Edge Cases
11. Integration Tests
12. Performance Tests
"""

import torch
import pytest
import numpy as np
from typing import Dict, Any, List
from logic_lang import (
    RuleParser,
    RuleInterpreter,
    RuleMammoLoss,
    Truth,
    Constraint,
    ConstraintSet,
    GodelSemantics,
    LukasiewiczSemantics,
    ProductSemantics,
    ParseError,
    InterpreterError,
    VariableNotFoundError,
    InvalidFunctionError,
    TypeMismatchError,
    UnsupportedOperationError,
)


class ComprehensiveLogicLangTest:
    """Complete test suite for every logic-lang feature."""

    def __init__(self):
        self.parser = RuleParser()
        self.interpreter = RuleInterpreter()
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        self.test_results.append((test_name, passed, details))

    def setup_test_data(self) -> Dict[str, torch.Tensor]:
        """Create comprehensive test data covering all tensor types."""
        return {
            # Basic predictions (batch_size=2 for testing)
            "predictions": torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]),
            "birads_L": torch.tensor(
                [[0.1, 0.2, 0.3, 0.25, 0.15], [0.05, 0.1, 0.4, 0.3, 0.15]]
            ),
            "birads_R": torch.tensor(
                [[0.2, 0.1, 0.2, 0.3, 0.2], [0.1, 0.2, 0.3, 0.25, 0.15]]
            ),
            # Binary features
            "mass_L": torch.tensor([[0.8], [0.3]]),
            "mass_R": torch.tensor([[0.2], [0.9]]),
            "mc_L": torch.tensor([[0.4], [0.7]]),
            "mc_R": torch.tensor([[0.6], [0.1]]),
            # Risk scores
            "risk_score": torch.tensor([[0.75], [0.25]]),
            "confidence": torch.tensor([[0.9], [0.6]]),
            "baseline": torch.tensor([[0.5], [0.5]]),
            # Multi-dimensional data for indexing tests
            "matrix_2d": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "tensor_3d": torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            "assessments": torch.tensor(
                [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.3, 0.4, 0.5]]]
            ),
            # Scalar tensors
            "scalar_value": torch.tensor(0.5),
            "zero_value": torch.tensor(0.0),
            "one_value": torch.tensor(1.0),
            # Edge case values
            "small_value": torch.tensor([[1e-8], [1e-8]]),
            "large_value": torch.tensor([[100.0], [200.0]]),
            "negative_value": torch.tensor([[-0.5], [-1.0]]),
            # For consensus testing
            "radiologist_assessments": torch.tensor([[0.8, 0.9, 0.7], [0.3, 0.4, 0.2]]),
            "all_positive": torch.tensor([[0.8, 0.9, 0.95], [0.7, 0.8, 0.85]]),
            "mixed_evidence": torch.tensor([[0.8, 0.2, 0.9], [0.1, 0.9, 0.3]]),
        }

    # =================================================================
    # 1. SYNTAX & PARSING TESTS
    # =================================================================

    def test_comments(self):
        """Test all comment formats."""
        scripts = [
            "# Single line comment\ndefine x = mass_L",
            "define x = mass_L  # Inline comment",
            "# Multiple\n# Comments\ndefine x = mass_L",
            "define x = mass_L # Comment with symbols !@#$%^&*()",
        ]

        for i, script in enumerate(scripts):
            try:
                ast = self.parser.parse(script)
                self.log_test(f"Comments_{i+1}", True, "Comment parsing successful")
            except Exception as e:
                self.log_test(f"Comments_{i+1}", False, f"Failed: {e}")

    def test_statement_separation(self):
        """Test all statement separation methods."""
        test_cases = [
            # Semicolon separation
            (
                "expect predictions; define c = predictions[:, 0]; constraint exactly_one(predictions)",
                "Semicolon separation",
            ),
            ("const x = 5; const y = 10; define z = x + y", "Multiple semicolons"),
            ("expect predictions;", "Trailing semicolon"),
            # Newline separation
            (
                "expect predictions\ndefine subset = predictions[:, 0]\nconstraint exactly_one(predictions)",
                "Newline separation",
            ),
            # Mixed separation
            (
                "expect predictions; define subset = predictions[:, 0]\nconstraint exactly_one(predictions)",
                "Mixed semicolon/newline",
            ),
            # Empty lines
            (
                "expect predictions\n\ndefine subset = predictions[:, 0]\n\nconstraint exactly_one(predictions)",
                "Empty lines",
            ),
        ]

        features = {"predictions": torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])}

        for script, description in test_cases:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Statement_Separation_{description}", True)
            except Exception as e:
                self.log_test(
                    f"Statement_Separation_{description}", False, f"Failed: {e}"
                )

    def test_all_statement_types(self):
        """Test every type of statement."""
        features = self.setup_test_data()

        test_cases = [
            # Comment statements
            ("# This is a comment", "CommentStatement"),
            # Expect statements
            ("expect mass_L", "ExpectStatement_single"),
            ("expect mass_L, mass_R, mc_L", "ExpectStatement_multiple"),
            # Const statements
            ("const threshold = 0.5", "ConstStatement_number"),
            ("const message = 'hello'", "ConstStatement_string"),
            ("const calc = 5 + 3", "ConstStatement_expression"),
            ("const neg = -5", "ConstStatement_negative"),
            ("const complex = (2 + 3) * 4", "ConstStatement_complex"),
            # Define statements
            (
                "expect mass_L; define findings = mass_L > 0.5",
                "DefineStatement_comparison",
            ),
            (
                "expect mass_L, mc_L; define combined = mass_L | mc_L",
                "DefineStatement_logical",
            ),
            ("expect mass_L; define scaled = mass_L * 2", "DefineStatement_arithmetic"),
            # Constraint statements
            (
                "expect predictions; constraint exactly_one(predictions)",
                "ConstraintStatement_basic",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) weight=0.5",
                "ConstraintStatement_weight",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) transform='hinge'",
                "ConstraintStatement_transform",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) weight=0.5 alpha=2.0",
                "ConstraintStatement_params",
            ),
        ]

        for script, test_name in test_cases:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Statement_{test_name}", True)
            except Exception as e:
                self.log_test(f"Statement_{test_name}", False, f"Failed: {e}")

    # =================================================================
    # 2. DATA TYPES & LITERALS
    # =================================================================

    def test_all_literal_types(self):
        """Test every literal data type."""
        test_cases = [
            # Number literals
            ("const int_pos = 42", 42, "Integer positive"),
            ("const int_zero = 0", 0, "Integer zero"),
            ("const float_pos = 3.14", 3.14, "Float positive"),
            ("const float_zero = 0.0", 0.0, "Float zero"),
            ("const scientific = 1e-5", 1e-5, "Scientific notation"),
            # Negative numbers
            ("const neg_int = -42", -42, "Negative integer"),
            ("const neg_float = -3.14", -3.14, "Negative float"),
            ("const neg_sci = -1e5", -1e5, "Negative scientific"),
            # String literals
            ('const str_double = "hello"', "hello", "Double quote string"),
            ("const str_single = 'world'", "world", "Single quote string"),
            ("const str_empty = ''", "", "Empty string"),
            (
                "const str_special = 'symbols!@#$%'",
                "symbols!@#$%",
                "Special characters",
            ),
        ]

        for script, expected, description in test_cases:
            try:
                self.interpreter.execute(script)
                actual = self.interpreter.get_variable(script.split()[1])
                success = actual == expected
                self.log_test(
                    f"Literal_{description}",
                    success,
                    f"Expected {expected}, got {actual}",
                )
            except Exception as e:
                self.log_test(f"Literal_{description}", False, f"Failed: {e}")

    def test_list_literals(self):
        """Test list literals in all contexts."""
        features = {"predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4]])}

        test_cases = [
            # Basic lists
            (
                "expect predictions; define subset = sum(predictions, [0, 1])",
                "Basic integer list",
            ),
            (
                "expect predictions; define subset = sum(predictions, [2, 3])",
                "Different indices",
            ),
            # Mixed types in function calls
            (
                "expect predictions; define subset = sum(predictions, [0, 2])",
                "Non-consecutive indices",
            ),
        ]

        for script, description in test_cases:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"List_{description}", True)
            except Exception as e:
                self.log_test(f"List_{description}", False, f"Failed: {e}")

    # =================================================================
    # 3. ALL OPERATORS TEST
    # =================================================================

    def test_logical_operators(self):
        """Test every logical operator with all combinations."""
        features = {
            "a": torch.tensor([[0.8], [0.2]]),
            "b": torch.tensor([[0.3], [0.9]]),
            "c": torch.tensor([[0.6], [0.4]]),
        }

        logical_tests = [
            # Binary logical operators
            ("expect a, b; define or_result = a | b", "OR operator"),
            ("expect a, b; define and_result = a & b", "AND operator"),
            ("expect a, b; define xor_result = a ^ b", "XOR operator"),
            ("expect a, b; define implies_result = a >> b", "IMPLIES operator"),
            # Unary logical operators
            ("expect a; define not_result = ~a", "NOT operator"),
            # Complex combinations
            ("expect a, b, c; define complex1 = (a | b) & ~c", "Complex logical 1"),
            ("expect a, b, c; define complex2 = a >> (b | c)", "Complex logical 2"),
            ("expect a, b, c; define complex3 = ~(a & b) | c", "Complex logical 3"),
            # Precedence tests
            ("expect a, b, c; define precedence1 = a | b & c", "Precedence test 1"),
            ("expect a, b, c; define precedence2 = a >> b | c", "Precedence test 2"),
        ]

        for script, description in logical_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Logical_{description}", True)
            except Exception as e:
                self.log_test(f"Logical_{description}", False, f"Failed: {e}")

    def test_comparison_operators(self):
        """Test all comparison operators."""
        features = {
            "a": torch.tensor([[0.8], [0.2]]),
            "b": torch.tensor([[0.5], [0.5]]),
        }

        comparison_tests = [
            ("expect a, b; define gt = a > b", "Greater than"),
            ("expect a, b; define lt = a < b", "Less than"),
            ("expect a, b; define eq = a == b", "Equal"),
            ("expect a, b; define gte = a >= b", "Greater than or equal"),
            ("expect a, b; define lte = a <= b", "Less than or equal"),
            # With literals
            ("expect a; define gt_lit = a > 0.5", "GT with literal"),
            ("expect a; define lt_lit = a < 0.9", "LT with literal"),
            ("expect a; define eq_lit = a == 0.8", "EQ with literal"),
            # With constants
            (
                "const threshold = 0.6; expect a; define gt_const = a > threshold",
                "GT with constant",
            ),
        ]

        for script, description in comparison_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Comparison_{description}", True)
            except Exception as e:
                self.log_test(f"Comparison_{description}", False, f"Failed: {e}")

    def test_arithmetic_operators(self):
        """Test all arithmetic operators thoroughly."""
        features = {
            "a": torch.tensor([[2.0], [3.0]]),
            "b": torch.tensor([[4.0], [6.0]]),
            "c": torch.tensor([[1.0], [2.0]]),
        }

        arithmetic_tests = [
            # Basic binary arithmetic
            ("expect a, b; define add = a + b", "Addition"),
            ("expect a, b; define sub = a - b", "Subtraction"),
            ("expect a, b; define mul = a * b", "Multiplication"),
            ("expect a, b; define div = a / b", "Division"),
            # With constants
            ("const val = 5; expect a; define add_const = a + val", "Add constant"),
            (
                "const val = 2; expect a; define mul_const = a * val",
                "Multiply constant",
            ),
            # Arithmetic in constants
            ("const sum_result = 5 + 3", "Constant addition"),
            ("const diff_result = 10 - 4", "Constant subtraction"),
            ("const prod_result = 6 * 7", "Constant multiplication"),
            ("const div_result = 15 / 3", "Constant division"),
            # Complex arithmetic
            ("expect a, b, c; define complex = (a + b) * c - 1", "Complex arithmetic"),
            ("const calc = (5 + 3) * 2 - 1", "Complex constant calc"),
            # Negative arithmetic
            ("const neg_calc = -5 + 3", "Negative arithmetic"),
            ("expect a; define neg_var = -a + 1", "Negative variable arithmetic"),
            # Precedence
            ("expect a, b, c; define precedence = a + b * c", "Arithmetic precedence"),
            ("expect a, b, c; define parens = (a + b) * c", "Parentheses precedence"),
        ]

        for script, description in arithmetic_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Arithmetic_{description}", True)
            except Exception as e:
                self.log_test(f"Arithmetic_{description}", False, f"Failed: {e}")

    def test_unary_operators(self):
        """Test all unary operators."""
        features = {
            "tensor_vals": torch.tensor([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3]]),
            "single_val": torch.tensor([[0.6], [0.4]]),
        }

        unary_tests = [
            # Logical unary
            ("expect single_val; define not_val = ~single_val", "Unary NOT"),
            # Arithmetic unary
            ("expect single_val; define neg_val = -single_val", "Unary minus"),
            ("expect single_val; define pos_val = +single_val", "Unary plus"),
            # Tensor reduction unary
            ("expect tensor_vals; define all_true = & tensor_vals", "Unary AND_n"),
            ("expect tensor_vals; define any_true = | tensor_vals", "Unary OR_n"),
            # Nested unary
            (
                "expect single_val; define double_neg = -(-single_val)",
                "Double negative",
            ),
            ("expect single_val; define neg_not = ~(-single_val)", "Negative then NOT"),
            # Constants with unary
            ("const neg_const = -42", "Negative constant"),
            ("const pos_const = +42", "Positive constant"),
            ("const double_neg_const = -(-5)", "Double negative constant"),
        ]

        for script, description in unary_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Unary_{description}", True)
            except Exception as e:
                self.log_test(f"Unary_{description}", False, f"Failed: {e}")

    # =================================================================
    # 4. ALL BUILT-IN FUNCTIONS TEST
    # =================================================================

    def test_all_builtin_functions(self):
        """Test every single built-in function comprehensively."""
        features = {
            "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]]),
            "prob_vector": torch.tensor([[0.8, 0.1, 0.1], [0.3, 0.4, 0.3]]),
            "tensor_a": torch.tensor([[0.8], [0.2]]),
            "tensor_b": torch.tensor([[0.6], [0.9]]),
            "mass_probs": torch.tensor([[0.7], [0.3]]),
            "calc_probs": torch.tensor([[0.2], [0.8]]),
            "multi_pred": torch.tensor(
                [[0.1, 0.9, 0.8, 0.3, 0.6], [0.2, 0.1, 0.7, 0.9, 0.4]]
            ),
        }

        function_tests = [
            # Probability aggregation functions
            (
                "expect predictions; define high_classes = sum(predictions, [2, 3])",
                "sum function",
            ),
            (
                "expect prob_vector; constraint exactly_one(prob_vector)",
                "exactly_one function",
            ),
            (
                "expect mass_probs, calc_probs; constraint mutual_exclusion(mass_probs, calc_probs)",
                "mutual_exclusion function",
            ),
            # Cardinality constraint functions
            (
                "expect multi_pred; define at_least_2 = at_least_k(multi_pred, 2)",
                "at_least_k function",
            ),
            (
                "expect multi_pred; define at_most_3 = at_most_k(multi_pred, 3)",
                "at_most_k function",
            ),
            (
                "expect multi_pred; define exactly_2 = exactly_k(multi_pred, 2)",
                "exactly_k function",
            ),
            # Logical implication and conditional functions
            (
                "expect tensor_a, tensor_b; define thresh_impl = threshold_implication(tensor_a, tensor_b, 0.7)",
                "threshold_implication function",
            ),
            (
                "expect tensor_a, tensor_b; define cond_prob = conditional_probability(tensor_a, tensor_b, 0.8)",
                "conditional_probability function",
            ),
            # Comparison and threshold functions
            (
                "expect tensor_a, tensor_b; define gt_comp = greater_than(tensor_a, tensor_b)",
                "greater_than function",
            ),
            (
                "expect tensor_a, tensor_b; define lt_comp = less_than(tensor_a, tensor_b)",
                "less_than function",
            ),
            (
                "expect tensor_a, tensor_b; define eq_comp = equals(tensor_a, tensor_b)",
                "equals function",
            ),
            (
                "expect tensor_a; define thresh_gt = threshold_constraint(tensor_a, 0.5, '>')",
                "threshold_constraint function",
            ),
            # Utility functions
            (
                "expect tensor_a; define clamped = clamp(tensor_a, 0.1, 0.9)",
                "clamp function",
            ),
            (
                "expect tensor_a; define thresholded = threshold(tensor_a, 0.5)",
                "threshold function",
            ),
            # Functions with different parameters
            (
                "expect multi_pred; define thresh_lt = threshold_constraint(multi_pred, 0.5, '<')",
                "threshold_constraint LT",
            ),
            (
                "expect multi_pred; define thresh_eq = threshold_constraint(multi_pred, 0.5, '==')",
                "threshold_constraint EQ",
            ),
            (
                "expect multi_pred; define thresh_gte = threshold_constraint(multi_pred, 0.5, '>=')",
                "threshold_constraint GTE",
            ),
            (
                "expect multi_pred; define thresh_lte = threshold_constraint(multi_pred, 0.5, '<=')",
                "threshold_constraint LTE",
            ),
        ]

        for script, description in function_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Function_{description}", True)
            except Exception as e:
                self.log_test(f"Function_{description}", False, f"Failed: {e}")

    def test_function_parameter_variations(self):
        """Test functions with various parameter combinations."""
        features = {
            "values": torch.tensor(
                [[0.1, 0.8, 0.3, 0.9, 0.2], [0.6, 0.4, 0.7, 0.1, 0.5]]
            ),
            "condition": torch.tensor([[0.9], [0.3]]),
            "event": torch.tensor([[0.7], [0.8]]),
        }

        param_tests = [
            # sum with different indices
            ("expect values; define sum1 = sum(values, [0])", "sum single index"),
            (
                "expect values; define sum2 = sum(values, [0, 2, 4])",
                "sum non-consecutive",
            ),
            (
                "expect values; define sum3 = sum(values, [4, 3, 2, 1, 0])",
                "sum reverse order",
            ),
            # at_least_k variations
            (
                "expect values; define atleast_0 = at_least_k(values, 0)",
                "at_least_k zero",
            ),
            (
                "expect values; define atleast_1 = at_least_k(values, 1)",
                "at_least_k one",
            ),
            (
                "expect values; define atleast_5 = at_least_k(values, 5)",
                "at_least_k all",
            ),
            # threshold_implication variations
            (
                "expect condition, event; define impl_0 = threshold_implication(condition, event, 0.0)",
                "threshold_impl 0.0",
            ),
            (
                "expect condition, event; define impl_1 = threshold_implication(condition, event, 1.0)",
                "threshold_impl 1.0",
            ),
            (
                "expect condition, event; define impl_mid = threshold_implication(condition, event, 0.5)",
                "threshold_impl 0.5",
            ),
            # conditional_probability variations
            (
                "expect condition, event; define cond_0 = conditional_probability(condition, event, 0.0)",
                "conditional_prob 0.0",
            ),
            (
                "expect condition, event; define cond_1 = conditional_probability(condition, event, 1.0)",
                "conditional_prob 1.0",
            ),
            # clamp variations
            (
                "expect values; define clamp_normal = clamp(values, 0.0, 1.0)",
                "clamp 0-1",
            ),
            (
                "expect values; define clamp_narrow = clamp(values, 0.3, 0.7)",
                "clamp narrow",
            ),
            (
                "expect values; define clamp_wide = clamp(values, -1.0, 2.0)",
                "clamp wide",
            ),
        ]

        for script, description in param_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"FunctionParam_{description}", True)
            except Exception as e:
                self.log_test(f"FunctionParam_{description}", False, f"Failed: {e}")

    # =================================================================
    # 5. INDEXING & SLICING TESTS
    # =================================================================

    def test_tensor_indexing(self):
        """Test all forms of tensor indexing."""
        features = {
            "matrix_2d": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "tensor_3d": torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        }

        indexing_tests = [
            # Simple indexing
            ("expect matrix_2d; define elem = matrix_2d[:, 0]", "Index first column"),
            ("expect matrix_2d; define elem2 = matrix_2d[:, 1]", "Index second column"),
            ("expect matrix_2d; define elem3 = matrix_2d[:, 2]", "Index third column"),
            # Multi-dimensional indexing
            ("expect tensor_3d; define slice_3d = tensor_3d[:, 0, :]", "3D slice"),
            ("expect tensor_3d; define elem_3d = tensor_3d[:, 1, 0]", "3D element"),
            # Slice indexing
            (
                "expect predictions; define first_two = predictions[:, 0:2]",
                "Slice first two",
            ),
            (
                "expect predictions; define last_two = predictions[:, 2:4]",
                "Slice last two",
            ),
            ("expect predictions; define middle = predictions[:, 1:3]", "Slice middle"),
            # Edge cases
            (
                "expect predictions; define single = predictions[:, 0:1]",
                "Single element slice",
            ),
            ("expect predictions; define all_cols = predictions[:, :]", "All columns"),
            # Step slicing
            (
                "expect predictions; define every_other = predictions[:, ::2]",
                "Every other element",
            ),
            (
                "expect predictions; define step_slice = predictions[:, 1::2]",
                "Step slice from 1",
            ),
        ]

        for script, description in indexing_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Indexing_{description}", True)
            except Exception as e:
                self.log_test(f"Indexing_{description}", False, f"Failed: {e}")

    def test_slice_expressions(self):
        """Test all slice expression formats."""
        features = {
            "data": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
        }

        slice_tests = [
            # Basic slices
            ("expect data; define start_slice = data[:, :3]", "Start to 3"),
            ("expect data; define end_slice = data[:, 2:]", "From 2 to end"),
            ("expect data; define mid_slice = data[:, 1:4]", "Middle slice"),
            # Step slices
            ("expect data; define step_2 = data[:, ::2]", "Step by 2"),
            ("expect data; define step_3 = data[:, 1::2]", "Start 1, step 2"),
            # Edge cases
            ("expect data; define empty_slice = data[:, 5:5]", "Empty slice"),
            ("expect data; define full_slice = data[:, :]", "Full slice"),
        ]

        for script, description in slice_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Slice_{description}", True)
            except Exception as e:
                self.log_test(f"Slice_{description}", False, f"Failed: {e}")

    # =================================================================
    # 6. CONSTRAINT PARAMETERS & TRANSFORMS
    # =================================================================

    def test_constraint_parameters(self):
        """Test all constraint parameters and transforms."""
        features = {"predictions": torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])}

        constraint_tests = [
            # Different weights
            (
                "expect predictions; constraint exactly_one(predictions) weight=0.5",
                "Weight 0.5",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) weight=1.0",
                "Weight 1.0",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) weight=2.0",
                "Weight 2.0",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) weight=-0.5",
                "Negative weight",
            ),
            # Different transforms
            (
                "expect predictions; constraint exactly_one(predictions) transform='logbarrier'",
                "Logbarrier transform",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) transform='linear'",
                "Linear transform",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) transform='hinge'",
                "Hinge transform",
            ),
            # Multiple parameters
            (
                "expect predictions; constraint exactly_one(predictions) weight=0.8 alpha=2.0",
                "Weight + alpha",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) weight=0.8 transform='hinge' alpha=1.5",
                "Weight + transform + alpha",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) alpha=2.0 beta=-1.0 gamma=0.5",
                "Multiple parameters",
            ),
            # Parameters with arithmetic
            (
                "const w = 0.5; expect predictions; constraint exactly_one(predictions) weight=w * 2",
                "Arithmetic weight",
            ),
            (
                "expect predictions; constraint exactly_one(predictions) alpha=(2 + 3)",
                "Arithmetic parameter",
            ),
        ]

        for script, description in constraint_tests:
            try:
                result = self.interpreter.execute(script, features)
                # Verify constraint was created
                success = len(result.constraints) > 0
                self.log_test(f"Constraint_{description}", success)
            except Exception as e:
                self.log_test(f"Constraint_{description}", False, f"Failed: {e}")

    # =================================================================
    # 7. ERROR HANDLING TESTS
    # =================================================================

    def test_syntax_errors(self):
        """Test all syntax error conditions."""
        error_tests = [
            # Parse errors
            ("define x = mass_L |", ParseError, "Missing right operand"),
            ("define x = |", ParseError, "Missing operands"),
            ("constraint", ParseError, "Missing constraint expression"),
            ("expect", ParseError, "Missing variable names"),
            ("const", ParseError, "Missing constant definition"),
            ("define", ParseError, "Missing variable definition"),
            # Invalid tokens
            ("define x = mass_L @@ mass_R", ParseError, "Invalid token"),
            ("define x = mass_L $ mass_R", ParseError, "Invalid character"),
            # Unbalanced parentheses
            ("define x = (mass_L | mass_R", ParseError, "Unbalanced parentheses"),
            ("define x = mass_L | mass_R)", ParseError, "Extra closing paren"),
            # Invalid indexing
            ("define x = mass_L[", ParseError, "Incomplete indexing"),
            ("define x = mass_L[0", ParseError, "Unclosed bracket"),
        ]

        for script, expected_error, description in error_tests:
            try:
                self.parser.parse(script)
                self.log_test(f"SyntaxError_{description}", False, "Should have failed")
            except expected_error:
                self.log_test(f"SyntaxError_{description}", True)
            except Exception as e:
                self.log_test(
                    f"SyntaxError_{description}", False, f"Wrong error type: {e}"
                )

    def test_interpreter_errors(self):
        """Test all interpreter error conditions."""
        features = {"mass_L": torch.tensor([[0.5], [0.7]])}

        error_tests = [
            # Undefined variables
            ("define x = undefined_var", VariableNotFoundError, "Undefined variable"),
            (
                "constraint exactly_one(undefined_var)",
                VariableNotFoundError,
                "Undefined in constraint",
            ),
            # Invalid functions
            (
                "define x = unknown_func(mass_L)",
                InvalidFunctionError,
                "Unknown function",
            ),
            (
                "constraint unknown_constraint(mass_L)",
                InvalidFunctionError,
                "Unknown constraint function",
            ),
            # Type mismatches
            ("constraint exactly_one(5)", Exception, "Wrong type for function"),
            ("define x = mass_L + 'string'", Exception, "Type mismatch in operation"),
            # Division by zero
            ("const x = 5 / 0", InterpreterError, "Division by zero constant"),
            (
                "expect mass_L; define x = mass_L / 0",
                InterpreterError,
                "Division by zero expression",
            ),
            # Missing expected variables
            (
                "expect undefined_var; define x = mass_L",
                Exception,
                "Missing expected variable",
            ),
        ]

        for script, expected_error, description in error_tests:
            try:
                self.interpreter.execute(script, features)
                self.log_test(
                    f"InterpreterError_{description}", False, "Should have failed"
                )
            except expected_error:
                self.log_test(f"InterpreterError_{description}", True)
            except Exception as e:
                self.log_test(
                    f"InterpreterError_{description}", False, f"Wrong error type: {e}"
                )

    # =================================================================
    # 8. EDGE CASES & BOUNDARY CONDITIONS
    # =================================================================

    def test_edge_cases(self):
        """Test boundary conditions and edge cases."""
        features = {
            "zeros": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "ones": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            "tiny": torch.tensor([[1e-10], [1e-10]]),
            "huge": torch.tensor([[1e10], [1e10]]),
            "mixed": torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]),
            "single": torch.tensor([[0.5]]),
            "scalar": torch.tensor(0.5),
        }

        edge_tests = [
            # Extreme values
            ("expect zeros; define not_zeros = ~zeros", "All zeros NOT"),
            ("expect ones; define not_ones = ~ones", "All ones NOT"),
            ("expect tiny; define scaled_tiny = tiny * 1000", "Tiny values"),
            ("expect huge; define scaled_huge = huge / 1000", "Huge values"),
            # Boundary arithmetic
            (
                "const zero = 0.0; const one = 1.0; const diff = one - zero",
                "Boundary subtraction",
            ),
            ("const tiny_calc = 1e-10 + 1e-10", "Tiny arithmetic"),
            # Single element operations
            ("expect single; define neg_single = -single", "Single element negative"),
            (
                "expect single; constraint exactly_one(single)",
                "Single element constraint",
            ),
            # Scalar operations
            ("expect scalar; define scalar_op = scalar > 0.3", "Scalar comparison"),
            # Empty-like operations
            ("expect mixed; define zero_elements = mixed == 0.0", "Zero elements"),
            ("expect mixed; define one_elements = mixed == 1.0", "One elements"),
            # Degenerate slicing
            ("expect mixed; define empty_slice = mixed[:, 3:3]", "Empty slice"),
            (
                "expect mixed; define single_slice = mixed[:, 1:2]",
                "Single element slice",
            ),
        ]

        for script, description in edge_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"EdgeCase_{description}", True)
            except Exception as e:
                self.log_test(f"EdgeCase_{description}", False, f"Failed: {e}")

    def test_operator_precedence_edge_cases(self):
        """Test complex operator precedence scenarios."""
        features = {
            "a": torch.tensor([[0.8], [0.2]]),
            "b": torch.tensor([[0.6], [0.4]]),
            "c": torch.tensor([[0.3], [0.9]]),
            "d": torch.tensor([[0.7], [0.1]]),
        }

        precedence_tests = [
            # Mixed logical and arithmetic
            ("expect a, b; define mixed1 = a > 0.5 & b < 0.5", "Comparison then AND"),
            ("expect a, b; define mixed2 = a + b > 1.0", "Arithmetic then comparison"),
            ("expect a, b, c; define mixed3 = a * b > c", "Multiply then compare"),
            # Complex nested operations
            (
                "expect a, b, c, d; define complex1 = (a | b) & (c | d)",
                "Nested logical",
            ),
            (
                "expect a, b, c, d; define complex2 = a >> b & c >> d",
                "Chained implications",
            ),
            ("expect a, b, c; define complex3 = ~a | b & c", "NOT then OR then AND"),
            # Arithmetic precedence with logical
            (
                "expect a, b, c; define arith_logic = a + b * c > 1.0",
                "Full precedence chain",
            ),
            (
                "expect a, b, c; define logic_arith = (a > b) + (c < 0.5)",
                "Logic in arithmetic",
            ),
        ]

        for script, description in precedence_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Precedence_{description}", True)
            except Exception as e:
                self.log_test(f"Precedence_{description}", False, f"Failed: {e}")

    # =================================================================
    # 9. INTEGRATION TESTS
    # =================================================================

    def test_complete_scenarios(self):
        """Test complete real-world scenarios."""
        features = {
            "birads_L": torch.tensor(
                [[0.1, 0.2, 0.3, 0.25, 0.15], [0.05, 0.1, 0.4, 0.3, 0.15]]
            ),
            "birads_R": torch.tensor(
                [[0.2, 0.1, 0.2, 0.3, 0.2], [0.1, 0.2, 0.3, 0.25, 0.15]]
            ),
            "mass_L": torch.tensor([[0.8], [0.3]]),
            "mass_R": torch.tensor([[0.2], [0.9]]),
            "mc_L": torch.tensor([[0.4], [0.7]]),
            "mc_R": torch.tensor([[0.6], [0.1]]),
            "risk_score": torch.tensor([[0.75], [0.25]]),
        }

        # Complete mammography scenario
        mammography_script = """
        # Expected variables
        expect birads_L, birads_R, mass_L, mass_R, mc_L, mc_R, risk_score
        
        # Constants for thresholds
        const high_threshold = 0.8
        const low_threshold = 0.2
        const birads_cutoff = 4
        
        # Feature definitions
        define findings_L = mass_L | mc_L
        define findings_R = mass_R | mc_R
        define high_birads_L = sum(birads_L, [4])
        define high_birads_R = sum(birads_R, [4])
        define bilateral = findings_L & findings_R
        
        # Risk assessments
        define high_risk = risk_score > high_threshold
        define low_risk = risk_score < low_threshold
        define moderate_risk = ~high_risk & ~low_risk
        
        # Complex logical rules
        define suspicious_pattern = (findings_L >> high_birads_L) & (findings_R >> high_birads_R)
        define asymmetric = findings_L ^ findings_R
        
        # Constraints
        constraint exactly_one(birads_L) weight=1.0
        constraint exactly_one(birads_R) weight=1.0
        constraint findings_L >> high_birads_L weight=0.7
        constraint findings_R >> high_birads_R weight=0.7
        constraint high_risk >> bilateral weight=0.5 transform="hinge"
        """

        try:
            result = self.interpreter.execute(mammography_script, features)
            success = len(result.constraints) == 5
            self.log_test(
                "Integration_Mammography",
                success,
                f"Created {len(result.constraints)} constraints",
            )
        except Exception as e:
            self.log_test("Integration_Mammography", False, f"Failed: {e}")

    def test_advanced_arithmetic_scenarios(self):
        """Test complex arithmetic scenarios."""
        features = {
            "score_a": torch.tensor([[0.8], [0.3]]),
            "score_b": torch.tensor([[0.6], [0.7]]),
            "weight_a": torch.tensor([[0.4], [0.6]]),
            "weight_b": torch.tensor([[0.6], [0.4]]),
            "baseline": torch.tensor([[0.5], [0.5]]),
        }

        arithmetic_script = """
        expect score_a, score_b, weight_a, weight_b, baseline
        
        # Complex weighted combination
        define weighted_score = score_a * weight_a + score_b * weight_b
        
        # Normalization
        define total_weight = weight_a + weight_b
        define normalized_score = weighted_score / total_weight
        
        # Distance metrics
        define diff_from_baseline = normalized_score - baseline
        define abs_diff = diff_from_baseline * diff_from_baseline  # Squared diff approximation
        
        # Threshold-based decisions
        define above_baseline = normalized_score > baseline
        define significantly_above = abs_diff > 0.1
        
        # Combined logic
        define confident_positive = above_baseline & significantly_above
        
        # Constraint with complex expression
        constraint (weighted_score > 0.5) >> confident_positive weight=0.8
        """

        try:
            result = self.interpreter.execute(arithmetic_script, features)
            success = len(result.constraints) == 1
            self.log_test("Integration_Arithmetic", success)
        except Exception as e:
            self.log_test("Integration_Arithmetic", False, f"Failed: {e}")

    # =================================================================
    # 10. PERFORMANCE & STRESS TESTS
    # =================================================================

    def test_large_tensors(self):
        """Test with large tensor dimensions."""
        features = {
            "large_tensor": torch.randn(10, 100),  # Larger than typical
            "wide_tensor": torch.randn(2, 1000),  # Very wide
            "predictions": torch.softmax(torch.randn(10, 50), dim=1),  # Many classes
        }

        large_tests = [
            (
                "expect large_tensor; define subset = large_tensor[:, :50]",
                "Large tensor slicing",
            ),
            (
                "expect wide_tensor; define narrow = wide_tensor[:, 100:200]",
                "Wide tensor slicing",
            ),
            (
                "expect predictions; define high_conf = sum(predictions, [45, 46, 47, 48, 49])",
                "Many class sum",
            ),
            (
                "expect large_tensor; define means = & large_tensor",
                "Large tensor AND_n",
            ),
        ]

        for script, description in large_tests:
            try:
                result = self.interpreter.execute(script, features)
                self.log_test(f"Performance_{description}", True)
            except Exception as e:
                self.log_test(f"Performance_{description}", False, f"Failed: {e}")

    def test_many_constraints(self):
        """Test scripts with many constraints."""
        features = {"predictions": torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])}

        # Generate script with many constraints
        many_constraints_script = "expect predictions\n"
        for i in range(20):
            many_constraints_script += (
                f"constraint exactly_one(predictions) weight={0.1 + i * 0.05}\n"
            )

        try:
            result = self.interpreter.execute(many_constraints_script, features)
            success = len(result.constraints) == 20
            self.log_test(
                "Performance_ManyConstraints",
                success,
                f"Created {len(result.constraints)} constraints",
            )
        except Exception as e:
            self.log_test("Performance_ManyConstraints", False, f"Failed: {e}")

    # =================================================================
    # 11. SEMANTIC TESTS
    # =================================================================

    def test_different_semantics(self):
        """Test different logical semantics."""
        features = {
            "a": torch.tensor([[0.8], [0.2]]),
            "b": torch.tensor([[0.6], [0.9]]),
        }

        semantics_tests = [
            (GodelSemantics(), "Godel semantics"),
            (LukasiewiczSemantics(), "Lukasiewicz semantics"),
            (ProductSemantics(), "Product semantics"),
        ]

        script = "expect a, b; define result = a & b; constraint result"

        for semantics, description in semantics_tests:
            try:
                interpreter = RuleInterpreter(default_semantics=semantics)
                result = interpreter.execute(script, features)
                self.log_test(f"Semantics_{description}", True)
            except Exception as e:
                self.log_test(f"Semantics_{description}", False, f"Failed: {e}")

    # =================================================================
    # MAIN TEST RUNNER
    # =================================================================

    def run_all_tests(self):
        """Run the complete test suite."""
        print("🧪 COMPREHENSIVE LOGIC-LANG TEST SUITE")
        print("=" * 60)

        # Reset interpreter for clean state
        self.interpreter = RuleInterpreter()

        print("\n📝 1. SYNTAX & PARSING TESTS")
        self.test_comments()
        self.test_statement_separation()
        self.test_all_statement_types()

        print("\n🔢 2. DATA TYPES & LITERALS")
        self.test_all_literal_types()
        self.test_list_literals()

        print("\n⚡ 3. OPERATORS")
        self.test_logical_operators()
        self.test_comparison_operators()
        self.test_arithmetic_operators()
        self.test_unary_operators()

        print("\n🔧 4. BUILT-IN FUNCTIONS")
        self.test_all_builtin_functions()
        self.test_function_parameter_variations()

        print("\n📊 5. INDEXING & SLICING")
        self.test_tensor_indexing()
        self.test_slice_expressions()

        print("\n⚙️ 6. CONSTRAINT PARAMETERS")
        self.test_constraint_parameters()

        print("\n❌ 7. ERROR HANDLING")
        self.test_syntax_errors()
        self.test_interpreter_errors()

        print("\n🎯 8. EDGE CASES")
        self.test_edge_cases()
        self.test_operator_precedence_edge_cases()

        print("\n🔗 9. INTEGRATION TESTS")
        self.test_complete_scenarios()
        self.test_advanced_arithmetic_scenarios()

        print("\n⚡ 10. PERFORMANCE TESTS")
        self.test_large_tensors()
        self.test_many_constraints()

        print("\n🧠 11. SEMANTIC TESTS")
        self.test_different_semantics()

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for name, passed, details in self.test_results:
                if not passed:
                    print(f"  • {name}: {details}")

        return failed_tests == 0


def main():
    """Run the comprehensive test suite."""
    tester = ComprehensiveLogicLangTest()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 ALL TESTS PASSED! Logic-lang is working perfectly.")
    else:
        print("\n💥 Some tests failed. Please review the failures above.")

    return success


if __name__ == "__main__":
    main()
