"""
Unit tests for Code Validator

Tests the validation logic that checks miner code before sandbox execution.
"""

import pytest

from tournament.pipeline.validator import CodeValidator, ValidationResult


class TestCodeValidator:
    """Tests for code validation logic."""

    def test_valid_code_passes(self):
        """Code with required inner_steps function passes validation."""
        code = """
def inner_steps(model, data_iterator, optimizer, num_steps, device):
    return result
"""
        validator = CodeValidator(check_imports=False)
        result = validator.validate(code)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_function_fails(self):
        """Code missing required function fails validation."""
        code = """
def some_other_function():
    pass
# Missing inner_steps
"""
        validator = CodeValidator(check_imports=False)
        result = validator.validate(code)

        assert result.valid is False
        assert any("inner_steps" in error for error in result.errors)

    def test_missing_all_functions_fails(self):
        """Code with no required function fails validation."""
        code = """
def some_other_function():
    pass
"""
        validator = CodeValidator(check_imports=False)
        result = validator.validate(code)

        assert result.valid is False
        assert len(result.errors) > 0
        assert "inner_steps" in result.errors[0]

    def test_syntax_error_fails(self):
        """Code with syntax errors fails validation."""
        code = """
def setup_model(model_path):
    return model
    
def setup_data(data_path, batch_size, seq_len
    # Missing closing parenthesis
    return iterator
"""
        validator = CodeValidator(check_imports=False)
        result = validator.validate(code)

        assert result.valid is False
        assert len(result.errors) > 0
        assert "Syntax error" in result.errors[0]

    def test_forbidden_import_os_fails(self):
        """Code importing os module fails validation."""
        code = """
import os

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    return result
"""
        validator = CodeValidator(check_imports=True)
        result = validator.validate(code)

        assert result.valid is False
        assert any("os" in error for error in result.errors)

    def test_forbidden_import_subprocess_fails(self):
        """Code importing subprocess fails validation."""
        code = """
import subprocess

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    return result
"""
        validator = CodeValidator(check_imports=True)
        result = validator.validate(code)

        assert result.valid is False
        assert any("subprocess" in error for error in result.errors)

    def test_forbidden_import_socket_fails(self):
        """Code importing socket fails validation."""
        code = """
from socket import socket

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    return result
"""
        validator = CodeValidator(check_imports=True)
        result = validator.validate(code)

        assert result.valid is False
        assert any("socket" in error for error in result.errors)

    def test_allowed_import_torch_passes(self):
        """Code importing torch (allowed) passes validation."""
        code = """
import torch

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    return result
"""
        validator = CodeValidator(check_imports=True)
        result = validator.validate(code)

        assert result.valid is True

    def test_allowed_import_numpy_passes(self):
        """Code importing numpy (allowed) passes validation."""
        code = """
import numpy as np

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    return result
"""
        validator = CodeValidator(check_imports=True)
        result = validator.validate(code)

        assert result.valid is True

    def test_multiple_errors_reported(self):
        """All validation errors are reported together."""
        code = """
import os
import subprocess

def some_other_function():
    pass
# Missing inner_steps
"""
        validator = CodeValidator(check_imports=True)
        result = validator.validate(code)

        assert result.valid is False
        assert len(result.errors) >= 2  # Missing functions + forbidden imports

    def test_check_imports_disabled(self):
        """When check_imports=False, forbidden imports are ignored."""
        code = """
import os

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    return result
"""
        validator = CodeValidator(check_imports=False)
        result = validator.validate(code)

        assert result.valid is True

    def test_async_functions_detected(self):
        """Async functions are also detected as defined functions."""
        code = """
async def inner_steps(model, data_iterator, optimizer, num_steps, device):
    return result
"""
        validator = CodeValidator(check_imports=False)
        result = validator.validate(code)

        assert result.valid is True

    def test_nested_functions_counted(self):
        """ast.walk() finds nested functions (this is acceptable behavior)."""
        code = """
def outer():
    def inner_steps(model, data_iterator, optimizer, num_steps, device):
        return result
"""
        validator = CodeValidator(check_imports=False)
        result = validator.validate(code)

        # ast.walk() traverses the entire tree, so nested functions are found
        # This is acceptable - miners shouldn't nest functions anyway
        assert result.valid is True

