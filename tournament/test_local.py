#!/usr/bin/env python3
"""
Local testing script for miners.

This allows miners to test their code locally before submitting (and paying).
No network access required, no payment needed.

Usage:
    uv run python -m tournament.test_local train.py
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def validate_code_structure(code_path: Path) -> tuple[bool, list[str]]:
    """Validate code has required structure.
    
    Args:
        code_path: Path to train.py file
        
    Returns:
        Tuple of (is_valid, errors)
    """
    import ast
    
    errors = []
    
    if not code_path.exists():
        return False, [f"File not found: {code_path}"]
    
    code = code_path.read_text()
    
    # Check syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error at line {e.lineno}: {e.msg}"]
    
    # Check for inner_steps function
    has_inner_steps = False
    has_result_class = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            has_inner_steps = True
            # Check parameters
            if len(node.args.args) < 5:
                errors.append("inner_steps must have at least 5 parameters")
        
        if isinstance(node, ast.ClassDef) and node.name == "InnerStepsResult":
            has_result_class = True
    
    if not has_inner_steps:
        errors.append("Missing required function: inner_steps")
    
    if not has_result_class:
        errors.append("Missing required class: InnerStepsResult")
    
    # Check for forbidden imports
    forbidden = ["os", "subprocess", "socket", "http", "urllib", "requests", "pickle"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(fb in alias.name for fb in forbidden):
                    errors.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and any(fb in node.module for fb in forbidden):
                errors.append(f"Forbidden import: {node.module}")
    
    return len(errors) == 0, errors




def main():
    parser = argparse.ArgumentParser(
        description="Validate your training code structure before submitting"
    )
    parser.add_argument(
        "code_path",
        type=Path,
        help="Path to your train.py file"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TEMPLAR TOURNAMENT - CODE VALIDATION")
    logger.info("="*60)
    logger.info(f"Validating: {args.code_path}")
    logger.info("")
    
    # Validate code structure
    logger.info("Checking code structure...")
    is_valid, errors = validate_code_structure(args.code_path)
    
    if not is_valid:
        logger.error("❌ VALIDATION FAILED:")
        for error in errors:
            logger.error(f"   - {error}")
        logger.info("")
        logger.info("💡 Fix these errors before submitting")
        logger.info("")
        return 1
    
    logger.info("✅ Code structure valid")
    logger.info("   - inner_steps function found")
    logger.info("   - InnerStepsResult class found")
    logger.info("   - No forbidden imports")
    logger.info("")
    logger.info("="*60)
    logger.info("✅ VALIDATION PASSED - Ready to submit!")
    logger.info("="*60)
    logger.info("")
    logger.info("💡 To submit (costs 0.1 TAO):")
    logger.info("   uv run python -m neurons.miner train.py \\")
    logger.info("       --wallet.name mywallet \\")
    logger.info("       --wallet.hotkey myhotkey \\")
    logger.info("       --payment-recipient <validator_hotkey> \\")
    logger.info("       --validator-api <validator_url>")
    logger.info("")
    logger.info("   Example:")
    logger.info("   uv run python -m neurons.miner train.py \\")
    logger.info("       --wallet.name miner \\")
    logger.info("       --wallet.hotkey default \\")
    logger.info("       --payment-recipient 5GEKtrNMzRE3Xh7x7csKS1eGrZ7oSAzYYSQgxhZv3QUdVr9a \\")
    logger.info("       --validator-api http://154.54.100.65:8000")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

