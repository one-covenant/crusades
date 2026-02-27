"""
Verify your train.py passes validator checks before submitting.

Usage:
    uv run local_test/verify.py

This script runs the SAME checks as the production validator:
0. Security scan (AST) — forbidden imports, patterns, backend modifications
1. final_logits must not be None
2. Logits must be 3D (batch, seq_len-1, vocab)
3. Sequence length must match expected
4. Token count matches expected
5. Loss must be positive, valid, and close to reference
6. 100% of parameters must be trainable (no frozen layers)
7. 80% of parameter elements must change during training
8. Gradient relative error |g - g_truth| / |g_truth| must be small
9. Final weight relative error |w - w_ref| / |w_ref| must be small

Fix any failures before submitting to avoid failed evaluations!
"""

import ast
import gc
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from crusades.core.security_defs import (
    ALLOWED_TORCH_ASSIGNMENT_PREFIXES,
    ALLOWED_TORCH_SUBMODULE_IMPORTS,
    FORBIDDEN_ASSIGNMENT_ATTRS,
    FORBIDDEN_ATTR_CALLS,
    FORBIDDEN_BACKEND_TOGGLE_ATTRS,
    FORBIDDEN_BUILTINS,
    FORBIDDEN_CUDNN_ATTRS,
    FORBIDDEN_DIRECT_CALLS,
    FORBIDDEN_DOTTED_MODULES,
    FORBIDDEN_GC_ATTRS,
    FORBIDDEN_GRAD_TOGGLE_CALLS,
    FORBIDDEN_IMPORT_SUBSTRINGS,
    FORBIDDEN_INTROSPECTION_ATTRS,
    FORBIDDEN_MODULES,
    FORBIDDEN_NAMES,
    FORBIDDEN_OBJECT_DUNDER_ATTRS,
    FORBIDDEN_STRINGS,
    FORBIDDEN_SYS_MODULE_NAMES,
    FORBIDDEN_TIMER_ATTRS,
    FORBIDDEN_TORCH_ATTRIBUTE_ALIASES,
    FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS,
    FORBIDDEN_TORCH_CONFIG_MODULES,
    FORBIDDEN_TORCH_SYMBOL_IMPORTS,
)

# =========================================================================
# Security checks — policy loaded from crusades.core.security_defs
# =========================================================================

# Local aliases for backward-compat with the rest of this file
_FORBIDDEN_STRINGS = FORBIDDEN_STRINGS
_FORBIDDEN_MODULES = FORBIDDEN_MODULES
_FORBIDDEN_DOTTED_MODULES = FORBIDDEN_DOTTED_MODULES
_FORBIDDEN_TORCH_SYMBOL_IMPORTS = FORBIDDEN_TORCH_SYMBOL_IMPORTS
_FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS = FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS
_FORBIDDEN_TORCH_ATTRIBUTE_ALIASES = FORBIDDEN_TORCH_ATTRIBUTE_ALIASES
_BLOCKED_BUILTINS = FORBIDDEN_BUILTINS


def _is_main_guard(node: ast.AST) -> bool:
    """Check if an AST node is an `if __name__ == "__main__":` block.

    Handles both `__name__ == "__main__"` and `"__main__" == __name__`
    for consistency with _collect_main_guard_nodes.
    """
    if not (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], (ast.Eq, ast.Is))
        and len(node.test.comparators) == 1
    ):
        return False
    left, right = node.test.left, node.test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _collect_main_guard_nodes(tree: ast.AST) -> set[int]:
    skip_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_main_guard = False
            if isinstance(test, ast.Compare) and len(test.ops) == 1:
                op = test.ops[0]
                if isinstance(op, (ast.Eq, ast.Is)):
                    left, right = test.left, test.comparators[0]
                    if (
                        isinstance(left, ast.Name)
                        and left.id == "__name__"
                        and isinstance(right, ast.Constant)
                        and right.value == "__main__"
                    ) or (
                        isinstance(right, ast.Name)
                        and right.id == "__name__"
                        and isinstance(left, ast.Constant)
                        and left.value == "__main__"
                    ):
                        is_main_guard = True
            if is_main_guard:
                for child in ast.walk(node):
                    skip_ids.add(id(child))
    return skip_ids


def _forbidden_name_binding_reason(node: ast.AST) -> str | None:
    """Return violation text when ``node`` binds or modifies __name__."""
    forbidden_name = "__name__"

    if (
        isinstance(node, ast.Name)
        and node.id == forbidden_name
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ):
        return "modification of __name__ is forbidden"

    if isinstance(node, ast.alias):
        if node.asname == forbidden_name:
            return "aliasing import to __name__ is forbidden"
        if node.name == forbidden_name:
            return "importing __name__ is forbidden"

    if isinstance(node, ast.arg) and node.arg == forbidden_name:
        return "using __name__ as an argument is forbidden"

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.name == forbidden_name:
            return "defining __name__ is forbidden"

    if isinstance(node, ast.ExceptHandler) and node.name == forbidden_name:
        return "binding __name__ in exception handler is forbidden"

    if isinstance(node, ast.MatchAs) and getattr(node, "name", None) == forbidden_name:
        return "binding __name__ in match pattern is forbidden"

    if isinstance(node, ast.MatchStar) and getattr(node, "name", None) == forbidden_name:
        return "binding __name__ in match pattern is forbidden"

    if isinstance(node, ast.MatchMapping) and getattr(node, "rest", None) == forbidden_name:
        return "binding __name__ in match mapping pattern is forbidden"

    if isinstance(node, ast.Global) and forbidden_name in node.names:
        return "declaring global __name__ is forbidden"

    if isinstance(node, ast.Nonlocal) and forbidden_name in node.names:
        return "declaring nonlocal __name__ is forbidden"

    if hasattr(ast, "TypeVar") and isinstance(node, ast.TypeVar):
        if node.name == forbidden_name:
            return "using __name__ as a type parameter is forbidden"

    return None


def _scan_for_dangerous_patterns(tree: ast.AST) -> list[str]:
    violations = []

    _forbidden_names = FORBIDDEN_NAMES

    # Track names that currently alias the torch module
    torch_aliases: set[str] = {"torch"}

    # Track names that alias torch submodules (e.g. "F" from
    # "import torch.nn.functional as F").  Attribute assignment on any
    # of these is forbidden — legitimate code never monkey-patches torch.
    torch_submodule_aliases: set[str] = set()

    for node in ast.walk(tree):
        # --- Torch alias tracking ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    local_name = alias.asname or alias.name
                    if local_name != "torch":
                        line = getattr(node, "lineno", "?")
                        violations.append(f"Line {line}: aliasing torch is forbidden")
                    torch_aliases.add(local_name)
                elif alias.name.startswith("torch.") and alias.asname:
                    if alias.name in ALLOWED_TORCH_SUBMODULE_IMPORTS:
                        torch_submodule_aliases.add(alias.asname)

        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id != "torch":
                            line = getattr(node, "lineno", "?")
                            violations.append(f"Line {line}: aliasing torch is forbidden")
                        torch_aliases.add(target.id)

        # Block attribute mutation on torch modules / submodule aliases.
        # Covers Assign, AugAssign (+=), and Delete (del) targets.
        _attr_targets: list[ast.Attribute] = []
        if isinstance(node, ast.Assign):
            _attr_targets = [t for t in node.targets if isinstance(t, ast.Attribute)]
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute):
            _attr_targets = [node.target]
        elif isinstance(node, ast.Delete):
            _attr_targets = [t for t in node.targets if isinstance(t, ast.Attribute)]

        for target in _attr_targets:
            root_node = target.value
            while isinstance(root_node, ast.Attribute):
                root_node = root_node.value
            if not isinstance(root_node, ast.Name):
                continue
            root_name = root_node.id

            if root_name in torch_submodule_aliases:
                line = getattr(node, "lineno", "?")
                violations.append(
                    f"Line {line}: mutating {root_name}.{target.attr} is forbidden"
                    " (monkey-patching torch modules is not allowed)"
                )
                continue

            if root_name in torch_aliases:
                if isinstance(target.value, ast.Name):
                    line = getattr(node, "lineno", "?")
                    violations.append(
                        f"Line {line}: mutating {root_name}.{target.attr} is forbidden"
                        " (monkey-patching torch modules is not allowed)"
                    )
                else:
                    parts: list[str] = []
                    walk = target.value
                    while isinstance(walk, ast.Attribute):
                        parts.append(walk.attr)
                        walk = walk.value
                    if isinstance(walk, ast.Name):
                        parts.append(walk.id)
                        parent_path = ".".join(reversed(parts))
                        if any(
                            parent_path == pfx or parent_path.startswith(pfx + ".")
                            for pfx in ALLOWED_TORCH_ASSIGNMENT_PREFIXES
                        ):
                            continue
                        line = getattr(node, "lineno", "?")
                        violations.append(
                            f"Line {line}: mutating {parent_path}.{target.attr}"
                            " is forbidden (monkey-patching torch modules is not allowed)"
                        )

        # Block rebinding sensitive torch attributes to local names
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in torch_aliases
            and node.value.attr in _FORBIDDEN_TORCH_ATTRIBUTE_ALIASES
        ):
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: aliasing torch.{node.value.attr} is forbidden")

        # Block bare-name references to dangerous builtins
        if isinstance(node, ast.Name) and node.id in _forbidden_names:
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: reference to '{node.id}' is forbidden")

        name_binding_violation = _forbidden_name_binding_reason(node)
        if name_binding_violation:
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: {name_binding_violation}")

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_OBJECT_DUNDER_ATTRS:
            if isinstance(node.value, ast.Name) and node.value.id == "object":
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: object.{node.attr} is forbidden")

        # Block torch._C access (including through aliases)
        if isinstance(node, ast.Attribute) and node.attr == "_C":
            if isinstance(node.value, ast.Name) and node.value.id in torch_aliases:
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: torch._C access is forbidden")

        # Block torch._dynamo.config and torch._inductor.config writes
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute):
                    if target.value.attr == "config" and isinstance(
                        target.value.value, ast.Attribute
                    ):
                        if target.value.value.attr in FORBIDDEN_TORCH_CONFIG_MODULES:
                            line = getattr(node, "lineno", "?")
                            violations.append(
                                f"Line {line}: modifying torch.{target.value.value.attr}.config is forbidden"
                            )

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "__class__":
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: __class__ assignment is forbidden")

        if isinstance(node, ast.Attribute) and node.attr == "__class__":
            if not isinstance(getattr(node, "_parent", None), ast.AnnAssign):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: __class__ access is forbidden")

        # Block timer-related attribute access on ANY object
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_TIMER_ATTRS:
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: accessing .{node.attr} is forbidden")

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in FORBIDDEN_ASSIGNMENT_ATTRS:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: overriding {target.attr} is forbidden")

        if isinstance(node, ast.Attribute) and node.attr == "__slots__":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: __slots__ modification is forbidden")

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_GC_ATTRS:
            if isinstance(node.value, ast.Name) and node.value.id == "gc":
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: gc.{node.attr}() is forbidden")

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CUDNN_ATTRS:
            if (
                isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "cudnn"
            ):
                line = getattr(node, "lineno", "?")
                violations.append(
                    f"Line {line}: setting torch.backends.cudnn.{node.attr} is forbidden"
                )

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_BACKEND_TOGGLE_ATTRS:
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: {func.attr}() is forbidden")
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_GRAD_TOGGLE_CALLS:
                line = getattr(node, "lineno", "?")
                violations.append(
                    f"Line {line}: {func.attr}() is forbidden"
                    " (disabling gradients would bypass verification)"
                )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_DIRECT_CALLS:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: {node.func.id}() is forbidden")
                if node.func.id in _BLOCKED_BUILTINS:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: {node.func.id}() is forbidden")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_ATTR_CALLS:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: .{node.func.attr}() is forbidden")
                if node.func.attr in _BLOCKED_BUILTINS:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: .{node.func.attr}() is forbidden")
                if node.func.attr == "compile":
                    if not (
                        isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
                    ):
                        line = getattr(node, "lineno", "?")
                        violations.append(
                            f"Line {line}: .compile() is forbidden (only torch.compile allowed)"
                        )

        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module in _FORBIDDEN_MODULES or alias.name.startswith("importlib"):
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: import {alias.name} is forbidden")
                for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                    if substr in alias.name:
                        line = getattr(node, "lineno", "?")
                        violations.append(f"Line {line}: import {alias.name} is forbidden")
                for forbidden_path in _FORBIDDEN_DOTTED_MODULES:
                    if alias.name == forbidden_path or alias.name.startswith(forbidden_path + "."):
                        line = getattr(node, "lineno", "?")
                        violations.append(f"Line {line}: import {alias.name} is forbidden")

        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: star imports (from ... import *) are forbidden")
            if not node.module:
                continue
            base_module = node.module.split(".")[0]
            if base_module in _FORBIDDEN_MODULES or node.module.startswith("importlib"):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: from {node.module} import is forbidden")
            if node.module == "torch":
                for alias in node.names:
                    if alias.name in _FORBIDDEN_TORCH_SYMBOL_IMPORTS:
                        line = getattr(node, "lineno", "?")
                        violations.append(f"Line {line}: importing torch.{alias.name} is forbidden")
            if node.module.startswith("torch.backends"):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS:
                        line = getattr(node, "lineno", "?")
                        violations.append(
                            f"Line {line}: importing torch backend toggle is forbidden"
                        )
            for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                if substr in node.module:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: from {node.module} import is forbidden")
            for forbidden_path in _FORBIDDEN_DOTTED_MODULES:
                if node.module == forbidden_path or node.module.startswith(forbidden_path + "."):
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: from {node.module} import is forbidden")
            for alias in node.names:
                for substr in FORBIDDEN_IMPORT_SUBSTRINGS:
                    if substr in alias.name:
                        line = getattr(node, "lineno", "?")
                        violations.append(f"Line {line}: import {alias.name} is forbidden")
                full_path = f"{node.module}.{alias.name}"
                for forbidden_path in _FORBIDDEN_DOTTED_MODULES:
                    if full_path == forbidden_path or full_path.startswith(forbidden_path + "."):
                        line = getattr(node, "lineno", "?")
                        violations.append(f"Line {line}: import {full_path} is forbidden")
                if full_path in ALLOWED_TORCH_SUBMODULE_IMPORTS:
                    local_name = alias.asname or alias.name
                    torch_submodule_aliases.add(local_name)

        # Block forbidden builtin calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: {node.func.id}() is forbidden")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_BUILTINS:
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: .{node.func.attr}() is forbidden")

        # Block torch.load (uses pickle internally), including aliased torch names
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in torch_aliases
        ):
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: torch.load() is forbidden (uses pickle internally)")

        # Block numpy.ctypeslib access
        if isinstance(node, ast.Attribute) and node.attr == "ctypeslib":
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: ctypeslib access is forbidden")

        if isinstance(node, ast.Name) and node.id == "__builtins__":
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: __builtins__ access is forbidden")
        if isinstance(node, ast.Attribute) and node.attr == "__builtins__":
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: __builtins__ access is forbidden")

        if isinstance(node, ast.Attribute) and node.attr == "modules":
            if isinstance(node.value, ast.Name) and node.value.id in FORBIDDEN_SYS_MODULE_NAMES:
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: sys.modules access is forbidden")

        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: __dict__ access is forbidden")

        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_INTROSPECTION_ATTRS:
            line = getattr(node, "lineno", "?")
            violations.append(f"Line {line}: .{node.attr} access is forbidden")

        if isinstance(node, ast.Attribute) and node.attr == "optimizer":
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                line = getattr(node, "lineno", "?")
                violations.append(f"Line {line}: accessing .optimizer attribute is forbidden")

        # Block dangerous builtins used as decorators
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Name) and deco.id in _forbidden_names:
                    line = getattr(deco, "lineno", "?")
                    violations.append(f"Line {line}: decorator @{deco.id} is forbidden")

    return violations


def validate_code_structure(code: str) -> list[str]:
    """Validate that train.py passes all security checks (same as production)."""
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error at line {exc.lineno}: {exc.msg}"]

    # Scan the FULL tree — do NOT strip `if __name__ == "__main__":` blocks.
    # Attackers can force them to execute via `__name__: str = "__main__"`
    # (AnnAssign) or other reassignment tricks, so they must be scanned.
    violations.extend(_scan_for_dangerous_patterns(tree))

    # Scan string literals for forbidden patterns
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pattern in _FORBIDDEN_STRINGS:
                if pattern in node.value:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: forbidden string pattern '{pattern}' detected")

    # Scan bytes().decode() and b"...".decode() for forbidden patterns
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "decode"
        ):
            inner = node.func.value
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in ("bytes", "bytearray")
            ):
                try:
                    if inner.args and len(inner.args) == 1 and isinstance(inner.args[0], ast.List):
                        int_values = []
                        for elt in inner.args[0].elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                                int_values.append(elt.value)
                            else:
                                raise ValueError("non-constant")
                        decoded_str = bytes(int_values).decode()
                        for pattern in _FORBIDDEN_STRINGS:
                            if pattern in decoded_str:
                                line = getattr(node, "lineno", "?")
                                violations.append(
                                    f"Line {line}: forbidden string via bytes().decode()"
                                )
                    else:
                        raise ValueError("not simple")
                except (ValueError, UnicodeDecodeError):
                    line = getattr(node, "lineno", "?")
                    violations.append(
                        f"Line {line}: dynamic bytes().decode() construction is forbidden"
                    )
            elif (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "fromhex"
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id in ("bytes", "bytearray")
                and inner.args
                and len(inner.args) == 1
                and isinstance(inner.args[0], ast.Constant)
                and isinstance(inner.args[0].value, str)
            ):
                try:
                    decoded_str = bytes.fromhex(inner.args[0].value).decode()
                    for pattern in _FORBIDDEN_STRINGS:
                        if pattern in decoded_str:
                            line = getattr(node, "lineno", "?")
                            violations.append(
                                f"Line {line}: forbidden string via fromhex().decode()"
                            )
                except (ValueError, UnicodeDecodeError):
                    line = getattr(node, "lineno", "?")
                    violations.append(
                        f"Line {line}: dynamic fromhex().decode() construction is forbidden"
                    )
            elif (
                isinstance(inner, ast.BinOp)
                and isinstance(inner.op, ast.Add)
                and isinstance(inner.left, ast.Constant)
                and isinstance(inner.left.value, bytes)
                and isinstance(inner.right, ast.Constant)
                and isinstance(inner.right.value, bytes)
            ):
                try:
                    decoded_str = (inner.left.value + inner.right.value).decode()
                except UnicodeDecodeError:
                    decoded_str = ""
                for pattern in _FORBIDDEN_STRINGS:
                    if pattern in decoded_str:
                        line = getattr(node, "lineno", "?")
                        violations.append(
                            f"Line {line}: forbidden string via byte concatenation + decode()"
                        )
            elif isinstance(inner, ast.Constant) and isinstance(inner.value, bytes):
                try:
                    decoded_str = inner.value.decode()
                except UnicodeDecodeError:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: undecodable byte literal in .decode() call")
                    continue
                for pattern in _FORBIDDEN_STRINGS:
                    if pattern in decoded_str:
                        line = getattr(node, "lineno", "?")
                        violations.append(f"Line {line}: forbidden string via b'...'.decode()")

    # Catch str(bytearray([...]), 'ascii') / str(bytes([...]), 'utf-8') —
    # a variation of bytes obfuscation that doesn't use .decode().
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) >= 2
        ):
            inner = node.args[0]
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                if inner.func.id in ("bytes", "bytearray"):
                    try:
                        if (
                            inner.args
                            and len(inner.args) == 1
                            and isinstance(inner.args[0], ast.List)
                        ):
                            int_values = []
                            for elt in inner.args[0].elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                                    int_values.append(elt.value)
                                else:
                                    raise ValueError("non-constant element")
                            decoded_str = bytes(int_values).decode()
                            for pattern in _FORBIDDEN_STRINGS:
                                if pattern in decoded_str:
                                    line = getattr(node, "lineno", "?")
                                    violations.append(
                                        f"Line {line}: forbidden string via str({inner.func.id}(...))"
                                    )
                        else:
                            raise ValueError("not simple")
                    except (ValueError, UnicodeDecodeError):
                        line = getattr(node, "lineno", "?")
                        violations.append(
                            f"Line {line}: dynamic str({inner.func.id}(...)) construction is forbidden"
                        )

    # Scan attribute names (e.g. _e._perf_counter) — AST string scan only
    # catches ast.Constant values, not ast.Attribute.attr names
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            for pattern in _FORBIDDEN_STRINGS:
                if node.attr == pattern:
                    line = getattr(node, "lineno", "?")
                    violations.append(f"Line {line}: forbidden attribute name '{node.attr}'")

    # Scan for str.join() obfuscation: "".join(["s","e","t","a","t","t","r"])
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
            and node.args
            and len(node.args) == 1
        ):
            arg = node.args[0]
            if isinstance(arg, (ast.List, ast.Tuple)):
                chars = []
                all_const = True
                for elt in arg.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        chars.append(elt.value)
                    else:
                        all_const = False
                        break
                if all_const and chars:
                    joined = node.func.value.value.join(chars)
                    for pattern in _FORBIDDEN_STRINGS:
                        if pattern in joined:
                            line = getattr(node, "lineno", "?")
                            violations.append(
                                f"Line {line}: forbidden string constructed via str.join()"
                            )
            elif (
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Name)
                and arg.func.id == "reversed"
                and arg.args
                and len(arg.args) == 1
                and isinstance(arg.args[0], ast.Constant)
                and isinstance(arg.args[0].value, str)
            ):
                joined = node.func.value.value.join(reversed(arg.args[0].value))
                for pattern in _FORBIDDEN_STRINGS:
                    if pattern in joined:
                        line = getattr(node, "lineno", "?")
                        violations.append(
                            f"Line {line}: forbidden string constructed via reversed join"
                        )

    # Scan for string concatenation obfuscation: "__set" + "attr__"
    # Walk full BinOp tree recursively to catch multi-level concat like "a" + "b" + "c"
    def _collect_concat_parts(node: ast.AST) -> list[str] | None:
        """Recursively collect all string constants from chained Add BinOps."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left_parts = _collect_concat_parts(node.left)
            right_parts = _collect_concat_parts(node.right)
            if left_parts is not None and right_parts is not None:
                return left_parts + right_parts
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            parts = _collect_concat_parts(node)
            if parts and len(parts) >= 2:
                combined = "".join(parts)
                for pattern in _FORBIDDEN_STRINGS:
                    if pattern in combined:
                        line = getattr(node, "lineno", "?")
                        violations.append(
                            f"Line {line}: forbidden string constructed via concatenation"
                        )

    # Scan for %-format obfuscation: "%s%s" % ("__set", "attr__") or "%s_back" % "f"
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        ):
            parts: list[str] | None = None
            if isinstance(node.right, ast.Tuple):
                parts = []
                for elt in node.right.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        parts.append(elt.value)
                    else:
                        parts = None
                        break
            elif isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                parts = [node.right.value]
            if parts is not None:
                try:
                    formatted = node.left.value % tuple(parts)
                    for pattern in _FORBIDDEN_STRINGS:
                        if pattern in formatted:
                            line = getattr(node, "lineno", "?")
                            violations.append(
                                f"Line {line}: forbidden string constructed via %-format"
                            )
                except (TypeError, ValueError):
                    pass

    # Scan for f-string obfuscation: f"{'__set'}{'attr__'}"
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = []
            all_const = True
            for val in node.values:
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    parts.append(val.value)
                elif (
                    isinstance(val, ast.FormattedValue)
                    and isinstance(val.value, ast.Constant)
                    and isinstance(val.value.value, str)
                    and val.format_spec is None
                    and val.conversion == -1
                ):
                    parts.append(val.value.value)
                else:
                    all_const = False
                    break
            if all_const and parts:
                combined = "".join(parts)
                for pattern in _FORBIDDEN_STRINGS:
                    if pattern in combined:
                        line = getattr(node, "lineno", "?")
                        violations.append(f"Line {line}: forbidden string constructed via f-string")

    inner_steps_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            inner_steps_found = True
            args = node.args
            if len(args.args) < 5:
                violations.append(f"inner_steps has {len(args.args)} args, expected at least 5")
            break

    if not inner_steps_found:
        violations.append("Missing required function: inner_steps")

    return violations


# =========================================================================
# Data classes
# =========================================================================


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


@dataclass
class GradientInfo:
    grad_norm: float
    grad_vector: list
    layers_with_grad: int
    total_layers: int


class GradientCapturingOptimizer:
    """Same wrapper as the production validator uses."""

    __slots__ = (
        "_opt_impl",
        "model",
        "captured_gradients",
        "step_count",
        "num_steps",
        "_grad_snapshot_gpu",
        "_initialized",
    )

    _PUBLIC_ATTRS = frozenset(
        {
            "step",
            "zero_grad",
            "param_groups",
            "state",
            "state_dict",
            "load_state_dict",
            "add_param_group",
            "finalize_gradients",
            "captured_gradients",
            "num_steps",
            "model",
            "step_count",
        }
    )

    def __init__(self, optimizer, model, num_steps):
        object.__setattr__(self, "_opt_impl", optimizer)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "captured_gradients", None)
        object.__setattr__(self, "step_count", 0)
        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "_grad_snapshot_gpu", None)
        object.__setattr__(self, "_initialized", True)

    def __getattribute__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return object.__getattribute__(self, name)
        if name in GradientCapturingOptimizer._PUBLIC_ATTRS:
            return object.__getattribute__(self, name)
        if name.startswith("_"):
            raise AttributeError(f"Access to '{name}' is not allowed on optimizer wrapper")
        return object.__getattribute__(self, name)

    def __dir__(self):
        return sorted(GradientCapturingOptimizer._PUBLIC_ATTRS)

    def __setattr__(self, name, value):
        try:
            initialized = object.__getattribute__(self, "_initialized")
        except AttributeError:
            initialized = False
        if initialized:
            raise AttributeError(
                f"Cannot modify attribute '{name}' on optimizer wrapper (read-only after init)"
            )
        object.__setattr__(self, name, value)

    def step(self, *args, **kwargs):
        current_step = object.__getattribute__(self, "step_count")
        object.__setattr__(self, "step_count", current_step + 1)

        if current_step == object.__getattribute__(self, "num_steps") - 1:
            snapshot = []
            model = object.__getattribute__(self, "model")
            for param in model.parameters():
                if param.grad is not None:
                    snapshot.append(param.grad.detach().clone())
                else:
                    snapshot.append(None)
            object.__setattr__(self, "_grad_snapshot_gpu", snapshot)

        opt = object.__getattribute__(self, "_opt_impl")
        return opt.step(*args, **kwargs)

    def finalize_gradients(self) -> None:
        snapshot = object.__getattribute__(self, "_grad_snapshot_gpu")
        if snapshot is None:
            return

        grad_vectors_cpu = []
        total_norm_sq = 0.0
        layers_with_grad = 0
        layers_without_grad = 0

        for grad_gpu in snapshot:
            if grad_gpu is not None:
                grad_flat = grad_gpu.cpu().float().view(-1)
                total_norm_sq += grad_flat.pow(2).sum().item()
                if grad_flat.abs().sum().item() > 1e-10:
                    layers_with_grad += 1
                    grad_vectors_cpu.append(grad_flat)
                else:
                    layers_without_grad += 1
                    grad_vectors_cpu.append(grad_flat)
            else:
                layers_without_grad += 1
                grad_vectors_cpu.append(None)

        total_layers = layers_with_grad + layers_without_grad
        object.__setattr__(
            self,
            "captured_gradients",
            GradientInfo(
                grad_norm=total_norm_sq**0.5,
                grad_vector=grad_vectors_cpu,
                layers_with_grad=layers_with_grad,
                total_layers=total_layers,
            ),
        )
        object.__setattr__(self, "_grad_snapshot_gpu", None)

    def zero_grad(self, set_to_none=False):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.param_groups

    @param_groups.setter
    def param_groups(self, value):
        opt = object.__getattribute__(self, "_opt_impl")
        opt.param_groups = value

    def state_dict(self):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.state_dict()

    def load_state_dict(self, state_dict):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.add_param_group(param_group)

    @property
    def state(self):
        opt = object.__getattribute__(self, "_opt_impl")
        return opt.state

    def __getattr__(self, name):
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            raise AttributeError(f"Access to '{name}' is not allowed on optimizer wrapper")
        opt = object.__getattribute__(self, "_opt_impl")
        return getattr(opt, name)


def _enforce_backend_state():
    """Set torch backend to SAME values as production validator."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def _check_backend_state() -> list[str]:
    """Check torch backend settings match production. Returns violations."""
    violations = []
    if torch.backends.cudnn.deterministic:
        violations.append("cudnn.deterministic was set to True")
    if not torch.backends.cudnn.benchmark:
        violations.append("cudnn.benchmark was set to False")
    if not torch.backends.cudnn.allow_tf32:
        violations.append("cudnn.allow_tf32 was set to False")
    if torch.get_float32_matmul_precision() != "high":
        violations.append(
            f"float32_matmul_precision is '{torch.get_float32_matmul_precision()}', expected 'high'"
        )
    if not torch.backends.cuda.flash_sdp_enabled():
        violations.append("flash_sdp was disabled")
    if not torch.backends.cuda.mem_efficient_sdp_enabled():
        violations.append("mem_efficient_sdp was disabled")
    if not torch.backends.cuda.math_sdp_enabled():
        violations.append("math_sdp was disabled")
    return violations


def load_train_module(train_path: Path):
    """Load train.py as a module."""
    spec = importlib.util.spec_from_file_location("train", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_gradients(model: torch.nn.Module) -> GradientInfo:
    """Capture gradient information from model after backward pass."""
    grad_vectors_cpu = []
    total_norm_sq = 0.0
    layers_with_grad = 0
    layers_without_grad = 0

    for param in model.parameters():
        if param.grad is not None:
            grad_flat = param.grad.detach().cpu().float().view(-1)
            total_norm_sq += grad_flat.pow(2).sum().item()
            if grad_flat.abs().sum().item() > 1e-10:
                layers_with_grad += 1
                grad_vectors_cpu.append(grad_flat)
            else:
                layers_without_grad += 1
                grad_vectors_cpu.append(grad_flat)
        else:
            layers_without_grad += 1
            grad_vectors_cpu.append(None)

    return GradientInfo(
        grad_norm=total_norm_sq**0.5,
        grad_vector=grad_vectors_cpu,
        layers_with_grad=layers_with_grad,
        total_layers=layers_with_grad + layers_without_grad,
    )


def run_reference(model, data_iterator, optimizer, num_steps, device):
    """Run reference training with SAME backend settings as production validator."""
    _enforce_backend_state()

    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    grad_info = None

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        loss.backward()

        if step == num_steps - 1:
            grad_info = capture_gradients(model)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = float(loss.item())

    result = InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
    return result, grad_info


def main():
    print("=" * 70)
    print("VERIFYING train.py - Same checks as production validator")
    print("=" * 70)
    print()

    results: list[tuple[str, bool, str]] = []

    # Load configuration
    project_root = Path(__file__).parent.parent
    hparams_path = project_root / "hparams" / "hparams.json"
    hparams = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)

    batch_size = hparams.get("benchmark_batch_size", 4)
    seq_len = hparams.get("benchmark_sequence_length", 1024)
    num_steps = hparams.get("eval_steps", 5)

    verification = hparams.get("verification", {})
    max_loss_difference = verification.get("max_loss_difference", 0.3)
    min_changed_ratio = verification.get("min_params_changed_ratio", 0.8)
    gradient_norm_ratio_max = verification.get("gradient_norm_ratio_max", 1.08)
    weight_relative_error_max = verification.get("weight_relative_error_max", 0.006)

    expected_tokens = batch_size * seq_len * num_steps
    expected_seq_len = seq_len - 1

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Steps per eval: {num_steps}")
    print(f"  Expected tokens: {expected_tokens:,}")
    print(f"  Expected logits seq_len: {expected_seq_len}")
    print(f"  Max loss difference: {max_loss_difference}")
    print(f"  Min params changed: {min_changed_ratio:.0%}")
    gradient_err = gradient_norm_ratio_max - 1.0
    print(f"  Max gradient relative error: {gradient_err:.4f} ({gradient_err * 100:.1f}%)")
    print(
        f"  Max weight relative error: {weight_relative_error_max:.4f}"
        f" ({weight_relative_error_max * 100:.1f}%)"
    )
    print()

    # Check paths
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"
    train_path = project_root / "local_test" / "train.py"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        sys.exit(1)

    if not train_path.exists():
        print(f"train.py not found at {train_path}")
        sys.exit(1)

    # =====================================================================
    # CHECK: Security scan
    # =====================================================================
    print("Security scan (same as production validator)...")
    code = train_path.read_text()
    security_violations = validate_code_structure(code)
    if security_violations:
        for v in security_violations:
            print(f"  [FAILED] {v}")
            results.append(("Security: " + v, False, v))
        print()
        print("  WARNING: Production validator would REJECT this code before execution.")
        print("  Continuing with remaining checks for debugging purposes...")
    else:
        print("  [PASSED] No forbidden patterns detected")
        results.append(("Security scan", True, "No forbidden patterns"))
    print()

    # Load miner's module
    print("Loading train.py...")
    train_module = load_train_module(train_path)

    if not hasattr(train_module, "inner_steps"):
        print("ERROR: train.py must have an 'inner_steps' function!")
        results.append(("inner_steps function", False, "Function not found"))
        _print_summary(results)
        sys.exit(1)
    print("  Found inner_steps function")
    print()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model with random init (same as validator)
    print("Loading model with RANDOM INITIALIZATION (same as validator)...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    )
    model = model.to(device)
    model.gradient_checkpointing_enable()
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Load data
    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
    print(f"Samples: {data.shape[0]:,}, Sequence length: {data.shape[1]}")
    print()

    def create_iterator():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    # Save initial state
    initial_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # === Run reference with CORRECT backend settings ===
    print("Running reference baseline...")
    use_fused = torch.cuda.is_available()
    optimizer_ref = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )
    reference, reference_grad = run_reference(
        model, create_iterator(), optimizer_ref, num_steps, device
    )
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Reference gradient norm: {reference_grad.grad_norm:.4f}")

    reference_final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print("  Captured reference final model state for weight verification")
    del optimizer_ref
    gc.collect()
    torch.cuda.empty_cache()
    print()

    # === Reset model ===
    model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    model.train()

    # === Warmup (same as validator) ===
    print("Running warmup (2 steps, not verified)...")
    warmup_optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )
    warmup_ok = True
    try:
        warmup_result = train_module.inner_steps(
            model=model,
            data_iterator=create_iterator(),
            optimizer=warmup_optimizer,
            num_steps=2,
            device=device,
        )
        if warmup_result is None:
            print("  [FAILED] inner_steps returned None during warmup!")
            results.append(("Warmup", False, "inner_steps returned None"))
            warmup_ok = False
        elif warmup_result.final_logits is None:
            print("  [WARNING] final_logits is None during warmup")
            print("  Warmup passed (with warning)")
        else:
            print("  Warmup passed")
    except Exception as e:
        print(f"  [FAILED] Warmup crashed: {e}")
        results.append(("Warmup", False, str(e)))
        warmup_ok = False
    print()

    if not warmup_ok:
        print("Cannot continue — warmup failed (code doesn't run).")
        _print_summary(results)
        sys.exit(1)

    del warmup_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # === Reset model again ===
    model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    model.train()

    # === Enforce CORRECT backend state (same as production validator) ===
    _enforce_backend_state()

    # === Run miner's code with GradientCapturingOptimizer ===
    print("Running your inner_steps with GradientCapturingOptimizer...")
    base_optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), fused=use_fused
    )
    capturing_optimizer = GradientCapturingOptimizer(base_optimizer, model, num_steps=num_steps)

    miner_result = None
    exec_error = None
    try:
        miner_result = train_module.inner_steps(
            model=model,
            data_iterator=create_iterator(),
            optimizer=capturing_optimizer,
            num_steps=num_steps,
            device=device,
        )
    except AttributeError as e:
        if "read-only" in str(e) or "protected" in str(e):
            exec_error = f"Tried to modify optimizer internals: {e}"
        else:
            exec_error = str(e)
    except Exception as e:
        exec_error = str(e)

    if exec_error:
        print(f"  [FAILED] inner_steps crashed: {exec_error}")
        results.append(("Execution", False, exec_error))
        print()
        print("Cannot continue — inner_steps crashed during timed eval.")
        _print_summary(results)
        sys.exit(1)

    # Finalize gradients
    capturing_optimizer.finalize_gradients()

    # Verify backend settings were not tampered with
    backend_violations = _check_backend_state()
    if backend_violations:
        print("  [WARNING] Backend settings changed during eval:")
        for v in backend_violations:
            print(f"    - {v}")
            results.append((f"Backend: {v}", False, v))
        print("  Production validator would REJECT this submission.")
    print()

    # Verify optimizer wrapper integrity
    if type(capturing_optimizer) is not GradientCapturingOptimizer:
        print("  [FAILED] Optimizer wrapper type was replaced!")
        results.append(("Optimizer integrity", False, "Wrapper type changed"))

    candidate_grad = capturing_optimizer.captured_gradients

    # Check return type
    ok = miner_result is not None and all(
        hasattr(miner_result, attr) for attr in ("final_logits", "total_tokens", "final_loss")
    )
    if not ok:
        print(
            "  [FAILED] inner_steps must return object with final_logits, total_tokens, final_loss"
        )
        results.append(("Return type", False, "Invalid return type"))
        _print_summary(results)
        sys.exit(1)

    candidate = InnerStepsResult(
        final_logits=miner_result.final_logits,
        total_tokens=miner_result.total_tokens,
        final_loss=miner_result.final_loss,
    )

    if candidate_grad is not None:
        print(f"  Candidate loss: {candidate.final_loss:.6f}")
        print(f"  Candidate gradient norm: {candidate_grad.grad_norm:.4f}")
    else:
        print(f"  Candidate loss: {candidate.final_loss:.6f}")
        print("  Candidate gradients: NOT CAPTURED")
    print(f"  Optimizer step count: {capturing_optimizer.step_count}")

    # =====================================================================
    # Run all verification checks
    # =====================================================================
    print()
    print("=" * 70)
    print("VERIFICATION: Running validator checks (same as production)")
    print("=" * 70)

    # CHECK: final_logits not None
    check = "final_logits not None"
    print(f"\n[CHECK] {check}")
    if candidate.final_logits is None:
        print("  [FAILED] final_logits is None! Must return actual logits tensor.")
        results.append((check, False, "final_logits is None"))
    else:
        print("  [PASSED]")
        results.append((check, True, ""))

    # CHECK: Logits shape is 3D
    check = "Logits shape 3D"
    print(f"\n[CHECK] {check}")
    if candidate.final_logits is not None:
        shape = candidate.final_logits.shape
        print(f"  Shape: {tuple(shape)}")
        if len(shape) != 3:
            print(f"  [FAILED] Expected 3D tensor, got {len(shape)}D")
            results.append((check, False, f"Got {len(shape)}D"))
        else:
            print("  [PASSED]")
            results.append((check, True, ""))
    else:
        print("  [SKIPPED] (final_logits is None)")
        results.append((check, False, "Skipped — no logits"))

    # CHECK: Sequence length
    check = "Sequence length"
    print(f"\n[CHECK] {check}")
    if candidate.final_logits is not None and len(candidate.final_logits.shape) >= 2:
        logits_seq_len = candidate.final_logits.shape[1]
        print(f"  Expected: {expected_seq_len}, Got: {logits_seq_len}")
        if logits_seq_len != expected_seq_len:
            print("  [FAILED] Sequence length mismatch — possible truncation!")
            results.append((check, False, f"Got {logits_seq_len}, expected {expected_seq_len}"))
        else:
            print("  [PASSED]")
            results.append((check, True, ""))
    else:
        print("  [SKIPPED] (no valid logits)")
        results.append((check, False, "Skipped — no logits"))

    # CHECK: Token count
    check = "Token count"
    print(f"\n[CHECK] {check}")
    print(f"  Expected: {expected_tokens}, Got: {candidate.total_tokens}")
    if candidate.total_tokens != expected_tokens:
        print("  [FAILED] Token count mismatch!")
        results.append((check, False, f"Got {candidate.total_tokens}, expected {expected_tokens}"))
    else:
        print("  [PASSED]")
        results.append((check, True, ""))

    # CHECK: Loss validity
    check = "Loss validity"
    print(f"\n[CHECK] {check}")
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Candidate loss: {candidate.final_loss:.6f}")

    if candidate.final_loss != candidate.final_loss:
        print("  [FAILED] Loss is NaN!")
        results.append((check, False, "Loss is NaN"))
    elif candidate.final_loss <= 0:
        print(f"  [FAILED] Loss must be positive, got {candidate.final_loss:.4f}")
        results.append((check, False, f"Non-positive loss: {candidate.final_loss:.4f}"))
    elif candidate.final_loss > 100:
        print(f"  [FAILED] Loss unreasonable: {candidate.final_loss:.4f}")
        results.append((check, False, f"Unreasonable loss: {candidate.final_loss:.4f}"))
    else:
        loss_diff = abs(candidate.final_loss - reference.final_loss)
        print(f"  Loss difference: {loss_diff:.4f} (max allowed: {max_loss_difference})")
        if loss_diff > max_loss_difference:
            print("  [FAILED] Loss difference too large!")
            results.append((check, False, f"Diff {loss_diff:.4f} > {max_loss_difference}"))
        else:
            print("  [PASSED]")
            results.append((check, True, ""))

    # CHECK: Trainable parameters
    check = "Trainable parameters (100%)"
    print(f"\n[CHECK] {check}")
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    ratio = trainable_params / total_params if total_params > 0 else 0.0
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({ratio:.1%})")
    if ratio < 1.0:
        print("  [FAILED] All parameters must be trainable!")
        results.append((check, False, f"Only {ratio:.1%} trainable"))
    else:
        print("  [PASSED]")
        results.append((check, True, ""))

    # CHECK: Parameters changed
    check = f"Parameters changed (>={min_changed_ratio:.0%})"
    print(f"\n[CHECK] {check}")
    total_elements = 0
    changed_elements = 0
    for name, param in model.named_parameters():
        if name in initial_state:
            initial = initial_state[name].to(param.device)
            diffs = (param.data - initial).abs()
            total_elements += param.numel()
            changed_elements += (diffs > 1e-6).sum().item()
    changed_ratio = changed_elements / total_elements if total_elements > 0 else 0.0
    print(f"  Changed: {int(changed_elements):,} / {total_elements:,} ({changed_ratio:.1%})")
    if changed_ratio < min_changed_ratio:
        print(f"  [FAILED] Only {changed_ratio:.1%} changed, need {min_changed_ratio:.0%}")
        results.append((check, False, f"{changed_ratio:.1%} < {min_changed_ratio:.0%}"))
    else:
        print("  [PASSED]")
        results.append((check, True, ""))

    # CHECK: Gradient capture
    check = "Gradients captured"
    print(f"\n[CHECK] {check}")
    if candidate_grad is None:
        print("  [FAILED] No gradients captured! optimizer.step() must be called via wrapper.")
        print(f"  Wrapper step_count = {capturing_optimizer.step_count} (expected {num_steps})")
        results.append((check, False, "No gradients captured"))
    else:
        print(f"  [PASSED] step_count={capturing_optimizer.step_count}")
        results.append((check, True, ""))

    # CHECK: Gradient relative error
    check = "Gradient relative error"
    relative_error_threshold = gradient_norm_ratio_max - 1.0
    print(f"\n[CHECK] {check}")
    print(f"  Max allowed: {relative_error_threshold:.4f} ({relative_error_threshold * 100:.1f}%)")

    if candidate_grad is None:
        print("  [SKIPPED] No candidate gradients")
        results.append((check, False, "No gradients to compare"))
    else:
        if candidate_grad.total_layers > 0:
            coverage = candidate_grad.layers_with_grad / candidate_grad.total_layers
            print(
                f"  Gradient coverage: {coverage:.1%}"
                f" ({candidate_grad.layers_with_grad}/{candidate_grad.total_layers})"
            )
            if coverage < 1.0:
                print("  [FAILED] Not all layers have gradients!")
                results.append(("Gradient coverage", False, f"{coverage:.1%} < 100%"))

        ref_vecs = reference_grad.grad_vector
        cand_vecs = candidate_grad.grad_vector
        if ref_vecs and cand_vecs and len(ref_vecs) == len(cand_vecs):
            diff_norm_sq = 0.0
            ref_norm_sq = 0.0
            shape_ok = True
            for ref_layer, cand_layer in zip(ref_vecs, cand_vecs):
                if ref_layer is None or cand_layer is None:
                    continue
                if ref_layer.shape != cand_layer.shape:
                    print(
                        f"  [FAILED] Gradient shape mismatch:"
                        f" {ref_layer.shape} vs {cand_layer.shape}"
                    )
                    shape_ok = False
                    break
                diff = cand_layer - ref_layer
                diff_norm_sq += (diff * diff).sum().item()
                ref_norm_sq += (ref_layer * ref_layer).sum().item()

            if not shape_ok:
                results.append((check, False, "Shape mismatch"))
            else:
                ref_norm = ref_norm_sq**0.5
                diff_norm = diff_norm_sq**0.5
                relative_error = (
                    diff_norm / ref_norm
                    if ref_norm > 0
                    else (0.0 if diff_norm == 0 else float("inf"))
                )

                print(f"  |g - g_truth|: {diff_norm:.6f}")
                print(f"  |g_truth|: {ref_norm:.6f}")
                print(f"  Relative error: {relative_error:.6f}")

                if not math.isfinite(relative_error):
                    print(f"  [FAILED] Non-finite relative error ({relative_error})")
                    results.append((check, False, f"Non-finite ({relative_error})"))
                elif relative_error > relative_error_threshold:
                    print(f"  [FAILED] {relative_error:.6f} > {relative_error_threshold:.6f}")
                    results.append(
                        (check, False, f"{relative_error:.6f} > {relative_error_threshold:.6f}")
                    )
                else:
                    print("  [PASSED]")
                    results.append((check, True, ""))
        else:
            print("  [FAILED] Gradient vectors unavailable or layer count mismatch")
            results.append((check, False, "Vectors unavailable"))

    # CHECK: Final weight verification
    check = "Weight relative error"
    print(f"\n[CHECK] {check}")
    print(
        f"  Max allowed: {weight_relative_error_max:.4f} ({weight_relative_error_max * 100:.1f}%)"
    )

    if reference_final_state is not None:
        w_diff_norm_sq = 0.0
        w_ref_norm_sq = 0.0
        w_total_elements = 0
        w_mismatched_layers = 0

        for name, param in model.named_parameters():
            if name not in reference_final_state:
                continue
            ref_param = reference_final_state[name].to(param.device)
            diff = param.data.float() - ref_param.float()

            layer_diff_sq = (diff * diff).sum().item()
            layer_ref_sq = (ref_param.float() * ref_param.float()).sum().item()

            w_diff_norm_sq += layer_diff_sq
            w_ref_norm_sq += layer_ref_sq
            w_total_elements += param.numel()

            if layer_ref_sq > 0:
                layer_rel_error = (layer_diff_sq**0.5) / (layer_ref_sq**0.5)
                if (
                    not math.isfinite(layer_rel_error)
                    or layer_rel_error > weight_relative_error_max
                ):
                    w_mismatched_layers += 1

        w_ref_norm = w_ref_norm_sq**0.5
        w_diff_norm = w_diff_norm_sq**0.5
        w_relative_error = (
            w_diff_norm / w_ref_norm
            if w_ref_norm > 0
            else (0.0 if w_diff_norm == 0 else float("inf"))
        )

        print(f"  |w_miner - w_ref|: {w_diff_norm:.6f}")
        print(f"  |w_ref|: {w_ref_norm:.6f}")
        print(f"  Relative error: {w_relative_error:.6f}")
        print(f"  Total elements: {w_total_elements:,}")
        print(f"  Mismatched layers: {w_mismatched_layers}")

        if not math.isfinite(w_relative_error):
            print(f"  [FAILED] Non-finite weight error ({w_relative_error})")
            results.append((check, False, f"Non-finite ({w_relative_error})"))
        elif w_relative_error > weight_relative_error_max:
            print(f"  [FAILED] {w_relative_error:.6f} > {weight_relative_error_max:.6f}")
            results.append(
                (check, False, f"{w_relative_error:.6f} > {weight_relative_error_max:.6f}")
            )
        else:
            print("  [PASSED]")
            results.append((check, True, ""))
    else:
        print("  [SKIPPED] (no reference final state available)")
        results.append((check, False, "Skipped — no reference state"))

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    _print_summary(results)

    all_passed = all(passed for _, passed, _ in results)
    sys.exit(0 if all_passed else 1)


def _print_summary(results: list[tuple[str, bool, str]]) -> None:
    passed = [(name, detail) for name, ok, detail in results if ok]
    failed = [(name, detail) for name, ok, detail in results if not ok]

    print()
    print("=" * 70)
    print(f"SUMMARY: {len(passed)} passed, {len(failed)} failed")
    print("=" * 70)

    if passed:
        print()
        print("PASSED:")
        for name, _ in passed:
            print(f"  [PASS] {name}")

    if failed:
        print()
        print("FAILED:")
        for name, detail in failed:
            print(f"  [FAIL] {name}: {detail}")

    print()
    if not failed:
        print("Your submission should pass validator evaluation!")
    else:
        print("Fix the issues above before submitting.")
    print("=" * 70)


if __name__ == "__main__":
    main()
