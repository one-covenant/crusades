"""
Shared security policy constants for miner code validation.

Single source of truth consumed by both the production validator
(environments/templar/env.py) and the local pre-submission checker
(local_test/verify.py).  Any additions or removals should be made
here so the two stay in sync automatically.
"""

# ---------------------------------------------------------------------------
# Forbidden string patterns
# ---------------------------------------------------------------------------
# Scanned inside string literals, bytes().decode(), str.join(), concatenation,
# f-strings, %-formatting, and attribute names throughout the AST.

FORBIDDEN_STRINGS: list[str] = [
    # Dunder escape hatches
    "__setattr__",
    "__delattr__",
    "__class__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__import__",
    "__builtins__",
    "__dict__",
    "__globals__",
    "__code__",
    "__func__",
    "__self__",
    "__init_subclass__",
    "__traceback__",
    "__getattribute__",
    "__build_class__",
    # Timer / profiling internals
    "perf_counter",
    "_perf_counter",
    "monotonic",
    "_monotonic",
    "_REAL_PC_ID",
    "_REAL_MONO_ID",
    "_REAL_SYNC_ID",
    "_cuda_synchronize",
    # CUDA event timing
    "elapsed_time",
    "_et_orig",
    "_cuda_elapsed_time",
    "_REAL_ET_ID",
    # Cross-entropy reference poisoning
    "_real_cross_entropy",
    "_REAL_CE_ID",
    # GC / introspection
    "get_objects",
    "get_referrers",
    "get_referents",
    # Validator optimizer internals
    "captured_gradients",
    "_opt_impl",
    "_grad_snapshot_gpu",
    "step_count",
    "GradientCapturingOptimizer",
    # Dynamic import / module access
    "importlib",
    "import_module",
    "sys.modules",
    # Dangerous builtins (as strings for obfuscation detection)
    "setattr",
    "getattr",
    "delattr",
    "globals",
    "locals",
    "builtins",
    # Frame / traceback traversal
    "tb_frame",
    "tb_next",
    "f_globals",
    "f_builtins",
    "f_locals",
    "f_back",
    "co_consts",
    "co_names",
    # Operator helpers that can bypass attribute guards
    "operator.attrgetter",
    "operator.methodcaller",
    "attrgetter",
    "methodcaller",
    # Torch internal access (even under aliases)
    "_C",
    "_dynamo",
    "_inductor",
    # Validator env internals
    "_CACHE",
    "initial_state",
    "_hidden_modules",
    "_sensitive_keys",
    # unittest.mock attribute patching
    "mock.patch",
    "MagicMock",
    "unittest.mock",
]

# ---------------------------------------------------------------------------
# Forbidden module imports
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES: set[str] = {
    "ctypes",
    "_ctypes",
    "gc",
    "subprocess",
    "sys",
    "os",
    "pathlib",
    "io",
    "_io",
    "socket",
    "http",
    "urllib",
    "requests",
    "shutil",
    "tempfile",
    "signal",
    "threading",
    "_thread",
    "multiprocessing",
    "inspect",
    "ast",
    "dis",
    "code",
    "codeop",
    "compileall",
    "pickle",
    "shelve",
    "marshal",
    "builtins",
    "_builtins",
    "operator",
    "types",
    "codecs",
    "base64",
    "pdb",
    "pprint",
    "runpy",
    "linecache",
    "pkgutil",
    "atexit",
    "site",
    "zipimport",
    # Stack inspection — reveals internal file paths and code structure
    "traceback",
    # Block importing the validator's own module (timer tampering attack)
    "env",
    "time",
    "__main__",
    "miner_train",
    # logging.Logger.manager.loggerDict holds refs to all loggers —
    # traversable to reach the env module's logger and back to env itself
    "logging",
    # unittest.mock.patch can replace any attribute on any object
    "unittest",
    "mock",
    # functools.partial can wrap forbidden functions
    "functools",
    # weakref can observe GC behavior and hold references
    "weakref",
}

# Dotted module paths blocked even when the base module is allowed
FORBIDDEN_DOTTED_MODULES: set[str] = {
    "torch.multiprocessing",
    "torch.utils.dlpack",
    "torch.distributed",
    "numpy.ctypeslib",
}

# Substring checked against import paths to block C++ extension compilation
FORBIDDEN_IMPORT_SUBSTRINGS: set[str] = {
    "cpp_extension",
}

# ---------------------------------------------------------------------------
# Torch-specific import / alias guards
# ---------------------------------------------------------------------------

# torch symbols that enable bypassing attribute-based guards when imported directly
FORBIDDEN_TORCH_SYMBOL_IMPORTS: set[str] = {
    "load",
    "compile",
    "_C",
    "_dynamo",
    "_inductor",
}

# Backend toggles that bypass attribute-call checks if imported as bare names
FORBIDDEN_TORCH_BACKEND_SYMBOL_IMPORTS: set[str] = set()

# torch attributes that should never be rebound or called via alias variables
FORBIDDEN_TORCH_ATTRIBUTE_ALIASES: set[str] = {
    "load",
    "compile",
    "_C",
    "_dynamo",
    "_inductor",
}

# torch internal config namespaces whose .config attrs must not be written
FORBIDDEN_TORCH_CONFIG_MODULES: set[str] = {
    "_dynamo",
    "_inductor",
}

# ---------------------------------------------------------------------------
# Forbidden bare-name references
# ---------------------------------------------------------------------------
# Builtins like __import__, exec, setattr are blocked as CALLS, but a miner
# can bypass by storing a reference: `_imp = __import__; _imp("sys")`.
# This set blocks them even when only *referenced* (ast.Name in Load context).

FORBIDDEN_NAMES: set[str] = {
    "__import__",
    "exec",
    "eval",
    "compile",
    "breakpoint",
    "setattr",
    "getattr",
    "delattr",
    "vars",
    "dir",
    "globals",
    "locals",
    "type",
    "memoryview",
    "open",
    "chr",
    "ord",
    "input",
    "classmethod",
    "staticmethod",
    "property",
    "__build_class__",
}

# Subset of builtins blocked when called (overlaps with FORBIDDEN_NAMES
# but kept separate for the call-site check).
FORBIDDEN_BUILTINS: set[str] = {
    "setattr",
    "getattr",
    "delattr",
    "vars",
    "dir",
    "globals",
    "locals",
    "type",
    "memoryview",
    "open",
    "chr",
    "ord",
    "breakpoint",
    "input",
    "classmethod",
    "staticmethod",
    "property",
}

# Dangerous functions blocked as direct calls: name(...) form
FORBIDDEN_DIRECT_CALLS: set[str] = {
    "exec",
    "eval",
    "compile",
    "__import__",
}

# Dangerous functions blocked as attribute calls: obj.attr(...) form
FORBIDDEN_ATTR_CALLS: set[str] = {
    "exec",
    "eval",
    "__import__",
}

# ---------------------------------------------------------------------------
# Timer / timing-related attribute guards
# ---------------------------------------------------------------------------

# Attribute names blocked on ANY object (ast.Attribute check)
FORBIDDEN_TIMER_ATTRS: set[str] = {
    "perf_counter",
    "_perf_counter",
    "monotonic",
    "_monotonic",
    "perf_counter_ns",
    "monotonic_ns",
    "_cuda_synchronize",
    "elapsed_time",
    "_et_orig",
}

# Attribute names that must never appear as assignment targets
FORBIDDEN_ASSIGNMENT_ATTRS: set[str] = {
    "synchronize",
    "elapsed_time",
}

# Allowed torch submodule import paths.  Only these may be imported via
# ``import torch.X.Y as Z`` or ``from torch.X import Y``.  The local
# alias (Z / Y) is tracked so that *attribute assignment* on it
# (e.g. ``F.cross_entropy = fake``) is blocked — legitimate optimisation
# never needs to monkey-patch torch internals.
ALLOWED_TORCH_SUBMODULE_IMPORTS: set[str] = {
    "torch.nn",
    "torch.nn.functional",
    "torch.amp",
    "torch.cuda",
    "torch.cuda.amp",
}

# Prefixes of ``torch.*`` dotted paths where attribute *assignment* is
# permitted.  Everything else under ``torch.*`` is blocked to prevent
# monkey-patching (e.g. ``torch.autograd.backward = fake``).
# Currently only ``torch.backends.*`` is needed for legitimate config
# such as ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction``.
ALLOWED_TORCH_ASSIGNMENT_PREFIXES: set[str] = {
    "torch.backends",
}

# Gradient-control functions that must never be called — disabling
# gradients would make backward() a no-op and bypass gradient verification.
FORBIDDEN_GRAD_TOGGLE_CALLS: set[str] = {
    "set_grad_enabled",
    "inference_mode",
}

# ---------------------------------------------------------------------------
# Attribute-level AST guards
# ---------------------------------------------------------------------------

# object.__setattr__ / object.__delattr__ — blocked on `object` receiver
FORBIDDEN_OBJECT_DUNDER_ATTRS: set[str] = {
    "__setattr__",
    "__delattr__",
}

# gc.get_objects() etc. — blocked when receiver is `gc`
FORBIDDEN_GC_ATTRS: set[str] = {
    "get_objects",
    "get_referrers",
    "get_referents",
}

# cuDNN backend settings — blocked as Store targets
FORBIDDEN_CUDNN_ATTRS: set[str] = {
    "deterministic",
    "benchmark",
}

# Backend toggle / precision calls — no longer blocked; verification
# catches any numerical divergence and miners bear the consequences of
# changing precision settings.  Kept as empty set for backward compat.
FORBIDDEN_BACKEND_TOGGLE_ATTRS: set[str] = set()

# sys.modules access — blocked when receiver matches these names
FORBIDDEN_SYS_MODULE_NAMES: set[str] = {
    "sys",
    "_sys",
}

# Dunder / frame / traceback attributes blocked on any object
FORBIDDEN_INTROSPECTION_ATTRS: set[str] = {
    "__globals__",
    "__code__",
    "__func__",
    "__self__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__init_subclass__",
    "__traceback__",
    "tb_frame",
    "tb_next",
    "f_globals",
    "f_builtins",
    "f_locals",
    "f_code",
    "f_back",
    "co_consts",
    "co_names",
    "__getattribute__",
}
