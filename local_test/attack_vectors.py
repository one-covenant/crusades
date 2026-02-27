import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Dummy:
    """Small object that safely absorbs arbitrary attribute/call usage."""

    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __getattr__(self, _name):
        return _Dummy()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _FakeFastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def post(self, *args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def get(self, *args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator


@contextmanager
def _patched_modules(overrides: dict[str, object]):
    sentinel = object()
    previous: dict[str, object] = {}
    for name, module in overrides.items():
        previous[name] = sys.modules.get(name, sentinel)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_security_defs():
    return _load_module("crusades.core.security_defs", ROOT / "src/crusades/core/security_defs.py")


def _make_torch_stub() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__getattr__ = lambda _name: _Dummy()
    torch.Tensor = object
    torch.cuda = _Dummy()
    torch.cuda.is_available = lambda: False
    torch.cuda.Event = types.SimpleNamespace(elapsed_time=lambda *_a, **_k: 0.0)
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.__getattr__ = lambda _name: _Dummy()
    torch.nn.Module = object
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    torch.nn.functional.__getattr__ = lambda _name: _Dummy()
    torch.nn.functional.cross_entropy = _Dummy()
    torch.optim = types.ModuleType("torch.optim")
    torch.optim.__getattr__ = lambda _name: _Dummy()
    torch.optim.Optimizer = object
    torch.backends = _Dummy()
    torch.set_grad_enabled = _Dummy()
    torch.set_default_dtype = _Dummy()
    torch.set_float32_matmul_precision = _Dummy()
    torch.get_float32_matmul_precision = lambda: "high"
    torch.manual_seed = _Dummy()
    return torch


def _make_transformers_stub() -> types.ModuleType:
    transformers = types.ModuleType("transformers")
    transformers.__getattr__ = lambda _name: _Dummy()
    return transformers


def _make_crusades_overrides(security_defs) -> dict[str, object]:
    crusades_pkg = types.ModuleType("crusades")
    core_pkg = types.ModuleType("crusades.core")
    crusades_pkg.core = core_pkg
    core_pkg.security_defs = security_defs
    return {
        "crusades": crusades_pkg,
        "crusades.core": core_pkg,
        "crusades.core.security_defs": security_defs,
    }


def _load_verify_validate():
    security_defs = _load_security_defs()
    torch_stub = _make_torch_stub()
    overrides = {
        **_make_crusades_overrides(security_defs),
        "torch": torch_stub,
        "torch.nn": torch_stub.nn,
        "torch.nn.functional": torch_stub.nn.functional,
        "transformers": _make_transformers_stub(),
    }
    with _patched_modules(overrides):
        verify_module = _load_module("verify_under_test", ROOT / "local_test/verify.py")
    return verify_module.validate_code_structure


def _load_env_validate():
    security_defs = _load_security_defs()
    torch_stub = _make_torch_stub()
    overrides = {
        **_make_crusades_overrides(security_defs),
        "torch": torch_stub,
        "torch.nn": torch_stub.nn,
        "torch.nn.functional": torch_stub.nn.functional,
        "torch.optim": torch_stub.optim,
        "fastapi": types.SimpleNamespace(FastAPI=_FakeFastAPI),
        "pydantic": types.SimpleNamespace(BaseModel=object),
    }
    with _patched_modules(overrides):
        env_module = _load_module("env_under_test", ROOT / "environments/templar/env.py")
    return env_module._validate_code_structure


def _verify_security_blocked(validate_code_structure, code: str) -> tuple[bool, str]:
    violations = validate_code_structure(code)
    security_violations = [v for v in violations if "inner_steps" not in v]
    if security_violations:
        return True, security_violations[0]
    return False, "No security violation (only structural checks or full pass)"


def _env_security_blocked(validate_code_structure, code: str) -> tuple[bool, str]:
    safe, message = validate_code_structure(code)
    reason = message or ("No validation message" if safe else "Validation failed without message")
    blocked = (not safe) and reason.startswith("Security violation:")
    return blocked, reason


def _bypass_cases() -> list[tuple[str, str]]:
    cases = []
    n = 0

    def add(name, code):
        nonlocal n
        n += 1
        cases.append((f"{n}. {name}", code))

    # === __name__ BINDING VECTORS ===
    add("MatchAs binding", "match '__main__':\n    case __name__:\n        pass\n")
    add("MatchStar binding", "match ['__main__']:\n    case [*__name__]:\n        pass\n")
    add(
        "MatchMapping binding", "match {'key': '__main__'}:\n    case {**__name__}:\n        pass\n"
    )
    add(
        "Exception handler binding",
        "try:\n    raise Exception()\nexcept Exception as __name__:\n    pass\n",
    )
    add("For loop binding", "for __name__ in ['__main__']:\n    pass\n")
    add("List comprehension binding", "_ = [__name__ for __name__ in ['__main__']]\n")
    add("Context manager binding", "with open('/dev/null') as __name__:\n    pass\n")
    add("Type parameters (generics)", "def hack[__name__]():\n    pass\n")
    add("Walrus operator binding", "(__name__ := '__main__')\n")
    add("Class namespace hijack", "class __name__:\n    pass\n")
    add("Global declaration", "def hack():\n    global __name__\n    __name__ = '__main__'\n")
    add(
        "Nonlocal declaration",
        "def outer():\n    x = 1\n    def inner():\n        nonlocal __name__\n",
    )
    add("ImportFrom __name__", "from math import __name__\n")
    add("Import alias to __name__", "import math as __name__\n")
    add("AnnAssign __name__", "__name__: str = '__main__'\n")
    add("AugAssign __name__", "__name__ += '_main_'\n")
    add("Del __name__", "del __name__\n")
    add("__name__ as function arg", "def hack(__name__='__main__'):\n    pass\n")
    add("Torch attr mutation __name__", "import torch\ntorch.nn.functional.__name__ = '__main__'\n")

    # === STRING OBFUSCATION ===
    add("str(bytearray) obfuscation", "x = str(bytearray([102, 95, 98, 97, 99, 107]), 'ascii')\n")
    add("bytes().decode() obfuscation", "x = bytes([102, 95, 98, 97, 99, 107]).decode()\n")
    add("b-literal .decode()", "x = b'f_back'.decode()\n")
    add("str.join() obfuscation", 'x = "".join(["f", "_", "b", "a", "c", "k"])\n')
    add("String concatenation", 'x = "f_" + "back"\n')
    add("%-format obfuscation (single)", 'x = "%s_back" % "f"\n')
    add("%-format obfuscation (tuple)", 'x = "%s%s" % ("f_", "back")\n')
    add("f-string constant parts", "x = f\"{'f'}_back\"\n")

    # === FORBIDDEN MODULE IMPORTS ===
    add("import os", "import os\n")
    add("import sys", "import sys\n")
    add("import subprocess", "import subprocess\n")
    add("import inspect", "import inspect\n")
    add("import ctypes", "import ctypes\n")
    add("import gc", "import gc\n")
    add("import time", "import time\n")
    add("import socket", "import socket\n")
    add("import pickle", "import pickle\n")
    add("import threading", "import threading\n")
    add("import ast", "import ast\n")
    add("import builtins", "import builtins\n")
    add("import operator", "import operator\n")
    add("import types", "import types\n")
    add("import functools", "import functools\n")
    add("import weakref", "import weakref\n")
    add("import logging", "import logging\n")
    add("from os import path", "from os import path\n")
    add("import unittest.mock", "import unittest.mock\n")
    add("Dotted torch.distributed", "import torch.distributed\n")

    # === FORBIDDEN BUILTINS / NAMES ===
    add("exec() call", "exec('print(1)')\n")
    add("eval() call", "eval('1+1')\n")
    add("compile() call", "compile('x=1', '<s>', 'exec')\n")
    add("__import__() call", "__import__('os')\n")
    add("setattr() reference", "f = setattr\n")
    add("getattr() reference", "f = getattr\n")
    add("delattr() reference", "f = delattr\n")
    add("globals() reference", "f = globals\n")
    add("locals() reference", "f = locals\n")
    add("open() reference", "f = open\n")
    add("vars() reference", "f = vars\n")
    add("dir() reference", "f = dir\n")
    add("type() reference", "f = type\n")
    add("memoryview() reference", "f = memoryview\n")
    add("chr() reference", "f = chr\n")
    add("ord() reference", "f = ord\n")
    add("breakpoint() call", "breakpoint()\n")

    # === FRAME / INTROSPECTION ACCESS ===
    add("f_globals attribute", "import torch\nx = torch.f_globals\n")
    add("f_back attribute", "import torch\nx = torch.f_back\n")
    add("f_locals attribute", "import torch\nx = torch.f_locals\n")
    add("__globals__ attribute", "import torch\nx = torch.__globals__\n")
    add("__code__ attribute", "import torch\nx = torch.__code__\n")
    add("__subclasses__ attribute", "import torch\nx = torch.__subclasses__\n")
    add("__traceback__ attribute", "import torch\nx = torch.__traceback__\n")
    add("co_consts attribute", "import torch\nx = torch.co_consts\n")
    add("gi_frame attribute", "import torch\nx = torch.gi_frame\n")
    add("cr_frame attribute", "import torch\nx = torch.cr_frame\n")
    add("ag_frame attribute", "import torch\nx = torch.ag_frame\n")
    add("tb_frame attribute", "import torch\nx = torch.tb_frame\n")
    add("__getattribute__ attribute", "import torch\nx = torch.__getattribute__\n")

    # === TORCH MONKEY-PATCHING ===
    add("torch._C access", "import torch\nx = torch._C\n")
    add("torch._dynamo access", "import torch\nx = torch._dynamo\n")
    add("torch.compile alias", "import torch\nc = torch.compile\n")
    add("torch.load alias", "import torch\nld = torch.load\n")
    add("from torch import _C", "from torch import _C\n")
    add("from torch import compile", "from torch import compile\n")
    add("torch._dynamo.config write", "import torch\ntorch._dynamo.config.suppress_errors = True\n")
    add(
        "torch.nn.functional.cross_entropy overwrite",
        "import torch\nimport torch.nn.functional as F\nF.cross_entropy = lambda *a, **k: None\n",
    )

    # === TIMER TAMPERING ===
    add("perf_counter attr access", "import torch\nx = torch.perf_counter\n")
    add("monotonic attr access", "import torch\nx = torch.monotonic\n")
    add("elapsed_time assignment", "import torch\ntorch.cuda.Event.elapsed_time = lambda *a: 0.0\n")
    add("synchronize assignment", "import torch\ntorch.cuda.synchronize = lambda: None\n")

    # === GRADIENT TOGGLE ===
    add("set_grad_enabled call", "import torch\ntorch.set_grad_enabled(False)\n")
    add("inference_mode call", "import torch\ntorch.inference_mode()\n")

    # === __builtins__ / __dict__ ACCESS ===
    add("__builtins__ name access", "x = __builtins__\n")
    add("__dict__ attribute", "import torch\nx = torch.__dict__\n")
    add("object.__setattr__", "import torch\nobject.__setattr__(torch, 'x', 1)\n")
    add("object.__delattr__", "import torch\nobject.__delattr__(torch, 'x')\n")

    # === sys.modules ACCESS ===
    add("sys.modules string", 'x = "sys.modules"\n')

    # === VALIDATOR INTERNAL STRINGS ===
    add("_CACHE string", 'x = "_CACHE"\n')
    add("initial_state string", 'x = "initial_state"\n')
    add("GradientCapturingOptimizer string", 'x = "GradientCapturingOptimizer"\n')
    add("captured_gradients string", 'x = "captured_gradients"\n')

    # === FORBIDDEN DECORATORS ===
    add("@exec decorator", "@exec\ndef f(): pass\n")
    add("@eval decorator", "@eval\ndef f(): pass\n")

    # === FORBIDDEN IMPORT SUBSTRINGS ===
    add("cpp_extension import", "from torch.utils import cpp_extension\n")

    return cases


def main() -> int:
    verify_validate = _load_verify_validate()
    env_validate = _load_env_validate()

    print("=" * 60)
    print("  Templar Crusades — Security Scanner Regression Suite")
    print("=" * 60)

    failed_cases = 0
    total_checks = 0
    bypassed: list[str] = []

    for case_name, case_code in _bypass_cases():
        verify_blocked, verify_reason = _verify_security_blocked(verify_validate, case_code)
        env_blocked, env_reason = _env_security_blocked(env_validate, case_code)
        total_checks += 2

        verify_status = "\033[32mBLOCKED\033[0m" if verify_blocked else "\033[31mBYPASSED\033[0m"
        env_status = "\033[32mBLOCKED\033[0m" if env_blocked else "\033[31mBYPASSED\033[0m"

        print(f"\n  [{case_name}]")
        print(f"    verify.py: {verify_status} | {verify_reason}")
        print(f"    env.py:    {env_status} | {env_reason}")

        if not verify_blocked:
            failed_cases += 1
            bypassed.append(f"{case_name} (verify.py)")
        if not env_blocked:
            failed_cases += 1
            bypassed.append(f"{case_name} (env.py)")

    passed_checks = total_checks - failed_cases
    print("\n" + "=" * 60)
    print(f"  Result: {passed_checks}/{total_checks} checks passed")
    if bypassed:
        print(f"\n  \033[31mBYPASSED ({len(bypassed)}):\033[0m")
        for b in bypassed:
            print(f"    - {b}")
    else:
        print("\n  \033[32mAll attack vectors blocked.\033[0m")
    print("=" * 60)
    return 0 if failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
