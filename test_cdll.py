# save as trace_te_import.py  and run:  python trace_te_import.py
import builtins
import ctypes
import functools
import importlib
import inspect
import pathlib
import sys
import time

# # 1 ───── show every Python import ────────────────────────────────────────────
# _orig_import = builtins.__import__


# def _imp(name, *a, **kw):
#     t0 = time.perf_counter()
#     mod = _orig_import(name, *a, **kw)
#     print(f"[IMPORT] {name:<45}  +{(time.perf_counter() - t0) * 1e3:6.1f} ms")
#     return mod


# builtins.__import__ = _imp

# 2 ───── show every dlopen that ctypes performs ─────────────────────────────
_orig_cdll_init = ctypes.CDLL.__init__


def _cdll(self, libname, *a, **kw):
    print(f"[CDLL ] dlopen('{libname}')")
    _orig_cdll_init(self, libname, *a, **kw)


ctypes.CDLL.__init__ = _cdll


# 3 ───── tap into TE’s two private helpers before they are run  ──────────────
def _patch_te_helpers():
    import transformer_engine.common as tec

    def banner(fn):
        @functools.wraps(fn)
        def _wrap(*a, **k):
            print(f"[TE] → {fn.__name__}")
            out = fn(*a, **k)
            print(f"[TE] ← {fn.__name__}")
            return out

        return _wrap

    tec._load_core_library = banner(tec._load_core_library)
    tec.load_framework_extension = banner(tec.load_framework_extension)


_patch_te_helpers()

# 4 ───── do the regular import that triggers everything ─────────────────────
import transformer_engine.pytorch as tex

print("✓ Transformer Engine ready")
