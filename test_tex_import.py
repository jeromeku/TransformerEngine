
from transformer_engine.common import _get_shared_object_file
import importlib
import sys
framework = "pytorch"
module_name = "transformer_engine"

# breakpoint()
# core_path = _get_shared_object_file("core")
# breakpoint()
# spec = importlib.util.spec_from_file_location(module_name, core_path)
# solib = importlib.util.module_from_spec(spec)
# sys.modules[module_name] = solib
# spec.loader.exec_module(solib)

# so_file_path = _get_shared_object_file(framework)


def import_core_lib():
    import ctypes
    import importlib.util as iu, pathlib as pl 
    core_so = pl.Path(iu.find_spec("transformer_engine").origin).parent.parent / "libtransformer_engine.so" 
    breakpoint()
    ctypes.CDLL(core_so, mode=ctypes.RTLD_GLOBAL)

import_core_lib()

