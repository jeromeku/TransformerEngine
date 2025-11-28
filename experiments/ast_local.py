# file: ast_fast_local.py
import ast
import inspect
import timeit
import dis
from types import FunctionType
from ast_call import ast_transform_ctx

class AttrToLocalTransformer(ast.NodeTransformer):
    """
    Transform function:

        def f(obj):
            return obj.value + obj.value

    into:

        def f(obj):
            _v = obj.value
            return _v + _v

    so that Python uses LOAD_FAST for `_v` instead of repeated LOAD_ATTR.
    """

    def __init__(self, obj_name: str, attr_name: str, local_name: str = "_v"):
        self.obj_name = obj_name
        self.attr_name = attr_name
        self.local_name = local_name

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        breakpoint()

        # Remove all annotations
        for arg in node.args.args:
            print(f"Removing annotation {arg.annotation.id} from {arg.arg}")
            arg.annotation = None
            
        # Insert `_v = obj.value` at top of function body
        assign = ast.Assign(
            targets=[ast.Name(id=self.local_name, ctx=ast.Store())],
            value=ast.Attribute(
                value=ast.Name(id=self.obj_name, ctx=ast.Load()),
                attr=self.attr_name,
                ctx=ast.Load(),
            ),
        )
        node.body.insert(0, assign)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        # Replace obj.value with _v
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == self.obj_name
            and node.attr == self.attr_name
        ):
            return ast.Name(id=self.local_name, ctx=ast.Load())
        return node

def transform_function(fn: FunctionType, obj_name: str, attr_name: str) -> FunctionType:
    src = inspect.getsource(fn)
    tree = ast.parse(src)
    with ast_transform_ctx(tree):
        transformer = AttrToLocalTransformer(obj_name, attr_name)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

    code_obj = compile(new_tree, filename="<ast>", mode="exec")
    namespace: dict = {}
    exec(code_obj, namespace)
    return namespace[fn.__name__]

# Demo types --------------------------------------------------------------

class CppBackedWrapper:
    # Pretend self.value is kept in sync with some C++ value
    def __init__(self, v: int):
        self.value = v

def node_attr(o: CppBackedWrapper):
    # naive implementation
    return o.value + o.value + 1

if __name__ == "__main__":
    o = CppBackedWrapper(123)

    new_fn = transform_function(node_attr, "o", "value")

    print("Original result:", node_attr(o))
    print("Transformed result:", new_fn(o))

    print("\nOriginal bytecode:")
    dis.dis(node_attr)

    print("\nTransformed bytecode:")
    dis.dis(new_fn)

    # micro benchmark
    def bench_original():
        node_attr(o)

    def bench_transformed():
        new_fn(o)

    n = 1_000_00  # one hundred thousand
    t1 = timeit.timeit(bench_original, number=n)
    t2 = timeit.timeit(bench_transformed, number=n)
    print(f"\nOriginal:   {t1:.6f} s")
    print(f"Transformed:{t2:.6f} s")

