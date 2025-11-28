# file: ast_ticked_transform.py
import ast
import inspect
from types import FunctionType
from contextlib import contextmanager

def dump_ast(tree: ast.Module):
    print(ast.dump(tree, indent=2))

@contextmanager
def ast_transform_ctx(tree):
    print(" ------------------  BEFORE  -------------------- ")
    dump_ast(tree)
    print()
    yield

    print(" ------------------  AFTER  -------------------- ")
    dump_ast(tree)
    print()
    print(ast.unparse(tree))

def _fast_ticked(obj) -> bool:
    # Stand-in for a C-level helper that does not need attribute lookup.
    # For demo: call the original method, but pretend it is faster.
    return obj.ticked()

class TickedCallTransformer(ast.NodeTransformer):
    """
    Transform `something.ticked()` into `_fast_ticked(something)`.
    Only when:
      - function is an Attribute named "ticked"
      - no positional or keyword arguments
    """

    def visit_Call(self, node: ast.Call):
        print("Visiting:")
        dump_ast(node)

        self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            attr = node.func
            if (
                attr.attr == "ticked"
                and not node.args
                and not node.keywords
            ):
                # Original base object
                obj_expr = attr.value
                return ast.Call(
                    func=ast.Name(id="_fast_ticked", ctx=ast.Load()),
                    args=[obj_expr],
                    keywords=[],
                )

        return node

def transform_function(fn: FunctionType) -> FunctionType:

    src = inspect.getsource(fn)
    tree = ast.parse(src)
    with ast_transform_ctx(tree):    
        transformer = TickedCallTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
    
    code_obj = compile(new_tree, filename="<ast>", mode="exec")
    namespace: dict = {"_fast_ticked": _fast_ticked}
    exec(code_obj, namespace)
    return namespace[fn.__name__]

# Demo objects -------------------------------------------------------------

class FakeTimeSeries:
    def __init__(self):
        self._tick_count = 0

    def tick(self):
        self._tick_count += 1

    def ticked(self) -> bool:
        # In the real library, this would query a C++-backed state
        return self._tick_count > 0

def hello():
    return "World"

def user_node(ts: "FakeTimeSeries"):
    s = hello()
    
    print(s)

    if ts.ticked():
        return "got_tick"
    else:
        return "no_tick"

if __name__ == "__main__":
    ts = FakeTimeSeries()
    new_fn = transform_function(user_node)

    # print("Original:", user_node(ts))
    # print("Transformed:", new_fn(ts))

    # ts.tick()
    # print("Original after tick:", user_node(ts))
    # print("Transformed after tick:", new_fn(ts))
