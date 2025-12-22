# file: ast_output_binop.py
import ast
import inspect
from types import FunctionType
from ast_call import ast_transform_ctx

class Sink:
    def __init__(self, name: str):
        self.name = name
        self.values = []

    def output(self, value):
        # “Slow” path: attribute lookup + bound method
        self.values.append(("output_method", value))

    def __add__(self, other):
        # “Fast” path: binary operator via nb_add slot
        self.values.append(("add_slot", other))
        return self  # allow chaining if you want

class OutputToBinOpTransformer(ast.NodeTransformer):
    """
    Transform `obj.output(arg)` into `obj + arg`.
    Only when:
      - function is Attribute named "output"
      - exactly one positional argument
      - no keyword arguments
    """

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            attr = node.func
            if attr.attr == "output" and len(node.args) == 1 and not node.keywords:
                obj_expr = attr.value
                arg_expr = node.args[0]
                return ast.BinOp(
                    left=obj_expr,
                    op=ast.Add(),
                    right=arg_expr,
                )

        return node

def transform_function(fn: FunctionType) -> FunctionType:
    src = inspect.getsource(fn)


    tree = ast.parse(src)
    with ast_transform_ctx(tree):
        transformer = OutputToBinOpTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

    code_obj = compile(new_tree, filename="<ast>", mode="exec")
    namespace: dict = {"Sink": Sink}
    exec(code_obj, namespace)
    return namespace[fn.__name__]

# User-facing node --------------------------------------------------------

def user_node(s: Sink, x: int):
    # What the user writes
    s.output(x)
    s.output(x + 1)
    return s

if __name__ == "__main__":
    s1 = Sink("original")
    s2 = Sink("transformed")

    # Run original
    user_node(s1, 10)

    # Run transformed
    new_fn = transform_function(user_node)
    new_fn(s2, 10)

    print("Original values:", s1.values)
    print("Transformed values:", s2.values)
