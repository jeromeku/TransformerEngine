# file: node_stateful_transform.py
import ast
import inspect
from functools import wraps
from types import FunctionType

def dump_ast(tree: ast.Module):
    print(ast.dump(tree, indent=2))

class ToGeneratorTransformer(ast.NodeTransformer):
    """
    Transform a function body into:

        def fn(...):
            while True:
                <original_body, but Return -> Yield>

    This mimics what CSP does to make nodes stateful generators.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.decorator_list = []
        self.generic_visit(node)  # First transform Returns inside
        
        # Wrap original body in "while True: ..."
        while_node = ast.While(
            test=ast.Constant(value=True),
            body=node.body,
            orelse=[],
        )

        node.body = [while_node]
        return node

    def visit_Return(self, node: ast.Return):
        # Replace "return X" with "yield X"
        y_expr = ast.Yield(value=node.value)
        y_statement = ast.Expr(y_expr)
        return y_statement
    
def _transform_to_generator(fn: FunctionType) -> FunctionType:
    src = inspect.getsource(fn)
    tree = ast.parse(src)
    print("Original func")
    dump_ast(tree)
    # We expect our function at module top-level
    transformer = ToGeneratorTransformer()
    
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    print("After xform")
    dump_ast(new_tree)
    print(ast.unparse(new_tree))

    code_obj = compile(new_tree, filename="<ast>", mode="exec")
    namespace: dict = {}
    exec(code_obj, namespace)
    return namespace[fn.__name__]

def make_stateful(fn: FunctionType) -> FunctionType:
    """
    Decorator: transforms fn into a generator factory.
    For demo purposes we hard-code x = 1 inside.
    """
    gen_fn = _transform_to_generator(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # “Hot path” API: we just return a generator
        # Our transformed function expects the same args, but we can
        # feed inputs via closure, send, etc.
        return gen_fn(*args, **kwargs)

    return wrapper

def sum_node(s):
    s += 1
    return s

if __name__ == "__main__":
    og = sum_node(0)
    g = make_stateful(sum_node)(0)
    print(next(g))  # 1
    print(next(g))  
