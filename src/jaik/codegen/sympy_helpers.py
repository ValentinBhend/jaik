import sympy as sp
from typing import Callable


def _get_apply_pythagorean(angle_pairs) -> Callable:
    """Substitute s²+c²=1 for all known angle pairs, then simplify."""
    def _apply_pythagorean(expr):
        for s, c in angle_pairs:
            # s² → 1-c², then trigsimp handles sqrt(1-(1-c²)) etc.
            expr = expr.subs(s**2 + c**2, 1, simultaneous=True)
            expr = expr.subs(s**2, 1 - c**2, simultaneous=True)
        return expr
    return _apply_pythagorean

def _replace_after_cse(subs_list, red_list, _replace_fn):
    new_repl = [(sym, _replace_fn(expr)) for sym, expr in subs_list]
    new_red  = [_replace_fn(expr) for expr in red_list]
    return new_repl, new_red

# def _replace_sign(expr):
#     a = sp.Wild('a', properties=[lambda x: x.is_Symbol])
#     expr = expr.replace(a / sp.Abs(a), sp.sign(a)) # a / Abs(a) → sign(a)
#     return expr

def _replace_sign(expr):
    a = sp.Wild('a', properties=[lambda x: x.is_Symbol])
    expr = expr.replace( a / sp.Abs(a),  sp.sign(a))
    expr = expr.replace(-a / sp.Abs(a), -sp.sign(a))
    return expr

def _replace_hypot(expr):
    hypot = sp.Function('hypot')
    u = sp.Wild('u')
    v = sp.Wild('v')

    def _is_safe_arg(x):
        for node in sp.preorder_traversal(x):
            if node.is_Pow and node.exp == sp.Rational(1, 2):  # actual sqrt check
                inner = node.args[0]
                if not inner.is_nonnegative:
                    return False
        return True

    def _safe_hypot(u, v):
        if u.has(sp.I) or v.has(sp.I):
            return sp.sqrt(u**2 + v**2)
        if not (_is_safe_arg(u) and _is_safe_arg(v)):
            return sp.sqrt(u**2 + v**2)
        return hypot(u, v)

    def _safe_inv_hypot(u, v):
        if u.has(sp.I) or v.has(sp.I):
            return (u**2 + v**2) ** sp.Rational(-1, 2)
        if not (_is_safe_arg(u) and _is_safe_arg(v)):
            return (u**2 + v**2) ** sp.Rational(-1, 2)
        return 1 / hypot(u, v)

    expr = expr.replace(sp.sqrt(u**2 + v**2), _safe_hypot)
    expr = expr.replace((u**2 + v**2)**sp.Rational(-1, 2), _safe_inv_hypot)
    return expr