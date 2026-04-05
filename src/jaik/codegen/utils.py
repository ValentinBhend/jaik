import numpy as np
import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

def _vec(arr):
    """numpy 3-vector → sympy Matrix column, entries as rationals."""
    return sp.Matrix([sp.Float(arr[i]) for i in range(3)])

def _clean_hp(arr, tol=1e-10):
    """
    Round near-zero and near-unit entries to exact values.
    UR joint axes are exactly aligned with coordinate axes — any deviation
    from 0/±1 is purely floating point noise from DH→PoE conversion.
    """
    result = arr.copy().astype(float)
    result[np.abs(result) < tol] = 0.0
    result[np.abs(result - 1.0) < tol] = 1.0
    result[np.abs(result + 1.0) < tol] = -1.0
    return result

def _sym_rot(k_vec, s_theta, c_theta):
    """Rodrigues rotation: k_vec is sympy column vector, theta is sympy expr."""
    kx, ky, kz = k_vec
    c = c_theta
    s = s_theta
    t = 1 - c
    return sp.Matrix([
        [t*kx*kx + c,    t*kx*ky - s*kz, t*kx*kz + s*ky],
        [t*kx*ky + s*kz, t*ky*ky + c,    t*ky*kz - s*kx],
        [t*kx*kz - s*ky, t*ky*kz + s*kx, t*kz*kz + c   ],
    ])

def _generate_param_symbols(H_num, P_num):
    param_map = {}
    
    def _sym_vec(arr, name):
        vals = []
        for i in range(3):
            # Check if it's a significant constant (not 0 or 1)
            val = arr[i]
            if abs(val) > 1e-10 and abs(val - 1.0) > 1e-10 and abs(val + 1.0) > 1e-10:
                if val > 0:
                    symbol = sp.Symbol(f"{name}{i+1}", real=True, nonzero=True, positive=True)
                else:
                    symbol = sp.Symbol(f"{name}{i+1}", real=True, nonzero=True, negative=True)
                param_map[symbol] = float(val)
                vals.append(symbol)
            else:
                vals.append(sp.Integer(int(round(val))))
        return sp.Matrix(vals)

    H_sym = [_sym_vec(H_num[:, j], f"H{j}") for j in range(6)]
    P_sym = [_sym_vec(P_num[:, j], f"P{j}") for j in range(7)]
    
    return H_sym, P_sym, param_map

class _JnpPrinter(PythonCodePrinter):    
    def __init__(self, settings=None):
        super().__init__(settings)
        # Force these to be handled by our _print_ methods
        self.known_functions.pop('hypot', None)
        self.known_functions.pop('sign', None)
    
    def _print_Function(self, expr):
        # This handles the sp.Function('hypot') objects
        if expr.func.__name__ == 'hypot':
            return f"jnp.hypot({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        if expr.func.__name__ == 'sign':
            return f"jnp.sign({self._print(expr.args[0])})"
        
        # Fallback for sin, cos, etc.
        return super()._print_Function(expr)

    def _print_hypot(self, expr):
        arg0 = self._print(expr.args[0])
        arg1 = self._print(expr.args[1])
        return f"np.hypot({arg0}, {arg1})"
    # 2. Handle SymPy's symbolic sign function
    def _print_sign(self, expr):
        arg0 = self._print(expr.args[0])
        return f"jnp.sign({arg0})"
    
    def _print_sin(self, expr):
        return f"jnp.sin({self._print(expr.args[0])})"
    def _print_cos(self, expr):
        return f"jnp.cos({self._print(expr.args[0])})"
    def _print_sqrt(self, expr):
        return f"jnp.sqrt({self._print(expr.args[0])})"
    def _print_Pow(self, expr):
        b, e = expr.args
        if e == sp.Rational(1, 2):
            return f"jnp.sqrt({self._print(b)})"
        if e == sp.Rational(-1, 2) or e == -0.5:
            return f"(1.0 / jnp.sqrt({self._print(b)}))"
        if e == -1:
            return f"(1.0 / ({self._print(b)}))"
        return f"(({self._print(b)}) ** {self._print(e)})" # Wrap the base in its own parentheses!
    def _print_atan2(self, expr):
        return f"jnp.arctan2({self._print(expr.args[0])}, {self._print(expr.args[1])})"
    def _print_acos(self, expr):
        return f"jnp.arccos({self._print(expr.args[0])})"
    def _print_asin(self, expr):
        return f"jnp.arcsin({self._print(expr.args[0])})"
    def _print_Float(self, expr):
        return repr(float(expr))
    def _print_Integer(self, expr):
        return repr(int(expr))
    def _print_Rational(self, expr):
        return repr(float(expr))
    
class _NumbaPrinter(PythonCodePrinter):
    def __init__(self, settings=None):
        super().__init__(settings)
        # Force these to be handled by our _print_ methods
        self.known_functions.pop('hypot', None)
        self.known_functions.pop('sign', None)
    
    def _print_Function(self, expr):
        # This handles the sp.Function('hypot') objects
        if expr.func.__name__ == 'hypot':
            return f"np.hypot({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        if expr.func.__name__ == 'sign':
            return f"np.sign({self._print(expr.args[0])})"
        
        # Fallback for sin, cos, etc.
        return super()._print_Function(expr)

    def _print_hypot(self, expr):
        arg0 = self._print(expr.args[0])
        arg1 = self._print(expr.args[1])
        return f"np.hypot({arg0}, {arg1})"

    # 2. Handle SymPy's symbolic sign function
    def _print_sign(self, expr):
        arg0 = self._print(expr.args[0])
        return f"np.sign({arg0})"
    def _print_sin(self, expr):
        return f"np.sin({self._print(expr.args[0])})"
    def _print_cos(self, expr):
        return f"np.cos({self._print(expr.args[0])})"
    def _print_sqrt(self, expr):
        return f"np.sqrt({self._print(expr.args[0])})"
    def _print_Pow(self, expr):
        b, e = expr.args
        if e == sp.Rational(1, 2):
            return f"np.sqrt({self._print(b)})"
        if e == sp.Rational(-1, 2) or e == -0.5:
            return f"(1.0 / np.sqrt({self._print(b)}))"
        if e == -1:
            return f"(1.0 / ({self._print(b)}))"
        return f"(({self._print(b)}) ** {self._print(e)})" # Wrap the base in its own parentheses!
    def _print_atan2(self, expr):
        return f"np.arctan2({self._print(expr.args[0])}, {self._print(expr.args[1])})"
    def _print_acos(self, expr):
        return f"np.arccos({self._print(expr.args[0])})"
    def _print_asin(self, expr):
        return f"np.arcsin({self._print(expr.args[0])})"
    def _print_Float(self, expr):
        return repr(float(expr))
    def _print_Integer(self, expr):
        return repr(int(expr))
    def _print_Rational(self, expr):
        return repr(float(expr))