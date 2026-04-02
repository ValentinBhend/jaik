import sympy as sp

R_syms = [[sp.Symbol(f'r{i+1}{j+1}') for j in range(3)] for i in range(3)]
p_syms = [sp.Symbol(f'p{i+1}') for i in range(3)]
R_06   = sp.Matrix(R_syms)
p_0T   = sp.Matrix(p_syms)

print(R_06)
print(p_0T)

a = R_06 * p_0T
print(a)
print(a.shape)