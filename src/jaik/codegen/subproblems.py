import sympy as sp


def _sp1(p1, p2, k):
    """
    Subproblem 1: Find rotation about axis k that takes p1 to p2.
    Matches Rodrigues: R(k, theta) * p1 = p2
    """
    # 1. Ensure projections are perpendicular to the axis
    p1_p = p1 - k * k.dot(p1)
    p2_p = p2 - k * k.dot(p2)
    
    # 2. Basis vectors
    # u (cosine axis) is the direction of p1_p
    # v (sine axis) must be k cross u to follow right-hand rule
    u = p1_p
    v = k.cross(u)
    
    # 3. Component projections
    # p2_p = cos(theta)*u + sin(theta)*v
    y = p2_p.dot(v)
    x = p2_p.dot(u)
    
    mag = sp.sqrt(y**2 + x**2)
    return y / mag, x / mag

def _sp3(p1, p2, k, d):
    """Mechanical translation of SP3 to Sincos."""
    KxP = k.cross(p1)
    A1_r0 = KxP
    A1_r1 = -k.cross(KxP)
    
    A0 = -2 * p2.dot(A1_r0)
    A1v = -2 * p2.dot(A1_r1)
    norm_A_sq = A0**2 + A1v**2
    norm_A = sp.sqrt(norm_A_sq)
    
    p2_proj = p2 - k * k.dot(p1)
    p2_proj_sq = p2_proj.dot(p2_proj)
    KxP_sq = KxP.dot(KxP)
    
    b = d**2 - p2_proj_sq - KxP_sq
    
    # Least squares components
    x_ls0 = A1_r0.dot(-2 * p2) * b / norm_A_sq
    x_ls1 = A1_r1.dot(-2 * p2) * b / norm_A_sq
    
    # Discriminant for the two solutions
    xi = sp.sqrt(sp.expand(1 - b**2 / norm_A_sq))
    
    A_perp0 = A1v / norm_A
    A_perp1 = -A0 / norm_A
    
    # Raw sine and cosine components
    s1_raw, c1_raw = x_ls0 + xi * A_perp0, x_ls1 + xi * A_perp1
    s2_raw, c2_raw = x_ls0 - xi * A_perp0, x_ls1 - xi * A_perp1

    # return (s1_raw/norm_A, c1_raw/norm_A), (s2_raw/norm_A, c2_raw/norm_A)
    
    # Normalize to ensure unit vectors (sin^2 + cos^2 = 1)
    mag1 = sp.sqrt(s1_raw**2 + c1_raw**2)
    mag2 = sp.sqrt(s2_raw**2 + c2_raw**2)

    return (s1_raw/mag1, c1_raw/mag1), (s2_raw/mag2, c2_raw/mag2)

def _sp4(p, k, h, d):
    """
    Subproblem 4: Find rotation about axis k such that h.dot(R*p) = d
    Matches the sc1/sc2 logic from the working atan2 version.
    """
    A11 = k.cross(p)
    A1_r0 = A11          # This is the 'sine' direction basis
    A1_r1 = -k.cross(A11) # This is the 'cosine' direction basis
    
    A0 = h.dot(A1_r0)
    A1v = h.dot(A1_r1)
    b = d - h.dot(k) * k.dot(p)
    norm_A2 = A0**2 + A1v**2
    
    # These represent the 'Least Squares' center point in the (s, c) plane
    x_ls0 = A0 * b
    x_ls1 = A1v * b
    
    # xi is the 'distance' from the LS center to the valid circle intersections
    xi = sp.sqrt(sp.expand(norm_A2 - b**2))
    
    # Combine to get two solutions for (sin, cos)
    # Solution 1
    s1_raw = x_ls0 + xi * A1v
    c1_raw = x_ls1 - xi * A0
    # Solution 2
    s2_raw = x_ls0 - xi * A1v
    c2_raw = x_ls1 + xi * A0

    # mag = sp.sqrt(norm_A2) --> fails eval
    # return (s1_raw/mag, c1_raw/mag), (s2_raw/mag, c2_raw/mag)
    
    mag1 = sp.sqrt(s1_raw**2 + c1_raw**2)
    mag2 = sp.sqrt(s2_raw**2 + c2_raw**2)
    
    return (s1_raw/mag1, c1_raw/mag1), (s2_raw/mag2, c2_raw/mag2)