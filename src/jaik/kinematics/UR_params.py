import numpy as np


# ── robot parameter presets ───────────────────────────────────────────────────

# All UR-type robots share the same alpha pattern
_UR_ALPHA = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

# Named presets: (a2, a3, d1, d4, d5, d6) in meters, signed per UR convention
# Source: Universal Robots support articles
#   https://www.universal-robots.com/articles/ur/application-installation/
#   dh-parameters-for-calculations-of-kinematics-and-dynamics/
_UR_PARAMS = {
    # e-series
    "UR3e":    dict(a2=-0.24355,  a3=-0.2132,   d1=0.15185,  d4=0.13105,  d5=0.08535, d6=0.0921),
    "UR5e":    dict(a2=-0.425,    a3=-0.3922,   d1=0.1625,   d4=0.1333,   d5=0.0997,  d6=0.0996),
    "UR7e":    dict(a2=-0.425,    a3=-0.3922,   d1=0.1625,   d4=0.1333,   d5=0.0997,  d6=0.0996),
    "UR10e":   dict(a2=-0.6127,   a3=-0.57155,  d1=0.1807,   d4=0.17415,  d5=0.11985, d6=0.11655),
    "UR12e":   dict(a2=-0.6127,   a3=-0.57155,  d1=0.1807,   d4=0.17415,  d5=0.11985, d6=0.11655),
    "UR16e":   dict(a2=-0.4784,   a3=-0.36,     d1=0.1807,   d4=0.17415,  d5=0.11985, d6=0.11655),
    "UR20":    dict(a2=-0.8620,   a3=-0.7287,   d1=0.2363,   d4=0.2010,   d5=0.1593,  d6=0.1543),
    "UR30":    dict(a2=-0.6370,   a3=-0.5037,   d1=0.2363,   d4=0.2010,   d5=0.1593,  d6=0.1543),
    # CB3 series
    "UR3":     dict(a2=-0.24365,  a3=-0.21325,  d1=0.1519,   d4=0.11235,  d5=0.08535, d6=0.0819),
    "UR5":     dict(a2=-0.425,    a3=-0.39225,  d1=0.089159, d4=0.10915,  d5=0.09465, d6=0.0823),
    "UR10":    dict(a2=-0.612,    a3=-0.5723,   d1=0.1273,   d4=0.163941, d5=0.1157,  d6=0.0922),
    # heavy payload
    "UR8long": dict(a2=-0.8989,   a3=-0.7149,   d1=0.2186,   d4=0.1824,   d5=0.1361,  d6=0.1434),
    "UR15":    dict(a2=-0.6475,   a3=-0.5164,   d1=0.2186,   d4=0.1824,   d5=0.1361,  d6=0.1434),
    "UR18":    dict(a2=-0.475,    a3=-0.3389,   d1=0.2186,   d4=0.1824,   d5=0.1361,  d6=0.1434),
}