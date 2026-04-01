import numpy as np


def model_params():
    """UR10e robot parameters and configuration (NumPy)
    Returns a dict-like object with the same fields as the MATLAB `params` struct.
    """
    params = {}

    # DH Parameters
    params['a2'] = -0.6127
    params['a3'] = -0.57155
    params['d1'] = 0.1807
    params['d4'] = 0.17415
    params['d5'] = 0.11985
    params['d6'] = 0.11655

    params['a'] = np.array([0.0, params['a2'], params['a3'], 0.0, 0.0, 0.0])
    params['d'] = np.array([params['d1'], 0.0, 0.0, params['d4'], params['d5'], params['d6']])
    params['alpha'] = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])

    # Calibration offsets
    params['delta_a'] = np.array([2.36625875785141412e-06, 0.0955714149407106417,
                                   0.0028668800205046141, 6.01265538537812174e-05,
                                   -7.78325218726985484e-05, 0.0])
    params['delta_d'] = np.array([-0.000145303886550401939, 296.744101272290436,
                                   -305.284767441875829, 8.54069232073686635,
                                   -6.21696849946451469e-06, -0.000978508765798594138])
    params['delta_alpha'] = np.array([0.00029541528440946152, -0.0011034308757148861,
                                      0.00614706357153506962, -0.000816596798962399006,
                                      0.00091466809892049028, 0.0])
    params['delta_theta'] = np.array([3.59281525995347462e-07, -0.564457958804729154,
                                      6.75558885512157481, 0.0920584768235426093,
                                      -1.16024708789186359e-06, -2.26695349126240786e-07])

    # Physical parameters
    params['G_BASE'] = np.array([0.0, 0.0, -9.82])
    params['Ttau'] = 0.02
    params['Bv'] = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
    params['Fc'] = np.zeros(6)

    params['useUrconf'] = True

    if params['useUrconf']:
        params['m'] = np.array([7.369, 13.051, 3.989, 2.100, 1.980, 0.615])
        cm = np.array([[0.021, 0.000,  0.027],
                       [0.380, 0.000,  0.158],
                       [0.240, 0.000,  0.068],
                       [0.000, 0.007,  0.018],
                       [0.000, 0.007,  0.018],
                       [0.000, 0.000, -0.026]])
        params['rc'] = cm.T

        Iraw = np.array([[0.03408,  2e-05,  -0.00425,  2e-05, 0.03529, 8e-05,  -0.00425, 8e-05, 0.02156],
                         [0.02814,  5e-05,  -0.01561,  5e-05, 0.77068, 2e-05,  -0.01561, 2e-05, 0.76943],
                         [0.01014,  8e-05,   0.00916,  8e-05, 0.30928, 0.000,   0.00916, 0.000,  0.30646],
                         [0.00296, -1e-05,  -0.00000, -1e-05, 0.00222,-0.00024, -0.00000,-0.00024,0.00258],
                         [0.00296, -1e-05,  -0.00000, -1e-05, 0.00222,-0.00024, -0.00000,-0.00024,0.00258],
                         [0.00040,  0.000,  -0.00000,  0.000, 0.00041, 0.000,  -0.00000, 0.000,  0.00034]])

        I = np.zeros((3, 3, 6))
        for i in range(6):
            Ii = np.array([[Iraw[i, 0], Iraw[i, 1], Iraw[i, 2]],
                           [Iraw[i, 3], Iraw[i, 4], Iraw[i, 5]],
                           [Iraw[i, 6], Iraw[i, 7], Iraw[i, 8]]])
            Ii = 0.5 * (Ii + Ii.T)
            I[:, :, i] = Ii
        params['I'] = I
    else:
        params['m'] = np.array([5.0, 12.0, 5.0, 3.0, 1.5, 0.5])
        params['rc'] = np.array([[0,        params['a2']/2, params['a3']/2,  0,           0,           0],
                                 [0,        0,           0,            0,           0,           0],
                                 [params['d1']/2, 0,       0,            params['d4']/2, params['d5']/2, params['d6']/2]])
        I = np.zeros((3,3,6))
        L = params['d1']; m = params['m'][0]; I[:, :, 0] = np.diag([1/12*m*L**2, 1/12*m*L**2, 1e-3*m])
        L = abs(params['a2']); m = params['m'][1]; I[:, :, 1] = np.diag([1e-3*m, 1/12*m*L**2, 1/12*m*L**2])
        L = abs(params['a3']); m = params['m'][2]; I[:, :, 2] = np.diag([1e-3*m, 1/12*m*L**2, 1/12*m*L**2])
        L = params['d4']; m = params['m'][3]; I[:, :, 3] = np.diag([1/12*m*L**2, 1/12*m*L**2, 1e-3*m])
        L = params['d5']; m = params['m'][4]; I[:, :, 4] = np.diag([1/12*m*L**2, 1/12*m*L**2, 1e-3*m])
        L = params['d6']; m = params['m'][5]; I[:, :, 5] = np.diag([1/12*m*L**2, 1/12*m*L**2, 1e-3*m])
        params['I'] = I

    params['Xtcp'] = np.eye(4)
    params['oriPolicy'] = 'principal'
    params['eps_sing'] = 1e-9
    params['eps'] = 1e-12

    return params
