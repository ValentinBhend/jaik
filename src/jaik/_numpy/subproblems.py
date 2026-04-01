from math import atan2, sqrt
import numpy as np
from scipy import linalg as sci

#Returns theta, is_LS
def sp1(p1, p2, k):
   KxP = np.cross(k, p1)
   A = np.vstack((KxP, -np.cross(k, KxP)))
   x = np.dot(A, p2)
   theta = atan2(x[0], x[1])
   is_LS = abs(np.linalg.norm(p1) - np.linalg.norm(p2)) > 1e-8 or abs(np.dot(k,p1) - np.dot(k,p2)) > 1e-8
   return theta, is_LS


#Run subproblem
def sp2E(p0, p1, p2, k1, k2): # TODO cleanup
   KxP1 = np.cross(k1, p1)
   KxP2 = np.cross(k2, p2)
   
   #np.block appends two arrays together
   A1 = np.block([KxP1, -np.cross(k1, KxP1)])
   A2 = np.block([KxP2, -np.cross(k2, KxP2)])

   #Append A1, A2 then reshape into desired shape
   A = np.block([A1, -A2])
   A = np.reshape(A, (4,3))

   p = -k1*np.dot(k1, p1) + k2*np.dot(k2, p2) - p0
   
   radius_1_sq = np.dot(KxP1, KxP1)
   radius_2_sq = np.dot(KxP2, KxP2)

   alpha = radius_1_sq/(radius_1_sq+radius_2_sq)
   beta  = radius_2_sq/(radius_1_sq+radius_2_sq) 

   #reshape vector to matrix w/ opposite dimensions
   M_inv = np.eye(3) + (k1 * np.reshape(k1, (3,1)))*(alpha/(1.0-alpha)) 

   AAT_inv = 1.0/(radius_1_sq+radius_2_sq)*(M_inv + M_inv@np.array([k2]).T*k2 @ \
             M_inv*beta/(1.0-k2@M_inv@np.array([k2]).T*beta))

   x_ls = np.dot(A@AAT_inv,p)

   n_sym = np.reshape(np.cross(k1, k2), (3,1))
   pinv_A1 = A1/radius_1_sq
   pinv_A2 = A2/radius_2_sq

   A_perp_tilde = np.reshape(np.block([pinv_A1, pinv_A2]), (4, 3)) @ n_sym

   num = (pow(np.linalg.norm(x_ls[2:4], 2), 2) - 1.0)*pow(np.linalg.norm(A_perp_tilde[0:2], 2), 2) - \
         (pow(np.linalg.norm(x_ls[0:2], 2), 2) - 1.0)*pow(np.linalg.norm(A_perp_tilde[2:4], 2), 2)
   #den was in form [[den]], so we converted to integer through indexing
   den = 2*(np.reshape(x_ls[0:2], (1,2)) @ A_perp_tilde[0:2] * pow(np.linalg.norm(A_perp_tilde[2:4], 2), 2) - \
         np.reshape(x_ls[2:4], (1,2)) @ A_perp_tilde[2:4] * pow(np.linalg.norm(A_perp_tilde[0:2], 2), 2))[0][0]
   
   xi = num/den
   #We want sc as vector/list, so we flatten both arrays
   sc = x_ls.flatten() + xi*A_perp_tilde.flatten()

   theta1 = atan2(sc[0], sc[1])
   theta2 = atan2(sc[2], sc[3])

   return [theta1, theta2]



#Code called to run subproblem
def sp2(p1, p2, k1, k2):
   #Rescale for least-squares
   p1_norm = p1 / np.linalg.norm(p1)
   p2_norm = p2 / np.linalg.norm(p2)

   KxP1 = np.cross(k1, p1_norm)
   KxP2 = np.cross(k2, p2_norm)

   #np.block appends two arrays together
   A1 = np.vstack((KxP1, -np.cross(k1, KxP1)))
   A2 = np.vstack((KxP2, -np.cross(k2, KxP2)))

   radius_1_sq = np.dot(KxP1, KxP1)
   radius_2_sq = np.dot(KxP2, KxP2)
   
   k1_d_p1 = np.dot(k1, p1_norm)
   k2_d_p2 = np.dot(k2, p2_norm)
   k1_d_k2 = np.dot(k1, k2)

   ls_frac = 1 / (1 - k1_d_k2**2)

   alpha_1 = ls_frac * (k1_d_p1 - (k1_d_k2 * k2_d_p2))
   alpha_2 = ls_frac * (k2_d_p2 - (k1_d_k2 * k1_d_p1))

   x_ls_1 = alpha_2 * np.dot(A1, k2) / radius_1_sq
   x_ls_2 = alpha_1 * np.dot(A2, k1) / radius_2_sq
   x_ls = np.concatenate((x_ls_1, x_ls_2))

   is_LS = np.linalg.norm(x_ls[0]) >= 1
   if is_LS:
      theta1 = atan2(x_ls[0], x_ls[1])
      theta2 = atan2(x_ls[2], x_ls[3])
   else:
      n_sym = np.cross(k1, k2)
      pinv_A1 = A1 / radius_1_sq
      pinv_A2 = A2 / radius_2_sq
      A_perp_tilde = np.vstack((pinv_A1, pinv_A2)) @ n_sym

      xi = sqrt(1.0 - np.linalg.norm(x_ls[:2])**2) / np.linalg.norm(A_perp_tilde[0:2])
      sc_1 = x_ls + A_perp_tilde * xi
      sc_2 = x_ls - A_perp_tilde * xi

      theta1 = np.array([atan2(sc_1[0], sc_1[1]), atan2(sc_2[0], sc_2[1])])
      theta2 = np.array([atan2(sc_1[2], sc_1[3]), atan2(sc_2[2], sc_2[3])])
   return theta1, theta2, is_LS


def sp3(p1, p2, k, d):
   KxP = np.cross(k, p1)
   A1 = np.vstack((KxP, -np.cross(k, KxP)))  # (2, 3)
   A = -2 * p2 @ A1.T

   norm_A_sq = np.dot(A,A)
   norm_A = sqrt(norm_A_sq)
   b = d**2 - np.linalg.norm(p2 - np.dot(k, p1) * k)**2 - np.linalg.norm(KxP)**2

   x_ls = A1 @ (-2 * p2 * b / norm_A_sq)

   is_LS = x_ls @ x_ls > 1
   if is_LS:
      theta = atan2(x_ls[0], x_ls[1])
   else:
      xi = sqrt(1 - b**2 / norm_A_sq)

      A_perp_tilde = np.array([A[1], -A[0]]) # (2,)
      A_perp = A_perp_tilde / norm_A

      sc_1 = x_ls + xi * A_perp
      sc_2 = x_ls - xi * A_perp

      theta = np.array([atan2(sc_1[0], sc_1[1]), atan2(sc_2[0], sc_2[1])])
   return theta, is_LS


def sp4(p, k, h, d):
    A11 = np.cross(k, p)
    A1 = np.vstack((A11, -np.cross(k, A11)))   # (2, 3)
    A = h @ A1.T                                 # (2,)
    b = d - np.dot(h, k) * np.dot(k, p)
    norm_A2 = A @ A
    x_ls = A1 @ (h * b)
    is_LS = norm_A2 <= b**2
    if is_LS:
        theta = atan2(x_ls[0], x_ls[1])
    else:
        xi = sqrt(norm_A2 - b**2)
        A_perp_tilde = np.array([A[1], -A[0]])  # (2,)
        sc_1 = x_ls + xi * A_perp_tilde
        sc_2 = x_ls - xi * A_perp_tilde
        theta = np.array([atan2(sc_1[0], sc_1[1]), atan2(sc_2[0], sc_2[1])])
    return theta, is_LS
   

#Represents polynomials as coeffecient vectors
def cone_polynomials(p0_i, k_i, p_i, p_i_s, k2): # TODO cleanup
   #||A x + p_S - H k_2||^2 = 
   #-H^2 + P(H) +- sqrt(R(H))
   #Reperesent polynomials P_i, R_i as coefficient vectors
   #(Highest powers of H first)
   kiXk2 = np.cross(k_i, k2)
   kiXkiXk2 = np.cross(k_i, kiXk2)
   norm_kiXk2_sq = np.dot(kiXk2, kiXk2)

   kiXpi = np.cross(k_i, p_i)
   norm_kiXpi_sq = np.dot(kiXpi, kiXpi)

   delta = np.dot(k2 ,p_i_s)
   alpha = kiXkiXk2 @ np.reshape(p0_i, (3, 1)) / norm_kiXk2_sq
   beta =  kiXk2 @ np.reshape(p0_i, (3, 1)) / norm_kiXk2_sq

   P_const = norm_kiXpi_sq + np.dot(p_i_s, p_i_s) + 2 * alpha * delta
   #P = np.array([-2*alpha, P_const])
   R = np.array([-1, 2*delta, -pow(delta, 2)]) #-(H-delta_i)^2
   R[len(R) - 1] = R[len(R) - 1] + norm_kiXpi_sq*norm_kiXk2_sq #||A_i' k_2||^2 - (H-delta_i)^2
   return np.array([-2.0*alpha, P_const]), pow(2.0*beta, 2) * R


#Run subproblem
def sp5(p0, p1, p2, p3, k1, k2, k3): # TODO cleanup
   i_soln = 0

   p1_s = p0 + np.reshape(k1, (3, 1)) @ np.reshape(k1, (1, 3)) @ p1
   p3_s = p2 + np.reshape(k3, (3, 1)) @ np.reshape(k3, (1, 3)) @ p3
   delta1 = np.dot(k2, p1_s)
   delta3 = np.dot(k2, p3_s)   

   P_1, R_1 = cone_polynomials(p0, k1, p1, p1_s, k2)
   P_3, R_3 = cone_polynomials(p2, k3, p3, p3_s, k2)

   #Now solve the quadratic for H   
   P_13 = P_1 - P_3

   #Use 1D version of matrix
   P_13_sq = np.convolve(P_13[:, 0], P_13[:, 0])

   RHS = R_3 - R_1 - P_13_sq
   EQN = np.convolve(RHS, RHS)-4.0 * np.convolve(P_13_sq, R_1)

   all_roots = np.reshape(np.roots(EQN), (np.roots(EQN).size, 1))
   H_vec = all_roots[np.real(all_roots) == all_roots] #Find only those w/ imaginary part == 0

   H_vec = H_vec.real # for coder, removes non-existant imaginary part
   
   KxP1 = np.cross(k1, p1)
   KxP3 = np.cross(k3, p3)

   #np.concatenate appends two arrays together
   A1 = np.concatenate(([KxP1], [-np.cross(k1, KxP1)]), axis = 0).T
   A3 = np.concatenate(([KxP3], [-np.cross(k3, KxP3)]), axis = 0).T

   signs = [[1, 1, -1, -1], [1, -1, 1, -1]]
   J = np.array([[0, 1], [-1, 0]])

   #Setup return variables
   theta1 = []
   theta2 = []
   theta3 = []
   for i_H in range(len(H_vec)):
      H = H_vec[i_H]

      const_1 = k2 @ A1 * (H-delta1)
      const_3 = k2 @ A3 * (H-delta3)
      
      pm_1 = k2 @ A1 @ J * np.emath.sqrt(np.linalg.norm(k2 @ A1, 2)*np.linalg.norm(k2 @ A1, 2) - (H-delta1)*(H-delta1))
      pm_3 = k2 @ A3 @ J * np.emath.sqrt(np.linalg.norm(k2 @ A3, 2)*np.linalg.norm(k2 @ A3, 2) - (H-delta3)*(H-delta3))

      for i_sign in range(4):
         sign_1 = signs[0][i_sign]
         sign_3 = signs[1][i_sign]

         sc1 = const_1 + sign_1 * pm_1
         sc1 = sc1 / pow(np.linalg.norm(k2 @ A1, 2), 2)

         sc3 = const_3 + sign_3 * pm_3
         sc3 = sc3 / pow(np.linalg.norm(k2 @ A3, 2), 2)

         v1 = A1 @ sc1 + p1_s
         v3 = A3 @ sc3 + p3_s

         if abs(np.linalg.norm(v1-H*k2, 2) - np.linalg.norm(v3-H*k2, 2)) < 1E-6:
            i_soln = 1 + i_soln
            theta1.append(atan2(sc1[0], sc1[1]))
            theta2.append(sp1(v3, v1, k2)[0])
            theta3.append(atan2(sc3[0], sc3[1]))
   return theta1, theta2, theta3


def solve_2_ellipse_numeric(xm1, xn1, xm2, xn2): # TODO cleanup
   A_1 = np.transpose(xn1)@xn1
   a = A_1[0][0]
   b = 2*A_1[1][0]
   c = A_1[1][1]
   B_1 = 2*np.transpose(xm1)@xn1
   d = B_1[0][0]
   e = B_1[0][1]
   f = (np.transpose(xm1)@xm1-1)[0][0]

   A_2 = np.transpose(xn2)@xn2
   a1 = A_2[0][0]
   b1 = 2*A_2[1][0]
   c1 = A_2[1][1]
   B_2 = 2*np.transpose(xm2)@xn2
   d1 = B_2[0][0]
   e1 = B_2[0][1]
   fq = (np.transpose(xm2)@xm2-1)[0][0]

   z0 = f*a*pow(d1,2)+pow(a,2)*pow(fq,2)-d*a*d1*fq+pow(a1,2)*pow(f,2)-2*a*fq*a1*f-d*d1*a1*f+a1*pow(d,2)*fq

   z1 = e1*pow(d,2)*a1-fq*d1*a*b-2*a*fq*a1*e-f*a1*b1*d+2*d1*b1*a*f+2*e1*fq*pow(a,2)+pow(d1,2)*a*e-e1*d1*a*d-2*a*e1*a1*f-f*a1*d1*b+2*f*e*pow(a1,2)-fq*b1*a*d-e*a1*d1*d+2*fq*b*a1*d

   z2 = pow(e1,2)*pow(a,2)+2*c1*fq*pow(a,2)-e*a1*d1*b+fq*a1*pow(b,2)-e*a1*b1*d-fq*b1*a*b-2*a*e1*a1*e+2*d1*b1*a*e-c1*d1*a*d-2*a*c1*a1*f+pow(b1,2)*a*f+2*e1*b*a1*d+pow(e,2)*pow(a1,2)-c*a1*d1*d-e1*b1*a*d+2*f*c*pow(a1,2)-f*a1*b1*b+c1*pow(d,2)*a1+pow(d1,2)*a*c-e1*d1*a*b-2*a*fq*a1*c

   z3 = -2*a*a1*c*e1+e1*a1*pow(b,2)+2*c1*b*a1*d-c*a1*b1*d+pow(b1,2)*a*e-e1*b1*a*b-2*a*c1*a1*e-e*a1*b1*b-c1*b1*a*d+2*e1*c1*pow(a,2)+2*e*c*pow(a1,2)-c*a1*d1*b+2*d1*b1*a*c-c1*d1*a*b

   z4 = pow(a,2)*pow(c1,2)-2*a*c1*a1*c+pow(a1,2)*pow(c,2)-b*a*b1*c1-b*b1*a1*c+pow(b,2)*a1*c1+c*a*pow(b1,2)

   y = np.roots(np.array([z4, z3, z2, z1, z0]))

   y = np.real(y[np.real(y) == y]) #Grab only those with no imaginary parts
   
   x = -(a*fq+a*c1*y**2-a1*c*y**2+a*e1*y-a1*e*y-a1*f)/(a*b1*y+a*d1-a1*b*y-a1*d)

   return x, y #This could just return the calculations of x,y for optimization purposes

def sp6(H, K, P, d1, d2): # TODO cleanup
   k1Xp1 = np.cross(K[0], P[0])
   k1Xp2 = np.cross(K[1], P[1])
   k1Xp3 = np.cross(K[2], P[2])
   k1Xp4 = np.cross(K[3], P[3])

   A_1 = np.concatenate(([k1Xp1], [-np.cross(K[0], k1Xp1)]), axis = 0)
   A_2 = np.concatenate(([k1Xp2], [-np.cross(K[1], k1Xp2)]), axis = 0)
   A_3 = np.concatenate(([k1Xp3], [-np.cross(K[2], k1Xp3)]), axis = 0)
   A_4 = np.concatenate(([k1Xp4], [-np.cross(K[3], k1Xp4)]), axis = 0)
   A   = np.reshape(np.concatenate(([H[0]@A_1.T], [H[1]@A_2.T], [H[2]@A_3.T], [H[3]@A_4.T]), axis = 0), (2,4))
   
   x_min = np.linalg.lstsq(A, np.concatenate(([d1 - H[0]@np.reshape(K[0], (3,1))*K[0]@np.reshape(P[0], (3,1))-H[1]@np.reshape(K[1], (3,1))*K[1]@np.reshape(P[1], (3,1))], 
   [d2 -H[2]@np.reshape(K[2], (3,1))*K[2]@np.reshape(P[2], (3,1))-H[3]@np.reshape(K[3], (3,1))*K[3]@np.reshape(P[3], (3,1))]), axis = 0), rcond=None)

   x_min = x_min[0] #Removes residuals, matrix rank, and singularities   

   x_null = np.reshape(np.array(sci.null_space(A)), (2, 4)) #Save only the answer
   x_null_1 = np.array([x_null[0][0], x_null[0][2], x_null[1][0], x_null[1][2]])
   x_null_2 = np.array([x_null[0][1], x_null[0][3], x_null[1][1], x_null[1][3]])

   xi_1, xi_2 = solve_2_ellipse_numeric(x_min[0:2], np.reshape(x_null[0:1], (2,2)), x_min[2:4], np.reshape(x_null[1:2], (2,2)))

   theta1 = []
   theta2 = []
   for i in range(np.size(xi_1)):
      x = x_min + np.reshape(x_null_1*xi_1[i] + x_null_2*xi_2[i], (4,1))
      theta1.append(atan2(x[0], x[1]))
      theta2.append(atan2(x[2], x[3]))
   return [theta1, theta2]