'''
Created on May 25, 2015
Modified on Feb 05, 2018

@author: Xu He
@note: Logical operations on conceptors
https://github.com/he-xu/CAB
'''

import numpy as np;


def NOT(C, out_mode = "simple"):
  """
  Compute NOT operation of conceptor.
  
  @param R: conceptor matrix
  @param out_mode: output mode ("simple"/"complete")
  
  @return not_C: NOT C
  @return U: eigen vectors of not_C
  @return S: eigen values of not_C
  """
  
  dim = C.shape[0]
  
  not_C = np.eye(dim) - C
  

  if out_mode == "complete":
    U, S, _ = np.linalg.svd(not_C)
    return not_C, U, S
  else:
    return not_C
  
def AND(C, B, out_mode = "simple", tol = 1e-14):
  """
  Compute AND Operation of two conceptor matrices
  
  @param C: a conceptor matrix
  @param B: another conceptor matrix
  @param out_mode: output mode ("simple"/"complete")
  @param tol: adjust parameter for almost zero
  
  @return C_and_B: C AND B
  @return U: eigen vectors of C_and_B
  @return S: eigen values of C_and_B
  """
  
  dim = C.shape[0]
  
  UC, SC, _ = np.linalg.svd(C)
  UB, SB, _ = np.linalg.svd(B)
  
  num_rank_C = np.sum((SC > tol).astype(int))
  num_rank_B = np.sum((SB > tol).astype(int))
  
  UC0 = UC[:, num_rank_C:]
  UB0 = UB[:, num_rank_B:]
  
  W, sigma, _ = np.linalg.svd(UC0.dot(UC0.T) + UB0.dot(UB0.T))
  num_rank_sigma = np.sum((sigma > tol).astype(int))
  Wgk = W[:, num_rank_sigma:]
  
  C_and_B = Wgk.dot(np.linalg.inv(Wgk.T.dot(np.linalg.pinv(C, tol) + np.linalg.pinv(B, tol) - np.eye(dim)).dot(Wgk))).dot(Wgk.T)
  

  if out_mode =="complete":
    U, S, _ = np.linalg.svd(C_and_B)
    return C_and_B, U, S
  else:
    return C_and_B
  

def OR(R, Q, out_mode = "simple"):
  """
  Compute OR operation of two conceptor matrices
  
  @param R: a conceptor matrix
  @param Q: another conceptor matrix
  @param out_mode: output mode ("simple"/"complete")
  
  @return R_or_Q: R OR Q
  @return U: eigen vectors of R_or_Q
  @return S: eigen values of R_or_Q
  """
  
  R_or_Q = NOT(AND(NOT(R), NOT(Q)))


  if out_mode == "complete":
    U, S, _ = np.linalg.svd(R_or_Q)
    return R_or_Q, U, S
  else:
    return R_or_Q
  
  
def PHI(C, gamma):
  """
  aperture adaptation of conceptor C by factor gamma
  
  @param C: conceptor matrix
  @param gamma: adaptation parameter, 0 <= gamma <= Inf
  
  @return C_new: updated new conceptor matrix
  """
  
  dim = C.shape[0]
  
  if gamma == 0:
    U, S, _ = np.linalg.svd(C)
    S[S < 1] = np.zeros((np.sum((S < 1).astype(float)), 1))
    C_new = U.dot(S).dot(U.T)
  elif gamma == np.Inf:
    U, S, _ = np.linalg.svd(C)
    S[S > 0] = np.zeros((np.sum((S > 0).astype(float)), 1))
    C_new = U.dot(S).dot(U.T)
  else:
    C_new = C.dot(np.linalg.inv(C + gamma ** -2 * (np.eye(dim) - C)))
    
  return C_new
