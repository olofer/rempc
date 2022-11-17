import os
import argparse 
import numpy as np 

def infnorm(X):
  return np.max(np.abs(X))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dmin', type = int, default = 10)
  parser.add_argument('--dmax', type = int, default = 10)
  parser.add_argument('--stages', type = int, default = 50)
  parser.add_argument('--solves', type = int, default = 10)
  parser.add_argument('--tolerance', type = float, default = 1.0e-12, help = 'tolerance for test')
  parser.add_argument('--do-not-generate', action = 'store_true', help = 'try to read already existing files')

  args = parser.parse_args()

  generate_files = not args.do_not_generate

  if generate_files:
    textfiles = '1'
    cmdstr = './niqcheck {} {} {} {} {}'.format(args.stages, args.dmin, args.dmax, textfiles, args.solves)
    print('executing: {}'.format(cmdstr), flush = True)
    retval = os.system(cmdstr)
    # assert retval == 0
    if retval != 0:
      print('WARNING: nonzero return = {}'.format(retval))
 
  C = np.loadtxt('bigCee.txt')
  H = np.loadtxt('bigPhi.txt')
  L = np.loadtxt('bigEll.txt')

  print(C.shape)
  print(H.shape)
  print(L.shape)

  assert C.shape[1] == H.shape[0] and H.shape[0] == H.shape[1]

  LH = np.linalg.cholesky(H)
  Z = np.linalg.solve(LH, C.T)
  LY = np.linalg.cholesky(np.dot(Z.T, Z))
  assert np.all(L.shape == LY.shape)

  err1 = infnorm(L - LY) / infnorm(LY)
  print(err1)
  assert err1 < args.tolerance

  H_rank = np.linalg.matrix_rank(H)
  print('rank(H) = {}'.format(H_rank))
  C_rank = np.linalg.matrix_rank(C)
  print('rank(C) = {}'.format(C_rank))

  B = np.loadtxt('bigRhs.txt')
  XY = np.loadtxt('bigSol.txt')  # each column should be a solution with the column in B as rhs

  '''

  each column of B is [h;d], each column of XY is [x;y]
  this pair of vectors is supposed to fulfill the equation
  [H, C']         [x]          [-h]
  [C, 0 ] (times) [y] (equals) [ d]

  '''

  nd = H.shape[0]
  ne = C.shape[0]
  A = np.vstack((np.hstack((H, C.T)), np.hstack((C, np.zeros((ne, ne))))))
  print(A.shape)
  A_rank = np.linalg.matrix_rank(A)
  print('rank(A) = {}'.format(A_rank))

  assert np.all(B.shape == XY.shape)
  for i in range(B.shape[1]):
    B[:nd, i] = -B[:nd, i] 

  XY_ref = np.linalg.solve(A, B) 

  all_ok = True
  for i in range(XY.shape[1]):
    err2a = infnorm(XY[:nd, i] - XY_ref[:nd, i]) / infnorm(XY_ref[:nd, i])
    err2b = infnorm(XY[nd:, i] - XY_ref[nd:, i]) / infnorm(XY_ref[nd:, i])
    print([err2a, err2b])
    all_ok = all_ok and err2a < args.tolerance and err2b < args.tolerance

    #if err2a >= args.tolerance:
    #  print( np.abs(XY[:nd, i] - XY_ref[:nd, i]).flatten() )

  # crash if solution is incorrect
  assert all_ok 