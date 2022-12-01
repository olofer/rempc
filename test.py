#
# Basic (smoke) tests of the rempc package
#

import argparse
import numpy as np
import rempc 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--smoke', action = 'store_true') 
  parser.add_argument('--tripleint', action = 'store_true') 
  parser.add_argument('--horizon', type = int, default = 200) 
  # parser.add_argument('--profile-solver', type = int, default = -1) 

  args = parser.parse_args()

  if args.smoke:
    P = {}
    O = rempc.options_qpmpclti2f()
    O['eta'] = float(0.965)
    O['maxiters'] = int(42)
    O['verbosity'] = int(1)
    print(O)

    P.update({'n' : int(10)})
    P.update({'A' : np.reshape(np.arange(25), (5, 5), order = 'C')})
    P.update({'B' : np.reshape(np.arange(20), (5, 4), order = 'F')})
    P.update({'C' : np.reshape(np.arange(35), (7, 5), order = 'C')})
    P.update({'x' : np.zeros(5).reshape((5, 1))})
    P.update({'w' : np.zeros(1).reshape((1, 1))})
    P.update({'W' : np.ones(7).reshape((7, 1))})
    P.update({'r' : np.zeros(1).reshape((1, 1))})
    P.update({'Qx' : np.ones(5).reshape((5, 1))})
    P.update({'R' : np.ones(4).reshape((4, 1))})

    print(P['A'].flatten(order = 'K'))
    print('isfortran(A) = {}'.format(np.isfortran(P['A'])))

    print(P['B'].flatten(order = 'K'))
    print('isfortran(B) = {}'.format(np.isfortran(P['B'])))

    print(P['C'].flatten(order = 'K'))
    print('isfortran(C) = {}'.format(np.isfortran(P['C'])))

    R = rempc.qpmpclti2f(P, O)
    assert isinstance(R, dict)
    print(R.keys())
    assert R['isconverged'] == 1

  if args.tripleint:
    from scipy.signal import cont2discrete

    Ac = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype = float)
    Bc = np.array([[0], [0] ,[1]], dtype = float)
    Cc = np.array([[1, 0, 0]], dtype = float)
    Dc = np.array([[0]], dtype = float)
    Ts = 0.1
    dtsys = cont2discrete((Ac, Bc, Cc, Dc), Ts)

    N = args.horizon

    O = rempc.options_qpmpclti2f()
    O['verbosity'] = int(1)

    P = {}
    P.update({'n' : int(N)})
    P.update({'A' : dtsys[0]})
    P.update({'B' : dtsys[1]})
    P.update({'C' : dtsys[2]})
    P.update({'D' : dtsys[3]})

    P.update({'R' : np.array([[1.0e-6]])})
    P.update({'Qx' : np.array([[1.0e-6]])})
    P.update({'W' : np.array([[1.0]])})
    P.update({'x' : np.array([[0.0, 0.0, 0.0]])})  # initial state
    P.update({'r' : np.array([[1.0]])})  # target output
    P.update({'w' : np.array([[0.0]])})

    U = 1.50
    P.update({'F2' : np.array([[1.0], [-1.0]])})  # setup constraint |u(t)| <= U
    P.update({'f3' : np.array([[U], [U]])})

    P.update({'xreturn' : int(N)})

    R = rempc.qpmpclti2f(P, O)
    assert isinstance(R, dict)
    print(R.keys())
    assert R['isconverged'] == 1

    #if args.profile_solver >= 5:
    #  None

  print('reached end of {}'.format(__file__))
