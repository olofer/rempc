#
# Basic (smoke) tests of the rempc package
#
# Basic profiling demo like so:
#   python3 test.py --profile-solver --tripleint --horizon 250
#

import argparse
import numpy as np
import rempc 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--smoke', action = 'store_true') 
  parser.add_argument('--tripleint', action = 'store_true') 
  parser.add_argument('--horizon', type = int, default = 200) 
  parser.add_argument('--profile-solver', action = 'store_true') 
  parser.add_argument('--repeats', type = int, default = 20)
  parser.add_argument('--figext', type = str, default = 'pdf', help = 'figure file extension (e.g. pdf or png)')
  parser.add_argument('--dpi', type = float, default = 250.0, help = 'print to file PNG option')

  args = parser.parse_args()

  if args.tripleint:
    from scipy.signal import cont2discrete

  if args.profile_solver and args.tripleint:
    import matplotlib.pyplot as plt

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

    if args.profile_solver:
      O['verbosity'] = int(0)
      P['xreturn'] = int(0)
      nvec = np.arange(2, args.horizon + 1)
      clocks = np.tile(np.nan, (len(nvec), 2))
      print('profiling solver performance vs horizon length n={}..{}'.format(nvec[0], nvec[-1]))
      for k in range(len(nvec)):
        nk = int(nvec[k])
        P['n'] = nk 
        Rk = [rempc.qpmpclti2f(P, O) for _ in range(args.repeats)]
        all_good = [rk['isconverged'] for rk in Rk]
        assert np.sum(all_good) == args.repeats
        idx = np.argmin([rk['totalclock'] for rk in Rk])
        clocks[k, 0] = [rk['totalclock'] for rk in Rk][idx]
        clocks[k, 1] = [rk['solveclock'] for rk in Rk][idx]
      
      plt.plot(nvec, 1.0e3 * clocks[:, 0], label = 'total (solve + overhead)')
      plt.plot(nvec, 1.0e3 * clocks[:, 1], label = 'solve only')
      plt.xlabel('length of horizon')
      plt.ylabel('wall clock [ms] (best of {} repeats)'.format(args.repeats))
      plt.grid(True)
      plt.title('MPC solver profiling (nx={}, nu={}, ny={}, ni={})'.format(P['A'].shape[0], 
                                                                           P['B'].shape[1], 
                                                                           P['C'].shape[0], 
                                                                           P['f3'].size))
      plt.legend()
      plt.tight_layout()
      plt.savefig('test-profile-{}.{}'.format(args.horizon, args.figext), dpi = args.dpi)
      plt.close()

  print('reached end of {}'.format(__file__))
