#
# Basic check that the rempc package can be loaded and called.
#

import numpy as np
import rempc 

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

print(P['A'].flatten(order = 'K'))
print('isfortran(A) = {}'.format(np.isfortran(P['A'])))

print(P['B'].flatten(order = 'K'))
print('isfortran(B) = {}'.format(np.isfortran(P['B'])))

print(P['C'].flatten(order = 'K'))
print('isfortran(C) = {}'.format(np.isfortran(P['C'])))

R = rempc.qpmpclti2f(P, O)

assert R is None
print('reached end of {}'.format(__file__))
