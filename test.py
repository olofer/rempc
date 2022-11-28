#
# Basic check that the rempc package can be loaded and called.
#

import numpy as np
import rempc 

P = {}
O = rempc.options_qpmpclti2f()
O['eta'] = 0.965
O['maxiters'] = 42
print(O)

P.update({'n' : int(10)})
P.update({'A' : np.reshape(np.arange(20), (5, 4), order = 'C')})
P.update({'B' : np.reshape(np.arange(20), (5, 4), order = 'F')})

print(P['A'].flatten(order = 'K'))
print('isfortran(A) = {}'.format(np.isfortran(P['A'])))

print(P['B'].flatten(order = 'K'))
print('isfortran(B) = {}'.format(np.isfortran(P['B'])))

R = rempc.qpmpclti2f(P, O)

assert R is None
print('reached end of {}'.format(__file__))
