#!/bin/bash
octave --no-window-system --eval "build_qpmpclti2f; pkg load control; test_mpc_tripleint_2e2f_features;"
