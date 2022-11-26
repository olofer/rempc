#!/bin/bash

# This initial check and folder deletion is not necessary since
# the "install --user" call will rewrite it anyway if it exists.
# It is to avoid confusion when the build fails so that no old 
# version sits around on the path.

localsite=$(python3 -m site --user-site)
echo $localsite
localdir=$(ls -d $localsite/rempc_package*)
echo $localdir
if [ -d "$localdir" ]; then
  echo "$localdir EXISTS -- removing it now!"
  rm -rf $localdir
fi

rm -rf build
rm -rf dist
rm -rf rempc_package*.egg-info
python3 setup.py build
python3 setup.py install --user
python3 test.py
