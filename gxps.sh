#!/bin/bash

#ipython -i main.py -- -n fxp1 -o SGD_var_dup1 -wd 0.000005 --auto-start
ipython main.py -- -n gxp4 -o SGD_var_dup2_last2_mult -wd 0.00016 --last2-mult 3  --auto-start
ipython main.py -- -n gxp5 -o SGD_var_dup2_last2_mult -wd 0.00016 --last2-mult 30 --auto-start
