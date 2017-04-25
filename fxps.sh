#!/bin/bash

#ipython -i main.py -- -n fxp1 -o SGD_var_dup1 -wd 0.000005 --auto-start
ipython main.py -- -n fxp2 -o SGD_var_dup1 -wd 0.00001  --auto-start
ipython main.py -- -n fxp3 -o SGD_var_dup1 -wd 0.00002  --auto-start
ipython main.py -- -n fxp4 -o SGD_var_dup1 -wd 0.00004  --auto-start
ipython main.py -- -n fxp5 -o SGD_var_dup1 -wd 0.00008  --auto-start
ipython main.py -- -n fxp6 -o SGD_var_dup1 -wd 0.00016  --auto-start
ipython main.py -- -n fxp7 -o SGD_var_dup1 -wd 0.00032  --auto-start
ipython main.py -- -n fxp8 -o SGD_var_dup1 -wd 0.00064  --auto-start
