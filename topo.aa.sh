#!/bin/bash

ipython main.py -- -n topo.aa1 -o SGD_topo.a -wd 0.00004 --last1-mult 10 --auto-start
ipython main.py -- -n topo.aa2 -o SGD_topo.a -wd 0.00008 --last1-mult 10 --auto-start
ipython main.py -- -n topo.aa3 -o SGD_topo.a -wd 0.00016 --last1-mult 10 --auto-start
ipython main.py -- -n topo.aa4 -o SGD_topo.a -wd 0.00020 --last1-mult 10 --auto-start
ipython main.py -- -n topo.aa5 -o SGD_topo.a -wd 0.00032 --last1-mult 10 --auto-start
ipython main.py -- -n topo.aa6 -o SGD_topo.a -wd 0.00064 --last1-mult 10 --auto-start
