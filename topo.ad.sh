#!/bin/bash

ipython main.py -- -n topo.ad1 -o SGD_topo.a -wd 0.00016 --last1-mult 2 --auto-start
ipython main.py -- -n topo.ad2 -o SGD_topo.a -wd 0.00016 --last1-mult 5 --auto-start
ipython main.py -- -n topo.ad3 -o SGD_topo.a -wd 0.00016 --last1-mult 10 --auto-start
ipython main.py -- -n topo.ad4 -o SGD_topo.a -wd 0.00016 --last1-mult 20 --auto-start
ipython main.py -- -n topo.ad5 -o SGD_topo.a -wd 0.00016 --last1-mult 40 --auto-start
ipython main.py -- -n topo.ad6 -o SGD_topo.a -wd 0.00016 --last1-mult 80 --auto-start
ipython main.py -- -n topo.ad7 -o SGD_topo.a -wd 0.00016 --last1-mult 160 --auto-start
ipython main.py -- -n topo.ad8 -o SGD_topo.a -wd 0.00016 --last1-mult 320 --auto-start
ipython main.py -- -n topo.ad9 -o SGD_topo.a -wd 0.00016 --last1-mult 640 --auto-start
