#!/bin/bash

shared='-o SGD_topo.a -wd 0.00016 --last1-mult 10 --auto-start'

ipython main.py -- -n topo.ac1 --topo-base 1.0 $shared
ipython main.py -- -n topo.ac2 --topo-base 1.1 $shared
ipython main.py -- -n topo.ac3 --topo-base 1.2 $shared
ipython main.py -- -n topo.ac4 --topo-base 1.5 $shared
ipython main.py -- -n topo.ac5 --topo-base 1.8 $shared
ipython main.py -- -n topo.ac6 --topo-base 2.0 $shared
ipython main.py -- -n topo.ac7 --topo-base 2.3 $shared
ipython main.py -- -n topo.ac8 --topo-base 2.8 $shared
ipython main.py -- -n topo.ac9 --topo-base 3.0 $shared
ipython main.py -- -n topo.ac10 --topo-base 4.0 $shared
ipython main.py -- -n topo.ac11 --topo-base 6.0 $shared
