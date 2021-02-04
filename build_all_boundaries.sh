#!/bin/bash

set -e

python generate_boundaries.py -c 0.005 mnist --batch 100
python generate_boundaries.py fmnist -c 0.005 --batch 100
python generate_boundaries.py cifar10 -c 0.0015 -i 400 --batch 50