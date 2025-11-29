#!/usr/bin/env bash

NAME="femnist"

cd ../utils

python stats.py --name $NAME

cd ../$NAME