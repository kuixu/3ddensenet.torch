#!/bin/bash
ls $1/*.h5|awk '{print $1;system("th hdf2t7.lua -filepath "$1)}'
