#!/bin/bash 
aria2c -x10 -s10 -k1M https://storage.googleapis.com/corl_stare/stare.tar
aria2c -x10 -s10 -k1M https://storage.googleapis.com/corl_stare/data.tar
tar -xvf stare.tar
tar -xvf data.tar