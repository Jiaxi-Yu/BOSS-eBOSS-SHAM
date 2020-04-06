#!/usr/bin/env bash
salloc -N 1 -n 32 -c 2 -q interactive -L SCRATCH -C haswell -t 04:00:00

