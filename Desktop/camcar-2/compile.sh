#!/bin/bash
# Simple script to test syntactic correctness of CW2 program when programming on the PC.
# Requirement: gcc installed (standard on Mac, Windows needs installation of tools)

# parameter not used on PC: -mfloat-abi=hard

# commands to compile CW2 modules on PC (without linking)
gcc -c -I./resource -o camcar.o      camcar.c
gcc -c -I./resource -o detect_blob.o detect_blob.c
gcc -c -I./resource -o quickblob.o   quickblob.c

