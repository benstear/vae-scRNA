#!/usr/bin/env python

# Linux/bash file to run a python file(s) 
# https://askubuntu.com/questions/244378/running-python-file-in-terminal
# doing this:  python3 <filename>.py   calls the interpreter. we can abstract this
# by doing it all in the bash file... see line #1 of this file


# Make py file executable
chmod +x n_1078.py

# Run it
./n_1078.py


# you can keep your session running for further coding. Like this:
# python -i <file_name.py>