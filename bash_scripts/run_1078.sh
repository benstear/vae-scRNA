#!/bin/bash

# Linux/bash file to run a python file(s) 
# https://askubuntu.com/questions/244378/running-python-file-in-terminal
# doing this:  python3 <filename>.py   calls the interpreter. we can abstract this
# by doing it all in the bash file... see line #1 of this file

# Make py file executable
chmod +x /scr1/users/stearb/pyscripts/n_1078.py

# Run it
python /scr1/users/stearb/pyscripts/n_1078.py

#qacct -j 'jobnumber' > /scr1/users/stearb/results/qstats.txt

#python /scr1/users/stearb/pyscripts/n_13313.py
#python /scr1/users/stearb/pyscripts/n_27499.py


# you can keep your session running for further coding. Like this:
# python -i <file_name.py>


#scp stearb@respublica.research.chop.edu:/scr1/users/stearb/results/plots_1078/*  /users/stearb/desktop/metrics/n_1078/
