import csv
import glob

files = [ open(f, errors='ignore') for f in glob.glob("tweets/*.csv")]

_str = ''

for f in files:

    print (f'### reading {f.name} ###')

    for line in list(csv.reader(f)):

        if (len(line) == 0): 
            continue

        if (line[1] == True or line[2] == True): #if is QRT or RT
            continue

        _str += line[3]

with open('tweets/RAW.txt', 'w') as f:
    f.write(_str)