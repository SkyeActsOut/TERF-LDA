import csv
import glob

files = [ open(f, errors='ignore') for f in glob.glob("tweets/*.csv")]

ct = 0
ct_temp = 0

for f in files:

    print (f'### reading {f.name} ###')

    for line in list(csv.reader(f)):

        if (len(line) == 0): 
            continue

        if (line[1] == True or line[2] == True): #if is QRT or RT
            continue

        ct += 1
        ct_temp += 1
    
    print (ct_temp)
    ct_temp = 0

print (f"{ct} TWEETS")

f = open ('clean_data.txt')

words = len(f.read().split(' '))

print (f"TOTALLING {words} WORDS")