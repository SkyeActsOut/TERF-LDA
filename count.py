import csv
import glob
import json

files = [ open(f, errors='ignore') for f in glob.glob("tweets/*.csv")]

ct = 0
ct_temp = 0

_counts = {}

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
    _counts[f.name] = ct_temp
    ct_temp = 0

print (f"{ct} TWEETS")

_c = 0
for line in list(csv.reader(open('clean_text_dirty.csv'))):

    if (len(line) > 0):
        _c += len(''.join(line).split(' '))

print (f"TOTALLING {_c} WORDS")

files_1 = [ open(f, errors='ignore') for f in glob.glob("tweets_w_topic/*.csv")]
files_2 = [ open(f, errors='ignore') for f in glob.glob("tweets_w_support/*.csv")]

all_supports = []

for f1, f2 in zip(files_1, files_2):

    print (f'### reading {f1.name} ###')
    topics = {

        "name": f1.name,
        "Topic 1": [0, 0],
        "Topic 2": [0, 0],
        "Topic 3": [0, 0],
        "Topic 4": [0, 0]

    }

    for line_1, line_2 in zip(list(csv.reader(f1)), list(csv.reader(f2))):

        if (len(line_1) <= 6): 
            continue

        if (line_1[1] == True or line_1[2] == True): #if is QRT or RT
            continue

        _topic = line_1[6]
        _support = line_2[6]

        topics[_topic][0] += 1 / _counts[f1.name.replace('tweets_w_topic', 'tweets')]
        topics[_topic][1] += float(_support)

    all_supports.append(topics)

print (all_supports)

with open ('SUPPORTS_TOPICS.json', 'w') as f:

    json.dump(all_supports, f, indent=4)