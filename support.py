import csv
import glob
import copy

files = [ open(f, errors='ignore') for f in glob.glob("tweets/*.csv")]

for f in files:

    likes_total = 0

    new = copy.copy(list(csv.reader(f)))
    old = copy.copy(new)

    print (f'### reading {f.name} ###')

    for line in old:

        if (len(line) == 0): 
            continue

        if (line[1] == True or line[2] == True): #if is QRT or RT
            continue
    
        likes = line[5]

        if (likes == 'Likes'):
            continue

        likes_total += int(likes)

    # print (likes_total)

    for line in new:

        if (len(line) == 0): 
            continue

        if (line[1] == True or line[2] == True): #if is QRT or RT
            continue
    
        likes = line[5]

        if (likes == 'Likes'):
            continue

        line.append(int(likes) / likes_total)

    fname = f.name.replace('tweets', 'tweets_w_support')
    with open (fname, 'w') as _f:

        writer = csv.writer(_f)

        writer.writerows(new)