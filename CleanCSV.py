import csv

class CSV_Data:

    def __init__(self, f_name='clean_text_dirty.csv'):

        f = open(f_name)

        print (f'### reading {f.name} ###')

        self.new = []

        for line in list(csv.reader(f)):

            if (len(line) == 0): 
                continue
            
            self.new.append(''.join(line))

    def get (self) -> list:

        return self.new