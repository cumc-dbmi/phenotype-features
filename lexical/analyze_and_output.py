from Levenshtein import distance, ratio, jaro
from tqdm import tqdm
from fuzzywuzzy import fuzz
import csv
import gzip

partial_ratio = fuzz.partial_ratio

def percentified(func):
    def percentified_func(master_string, string):
        return func(master_string, string) / 100
    return percentified_func

analysis_to_run = [distance, ratio, jaro, percentified(partial_ratio)]

def run_test(master_string, string):
    return [func(master_string, string) for func in analysis_to_run]

def analyze():
    ''' Uses Levenshtein distance and ratio '''
    open('./exported.csv', 'a').close()
    new_csv = open('./exported.csv', 'w')
    writer = csv.writer(new_csv, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    timer = tqdm(total=1116595)
    skiprows = 0
    # #do something
    with gzip.open('./phenotype_paired_concepts.csv.gz', 'rt') as f:
        spamreader = csv.reader(f, delimiter=',', quotechar='"')
        header = next(spamreader)
        writer.writerow([header[6], header[14]] + ['distance', 'ratio', 'jaro', 'fuzz_partial_ratio'])
        timer.update(1)
        for line in spamreader:
            if len(line) > 1:
                master_string, string = line[6], line[14]
                values = run_test(master_string, string)
                writer.writerow([master_string, string] + values)
                timer.update(1)
    new_csv.close()