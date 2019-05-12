from Levenshtein import distance, ratio, jaro, jaro_winkler
from tqdm import tqdm
from fuzzywuzzy import fuzz
import csv
import gzip

partial_ratio = fuzz.partial_ratio

def percentified(func):
    def percentified_func(master_string, string):
        return func(master_string, string) / 100
    return percentified_func

analysis_to_run = [distance, ratio, jaro, jaro_winkler, percentified(partial_ratio)]
analysis_headers = ['Levenshtein_distance', 'Levenshtein_ratio', 'jaro', 'jaro_winkler', 'fuzz_partial_ratio']

def run_test(master_string, string):
    return [func(master_string, string) for func in analysis_to_run]

def analyze():
    ''' Uses Levenshtein distance and ratio '''
    gzip.open('./lexical_analysis_added_as_rows.csv.gz', 'wt').close()
    new_csv = gzip.open('./lexical_analysis_added_as_rows.csv.gz', 'wt')
    writer = csv.writer(new_csv, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    timer = tqdm(total=1116595)
    skiprows = 0
    # #do something
    with gzip.open('./phenotype_paired_concepts.csv.gz', 'rt') as f:
        spamreader = csv.reader(f, delimiter=',', quotechar='"')
        header = next(spamreader)
        writer.writerow(header + analysis_headers)
        timer.update(1)
        for line in spamreader:
            if len(line) > 1:
                master_string, string = line[6], line[14]
                values = run_test(master_string, string)
                writer.writerow(line + values)
                timer.update(1)
    new_csv.close()

'''
Levenshtein distance - minimum number of single character edits
Levenshtein ratio - similarity of two strings
Jaro - similarity score
Jaro-winkler - similarity score, which favors strings that match prefix from the beginning
fuzz partial ratio - gives a score based on how well parts of a string match another
'''
# analyze()