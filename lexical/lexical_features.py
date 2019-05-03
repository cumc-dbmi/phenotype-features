from fuzzywuzzy import fuzz
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import udf
from Levenshtein import distance, ratio, jaro, jaro_winkler
from fuzzywuzzy import fuzz
from pyspark.sql.functions import udf

def fuzz_partial_ratio(string1, string2):
    return fuzz.partial_ratio(string1, string2) / 100

distance_udf = udf(distance, IntegerType())
ratio_udf = udf(ratio, DoubleType())
jaro_udf = udf(jaro, DoubleType())
jaro_winkler_udf = udf(jaro_winkler, DoubleType())
fuzz_partial_ratio_udf = udf(fuzz_partial_ratio, DoubleType())

functions = [distance_udf, ratio_udf, jaro_udf, jaro_winkler_udf, fuzz_partial_ratio_udf]
headers = ['Levenshtein_distance', 'Levenshtein_ratio', 'jaro', 'jaro_winkler', 'fuzz_partial_ratio']
def add_to_lexical(df):
    what_to_select = list(df) + [package[0]('standard_name_1', 'standard_name_2').alias(package[1]) for package in zip(functions, headers)]
    df = df.select( what_to_select ).show(1)
    return df

'''
Levenshtein distance - minimum number of single character edits
Levenshtein ratio - similarity of two strings
Jaro - similarity score
Jaro-winkler - similarity score, which favors strings that match prefix from the beginning
fuzz partial ratio - gives a score based on how well parts of a string match another
'''