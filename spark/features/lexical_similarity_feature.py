from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, DoubleType
from Levenshtein import distance, ratio, jaro, jaro_winkler
from fuzzywuzzy import fuzz
from features.feature import *

class LexicalSimilarityFeature(Feature):
    
    def __init__(self, name:str):
        
        Feature.__init__(self, name)
        
    def annotate(self, training_set):
        
        #Levenshtein distance - minimum number of single character edits
        distance_udf = udf(lambda x, y: distance(x,y), IntegerType())
        #Levenshtein ratio - similarity of two strings
        ratio_udf = udf(lambda x, y: ratio(x,y), DoubleType())
        #Jaro - similarity score
        jaro_udf = udf(lambda x, y: jaro(x,y), DoubleType())
        #Jaro-winkler - similarity score, which favors strings that match prefix from the beginning
        jaro_winkler_udf = udf(lambda x, y: jaro_winkler(x,y), DoubleType())
        #fuzz partial ratio - gives a score based on how well parts of a string match another
        fuzz_partial_ratio_udf = udf(lambda x, y: fuzz.partial_ratio(x, y) / 100, DoubleType())
        
        training_set = training_set.withColumn("distance", distance_udf("concept_name_1", "concept_name_2")) \
            .withColumn("ratio", ratio_udf("concept_name_1", "concept_name_2")) \
            .withColumn("jaro", jaro_udf("concept_name_1", "concept_name_2")) \
            .withColumn("jaro_wrinkler", jaro_winkler_udf("concept_name_1", "concept_name_2")) \
            .withColumn("fuzz_partial_ratio", fuzz_partial_ratio_udf("concept_name_1", "concept_name_2"))
            
        return training_set