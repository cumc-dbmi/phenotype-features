from features.feature import *
from pyspark.sql.functions import col

class OmopCooccurrenceFeature(Feature):
    
    def __init__(self, name:str, cooccurrence_df=None):
        
        Feature.__init__(self, name)
        self.cooccurrence_df = cooccurrence_df
    
    def annotate(self, training_set):
        joined_set = training_set.join(self.cooccurrence_df, (training_set["concept_id_1"] == self.cooccurrence_df["standard_concept_id_1"]) & 
                 (training_set["concept_id_2"] == self.cooccurrence_df["standard_concept_id_2"]), "left_outer")
        columns = [training_set[fieldName] for fieldName in training_set.schema.fieldNames()] \
                    + [self.cooccurrence_df["normalized_count"].alias(self.name.replace(" ", "_"))]
        return joined_set.select(columns)