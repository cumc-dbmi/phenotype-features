from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from features.feature import *


class InformationContentFeature(Feature):

    def __init__(self, name: str, information_content=None, concept_ancestor=None):
        
        Feature.__init__(self, name)
        self.information_content = information_content
        self.concept_ancestor = concept_ancestor
    
    
    def annotate(self, training_set):
        
        #Extract the pairs of concepts from the training data and join to the information content table
        concept_pair = training_set.select("concept_id_1", "concept_id_2") \
            .join(self.information_content, col("concept_id_1") == col("concept_id"), "left_outer") \
            .select(col("concept_id_1"), 
                    col("concept_id_2"), 
                    col("information_content").alias("information_content_1")) \
            .join(self.information_content, col("concept_id_2") == col("concept_id"), "left_outer") \
            .select(col("concept_id_1"), 
                    col("concept_id_2"), 
                    col("information_content_1"), 
                    col("information_content").alias("information_content_2"))
        
        #Join to get all the ancestors of concept_id_1
        concept_id_1_ancestor = concept_pair.join(self.concept_ancestor, col('concept_id_1') == col('descendant_concept_id')) \
                        .select("concept_id_1", "concept_id_2", "ancestor_concept_id")

        #Join to get all the ancestors of concept_id_2
        concept_id_2_ancestor = concept_pair.join(self.concept_ancestor, col('concept_id_2') == col('descendant_concept_id')) \
                        .select("concept_id_1", "concept_id_2", "ancestor_concept_id")
        
        #Compute the summed information content of all ancestors of concept_id_1 and concept_id_2
        unioned_sum = concept_id_1_ancestor.union(concept_id_2_ancestor).distinct() \
                .join(self.information_content, col("ancestor_concept_id") == col("concept_id")) \
                .groupBy("concept_id_1", "concept_id_2").sum("information_content") \
                .withColumnRenamed("sum(information_content)", "ancestor_union_information_content")
        
        #Compute the summed information content of common ancestors of concept_id_1 and concept_id_2
        intersection_sum = concept_id_1_ancestor.intersect(concept_id_2_ancestor) \
                .join(self.information_content, col("ancestor_concept_id") == col("concept_id")) \
                .groupBy("concept_id_1", "concept_id_2").sum("information_content") \
                .withColumnRenamed("sum(information_content)", "ancestor_intersecion_information_content")
                
        #Compute the information content and probability of the most informative common ancestor (MICA)
        mica_ancestor = concept_id_1_ancestor.intersect(concept_id_2_ancestor) \
                .join(self.information_content, col("ancestor_concept_id") == col("concept_id")) \
                .groupBy("concept_id_1", "concept_id_2").max("information_content", "probability") \
                .withColumnRenamed("max(information_content)", "mica_information_content") \
                .withColumnRenamed("max(probability)", "mica_probability")
                
        #Join the MICA to pairs of concepts extracted from the training examples
        features = concept_pair.join(mica_ancestor, (concept_pair["concept_id_1"] == mica_ancestor["concept_id_1"]) 
                            & (concept_pair["concept_id_2"] == mica_ancestor["concept_id_2"]), "left_outer") \
                        .select([concept_pair[f] for f in concept_pair.schema.fieldNames()] 
                            + [col("mica_information_content"), col("mica_probability")])
        
        #Compute the lin measure
        features = features.withColumn("lin_measure", 2 * col('mica_information_content') / (col('information_content_1') * col('information_content_2')))
        
        #Compute the jiang measure
        features = features.withColumn("jiang_measure", (1 - (col('information_content_1') + col('information_content_2') - 2 * col('mica_information_content'))))
        
        #Compute the information coefficient
        features = features.withColumn('information_coefficient', col('lin_measure') * (1 - 1 / (1 + col('mica_information_content'))))
        
        #Compute the relevance_measure
        features = features.withColumn('relevance_measure', col('lin_measure') * (1 - col('mica_probability')))
        
        
        #Join to get the summed information content of the common ancestors of concept_id_1 and concept_id_2
        features = features.join(intersection_sum, (features["concept_id_1"] == intersection_sum["concept_id_1"]) 
                            & (features["concept_id_2"] == intersection_sum["concept_id_2"]), "left_outer") \
            .select([features[f] for f in features.schema.fieldNames()] + [col("ancestor_intersecion_information_content")])
            
        #Join to get the summed information content of the common ancestors of concept_id_1 and concept_id_2
        features = features.join(unioned_sum, (features["concept_id_1"] == unioned_sum["concept_id_1"]) 
                            & (features["concept_id_2"] == unioned_sum["concept_id_2"]), "left_outer") \
            .select([features[f] for f in features.schema.fieldNames()] + [col("ancestor_union_information_content")])
        
        #Compute the graph information content measure
        features = features.withColumn('graph_ic_measure', col('ancestor_intersecion_information_content') / col('ancestor_union_information_content'))
        
        
        feature_columns = [col("mica_information_content"), 
                           col("lin_measure"), 
                           col("jiang_measure"),
                           col("information_coefficient"), 
                           col("relevance_measure"), 
                           col("graph_ic_measure")]
        
        training_set = training_set.join(features, (training_set["concept_id_1"] == features["concept_id_1"]) 
                            & (training_set["concept_id_2"] == features["concept_id_2"]), "left_outer") \
                .select([training_set[f] for f in training_set.schema.fieldNames()] + feature_columns)
        
        return training_set