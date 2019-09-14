from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from features.feature import *

class SemanticSimilarityFeature(Feature):

    def __init__(self, name, concept_ancestor=None):
        
        Feature.__init__(self, name)
        self.concept_ancestor = concept_ancestor;
        self.feature_name = name.lower().replace(" ", "_")
    
    def annotate(self, training_set):
            
        #Join to get all the ancestors of concept_id_1
        joined_dataset = training_set.join(self.concept_ancestor, col('concept_id_1') == col('descendant_concept_id')) \
                        .select(col("concept_id_1"), 
                                col("concept_id_2"), 
                                col("ancestor_concept_id").alias("ancestor_concept_id_1"), 
                                col("min_levels_of_separation").alias("distance_1")) \

        #Join to get all the ancestors of concept_id_2
        joined_dataset = joined_dataset.join(self.concept_ancestor, col('concept_id_2') == col('descendant_concept_id')) \
                        .select(col("concept_id_1"), 
                                col("concept_id_2"),
                                col("ancestor_concept_id_1"),
                                col("distance_1"),
                                col("ancestor_concept_id").alias("ancestor_concept_id_2"), 
                                col("min_levels_of_separation").alias("distance_2"))
        
        #Filter to find the common ancestors of concept_id_1 and concept_id_2
        joined_dataset = joined_dataset.where("ancestor_concept_id_1 == ancestor_concept_id_2") \
                        .select(col("concept_id_1"),
                              col("concept_id_2"),
                              col("ancestor_concept_id_1").alias("common_ancestor_concept_id"),
                              col("distance_1"),
                              col("distance_2"))
        
        #Find the root concepts
        root_concepts = self.concept_ancestor.groupBy("descendant_concept_id") \
            .count().where("count == 1").withColumnRenamed("descendant_concept_id", "root_concept_id")
            
        #Retrieve all ancestor descendant relationships for the root concepts
        root_concept_relationships = root_concepts.join(self.concept_ancestor, root_concepts["root_concept_id"] == self.concept_ancestor["ancestor_concept_id"]) \
            .select(self.concept_ancestor["ancestor_concept_id"], 
                    self.concept_ancestor["descendant_concept_id"],
                    self.concept_ancestor["max_levels_of_separation"].alias("root_distance")) \
            .where("ancestor_concept_id <> descendant_concept_id")
        
        #Join to get all root concepts and their corresponding root_distance
        joined_dataset = joined_dataset.join(root_concept_relationships, joined_dataset["common_ancestor_concept_id"] == root_concept_relationships["descendant_concept_id"]) \
            .select("concept_id_1", "concept_id_2", "distance_1", "distance_2", "root_distance")
        
        #Compute the semantic similarity
        joined_dataset = joined_dataset.withColumn("semantic_similarity", 
                          2 * col("root_distance") / (2 * col("root_distance") + col("distance_1") + col("distance_2")))
        
        #Find the maximum semantic similarity
        joined_dataset = joined_dataset.groupBy("concept_id_1", "concept_id_2") \
            .max("semantic_similarity") \
            .withColumnRenamed("max(semantic_similarity)", self.feature_name)
        
        training_set = training_set.join(joined_dataset, (training_set["concept_id_1"] == joined_dataset["concept_id_1"])
                                & (training_set["concept_id_2"] == joined_dataset["concept_id_2"]), "left_outer") \
                        .select([training_set[f] for f in training_set.schema.fieldNames()] + [joined_dataset[self.feature_name]])

        training_set = training_set.withColumn("is_connected", col(self.feature_name).isNotNull().cast("integer"))

        return training_set
