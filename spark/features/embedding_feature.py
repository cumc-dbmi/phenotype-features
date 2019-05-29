from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from scipy import spatial
import numpy as np 
import math
from features.feature import *

class EmbeddingFeature(Feature):
    
    def __init__(self, name:str, vocab=None, weights=None):
        
        Feature.__init__(self, name)
        
        target = weights \
            .join(vocab, weights["wordId"] == vocab["id"]) \
            .select("standard_concept_id", "vector")
        
        context = weights \
            .join(vocab, weights["wordId"] == vocab["id"] + vocab.count()) \
            .select("standard_concept_id", "vector")
        
        joined = target.join(context, target["standard_concept_id"] == context["standard_concept_id"])
        
        semantic_score = udf(lambda v1, v2: ((Vectors.dense(v1) + Vectors.dense(v2)) / 2).toArray().tolist(), ArrayType(DoubleType()))

        self.embeddings = joined.select(
            target["standard_concept_id"], 
            semantic_score(target["vector"], context["vector"]).alias("vector")
        )

    def annotate(self, training_set):
        
        cosine_udf = udf(lambda v1, v2: 0.0 if v1 is None or v2 is None else float(1 - spatial.distance.cosine(v1, v2)), DoubleType())
        
        joined_data = training_set.join(self.embeddings, training_set["concept_id_1"] == self.embeddings["standard_concept_id"], "left_outer") \
            .select([col(f) for f in training_set.schema.fieldNames()] + [self.embeddings["vector"].alias("vector_1")])
        
        joined_data = joined_data.join(self.embeddings, training_set["concept_id_2"] == self.embeddings["standard_concept_id"], "left_outer") \
            .select([col(f) for f in joined_data.schema.fieldNames()] + [self.embeddings["vector"].alias("vector_2")])
        
        joined_data = joined_data.select([col(f) for f in training_set.schema.fieldNames()] 
                                         + [cosine_udf("vector_1", "vector_2").alias(self.name.replace(" ", "_") + "_cosine")])
        
        return joined_data