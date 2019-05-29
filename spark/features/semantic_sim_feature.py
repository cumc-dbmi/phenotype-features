from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from features.feature import *

class SemanticSimilarityFeature(Feature):

    def __init__(self, name: str, concept_ancestor=None):
        
        Feature.__init__(self, name)
        self.direct_child_parent_df = concept_ancestor.where("min_levels_of_separation=1") \
            .rdd.map(lambda r: (r[1], r[0])).groupByKey().collectAsMap()

    def recur_connect_path_bottom_up(self, concept_id):
        node_paths = []
        if concept_id in self.direct_child_parent_df:
            for parent_concept_id in self.direct_child_parent_df[concept_id]:
                parent_node_paths = self.recur_connect_path_bottom_up(parent_concept_id)

                if len(parent_node_paths) == 0:
                    node_paths.append(str(parent_concept_id) + '.' + str(concept_id))
                for parent_node_path in parent_node_paths:
                    node_paths.append(str(parent_node_path) + '.' + str(concept_id))
        return node_paths
    
    def calculate_pairwise_sim(self, node_paths_1, node_paths_2):
        max_score = 0
        best_node_path_1 = ''
        best_node_path_2 = ''
        if (len(node_paths_1)!=0) & (len(node_paths_2)!=0):
            for node_path_1 in node_paths_1:
                for node_path_2 in node_paths_2:
                    score = self.calculate_sim(node_path_1, node_path_2)
                    if max_score < score:
                        max_score = score
                        best_node_path_1 = node_path_1
                        best_node_path_2 = node_path_2
        return (max_score, best_node_path_1, best_node_path_2)
    
    def calculate_sim(self, node_path_1, node_path_2):
        node_path_1_ids = node_path_1.split(".")
        node_path_2_ids = node_path_2.split(".")
        max_iter = max(len(node_path_1_ids), len(node_path_2_ids))

        shared_distance = 0

        for i in range(max_iter): #0-8
            if (len(node_path_1_ids) > i) & (len(node_path_2_ids) > i):
                if node_path_1_ids[i] != node_path_2_ids[i]:
                    break
                shared_distance += 1
        return (shared_distance * 2) / (len(node_path_1_ids) + len(node_path_2_ids))
    
    
    def annotate(self, training_set):
        
        semantic_score = udf(lambda x,y: self.calculate_pairwise_sim(self.recur_connect_path_bottom_up(x),  \
                 self.recur_connect_path_bottom_up(y))[0], FloatType())
        
        training_set = training_set.withColumn(self.name.replace(" ", "_"), semantic_score('concept_id_1','concept_id_2'));
        training_set = training_set.withColumn("is_connected", col(self.name.replace(" ", "_")).isNotNull().cast("integer"))
        
        return training_set