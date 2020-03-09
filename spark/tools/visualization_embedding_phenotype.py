# -*- coding: utf-8 -*-
from common import *
from omop_vector_euclidean_distance import *
from levenshtein import *

import altair as alt
import os
import datetime
import pandas as pd
import random
import numpy as np
import logging
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import recall_score, precision_score
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT

from scipy.spatial import distance
from tsnecuda import TSNE

LOGGER = logging.getLogger(__name__)


# +
def euclidean_distance(v1, v2):
    return float(distance.euclidean(v1.toArray(), v2.toArray()))

def cosine_sim(v1, v2):
    return 1 - float(distance.cosine(np.array(v1), np.array(v2)))

features_udf = F.udf(lambda v: Vectors.dense(v), VectorUDT())
cosine_sim_udf = F.udf(cosine_sim, T.FloatType())
euclidean_distance_udf = F.udf(euclidean_distance, T.FloatType())


# -

def load_phenotype_definition(filepath):
    concept_list = []
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            concept_list.extend([int(c.strip()) for c in line.split(',')])
            line = fp.readline()
            cnt += 1
    return concept_list


def compute_tsne_2d_components(embedding_dict):
    embeddings = []
    for name, path in embedding_dict.items():
        if os.path.exists(path):
            embedding_pd = pd.read_parquet(path)
            vectors = np.array([np.array(v) for v in embedding_pd['vector'].to_numpy()])
            vectors_tsne_results = TSNE(n_components=2, perplexity=40, learning_rate=10).fit_transform(vectors)
            embedding_pd = embedding_pd.join(pd.DataFrame(data=vectors_tsne_results, columns=['tsne-2d-one', 'tsne-2d-two']))
            embeddings.append((name, embedding_pd))
        else:
            LOGGER.error('embedding {name} does not exisit at {path}'.format(name=name, 
                                                                             path=path))
    return embeddings


def concat_embeddings(embeddings_to_compare, phenotype_concept_list=None, phenotype_name=None):
    union_embedding_list = []
    for name, embedding in embeddings_to_compare:
        embedding = embedding[['standard_concept_id', 'tsne-2d-one', 'tsne-2d-two']]
        embedding['name'] = name
        if phenotype_concept_list and phenotype_name:
            embedding = embedding[embedding['standard_concept_id'].isin(phenotype_concept_list)]
            embedding['phenotype'] = phenotype_name
        union_embedding_list.append(embedding)
    return pd.concat(union_embedding_list)


def get_visit_embedding(embedding_folder):
    return create_file_path(embedding_folder, 'visit/embedding')


def get_time_window_180_embedding(embedding_folder):
    return create_file_path(embedding_folder, 'time_window_180/embedding')


def visualize_time_lines(patient_event, concept_id, num_patients=50):
    
    ra_patient = patient_event.where(F.col('standard_concept_id') == concept_id) \
        .groupBy('person_id', 'standard_concept_id').agg(F.min('date').alias('index_date')) \
        .withColumn('random_num', F.randn()) \
        .withColumn('rank', F.dense_rank().over(Window.orderBy('random_num'))) \
        .where(F.col('rank') <= num_patients)
    
    join_collection_udf = F.udf(lambda its: ' '.join(sorted([str(it[1]) for it in its], key=lambda x: (x[0], x[1]))), T.StringType())
    
    patient_timeline_pd = patient_event \
        .join(ra_patient, 'person_id') \
        .where(F.col('index_date').between(F.col('lower_bound'), F.col('upper_bound')))\
        .withColumn('date_concept_id', F.struct(F.col('index_date'), patient_event['standard_concept_id']))\
        .groupBy('person_id').agg(join_collection_udf(F.collect_list('date_concept_id')).alias('sequence'), 
                                  F.size(F.collect_list('date_concept_id')).alias('size')) \
        .where(F.col('size') > 1) \
        .select('person_id', 'sequence').toPandas()
    
    return patient_timeline_pd


# ### Visualize embeddings on a 2D plane

cumc_embeddings_folder = '/data/research_ops/omops/omop_2019q4_embeddings'
ccae_embeddings_folder = '/data/research_ops/omops/omop_ccae_embeddings'
mdcd_embeddings_folder = '/data/research_ops/omops/omop_mdcd_embeddings'
mdcr_embeddings_folder = '/data/research_ops/omops/omop_mdcr_embeddings'

embedding_dict = {
    'cumc_visit_glove': get_visit_embedding(cumc_embeddings_folder),
    'ccae_visit_glove': get_visit_embedding(ccae_embeddings_folder),
    'mdcd_visit_glove': get_visit_embedding(mdcd_embeddings_folder),
    'mdcr_visit_glove': get_visit_embedding(mdcr_embeddings_folder),
    'cumc_visit_mce': '/data/research_ops/omops/omop_2019q4_embeddings/mce_visit/embeddings'
}

concept = pd.read_parquet('/data/research_ops/omops/omop_2019q4/concept')

breast_cancer = load_phenotype_definition('/data/research_ops/phenotype_definitions/breast_cancer_concept_list.txt')
t2dm_concept_list = load_phenotype_definition('/data/research_ops/phenotype_definitions/t2dm_concept_list.txt')
chemotherapy = load_phenotype_definition('/data/research_ops/phenotype_definitions/chemotherapy_concept_list.txt')
ckdd = load_phenotype_definition('/data/research_ops/phenotype_definitions/ckdd_concept_list.txt')
dialysis = load_phenotype_definition('/data/research_ops/phenotype_definitions/dialysis_concept_list.txt')
cancer = load_phenotype_definition('/data/research_ops/phenotype_definitions/cancer_concept_list.txt')
mdd_bd = load_phenotype_definition('/data/research_ops/phenotype_definitions/mdd_bd.txt')

embeddings_to_compare = compute_tsne_2d_components(embedding_dict)

t2dm_embeddings = concat_embeddings(embeddings_to_compare, t2dm_concept_list, 't2dm')
breast_cancer_embeddings = concat_embeddings(embeddings_to_compare, breast_cancer, 'breast_cancer')
cancer_embeddings = concat_embeddings(embeddings_to_compare, cancer, 'cancer')
chemotherapy_embeddings = concat_embeddings(embeddings_to_compare, chemotherapy, 'chemotherapy')
ckdd_embeddings = concat_embeddings(embeddings_to_compare, ckdd, 'ckdd')
dialysis_embeddings = concat_embeddings(embeddings_to_compare, dialysis, 'dialysis')
mdd_bd_embeddings = concat_embeddings(embeddings_to_compare, mdd_bd, 'mdd_bd')
union_embeddings = pd.concat([chemotherapy_embeddings, 
                              ckdd_embeddings, 
                              dialysis_embeddings, 
                              mdd_bd_embeddings,
                              t2dm_embeddings], axis=0)

columns = union_embeddings.columns.to_list() + ['concept_name', 'domain_id']
union_embeddings = union_embeddings.merge(concept, left_on='standard_concept_id', right_on='concept_id')[columns]

alt.data_transformers.disable_max_rows()

alt.Chart(union_embeddings, title='embeddings').mark_point().encode(
    x='tsne-2d-one:Q',
    y='tsne-2d-two:Q',
    color='phenotype',
    facet=alt.Facet('name:O', columns=2),
    tooltip=['concept_name']
).interactive()

# ### Measure the average cosine distances for breast cancer

cumc_visit_pairwise_dist = EuclideanDistance(name='cumc_visit', path=get_pairwise_euclidean_distance_output(create_file_path(cumc_embeddings_folder, 'visit')))
ccae_visit_pairwise_dist = EuclideanDistance(name='ccae_visit', path=get_pairwise_euclidean_distance_output(create_file_path(ccae_embeddings_folder, 'visit')))
mdcd_visit_pairwise_dist = EuclideanDistance(name='mdcd_visit', path=get_pairwise_euclidean_distance_output(create_file_path(ccae_embeddings_folder, 'visit')))
mdcr_visit_pairwise_dist = EuclideanDistance(name='mdcr_visit', path=get_pairwise_euclidean_distance_output(create_file_path(ccae_embeddings_folder, 'visit')))

pd.concat([cumc_visit_pairwise_dist.compute_average_dist(breast_cancer), 
           cumc_visit_pairwise_dist.compute_random_average_dist(100),
           ccae_visit_pairwise_dist.compute_average_dist(breast_cancer),
           ccae_visit_pairwise_dist.compute_random_average_dist(100),
           mdcd_visit_pairwise_dist.compute_average_dist(breast_cancer),
           mdcd_visit_pairwise_dist.compute_random_average_dist(100),
           mdcr_visit_pairwise_dist.compute_average_dist(breast_cancer),
           mdcr_visit_pairwise_dist.compute_random_average_dist(100)], axis=1)

# +
# cumc_visit_pairwise_dist = EuclideanDistance(name='cumc_visit', path=get_pairwise_cosine_similarity_output(create_file_path(cumc_embeddings_folder, 'visit')))
# cumc_visit_pairwise_dist.get_closest(80809, 20, False)
# -


# ### Validation using phenotype pairs

def compute_recall_precision(pheMLrefset, embedding):
    pred_truth = pheMLrefset.join(embedding, F.col('concept_id_1') == F.col('standard_concept_id')) \
        .select('concept_id_1', 'concept_id_2', F.col('ground_truth').cast('int'), F.col('vector').alias('vector_1')) \
        .join(embedding, F.col('concept_id_2') == F.col('standard_concept_id')) \
        .select('concept_id_1', 'concept_id_2', 'ground_truth', 'vector_1', F.col('vector').alias('vector_2')) \
        .withColumn('cos_sim', cosine_sim_udf('vector_1', 'vector_2')) \
        .withColumn('prediction', (F.col('cos_sim') > 0.5).cast('int')) \
        .select('ground_truth', 'prediction').toPandas()
    recall = recall_score(pred_truth["ground_truth"], pred_truth["prediction"])
    precision = precision_score(pred_truth["ground_truth"], pred_truth["prediction"])
    return (recall, precision)


pheMLrefset = spark.read.parquet('/data/research_ops/phenotype_definitions/pheMLrefset')
pheMLrefset = pheMLrefset.select([F.col(f_n).alias(f_n.lower()) for f_n in pheMLrefset.schema.fieldNames()])
pheMLrefset = pheMLrefset.select('concept_id_1', 'concept_id_2', 'ground_truth')

cumc_visit = spark.read.parquet('/data/research_ops/omops/omop_2019q4_embeddings/visit/embedding')
mce_visit = spark.read.parquet('/data/research_ops/omops/omop_2019q4_embeddings/mce_visit/embeddings')
ccae_visit = spark.read.parquet('/data/research_ops/omops/omop_ccae_embeddings/visit/embedding')
mdcd_visit = spark.read.parquet('/data/research_ops/omops/omop_mdcd_embeddings/visit/embedding')
mdcr_visit = spark.read.parquet('/data/research_ops/omops/omop_mdcr_embeddings/visit/embedding')

intersect_concept_ids = cumc_visit.join(ccae_visit, 'standard_concept_id') \
    .select('standard_concept_id').join(mdcd_visit, 'standard_concept_id') \
    .select('standard_concept_id').join(mdcr_visit, 'standard_concept_id') \
    .select('standard_concept_id').join(mce_visit, 'standard_concept_id') \
    .select('standard_concept_id')

cumc_visit = cumc_visit.join(intersect_concept_ids, 'standard_concept_id')
ccae_visit = ccae_visit.join(intersect_concept_ids, 'standard_concept_id')
mdcd_visit = mdcd_visit.join(intersect_concept_ids, 'standard_concept_id')
mdcr_visit = mdcr_visit.join(intersect_concept_ids, 'standard_concept_id')
mce_visit = mce_visit.join(intersect_concept_ids, 'standard_concept_id')

pheMLrefset = pheMLrefset.join(intersect_concept_ids, F.col('concept_id_1') == F.col('standard_concept_id')) \
    .select('concept_id_1', 'concept_id_2', 'ground_truth') \
    .join(intersect_concept_ids, F.col('concept_id_2') == F.col('standard_concept_id')) \
    .select('concept_id_1', 'concept_id_2', 'ground_truth') \

cumc_visit_recall,cumc_visit_precision = compute_recall_precision(pheMLrefset, cumc_visit)
ccae_visit_recall,ccae_visit_precision = compute_recall_precision(pheMLrefset, ccae_visit)
mdcd_visit_recall,mdcd_visit_precision = compute_recall_precision(pheMLrefset, mdcd_visit)
mdcr_visit_recall,mdcr_visit_precision = compute_recall_precision(pheMLrefset, mdcr_visit)
mce_visit_recall,mce_visit_precision = compute_recall_precision(pheMLrefset, mce_visit)

print(f'cumc_visit_glove (Recall: {cumc_visit_recall}, Precision: {cumc_visit_precision})')
print(f'ccae_visit_glove (Recall: {ccae_visit_recall}, Precision: {ccae_visit_precision})')
print(f'mdcd_visit_glove (Recall: {mdcd_visit_recall}, Precision: {mdcd_visit_precision})')
print(f'mdcr_visit_glove (Recall: {mdcr_visit_recall}, Precision: {mdcr_visit_precision})')
print(f'cumc_visit_mce (Recall: {mce_visit_recall}, Precision: {mce_visit_precision})')


# ### Patient similarity

def add_concept_name(alignment, concept):
    alignemnt = alignment.merge(concept, left_on='sequence_1', right_on='concept_id') \
        [['sequence_1', 'sequence_2', 'alignment_score', 'concept_name']] \

    alignemnt.rename(columns={'concept_name' : 'concept_name_1'}, inplace=True)
    
    alignemnt = alignemnt.merge(concept, left_on='sequence_2', right_on='concept_id') \
        [['sequence_1', 'sequence_2', 'alignment_score', 'concept_name_1', 'concept_name']]

    alignemnt.rename(columns={'concept_name' : 'concept_name_2'}, inplace=True)
    return alignemnt


def max_pool(embedding_pd, sequence):
    intersection = set(sequence).intersection(embedding_pd.index.to_list())
    return np.max(np.asarray(
        embedding_pd.loc[intersection, 'vector'].to_list()), axis=0)


#embedding_pd = pd.read_parquet(get_visit_embedding('/data/research_ops/omops/omop_2019q4_embeddings'))
embedding_pd = pd.read_parquet('/data/research_ops/omops/omop_2019q4_embeddings/mce_visit/embeddings')
embedding_pd.set_index('standard_concept_id', inplace=True)

patient_event = spark.read.parquet('/data/research_ops/omops/omop_2019q4_embeddings/visit/patient_event/')
patient_event = patient_event \
    .withColumn("lower_bound", F.unix_timestamp(F.date_add(F.from_unixtime(F.col('date'), 'yyyy-MM-dd'), -30), 'yyyy-MM-dd')) \
    .withColumn("upper_bound", F.unix_timestamp(F.date_add(F.from_unixtime(F.col('date'), 'yyyy-MM-dd'), 30), 'yyyy-MM-dd'))
patient_event.cache()

data = visualize_time_lines(patient_event, 312327, 5000)
data['sequence'] = [list(map(int, i.split())) for i in data['sequence']]
sequences = data['sequence'].to_list()
max_pooled_vector = np.asarray([max_pool(embedding_pd, s) for s in sequences])
vectors_tsne_results = TSNE(n_components=2, perplexity=40, learning_rate=10).fit_transform(max_pooled_vector)
data = data.join(pd.DataFrame(data=vectors_tsne_results, columns=['tsne-2d-one', 'tsne-2d-two']))

alt.data_transformers.disable_max_rows()
alt.Chart(data, title='max pooled vectors').mark_point().encode(
    x='tsne-2d-one:Q',
    y='tsne-2d-two:Q',
    tooltip=['person_id']
).interactive()

sequences = data[(data['tsne-2d-one'] > -15) & (data['tsne-2d-one'] < -5) & (data['tsne-2d-two'] > -25) & (data['tsne-2d-two'] < -15)]['sequence']

sequences = data['sequence'].to_list()[0:100]

# +
# sequence_1 = [140648, 319826, 46271022, 80809]
# sequence_2 = [319826, 381290, 80809, 80809]
# sequence_1 = [c for c in sorted_alignments[0][1]['sequence_1'].to_list() if c != 0]
# sequence_2 = [c for c in sorted_alignments[0][1]['sequence_2'].to_list() if c != 0]
# -

patient_similarity = PatientSimilarity(max_cost=1, is_similarity=False)

# +
# sim, alignment, scoring_mat, direction_mat = patient_similarity.compute(cumc_visit_pairwise_dist.get_metric(sequence_1, sequence_2), sequence_1, sequence_2)
# -

alignments = []
total = len(sequences) * (len(sequences) - 1) / 2
counter = 0
start_time = datetime.datetime.now()
for i, sequence_1 in enumerate(sequences):
    for j, sequence_2 in enumerate(sequences):
        if i < j:
#             sequence_1 = [int(c) for c in seq_1.split(' ')]
#             sequence_2 = [int(c) for c in seq_2.split(' ')]
            #matrix = cumc_visit_pairwise_dist.get_metric(sequence_1, sequence_2)
            score, alignment, _, _ = patient_similarity.match(sequence_1, sequence_2)
            alignments.append((score, alignment, i, j))
            end_time = datetime.datetime.now()
            counter += 1
print(f'Took {(end_time - start_time).seconds} for this iteration, finished: {counter} and remaining: {total - counter}')

sorted_alignments = sorted(alignments, key=lambda x: x[0], reverse=False)

pd.set_option('display.max_rows', 20)

sorted_alignments[0:10]

same_pairs = []
for t in sorted_alignments:
    same_pairs.append(t[1][t[1]['sequence_1'] == t[1]['sequence_2']][['sequence_1']])

pd.concat(same_pairs, axis=0).get('sequence_1').value_counts()

for i, a in enumerate(sorted_alignments):
    print(f'Index: {i} Score: {a[0]}\n{a[1]}\n')

add_concept_name(sorted_alignments[13][1], concept)


