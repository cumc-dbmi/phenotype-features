# -*- coding: utf-8 -*-
from common import *
from omop_vector_euclidean_distance import *
from levenshtein import *

import altair as alt
import os
import pandas as pd
import random
import numpy as np
import logging
from sklearn.metrics.pairwise import euclidean_distances
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from tsnecuda import TSNE
from Bio import pairwise2

LOGGER = logging.getLogger(__name__)


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


# ### Visualize embeddings on a 2D plane

cumc_embeddings_folder = '/data/research_ops/omops/omop_2019q4_embeddings'
ccae_embeddings_folder = '/data/research_ops/omops/omop_ccae_embeddings'
mdcd_embeddings_folder = '/data/research_ops/omops/omop_mdcd_embeddings'
mdcr_embeddings_folder = '/data/research_ops/omops/omop_mdcr_embeddings'

embedding_dict = {
    'cumc_visit': get_visit_embedding(cumc_embeddings_folder),
    'ccae_visit': get_visit_embedding(ccae_embeddings_folder),
    'mdcd_visit': get_visit_embedding(mdcd_embeddings_folder),
    'mdcr_visit': get_visit_embedding(mdcr_embeddings_folder)
}

concept = pd.read_parquet('/data/research_ops/omops/omop_2019q4/concept')

breast_cancer = load_phenotype_definition('/data/research_ops/phenotype_definitions/breast_cancer_concept_list.txt')
t2dm_concept_list = load_phenotype_definition('/data/research_ops/phenotype_definitions/t2dm_concept_list.txt')

embeddings_to_compare = compute_tsne_2d_components(embedding_dict)

t2dm_embeddings = concat_embeddings(embeddings_to_compare, t2dm_concept_list, 't2dm')
breast_cancer_embeddings = concat_embeddings(embeddings_to_compare, breast_cancer, 'breast_cancer')
union_embeddings = pd.concat([t2dm_embeddings, breast_cancer_embeddings], axis=0)

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

pd.concat([cumc_visit_pairwise_dist.compute_average_dist(breast_cancer), 
           cumc_visit_pairwise_dist.compute_random_average_dist(205),
           ccae_visit_pairwise_dist.compute_average_dist(breast_cancer),
           ccae_visit_pairwise_dist.compute_random_average_dist(392)], axis=1)

cumc_visit_pairwise_dist = EuclideanDistance(name='cumc_visit', path=get_pairwise_cosine_similarity_output(create_file_path(cumc_embeddings_folder, 'visit')))


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


patient_event = spark.read.parquet('/data/research_ops/omops/omop_2019q4_embeddings/visit/patient_event/')
patient_event = patient_event \
    .withColumn("lower_bound", F.unix_timestamp(F.date_add(F.from_unixtime(F.col('date'), 'yyyy-MM-dd'), -30), 'yyyy-MM-dd')) \
    .withColumn("upper_bound", F.unix_timestamp(F.date_add(F.from_unixtime(F.col('date'), 'yyyy-MM-dd'), 30), 'yyyy-MM-dd'))
patient_event.cache()

data = visualize_time_lines(patient_event, 80809, 30)

sequences = data['sequence'].to_list()

sequence_1 = [int(c) for c in sequences[20].split(' ')]
sequence_2 = [int(c) for c in sequences[1].split(' ')]

sequence_1 = [1000560, 1126658, 1125315]
sequence_2 = [1125315, 1124957, 1125315, 1125315, 1112807]

patient_similarity = PatientSimilarity(max_cost=0, is_similarity=True)

sim, alignment, scoring_mat, direction_mat = patient_similarity.compute(cumc_visit_pairwise_dist, sequence_1, sequence_2)

alignment

alignments = []
for i, seq_1 in enumerate(sequences):
    for j, seq_2 in enumerate(sequences):
        if i < j:
            sequence_1 = [int(c) for c in seq_1.split(' ')]
            sequence_2 = [int(c) for c in seq_2.split(' ')]
            score, alignment, _, _ = patient_similarity.compute(cumc_visit_pairwise_dist, sequence_1, sequence_2)
            alignments.append((score, alignment))

sorted_alignments = sorted(alignments, key=lambda x: x[0], reverse=True)

sorted_alignments

# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

breast_cancer_cohort = pd.concat([cumc_visit_glove[cumc_visit_glove['standard_concept_id'].isin(breast_cancer_concepts)]])
breast_cancer_cohort['phenotype'] = 'breast_cancer'
t2dm_corhort = pd.concat([cumc_visit_glove[cumc_visit_glove['standard_concept_id'].isin(t2dm_concept_list)]])
t2dm_corhort['phenotype'] = 't2dm'

breast_cancer_cohort.append(t2dm_corhort)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    data=cumc_visit_glove,
    legend="full",
    alpha=0.5
)
