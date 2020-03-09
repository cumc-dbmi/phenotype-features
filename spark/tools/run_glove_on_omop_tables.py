# -*- coding: utf-8 -*-
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

from glove import Glove
from glove import Corpus

import numpy as np
import argparse
import re
import os

GLOVE = 'glove'
CORPUS_MODEL = 'corpus.model'
GLOVE_MODEL = 'glove.model'


# +
def get_corpus_model_path(folder):
    return create_file_path(folder, CORPUS_MODEL)

def get_glove_model_path(folder):
    return create_file_path(folder, GLOVE_MODEL)


# -

def read_corpus(filename):
    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)
    with open(filename, 'r') as datafile:
        for line in datafile:
            yield line[0:-1].split(' ')


def create_file_path(input_folder, table_name):
    if input_folder[-1] == '/':
        file_path = input_folder + table_name
    else:
        file_path = input_folder + '/' + table_name
    return file_path


def train_glove(sequence_file_path, output_folder, no_components, epochs, window_size):
    corpus_model = Corpus()
    corpus_model.fit(read_corpus(sequence_file_path), window=100000)
    corpus_model.save(get_corpus_model_path(output_folder))
    
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    
    glove = Glove(no_components=int(no_components), learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=int(epochs), no_threads=50, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    glove.save(get_glove_model_path(output_folder))


def run_glove(input_folder, output_folder, num_components, num_epochs, window_size):
    pattern = re.compile('.*\\.csv$')
    for file_path in os.listdir(input_folder):
        if re.match(pattern, file_path):
            train_glove(create_file_path(input_folder, file_path), output_folder, num_components, num_epochs, window_size)
            break


def generate_vocab_df(spark, glove):
    vocab_schema = T.StructType([T.StructField('id', T.IntegerType(), True),
                                 T.StructField('standard_concept_id', T.IntegerType(), True)])
    vocab_df = spark.sparkContext \
        .parallelize([(int(k), int(v)) for k, v in glove.inverse_dictionary.items()]) \
        .toDF(vocab_schema)

    return vocab_df


def generate_vector_df(spark, glove, vocab_df):
    vector_schema = T.StructType([T.StructField('id', T.IntegerType(), True),
                                  T.StructField('vector', T.ArrayType(T.DoubleType()), True)])

    vector_df = spark.sparkContext \
        .parallelize([(i, [float(d) for d in glove.word_vectors[i]]) for i in range(len(glove.word_vectors))]) \
        .map(lambda t: T.Row(id=t[0], vector=t[1])).toDF(vector_schema)

    return vector_df


def export_vector_df(vector_df, vocab_df, output_folder):
    vector_df = vector_df.join(vocab_df, 'id').select('standard_concept_id', 'vector')
    vector_df.withColumn('vector', F.array_join(F.col('vector'), ',')) \
        .select('standard_concept_id', 'vector').repartition(1) \
        .write.option('header', 'true').mode('overwrite').csv(create_file_path(output_folder, 'embedding_csv'))


def main(input_folder, output_folder, num_components, num_epochs, window_size):
    
    run_glove(input_folder, output_folder, num_components, num_epochs, window_size)

    glove = Glove.load(get_glove_model_path(output_folder))
    
    spark = SparkSession.builder.appName('Extract embeddings').getOrCreate()
    
    vocab_df = generate_vocab_df(spark, glove)

    vector_df = generate_vector_df(spark, glove, vocab_df)

    export_vector_df(vector_df, vocab_df, output_folder)


# +
# output_embeddings = ccae_embeddings.repartition(200).join(concept, ccae_embeddings['standard_concept_id'] == concept['concept_id']) \
#     .select('concept_id', 'concept_name', 'embedding') \
#     .withColumn('embedding', F.array_join(F.split('embedding', ','), '\t')).orderBy('concept_id')
# -

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The input folder',
                        required=True)
    
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The input folder',
                        required=True)

    parser.add_argument('-n',
                        '--no_components',
                        dest='no_components',
                        action='store',
                        help='vector size',
                        required=False,
                        default=200)

    parser.add_argument('-e',
                        '--epochs',
                        dest='epochs',
                        action='store',
                        help='The number of epochs that the algorithm will run',
                        required=False,
                        default=100)
    
    parser.add_argument('-w',
                        '--window_size',
                        dest='window_size',
                        action='store',
                        help='The window',
                        required=False,
                        default=10)

    ARGS = parser.parse_args()

    main(ARGS.input_folder, ARGS.output_folder, ARGS.no_components, ARGS.epochs, ARGS.window_size)
