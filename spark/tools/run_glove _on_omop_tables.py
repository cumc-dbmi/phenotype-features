import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

from glove import Glove
from glove import Corpus

import numpy as np
import argparse
import re
import os


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


def train_glove(sequence_folder, sequence_file_path, no_components, epochs):
    corpus_model = Corpus()
    corpus_model.fit(read_corpus(create_file_path(sequence_folder, sequence_file_path)), window=10)
    corpus_model.save(create_file_path(sequence_folder, 'corpus.model'))
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    glove = Glove(no_components=int(no_components), learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=int(epochs), no_threads=50, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    glove.save(create_file_path(sequence_folder, 'glove.model'))


def run_glove(input_folder, output_folder, num_components, num_epochs):
    pattern = re.compile('.*\\.csv$')
    for file_path in os.listdir(input_folder):
        if re.match(pattern, file_path):
            train_glove(output_folder, file_path, num_components, num_epochs)
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

    vector_df.withColumn('embedding', F.array_join(F.col('vector'), ',')) \
        .select('standard_concept_id', 'embedding').repartition(1) \
        .write.mode('overwrite').option('header', 'true').csv(create_file_path(output_folder, 'embeding_csv'))


def main(input_folder, output_folder, num_components, num_epochs):
    spark = SparkSession.builder.appName('Extract embeddings').getOrCreate()

    run_glove(input_folder, output_folder, num_components, num_epochs)

    glove = Glove.load(create_file_path(input_folder, 'glove.model'))

    vocab_df = generate_vocab_df(spark, glove)

    vector_df = generate_vector_df(spark, glove, vocab_df)

    export_vector_df(vector_df, vocab_df, input_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
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

    ARGS = parser.parse_args()

    main(ARGS.input_folder, ARGS.input_folder, ARGS.no_components, ARGS.epochs)
