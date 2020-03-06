from common import *
from generate_glove_training_data import *

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import Window

import argparse
import datetime
from itertools import islice
from os import path

NUM_PARTITIONS = 800


def extract_mce_embeddings(mce_vec_path, mce_embedding_output):
    rdd = spark.sparkContext.textFile(mce_vec_path)
    rdd.mapPartitionsWithIndex(
        lambda idx, it: islice(it, 1, None) if idx == 0 else it 
    ).filter(lambda l: '</s>' not in l).map(lambda l: l.split(' ')) \
        .map(lambda parts: Row(standard_concept_id=int(parts[0]), vector=[float(p) for p in parts[1:-1]])).toDF() \
        .write.mode('overwrite').parquet(mce_embedding_output)


def create_time_window_partitions(patient_event, start_date, end_date, output_path):
    
    patient_event = patient_event \
        .withColumnRenamed('date', 'unix_date') \
        .withColumn('date', F.from_unixtime('unix_date').cast(T.DateType())) \
        .withColumn('start_date', F.lit(start_date).cast(T.DateType())) \
        .withColumn('end_date', F.lit(end_date).cast(T.DateType())) \
        .where(F.col('date') >= F.col('start_date')) \
        .where(F.col('date') <= F.col('end_date'))

    patient_event.repartition(NUM_PARTITIONS, 'person_id', 'date') \
        .write.mode('overwrite').parquet(output_path)

def generate_sequences(patient_event, output_path):

    patient_event.groupBy('person_id', 'date').agg(F.collect_list('standard_concept_id').alias('concept_list')) \
        .withColumn('date_str', F.col('date').cast('string')) \
        .withColumn('concept_list', F.col('concept_list').cast('string')) \
        .withColumn('date_concept_list', F.struct('date', 'date_str', 'concept_list')) \
        .groupBy('person_id').agg(F.collect_list('date_concept_list').alias('date_concept_list')) \
        .rdd.map(lambda row: (row[0], sorted(row[1], key=lambda x:x[0]))) \
        .map(lambda row: str(row[0]) + ', [' + ', '.join(['[' + str(datetime.datetime.strptime(dc[1], '%Y-%m-%d').timestamp()) + ', ' + dc[2] + ']' for dc in row[1]]) + ']') \
        .repartition(1).saveAsTextFile(output_path)


def generate_patient_sequence(spark, 
                              input_folder, 
                              output_folder, 
                              omop_table_list, 
                              start_year):
    domain_tables = []
    for domain_table_name in omop_table_list:
        domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    patient_event = join_domain_tables(domain_tables)

    start_date = str(datetime.date(int(start_year), 1, 1))
    end_date = datetime.datetime.now()
    
    patient_event = create_time_window_partitions(patient_event, 
                                                  start_date, 
                                                  end_date,
                                                  get_patient_event_folder(output_folder))    
    
    patient_event = spark.read.parquet(get_patient_event_folder(output_folder))
    generate_sequences(patient_event, get_patient_sequence_folder(output_folder))


def parse_args():
    
    parser = argparse.ArgumentParser(description='Aruguments for generating patient sequences from OMOP tables')
    
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The input folder that contains the omop tables',
                        required=True)
    
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The output folder that stores the patient sequence data',
                        required=True)

    parser.add_argument('-t',
                        '--omop_table_list',
                        dest='omop_table_list',
                        nargs='+',
                        help='The list of omop tables',
                        required=True)
    
    parser.add_argument('-s',
                        '--start_year',
                        type=int,
                        dest='start_year',
                        action='store',
                        help='The start year for the time window',
                        default=1985,
                        required=False)

    return parser.parse_args()


if __name__ == "__main__":

    spark = SparkSession.builder.appName('Generating MCE training data').getOrCreate()
    ARGS = parse_args()
    generate_patient_sequence(spark, 
                              ARGS.input_folder, 
                              ARGS.output_folder, 
                              ARGS.omop_table_list,
                              ARGS.start_year)
