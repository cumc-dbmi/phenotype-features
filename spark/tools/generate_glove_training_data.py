import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import Window

import argparse
import datetime
from os import path
from run_glove_on_omop_tables import *
from extract_glove_embeddings import *

# +
NUM_PARTITIONS = 800

DOMAIN_KEY_FIELDS = {
    'condition_occurrence_id' : ('condition_concept_id', 'condition_start_date', 'condition'),
    'procedure_occurrence_id' : ('procedure_concept_id', 'procedure_date', 'procedure'),
    'drug_exposure_id' : ('drug_concept_id', 'drug_exposure_start_date', 'drug'),
    'measurement_id' : ('measurement_concept_id', 'measurement_date', 'measurement')
}


# -

def get_logger(spark):
    log4jLogger = spark.sparkContext._jvm.org.apache.log4j
    return log4jLogger.LogManager.getLogger(__name__)


# +
def create_file_path(input_folder, table_name):

    if input_folder[-1] == '/':
        file_path = input_folder + table_name
    else:
        file_path = input_folder + '/' + table_name

    return file_path

def get_patient_event_folder(output_folder):
    return create_file_path(output_folder, 'patient_event')

def get_patient_sequence_folder(output_folder):
    return create_file_path(output_folder, 'patient_sequence')

def get_patient_sequence_csv_folder(output_folder):
    return create_file_path(output_folder, 'patient_sequence_csv')

def write_sequences_to_csv(spark, patient_sequence_path, patient_sequence_csv_path):
    spark.read.parquet(patient_sequence_path).select('concept_list').repartition(1) \
        .write.mode('overwrite').option('header', 'false').csv(patient_sequence_csv_path)

def get_key_fields(domain_table):
    field_names = domain_table.schema.fieldNames()
    for k, v in DOMAIN_KEY_FIELDS.items():
        if k in field_names:
            return v
    return (get_concept_id_field(domain_table), get_domain_date_field(domain_table), get_domain_field(domain_table))
    
def get_domain_date_field(domain_table):
    #extract the domain start_date column
    return [f for f in domain_table.schema.fieldNames() if 'date' in f][0]
    
def get_concept_id_field(domain_table):
    return [f for f in domain_table.schema.fieldNames() if 'concept_id' in f][0]

def get_domain_field(domain_table):
    return get_concept_id_field(domain_table).replace('_concept_id', '')


# -

def join_domain_tables(domain_tables):

    patient_event = None

    for domain_table in domain_tables:
        
        #extract the domain concept_id from the table fields. E.g. condition_concept_id from condition_occurrence
        #extract the domain start_date column
        #extract the name of the table
        concept_id_field, date_field, table_domain_field = get_key_fields(domain_table) 
        
        domain_table = domain_table.where(F.col(concept_id_field) != 0) \
            .withColumn('date', F.unix_timestamp(F.to_date(F.col(date_field)), 'yyyy-MM-dd')) \
            .select(domain_table['person_id'],
                    domain_table[concept_id_field].alias('standard_concept_id'),
                    F.col('date'),
                    domain_table['visit_occurrence_id'],
                    F.lit(table_domain_field).alias('domain')) \
            .distinct()
            
        if patient_event == None:
            patient_event = domain_table
        else:
            patient_event = patient_event.unionAll(domain_table)

    return patient_event

def preprocess_domain_table(spark, input_folder, domain_table_name):
    
    domain_table = spark.read.parquet(create_file_path(input_folder, domain_table_name))
    #lowercase the schema fields
    domain_table = domain_table.select([F.col(f_n).alias(f_n.lower()) for f_n in domain_table.schema.fieldNames()])    
    _, _, domain_field = get_key_fields(domain_table) 
    
    if domain_field == 'drug' \
        and path.exists(create_file_path(input_folder, 'concept')) \
        and path.exists(create_file_path(input_folder, 'concept_ancestor')):
        
        concept = spark.read.parquet(create_file_path(input_folder, 'concept'))
        concept_ancestor = spark.read.parquet(create_file_path(input_folder, 'concept_ancestor'))
        domain_table = roll_up_to_drug_ingredients(domain_table, concept, concept_ancestor)
        
    return domain_table


def roll_up_to_drug_ingredients(drug_exposure, concept, concept_ancestor):
    
    #lowercase the schema fields
    drug_exposure = drug_exposure.select([F.col(f_n).alias(f_n.lower()) for f_n in drug_exposure.schema.fieldNames()])
    
    drug_ingredient = drug_exposure.select('drug_concept_id').distinct() \
        .join(concept_ancestor, F.col('drug_concept_id') == F.col('descendant_concept_id')) \
        .join(concept, F.col('ancestor_concept_id') == F.col('concept_id')) \
        .where(concept['concept_class_id'] == 'Ingredient') \
        .select(F.col('drug_concept_id'), F.col('concept_id').alias('ingredient_concept_id'))
    
    drug_ingredient_fields = [F.coalesce(F.col('ingredient_concept_id'), F.col('drug_concept_id')).alias('drug_concept_id')]
    drug_ingredient_fields.extend([F.col(field_name) for field_name in drug_exposure.schema.fieldNames() if field_name != 'drug_concept_id'])
    
    drug_exposure = drug_exposure.join(drug_ingredient, 'drug_concept_id', 'left_outer') \
        .select(drug_ingredient_fields)
    
    return drug_exposure


def create_time_window_partitions(patient_event, start_date, end_date, window_size, sub_window_size, output_path):
    
    # if the window_size is bigger than SUB_WINOW_THRESHOLD, create the sub_windows for scalable computations
    sub_window_size = sub_window_size if window_size > sub_window_size else window_size
    
    patient_event = patient_event \
        .withColumnRenamed('date', 'unix_date') \
        .withColumn('date', F.from_unixtime('unix_date').cast(T.DateType())) \
        .withColumn('start_date', F.lit(start_date).cast(T.DateType())) \
        .withColumn('end_date', F.lit(end_date).cast(T.DateType())) \
        .withColumn('patient_window', (F.datediff('date', 'start_date') / window_size).cast(T.IntegerType())) \
        .withColumn('sub_patient_window', (F.datediff('date', 'start_date') / sub_window_size).cast(T.IntegerType())) \
        .where(F.col('date') >= F.col('start_date')) \
        .where(F.col('date') <= F.col('end_date'))

    patient_event.repartition(NUM_PARTITIONS, 'person_id', 'sub_patient_window') \
        .write.mode('overwrite').parquet(output_path)

def create_visit_partitions(patient_event, output_path):
    
    patient_event = patient_event \
        .where(F.col('visit_occurrence_id').isNotNull()) \
        .withColumnRenamed('visit_occurrence_id', 'patient_window') \
    
    patient_event.repartition(NUM_PARTITIONS, 'person_id', 'patient_window') \
        .write.mode('overwrite').parquet(output_path)


def generate_sequences(patient_event, output_path, is_visit_based):
    
    # udf for sorting a list of tuples (date, standard_concept_id) by the key and value 
    # and concatenating standard_concept_ids to form a sequence
    join_collection_udf = F.udf(lambda its: ' '.join(sorted([str(it[1]) for it in its], key=lambda x: (x[0], x[1]))), T.StringType())
    
    # if is_visit_based is enabled, patient_window is same as sub-window, it's not necessary to perform a hierarchical groupby operations
    if is_visit_based:
        patient_event = patient_event \
            .groupBy('person_id', 'patient_window') \
            .agg(join_collection_udf(F.collect_list(F.struct('date', 'standard_concept_id'))).alias('concept_list'), 
                 F.size(F.collect_list('standard_concept_id')).alias('collection_size')) \
            .where(F.col('collection_size') > 1)
    else:
        # if time_window is enabled, it's necessary to perform a hierarchical groupby operations in order to scale.
        # group by the sub-window and concatenate the events     
        patient_event = patient_event \
            .groupBy('person_id', 'sub_patient_window', 'patient_window') \
            .agg(join_collection_udf(F.collect_list(F.struct('date', 'standard_concept_id'))).alias('sub_window_concept_list'), 
                 F.size(F.collect_list('standard_concept_id')).alias('sub_window_collection_size'))

        # group by the patient window and concatenate the sub-window patient sequence
        patient_event = patient_event \
            .groupBy('person_id', 'patient_window') \
            .agg(join_collection_udf(F.collect_list(F.struct('sub_patient_window', 'sub_window_concept_list'))).alias('concept_list'),
                 F.sum('sub_window_collection_size').alias('collection_size')) \
            .where(F.col('collection_size') > 1)
    
    patient_event.write.mode('overwrite').parquet(output_path)

def generate_patient_sequence(spark, 
                              input_folder, 
                              output_folder, 
                              omop_table_list, 
                              is_visit_based, 
                              start_year, 
                              window_size, 
                              sub_window_size):
    
    logger = get_logger(spark)
    logger.info('Started merging patient events from {omop_table_list}'.format(omop_table_list=omop_table_list))

    domain_tables = []
    for domain_table_name in omop_table_list:
        logger.info('Processing {domain_table_name}'.format(domain_table_name=domain_table_name))
        domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    patient_event = join_domain_tables(domain_tables)

    if is_visit_based:
        logger.info('Generating event sequences using visit_occurrence_id for {omop_table_list}' \
                         .format(omop_table_list=omop_table_list))
        
        patient_event = create_visit_partitions(patient_event, 
                                                get_patient_event_folder(output_folder))
    else:
        start_date = str(datetime.date(int(start_year), 1, 1))
        end_date = datetime.datetime.now()
        
        logger.info('''Generating event sequences using the time window configuration 
            start_date={start_date}
            end_date={end_date}
            window_size={window_size}
            omop_table_list={omop_table_list}'''.format(start_date=start_date,
                                                        end_date=end_date,
                                                        window_size=window_size,
                                                        omop_table_list=omop_table_list))
        
        patient_event = create_time_window_partitions(patient_event, 
                                                      start_date, 
                                                      end_date, 
                                                      window_size,
                                                      sub_window_size,
                                                      get_patient_event_folder(output_folder))
    
    patient_event = spark.read.parquet(get_patient_event_folder(output_folder))
    generate_sequences(patient_event, get_patient_sequence_folder(output_folder), is_visit_based)
    write_sequences_to_csv(spark, get_patient_sequence_folder(output_folder), get_patient_sequence_csv_folder(output_folder))


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
    
    parser.add_argument('-v',
                        '--is_visit_based',
                        dest='is_visit_based',
                        action='store_true',
                        default=False,
                        required=False)
    
        
    parser.add_argument('-n',
                        '--no_components',
                        type=int,
                        dest='no_components',
                        action='store',
                        help='vector size',
                        required=False,
                        default=200)

    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        dest='epochs',
                        action='store',
                        help='The number of epochs that the algorithm will run',
                        required=False,
                        default=100)
    
    parser.add_argument('-s',
                        '--start_year',
                        type=int,
                        dest='start_year',
                        action='store',
                        help='The start year for the time window',
                        default=1985,
                        required=False)
    
    parser.add_argument('-w',
                        '--window_size',
                        type=int,
                        dest='window_size',
                        action='store',
                        help='The size of the time window',
                        default=60,
                        required=False)
    
    parser.add_argument('-sw',
                        '--sub_window_size',
                        type=int,
                        dest='sub_window_size',
                        action='store',
                        help='The size of the time window',
                        default=30,
                        required=False)

    return parser.parse_args()

if __name__ == "__main__":

    ARGS = parse_args()

    spark = SparkSession.builder.appName('Training Glove').getOrCreate()
    log4jLogger = spark.sparkContext._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info('Started running the application for generating patient sequences')
    generate_patient_sequence(spark, 
                              ARGS.input_folder, 
                              ARGS.output_folder, 
                              ARGS.omop_table_list, 
                              ARGS.is_visit_based, 
                              ARGS.start_year, 
                              ARGS.window_size,
                              ARGS.sub_window_size)
    LOGGER.info('Finished generating patient sequences')
    
    LOGGER.info('Started running the Glove algorithm')
    run_glove(get_patient_sequence_csv_folder(ARGS.output_folder), ARGS.output_folder, ARGS.no_components, ARGS.epochs, ARGS.window_size)
    LOGGER.info('Finished running the Glove algorithm')

    LOGGER.info('Started extracting embeddings')
    extract_embedding(spark, get_glove_model_path(ARGS.output_folder), get_embedding_output(ARGS.output_folder))
    LOGGER.info('Finished extracting embeddings')
