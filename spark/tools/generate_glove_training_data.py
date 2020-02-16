import logging
import argparse
import pyspark.sql.functions as F
import pyspark.sql.types as T
import datetime
from pyspark.sql import SparkSession
from pyspark.sql import Window

def join_domain_tables(domain_tables):

    patient_event = None

    for domain_table in domain_tables:
        #extract the domain concept_id from the table fields. E.g. condition_concept_id from condition_occurrence
        concept_id_field = [f.name for f in domain_table.schema.fields if 'concept_id' in f.name][0]

        #extract the domain start_date column
        date_field = [f.name for f in domain_table.schema.fields if 'date' in f.name][0]

        #extract the name of the table
        table_domain_field = concept_id_field.replace('_concept_id', '')

        domain_table = domain_table.withColumn('date', F.unix_timestamp(F.to_date(F.col(date_field)), 'yyyy-MM-dd')) \
            .select(domain_table['person_id'],
                    domain_table[concept_id_field].alias('standard_concept_id'),
                    F.col('date'),
                    F.lit(table_domain_field).alias('domain')) \
            .where(F.col('standard_concept_id') != 0)

        if patient_event == None:
            patient_event = domain_table
        else:
            patient_event = patient_event.union(domain_table)

    return patient_event

def join_domain_tables_to_visit(domain_tables, visit_occurrence):
    
    """Join domain tables to visit_occurrence
    
    Keyword arguments:
    domain_tables -- the array containing the OMOOP domain tabls except visit_occurrence
    visit_occurrence -- the OMOP visit_occurrence table
    
    The function is to join each domain table to visit_occurrence to limit the records that have a valid visit_occurrence_id.
    In addition, the output columns of the domain table is converted to the same standard format as the following 
    (person_id, time_window, standard_concept_id, domain). In this case, co-occurrence is defined as those concept ids that have co-occurred within the same visit. 
    
    """
    
    patient_event = None
    
    for domain_table in domain_tables:
        #extract the domain concept_id from the table fields. E.g. condition_concept_id from condition_occurrence
        concept_id_field = [f.name for f in domain_table.schema.fields if "concept_id" in f.name][0]
        #extract the name of the table
        table_domain_field = concept_id_field.replace("_concept_id", "")
        #limit the domain records to those which have a visit_occurrence_id
        joined_domain_table = domain_table \
            .join(visit_occurrence, domain_table["visit_occurrence_id"] == visit_occurrence["visit_occurrence_id"])
        #standardize the output columns
        domain_table = joined_domain_table \
            .select(domain_table["person_id"], 
                domain_table["visit_occurrence_id"].alias("partition"), 
                domain_table[concept_id_field].alias("standard_concept_id"), 
                F.lit(table_domain_field).alias("domain"))

        if patient_event == None:
            patient_event = domain_table
        else:
            patient_event = patient_event.union(domain_table)
        
    return patient_event


def roll_up_to_drug_ingredients(patient_event, concept, concept_ancestor):

    drug_ingredient = patient_event.where('domain="drug"').select('standard_concept_id').distinct() \
        .join(concept_ancestor, F.col('standard_concept_id') == F.col('descendant_concept_id')) \
        .join(concept, F.col('ancestor_concept_id') == F.col('concept_id')) \
        .where(concept['concept_class_id'] == 'Ingredient') \
        .select(F.col('standard_concept_id').alias('drug_concept_id'), F.col('concept_id').alias('ingredient_concept_id'))
    
    patient_event_fields = [F.col(field_name) for field_name in patient_event.schema.fieldNames() if field_name != 'standard_concept_id']
    patient_event_fields.extend([F.coalesce(F.col('ingredient_concept_id'), F.col('standard_concept_id')).alias('standard_concept_id')])
    
    patient_event = patient_event.join(drug_ingredient,
           (F.col('standard_concept_id') == F.col('drug_concept_id')) &
           (F.col('domain') == 'drug'), 'left_outer') \
        .select(patient_event_fields)
    
    return patient_event

def creat_partitions(patient_event, start_date, end_date, window_size):

    patient_data = patient_event \
        .withColumnRenamed('date', 'unix_date') \
        .withColumn('date', F.from_unixtime('unix_date').cast(T.DateType())) \
        .withColumn('start_date', F.lit(start_date).cast(T.DateType())) \
        .withColumn('end_date', F.lit(end_date).cast(T.DateType())) \
        .withColumn('partition', (F.datediff('date', 'start_date') / window_size).cast(T.IntegerType()))

    return patient_data

def generate_sequences(patient_event, output_path):

    join_collection_udf = F.udf(lambda its: ' '.join([str(item) for item in its]), T.StringType())

    patient_event.select('person_id', 'standard_concept_id', 'partition').distinct() \
        .groupBy('person_id', 'partition') \
        .agg(join_collection_udf(F.collect_list('standard_concept_id')).alias('concept_list'),
             F.size(F.collect_list('standard_concept_id')).alias('collection_size')) \
        .where(F.col('collection_size') > 1) \
        .select('concept_list').repartition(1) \
        .write.mode('overwrite').option('header', 'false').csv(output_path)

def create_file_path(input_folder, table_name):

    if input_folder[-1] == '/':
        file_path = input_folder + table_name
    else:
        file_path = input_folder + '/' + table_name

    return file_path

if __name__ == "__main__":

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
                        '--visit_occurrence',
                        dest='visit_occurrence',
                        action='store_true',
                        default=False,
                        required=False)
    
    parser.add_argument('-s',
                        '--start_year',
                        dest='start_year',
                        action='store',
                        help='The start year for the time window',
                        default=1985,
                        required=False)
    
    parser.add_argument('-w',
                        '--window_size',
                        dest='window_size',
                        action='store',
                        help='The size of the time window',
                        default=30,
                        required=False)

    
    ARGS = parser.parse_args()
    
    spark = SparkSession.builder.appName("Generate patient sequences").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    log4jLogger = spark.sparkContext._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.warn("pyspark script logger initialized")

    domain_tables = []
    for domain_table_name in ARGS.omop_table_list:
        domain_table = spark.read.parquet(create_file_path(ARGS.input_folder, domain_table_name))
        domain_tables.append(domain_table)

    concept = spark.read.parquet(create_file_path(ARGS.input_folder, 'concept'))
    concept_ancestor = spark.read.parquet(create_file_path(ARGS.input_folder, 'concept_ancestor'))
    
    if ARGS.visit_occurrence:
        
        LOGGER.warn('Generate event sequences using visit_occurrence for {omop_table_list}' \
                         .format(omop_table_list=ARGS.omop_table_list))
        
        visit_occurrence = spark.read.parquet(create_file_path(ARGS.input_folder, 'visit_occurrence'))
        patient_event = join_domain_tables_to_visit(domain_tables, visit_occurrence)
        patient_event = roll_up_to_drug_ingredients(patient_event, concept, concept_ancestor)
    else:
        
        start_date = str(datetime.date(int(ARGS.start_year), 1, 1))
        end_date = datetime.datetime.now()
        window_size = ARGS.window_size
        
        LOGGER.warn('''Generate event sequences using the time window configuration 
        start_date={start_date}
        end_date={end_date}
        window_size={window_size}
        omop_table_list={omop_table_list}'''.format(start_date=start_date,
                                                    end_date=end_date,
                                                    window_size=window_size,
                                                    omop_table_list=ARGS.omop_table_list))
        
        patient_event = join_domain_tables(domain_tables)
        patient_event = roll_up_to_drug_ingredients(patient_event, concept, concept_ancestor)
        patient_event = creat_partitions(patient_event, start_date, end_date, window_size)
        

    generate_sequences(patient_event, output_path=ARGS.output_folder)
