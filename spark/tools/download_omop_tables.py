import configparser
import argparse
from pyspark.sql import SparkSession


def find_num_of_records(domain_table_name, properties, column_name, spark_session):
    table_max_id = spark_session.read.format("jdbc") \
        .option("driver", properties['driver']) \
        .option("url", properties['base_url']) \
        .option("dbtable", "(SELECT MAX({}) AS {} FROM {}) as {}".format(column_name, column_name, domain_table_name, column_name)) \
        .option("user", properties['user']) \
        .option("password", properties['password']) \
        .load() \
        .select("{}".format(column_name)).collect()[0]['{}'.format(column_name)]
    return table_max_id


def download_omop_tables(domain_table, column_name, properties, output_folder, spark_session):
    table = spark_session.read.format("jdbc") \
        .option("url", properties['base_url']) \
        .option("dbtable", "%s" % domain_table) \
        .option("user", properties['user']) \
        .option("password", properties['password']) \
        .option("numPartitions", 16) \
        .option("partitionColumn", column_name) \
        .option("lowerBound", 1) \
        .option("upperBound", find_num_of_records(domain_table,properties, column_name, spark_session)) \
        .load()
    table.write.mode('overwrite').parquet(output_folder + '/' + str(domain_table) + '/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for downloading OMOP tables')

    parser.add_argument('-c',
                        '--credential_path',
                        dest='credential_path',
                        action='store',
                        help='The path for your database credentials',
                        required=True)

    parser.add_argument('-tc',
                        '--domain_table_list',
                        dest='domain_table_list',
                        nargs='+',
                        action='store',
                        help='The list of domain tables you want to download',
                        required=True)

    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The output folder that stores the domain tables download destination',
                        required=True)

    ARGS = parser.parse_args()
    spark = SparkSession.builder.appName("Download OMOP tables").getOrCreate()
    domain_table_list = ARGS.domain_table_list
    credential_path = ARGS.credential_path
    download_folder = ARGS.output_folder
    config = configparser.ConfigParser()
    config.read(credential_path)
    properties = config.defaults()
    downloaded_tables = []
    OMOP_table_dict = {'condition_occurrence': 'condition_occurrence_id', 'measurement': 'measurement_id',
                       'drug_exposure': 'drug_exposure_id', 'procedure_occurrence': 'procedure_occurrence_id',
                       'observation': 'observation_id', 'visit_occurrence': 'visit_occurrence_id'}
    for item in domain_table_list:
        try:
            download_omop_tables(item, OMOP_table_dict.get(item), properties, download_folder, spark)
            downloaded_tables.append(item)
            print('table: ' + str(item) + ' is downloaded')
        except Exception as e:
            print(str(e))
    print('The following tables were downloaded:' + str(downloaded_tables))
