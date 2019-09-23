import configparser
from time import sleep
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

def download_table(domain_table_name, base_url, properties, output_folder):
    
    spark = SparkSession.builder.config(conf=SparkConf()) \
            .appName('Download dataset {}'.format(domain_table_name)).getOrCreate()
    
    output_path = output_folder + '/' + domain_table_name
    
    print('{}'.format(output_path))
    
    domain_table = spark.read \
        .jdbc(base_url, 'dbo.{}'.format(domain_table_name), properties=properties)
    
    domain_table.write.parquet(output_path)
    
    return '{domain_table_name} is saved into {output_path}'.format(domain_table_name=domain_table_name,
                                                                    output_path=output_path)
    
def download_omop_tables(domain_tables, database_property_file, output_folder):

    config = configparser.ConfigParser()
    config.read(database_property_file)
    properties = config.defaults()
    base_url = properties["base_url"]
       
    with ThreadPoolExecutor() as executor:
        
        futures = []
        for domain_table_name in domain_tables:
            futures.append(executor.submit(download_table, domain_table_name, base_url, properties, output_folder))
        
        print('Started downloading data for {}'.format(','.join(domain_tables)))
        
        sleep(1)
        
        while any([future.running() for future in futures]):
            
            for finished_job in [future for future in futures if future.done()]:
                print(finished_job.result())
            
            running_jobs = [future for future in futures if future.running()]
            
            print('{} remining jobs'.format(len(running_jobs)))
            
            sleep(60)
        
        print('Completed downloading data for {}'.format(','.join(domain_tables)))
               
if __name__ == "__main__":
               
    download_omop_tables(['condition_occurrence', 
                          'procedure_occurrence', 
                          'drug_exposure',
                          'observation',
                          'visit_occurrence',
                          'measurement',
                          'concept',
                          'concept_ancestor'], database_property_file='omop_database_properties.ini', output_folder='omop')
