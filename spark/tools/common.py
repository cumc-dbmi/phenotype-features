DOMAIN_KEY_FIELDS = {
    'condition_occurrence_id' : ('condition_concept_id', 'condition_start_date', 'condition'),
    'procedure_occurrence_id' : ('procedure_concept_id', 'procedure_date', 'procedure'),
    'drug_exposure_id' : ('drug_concept_id', 'drug_exposure_start_date', 'drug'),
    'measurement_id' : ('measurement_concept_id', 'measurement_date', 'measurement')
}


# +
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

def get_pairwise_euclidean_distance_output(output_folder):
    return create_file_path(output_folder, 'pairwise_euclidean_distance.pickle')

def get_pairwise_cosine_similarity_output(output_folder):
    return create_file_path(output_folder, 'pairwise_cosine_similarity.pickle')

def write_sequences_to_csv(spark, patient_sequence_path, patient_sequence_csv_path):
    spark.read.parquet(patient_sequence_path).select('concept_list').repartition(1) \
        .write.mode('overwrite').option('header', 'false').csv(patient_sequence_csv_path)
# -


