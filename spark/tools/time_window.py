from abc import ABC, abstractmethod
import datetime

from common import *

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import Row

COOCCURRENCE_MATRIX = 'cooccurrence_matrix'
CONCEPT_OCCURRENCE = 'concept_dictionary'


class TimeWindowAbstract(ABC):
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _apply_time_window(self, patient_event):
        raise NotImplemented()
        
    @abstractmethod
    def _self_join_logic(self, patient_event_1, patient_event_2):
        raise NotImplemented()
        
    @abstractmethod
    def _group_events_rdd(self, patient_event):
        raise NotImplemented()


class FixedTimeWindow(TimeWindowAbstract):
    
    def __init__(self, start_date, end_date, window_size, sub_window_size=30):
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        # if the window_size is bigger than sub_window_size, create the sub_windows for scalable computations
        self.sub_window_size = sub_window_size if window_size > sub_window_size else window_size
        super().__init__()

    def _apply_time_window(self, patient_event):
        patient_event = patient_event \
            .withColumn('start_date', F.lit(self.start_date).cast(T.DateType())) \
            .withColumn('end_date', F.lit(self.end_date).cast(T.DateType())) \
            .withColumn('window', (F.datediff('date', 'start_date') / self.window_size).cast(T.IntegerType())) \
            .withColumn('sub_window', (F.datediff('date', 'start_date') / self.sub_window_size).cast(T.IntegerType())) \

        return patient_event.repartition('person_id', 'window', 'sub_window')
    
    def _self_join_logic(self, patient_event_1, patient_event_2):
        return (patient_event_1['person_id'] == patient_event_2['person_id']) \
                    & (patient_event_1['window'] == patient_event_2['window'])
    
    def _group_events_rdd(self, patient_event):
        return patient_event \
            .withColumn('date_concept_tuple', F.struct('date', 'standard_concept_id')) \
            .groupBy('person_id', 'window').agg(F.collect_list('date_concept_tuple')) \
            .rdd.map(lambda row: (row[0], row[1], sorted(row[2], key=lambda x:x[0])))


class ComputeCoccurrenceAbstract(ABC):
    
    def __init__(self, spark, time_window):
        self.spark = spark
        self.time_window = time_window

    def generate_cooccurrence(self, patient_event, output_folder):
        
        concept_dictionary = self._generate_concept_dictionary(patient_event, output_folder)
        cooccurrence_matrix = self._compute_cooccurrence(self.time_window._apply_time_window(patient_event))
        
        #Join the cooccurrence matrix to concept_dictionary
        cooccurrence_matrix = cooccurrence_matrix \
            .join(concept_dictionary, 
                  cooccurrence_matrix['standard_concept_id_1'] == concept_dictionary['standard_concept_id']) \
            .select(cooccurrence_matrix['standard_concept_id_1'],
                    cooccurrence_matrix['standard_concept_id_2'],
                    cooccurrence_matrix['normalized_count'],
                    concept_dictionary['id'].alias('id_1'))

        cooccurrence_matrix = cooccurrence_matrix \
            .join(concept_dictionary, 
                  cooccurrence_matrix['standard_concept_id_2'] == concept_dictionary['standard_concept_id']) \
            .select(cooccurrence_matrix['standard_concept_id_1'],
                    cooccurrence_matrix['standard_concept_id_2'],
                    cooccurrence_matrix['normalized_count'],
                    cooccurrence_matrix['id_1'],
                    concept_dictionary['id'].alias('id_2'))
        
        #Save the cooccurrence matrix
        cooccurrence_matrix \
            .select('id_1', 'id_2', 'normalized_count', 
                    'standard_concept_id_1', 'standard_concept_id_2') \
            .write.mode('overwrite').parquet(self._get_cooccurrence_matrix_path(output_folder))
    
    @abstractmethod
    def _compute_cooccurrence(self, patient_event):
        raise NotImplemented()
    
    def _generate_concept_dictionary(self, patient_event, output_folder):
        
        #Create the concept occurrence dictionary
        concept_dictionary = patient_event \
            .select('person_id', 'standard_concept_id').distinct() \
            .groupBy('standard_concept_id').count() \
            .withColumn('id', F.dense_rank().over(Window.orderBy('standard_concept_id')) - 1) \
            .select('id','standard_concept_id', 'count')
        
        concept_dictionary.write.mode('overwrite') \
            .parquet(self._get_concept_dictionary_path(output_folder))
        
        return self.spark.read.parquet(self._get_concept_dictionary_path(output_folder))
    
    def _get_concept_dictionary_path(self, output_folder):
        return create_file_path(output_folder, CONCEPT_OCCURRENCE)
    
    def _get_cooccurrence_matrix_path(self, output_folder):
        return create_file_path(output_folder, COOCCURRENCE_MATRIX)


class ComputeCoccurrenceIterator(ComputeCoccurrenceAbstract):
    
    def __init__(self, spark, time_window):
        assert isinstance(time_window, TimeWindowAbstract) 
        super().__init__(spark, time_window)
    
    def _compute_cooccurrence(self, patient_event):
        
        grouped_events = self.time_window._group_events_rdd(patient_event)
        
        cooccurrence_matrix = grouped_events \
            .map(lambda row: (row[0], row[1], sorted(row[2], key=lambda x:x[0]))) \
            .mapPartitions(ComputeCoccurrenceIterator.compute_coocurrence_per_partition) \
            .map(lambda tp: ((tp[0], tp[1]), tp[2])) \
            .reduceByKey(lambda v1, v2: v1 + v2) \
            .map(lambda t: Row(standard_concept_id_1=t[0][0], standard_concept_id_2=t[0][1], normalized_count=t[1])) \
            .toDF()
        
        return cooccurrence_matrix
    
    @staticmethod
    def compute_coocurrence_per_partition(iterator):
    
        coocurrence_partition = dict()

        for row in iterator:
            if len(row[2]) > 1:
                sequences = row[2]
                seq_length = len(sequences)
                for i in range(seq_length):
                    for j in range(i + 1, seq_length):
                        pre_date, prev_concept_id = sequences[i]
                        next_date, next_concept_id = sequences[j]

                        time_diff = (next_date - pre_date).days
                        cooccurrence = 1.0 / (time_diff + 1)

                        if (prev_concept_id, next_concept_id) not in coocurrence_partition:
                            coocurrence_partition[(prev_concept_id, next_concept_id)] = 0.0

                        if (next_concept_id, prev_concept_id) not in coocurrence_partition:
                            coocurrence_partition[(next_concept_id, prev_concept_id)] = 0.0

                        coocurrence_partition[(prev_concept_id, next_concept_id)] += cooccurrence
                        coocurrence_partition[(next_concept_id, prev_concept_id)] += cooccurrence

        return [(k[0],k[1], v) for k, v in coocurrence_partition.items()] 


class ComputeCoccurrenceSelfJoin(ComputeCoccurrenceAbstract):
    
    def __init__(self, spark, time_window):
        assert isinstance(time_window, TimeWindowAbstract) 
        super().__init__(spark, time_window)
    
    def _compute_cooccurrence(self, patient_event):
        
        #Make two copies of the patient_visit_concept dataframe for self-join
        pvc_1 = patient_event.rdd.toDF(patient_event.schema)
        pvc_2 = patient_event.rdd.toDF(patient_event.schema)
        
        cooccurrence_matrix = pvc_1 \
            .join(pvc_2, self.window._self_join_logic(pvc_1, pvc_2))
        
        #Create the cooccurrence matrix via a self-join where the concept_ids are NOT the same
        cooccurrence_matrix = cooccurrence_matrix \
            .where(pvc_1['standard_concept_id'] != pvc_2['standard_concept_id']) \
            .select(pvc_1['person_id'],
                    pvc_1['standard_concept_id'].alias('standard_concept_id_1'), 
                    pvc_2['standard_concept_id'].alias('standard_concept_id_2'),
                    pvc_1['date'].alias('date_1'),
                    pvc_2['date'].alias('date_2')) \
            .withColumn('time_diff', F.lit(1.0) / (F.abs(F.datediff('date_1', 'date_2')) + F.lit(1))) \
            .groupBy('standard_concept_id_1', 'standard_concept_id_2') \
            .agg(F.sum('time_diff').alias('normalized_count'))
        
        return cooccurrence_matrix
