import logging
import numpy as np
import random
import pandas as pd

class EuclideanDistance:
    
    def __init__(self, name: str, path: str):
        self.logger = logging.getLogger(name)
        self.name = name
        self.path = path
        self.pairwise_dist = pd.read_pickle(path)
        self.all_concept_ids = self.pairwise_dist.index.to_list()
    
    def compute_average_dist(self, concept_list, name=None):
        name = self.name if name is None else name
        intersection = set(concept_list).intersection(self.all_concept_ids)
        no_of_missing = len(concept_list) - len(intersection)
        no_of_concept_ids = len(intersection)
        values = self.pairwise_dist.loc[intersection, intersection].to_numpy().ravel()
        results = {
            'mean' : np.mean(values), 
            '25%' : np.quantile(values, 0.25), 
            'median' : np.median(values), 
            '75%' : np.quantile(values, 0.75), 
            'num_of_concepts' : no_of_concept_ids,
            'num_of_missings' : no_of_missing
        }
        return pd.Series(results, name=name)
    
    def get_long_form(self, concept_list):
        intersection = set(concept_list).intersection(self.all_concept_ids)
        values = self.pairwise_dist.loc[intersection, intersection]
        long_form = values.unstack()
        long_form.index.rename(['concept_id_1', 'concept_id_2'], inplace=True)
        long_form = long_form.to_frame('metric').reset_index()
        long_form[(long_form['concept_id_1'] != long_form['concept_id_2'])]
        return long_form
    
    def get_dist(self, concept_id_1, concept_id_2):
        if not self._validate_concept_id(concept_id_1) or not self._validate_concept_id(concept_id_2):
            return None
        return self.pairwise_dist.loc[concept_id_1, concept_id_2]
    
    def get_metric(self, list_1, list_2):
        filtered_list_1 = [c for c in list_1 if c in self.all_concept_ids]
        filtered_list_2 = [c for c in list_2 if c in self.all_concept_ids]
        return self.pairwise_dist.loc[filtered_list_1, filtered_list_2]
    
    def get_closest(self, concept_id, num=10, ascending=True):
        if not self._validate_concept_id(concept_id):
            return None
        index = self.pairwise_dist.loc[concept_id].index != concept_id
        return self.pairwise_dist.loc[concept_id][index].sort_values(ascending=ascending)[0:num]
    
    def compute_random_average_dist(self, k):
        return self.compute_average_dist(concept_list=self._get_ramdom_concept_ids(k), name=self.name + '_random_sample')
    
    def _get_ramdom_concept_ids(self, k):
        return random.sample(population=self.all_concept_ids, k=k)
    
    def _validate_concept_id(self, concept_id):
        if concept_id not in self.all_concept_ids:
            self.logger.error('{concept_id} does not exsit'.format(concept_id=concept_id))
            return False
        return True


