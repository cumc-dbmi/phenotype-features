class Concept:
    def __init__(self, concept_id: int, concept_name: str, concept_synonyms=None):
        self.concept_id = concept_id
        self.concept_name = concept_name
        self.concept_synonyms = concept_synonyms


class Feature:
    
    headers = ["default_header"]
    
    def __init__(self, name: str):
        self.name = name
    
    def description(self):
        return "This feature is {}".format(self.name)
    
    def annotate(self, training_set):
        pass