import os
import pandas as pd
import numpy as np

from common import *
from omop_vector_euclidean_distance import *

# MAX_COST = 0
DIAG = 0
LEFT = 1
TOP = 2
GAP = 0


class PatientSimilarity:
    
    def __init__(self, max_cost: int, is_similarity: bool):
        self.max_cost = max_cost
        self.func = max if is_similarity else min
    
    def match(self, sequence_1, sequence_2):
        
        row_size = len(sequence_1)
        column_size = len(sequence_2)

        alignment_scoring_matrix = np.zeros([row_size + 1, column_size + 1])
        alignment_scoring_matrix[0][1:] = [(i + 1) * self.max_cost for i in range(column_size)]
        alignment_scoring_matrix[1:, 0] = [(i + 1) * self.max_cost for i in range(row_size)]

        direction_matrix = np.zeros([row_size + 1, column_size + 1], dtype=int)
        direction_matrix[0][1:] = LEFT
        direction_matrix[1:, 0] = TOP

        for row in range(1, row_size + 1):
            for col in range(1, column_size + 1):
                val_1 = sequence_1[row-1]
                val_2 = sequence_2[col-1]
                
                val = 0 if val_1 == val_2 else 2

                prev_row = row - 1
                prev_col = col - 1

                diag = alignment_scoring_matrix[prev_row, prev_col]
                top = alignment_scoring_matrix[prev_row, col]
                left = alignment_scoring_matrix[row, prev_col]

                new_diag = alignment_scoring_matrix[prev_row, prev_col] + val
                new_top = alignment_scoring_matrix[prev_row, col] + self.max_cost
                new_left = alignment_scoring_matrix[row, prev_col] + self.max_cost

                candidates = [(new_diag, diag, DIAG),
                              (new_top, top, TOP),
                              (new_left, left, LEFT)]

                (new_value, prev_value, direction) = self.func(candidates, key=lambda x: (x[0], x[1]))

                alignment_scoring_matrix[row, col] = new_value
                direction_matrix[row, col] = direction
        
        normalized_score = alignment_scoring_matrix[-1, -1] * 2 / (len(sequence_1) + len(sequence_2))
        alignment = self._find_alignment(sequence_1, sequence_2, alignment_scoring_matrix, direction_matrix)
        return (normalized_score, alignment, alignment_scoring_matrix, direction_matrix)
    
    def match_approx(self, matrix, sequence_1, sequence_2):
        
        row_size, column_size = np.shape(matrix)

        alignment_scoring_matrix = np.zeros([row_size + 1, column_size + 1])
        alignment_scoring_matrix[0][1:] = [(i + 1) * self.max_cost for i in range(column_size)]
        alignment_scoring_matrix[1:, 0] = [(i + 1) * self.max_cost for i in range(row_size)]

        direction_matrix = np.zeros([row_size + 1, column_size + 1], dtype=int)
        direction_matrix[0][1:] = LEFT
        direction_matrix[1:, 0] = TOP

        for row in range(1, row_size + 1):
            for col in range(1, column_size + 1):
                val = matrix.iloc[row-1, col-1]

                prev_row = row - 1
                prev_col = col - 1

                diag = alignment_scoring_matrix[prev_row, prev_col]
                top = alignment_scoring_matrix[prev_row, col]
                left = alignment_scoring_matrix[row, prev_col]

                new_diag = alignment_scoring_matrix[prev_row, prev_col] + val
                new_top = alignment_scoring_matrix[prev_row, col] + self.max_cost
                new_left = alignment_scoring_matrix[row, prev_col] + self.max_cost

                candidates = [(new_diag, diag, DIAG),
                              (new_top, top, TOP),
                              (new_left, left, LEFT)]

                (new_value, prev_value, direction) = self.func(candidates, key=lambda x: (x[0], x[1]))

                alignment_scoring_matrix[row, col] = new_value
                direction_matrix[row, col] = direction
        
        normalized_score = alignment_scoring_matrix[-1, -1] * 2 / (len(sequence_1) + len(sequence_2))
        alignment = self._find_alignment(sequence_1, sequence_2, alignment_scoring_matrix, direction_matrix)
        return (normalized_score, alignment, alignment_scoring_matrix, direction_matrix)
    
    
    def _find_alignment(self, sequence_1, sequence_2, alignment_scoring_matrix, direction_matrix):
    
        local_seq_1 = sequence_1.copy()
        local_seq_2 = sequence_2.copy()

        init_row = len(local_seq_1)
        init_col = len(local_seq_2)
        aligned_seq_1 = []
        aligned_seq_2 = []
        alignment_score = []

        while (init_row > 0) | (init_col > 0):

            direction = direction_matrix[init_row, init_col]
            score = alignment_scoring_matrix[init_row, init_col]
            
            step_1 = GAP
            step_2 = GAP

            # diagonal
            if direction == DIAG:
                init_row -= 1
                init_col -= 1
                step_1 = local_seq_1.pop()
                step_2 = local_seq_2.pop()
            # top
            elif direction == TOP:
                init_row -= 1
                step_1 = local_seq_1.pop()
            # left
            elif direction == LEFT:
                init_col -= 1
                step_2 = local_seq_2.pop()
            else:
                raise Exception(f'Could not recognize the direction {direction}')

            aligned_seq_1.insert(0, step_1)
            aligned_seq_2.insert(0, step_2)
            alignment_score.insert(0, score)

        return pd.DataFrame(zip(aligned_seq_1, aligned_seq_2, alignment_score), columns=['sequence_1', 'sequence_2', 'alignment_score'])


