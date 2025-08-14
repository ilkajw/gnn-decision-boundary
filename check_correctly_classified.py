from config import DATASET_NAME
from index_sets_utils import graphs_correctly_classified
from math import comb

correct = graphs_correctly_classified(DATASET_NAME)

num_correct = len(correct)
num_incorrect = 188 - num_correct

print(f"# graphs not correctly classified: {num_incorrect}. \n"
      f"# graphs correctly classified: {num_correct} \n"
      f"# paths included under CORRECTLY_CLASSIFIED_ONLY: {comb(num_correct, 2)}")
