import os, sys
import
#from edit_distance.task import dataset_generator_genomic
#sys.path.insert(0, os.path.abspath('./'))

df = pd.read_csv("datasets/dataset_FINAL.tsv", sep='\t')
df.head()

print(sys.path)
#print(sys.modules)
#from edit_distance.task import dataset_generator_genomic as ds
from edit_distance import task
from edit_distance.task.dataset_generator_genomic import EditDistanceGenomicDatasetGenerator
