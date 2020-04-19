from scipy.spatial.distance import cosine
import pandas as pd
import csv
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import random
import math
from mpl_toolkits import mplot3d

abc = np.array([1,2,3])
egf = np.array([1,2,3])
print(not (abc == egf).all())