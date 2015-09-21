# package import
import multiprocessing
import numpy as np
multiprocessing.cpu_count()
from helperFunctions import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import cross_validation
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit


# locate the csv database
path = './transportData.csv'

# Import the csv as a dataframe.
data = pd.DataFrame.from_csv(path, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                               tupleize_cols=False, infer_datetime_format=False)

    # List of features:
    # ['ObsRPShape', 'Poros', 'Darcy', 'ConcIn', 'PvIn', 'pH', 'IonStr',
    # 'SaltType', 'ColumnLWRatio', 'PartDensity', 'PartIEP', 'pHIEPDist',
    # 'PartZeta', 'CollecZeta', 'PartDiam', 'CollecDiam', 'PartCollSizeRatio',
    # 'Coating', 'ConcHA', 'TypeNOM', 'Hamaker', 'Dispersivity', 'NanoSize',
    # 'mbEffluent', 'mbRetained', 'mbEffluent_norm', 'mbRetained_norm']

# Not all the features are sufficient to support ML (our database is small anyway), so let's drop some features.
# This consistutes the first drop round.
featuresToDrop = ['Hamaker','Dispersivity','NanoSize','pHIEPDist','PartCollSizeRatio','ColumnLWRatio',
                  'mbEffluent', 'mbRetained','mbEffluent_norm']
data = data.drop(featuresToDrop,1)


# rfecv seed (to control randomness and get repeatable results
SEED = 69

# set the number of model iterations, where the more the better. Each run takes several minutes, so 100-500 is best.
iterator1 = 10 # this is the number of model iterations to go through.



# Drop the fields that are subordinate to the calculated fields (i.e., the ones wrapped up in the dimensionless numbers
# (e.g., salttype and pH are in debyelength .
data = data.drop(['ObsRPShape','mbRetained_norm','Darcy','NMId','ObsRPShape','relPermValue','tempKelvin',
                  'PartZeta','PartIEP','PartDiam','CollecDiam','CollecZeta','IonStr','SaltType','pH'],1)

# assign the remaining data to the training data set.
trainingData = data

# Store the training data and target data as a matrices for import into ML.
trainingDataMatrix = trainingData.as_matrix()
targetDataRPShapeMatrix = targetDataRPShape
targetDataRFMatrix =  targetDataRF.as_matrix()

# Get a list of the trainingData features remaining. This is used later for plotting etc.
trainingDataNames =  list(trainingData)
# print trainingDataNames

stratShuffleSplitRFECVRandomForestClassification (nEstimators= 1000,
                                                  iterator1=10,
                                                  minSamplesSplit=2,
                                                  maxFeatures=None,
                                                  maxDepth=4,
                                                  nFolds=5,
                                                  targetDataMatrix = targetDataRPShapeMatrix,
                                                  trainingData = trainingData,
                                                  trainingDataMatrix = trainingDataMatrix,
                                                  SEED = 5)

## Output file summary
fileName1 = './outputFiles/f1_score_all.csv'
fileName2 = './outputFiles/class_IFIRS.csv'
fileName3 = './outputFiles/class_optimum_length.csv'
fileName4 = './outputFiles/class_sel_feature_importances.csv'
fileName5 = './outputFiles/class_rfecv_grid_scores.csv'