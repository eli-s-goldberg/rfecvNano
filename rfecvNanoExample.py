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

# Let's check to see how correlated our variables are. print data.corr() to inspect values.
# plot_corr(data,size=10)



# rfecv seed (to control randomness and get repeatable results
SEED = 69

# set the number of model iterations, where the more the better. Each run takes several minutes, so 100-500 is best.
iterator1 = 10 # this is the number of model iterations to go through.

# create the calculated data fields:
data['tempKelvin'] = 298.15 # the temperature is always assumed to be 25 degrees.
data['relPermValue'] = data.apply(relPermittivity,axis=1) # Calculate the relative permittivity value
data['debyeLength'] = data.apply(debyeLength,axis=1) # Calculate the debye length
data['pecletNumber'] = data.apply(pecletNumber,axis=1)
data['aspectRatio'] = data.PartDiam/data.CollecDiam
data['zetaRatio'] = data.PartZeta/data.CollecZeta
data['pHIepRatio'] = data.pH/data.PartIEP

# factorize the remaining training data fields
data['Coating'] = data['Coating'].factorize()[0]
data['SaltType'] = data['Coating'].factorize()[0]

# Do not include any experiments which have null values.
data = data.dropna()

# Set the target data fields. In this case, the 'ObsRPShape' and the 'mbRetained_norm' (i.e., retained fration, RF).
targetDataRPShape = data['ObsRPShape'].factorize()[0] # make sure that text values are converted to numeric values
targetDataRPShapeUniqueList = list(set(data['ObsRPShape'].unique()))

targetDataRF = data.mbRetained_norm


# Drop the fields that are subordinate to the calculated fields (i.e., the ones wrapped up in the dimensionless numbers
# (e.g., salttype and pH are in debyelength .
data = data.drop(['ObsRPShape','mbRetained_norm','Darcy','NMId','ObsRPShape','relPermValue','tempKelvin','TypeNOM',
                  'PartZeta','PartIEP','PartDiam','CollecDiam','CollecZeta','PvIn','IonStr','SaltType','pH'],1)

# assign the remaining data to the training data set.
trainingData = data

# Store the training data and target data as a matrices for import into ML.
trainingDataMatrix = trainingData.as_matrix()
targetDataRPShapeMatrix = targetDataRPShape
targetDataRFMatrix =  targetDataRF.as_matrix()

# Get a list of the trainingData features remaining. This is used later for plotting etc.
trainingDataNames =  list(trainingData)
# print trainingDataNames

stratShuffleSplitRFECVRandomForestClassification (nEstimators= 100,
                                                  iterator1=1,
                                                  minSamplesSplit=2,
                                                  maxFeatures=None,
                                                  maxDepth=4,
                                                  nFolds=5,
                                                  targetDataMatrix = targetDataRPShapeMatrix,
                                                  trainingData = trainingData,
                                                  trainingDataMatrix = trainingDataMatrix,
                                                  SEED = 5)

## Output file summary
fileName1 = 'f1_score_all.csv'
fileName2 = 'class_IFIRS.csv'
fileName3 = 'class_optimum_length.csv'
fileName4 = 'class_sel_feature_importances.csv'
fileName5 = 'class_rfecv_grid_scores.csv'