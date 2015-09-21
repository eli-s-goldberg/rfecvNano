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
import matplotlib
import matplotlib.pyplot as plt
from biokit.viz import corrplot
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['figure.figsize'] = (8,6)

# locate the csv database
path = './transportData.csv'

# Import the csv as a dataframe.
data = pd.DataFrame.from_csv(path, header=0, sep=',',
                             index_col=0, parse_dates=True, encoding=None,
                               tupleize_cols=False, infer_datetime_format=False)

    # List of features:
    # ['ObsRPShape', 'Poros', 'Darcy', 'ConcIn', 'PvIn', 'pH', 'IonStr',
    # 'SaltType', 'ColumnLWRatio', 'PartDensity', 'PartIEP', 'pHIEPDist',
    # 'PartZeta', 'CollecZeta', 'PartDiam', 'CollecDiam', 'PartCollSizeRatio',
    # 'Coating', 'ConcHA', 'TypeNOM', 'Hamaker', 'Dispersivity', 'NanoSize',
    # 'mbEffluent', 'mbRetained', 'mbEffluent_norm', 'mbRetained_norm']

# Not all the features are sufficient to support ML (our database is small anyway), so let's drop some features.
# This consistutes the first drop round.
featuresToDrop = ['Hamaker','Dispersivity','NanoSize','pHIEPDist',
                  'PartCollSizeRatio','ColumnLWRatio',
                  'mbEffluent', 'mbRetained','mbEffluent_norm']
data = data.drop(featuresToDrop,1)

# Let's put the data we can into our dimensionless quantities.
data['tempKelvin'] = 298.15 # the temperature is always assumed to be 25 degrees.
data['relPermValue'] = data.apply(relPermittivity,axis=1) # Calculate the rel permittivity
data['debyeLength'] = data.apply(debyeLength,axis=1) # Calculate the debye length
data['pecletNumber'] = data.apply(pecletNumber,axis=1)
data['aspectRatio'] = data.PartDiam/data.CollecDiam
data['zetaRatio'] = data.PartZeta/data.CollecZeta
data['pHIepRatio'] = data.pH/data.PartIEP

# Factorize the remaining training data fields to turn categories into numbers.
data['Coating'] = data['Coating'].factorize()[0]
data['SaltType'] = data['Coating'].factorize()[0]
data['NMId'] = data['NMId'].factorize()[0]
data['TypeNOM'] = data['TypeNOM'].factorize()[0]

# Do not include any experiments which have null values.
data = data.dropna()

# Set the target data fields. In this case, the 'ObsRPShape' and the 'mbRetained_norm' (i.e., retained fration, RF).
targetDataRPShape = data['ObsRPShape'].factorize()[0] # convert text to numeric values
print targetDataRPShape
targetDataRPShapeUniqueList = list(set(data['ObsRPShape'].unique()))
targetDataRF = data.mbRetained_norm
print targetDataRPShape
# Drop the fields that are subordinate to the calculated fields (i.e., the ones wrapped up in the dimensionless numbers
# (e.g., salttype and pH are in debyelength .

data = data.drop(['ObsRPShape', # target value (to be predicted)
                  'mbRetained_norm', # target value (to be predicted)
                  'Darcy', # included in peclet number
                  'ObsRPShape', # target value (to be predicted)
                  'relPermValue', # perfectly correlated wiht material and include in Debye
                  'tempKelvin', # assumed to be the same for each experiment (also in Debye)
                  'PartZeta', # included in dimensionless ratio of zetas
                  'PartIEP', # included in dimensionless ratio of pH to IEP
                  'PartDiam', # included in dimensionless aspect ratio
                  'CollecDiam', # included in dimensionless aspect ratio
                  'CollecZeta',# included in dimensionless aspect ratio
                  'IonStr', # included in Debye Length
                  'SaltType',# included in Debye Length
                  'pH'# included in dimensionless ratio of pH to IEP
                 ],1)

print list(data) # print out the remaining data field headers

# Make sure to install biokit dependencies with requirements.txt
# https://pypi.python.org/pypi/biokit/0.0.5

c = corrplot.Corrplot(data)
c.plot(upper='circle',fontsize = 10)

# assign the remaining data to the training data set.
trainingData = data


# Store the training data and target data as a matrices for import into ML.
trainingDataMatrix = trainingData.as_matrix() # all numbers, no headers
targetDataRPShapeMatrix  = targetDataRPShape
targetDataRFMatrix =  targetDataRF.as_matrix() # all numbers, no headers

# Get a list of the trainingData features remaining. This is used later for plotting etc.
trainingDataNames =  list(trainingData)
# print trainingDataNames

print targetDataRPShapeMatrix
print trainingData


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