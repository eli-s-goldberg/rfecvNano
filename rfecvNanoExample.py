import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

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
featuresToDrop = ['Hamaker','Dispersivity','NanoSize','pHIEPDist','PartCollSizeRatio','ColumnLWRatio',
                  'mbEffluent', 'mbRetained','mbRetained_norm','mbEffluent_norm']
data = data.drop(featuresToDrop,1)

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    from matplotlib import cm

    corr = df.corr()
    label = df.corr()
    mask = np.tri(corr.shape[0],k=-1)
    corr = np.ma.array(corr,mask=mask)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    cmap = cm.get_cmap('jet',10)
    cmap.set_bad('w')

    plt.xticks(range(len(label.columns)), label.columns, rotation=90)
    plt.yticks(range(len(label.columns)), label.columns)
    ax.imshow(corr,interpolation='nearest',cmap=cmap)
    plt.show()

plot_corr(data,size=10)
# Let's check to see how correlated our variables are
