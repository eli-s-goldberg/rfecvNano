# /usr/bin/python
# -*- coding: utf-8 -*-
# Developed by:  Eli Goldberg, MS, PhD
#                09/2015
#                © Eli Goldberg

# This file contains functions to support the import, use, and application of the rfecvNano model. Most, but not all,
# features include instructions for use. However, you're smart, and should be able to use these tools to build your own
# rfecv feature selection tool using the examples provided in the rfecvNanoExample.py.

def importNtpDatabase(sqliteFile, fileName, listOfFeatures):  # This function imports the nanotransport database
    '''
    This function imports the transportData csv database. Note that there are some values which are factorized
    are extracted and exported:
        ['NMId']
        ['SaltType']
        ['ObsRPShape']
        ['TypeNOM']
        ['Coating']

    Additionally, it creates an sqlite file that can be used to perform additional queries for data selection.
    :param sqliteFile: Desired SQLITE File Name
    :param fileName: CSV Database File Name
    :param listOfFeatures: Include list of features by #'ing out (i.e,. #'d out are included) - see example.
    :return:
    :example:

    # CSV Database File Name
    fileName = 'SQLiteImportData_v1.csv'

    # Desired SQLITE File Name
    sqliteFile = 'NanoTransportParameters1.sqlite'

    # Initialize dataframe to store database
    ntpDataset = pd.DataFrame()

    # Include list of features by #'ing out (i.e,. #'d out are included).
    listOfFeatures = ['PublicationTitle',
                      'ProfileID',
                      'Hamaker',
                      'Dispersivity',
                      'NanoSize',
                      'pHIEPDist',
                      'PartCollSizeRatio',
                      'ColumnLWRatio',
                      'mbEffluent',
                      'mbRetained',
                      'mbRetained_norm',
                      'mbEffluent_norm',
                      'Notes1',
                      'Notes2',
                      'Notes3',
                      'Notes4',
                      'Notes5']

    [sqliteFile, fileName, ntpDataset] = import_ntp_database(sqliteFile=sqliteFile, fileName=fileName,
                                                            listOfFeatures=listOfFeatures)
    '''
    import sqlite3
    import pandas as pd
    import os
    from os import getcwd, walk
    import fnmatch

    # Look through project directory and locate transport database (filename)
    matches = []  # initialize array to hold matches
    for root, dirnames, filenames in os.walk('..'):
        for filename in fnmatch.filter(filenames, fileName):
            matches.append(os.path.join(root, filename))

    print matches[0]
    # Move to the working directory that contains the transport database.
    os.chdir(os.path.dirname(matches[0]))

    # Label the current working directory
    transportDatabaseDir = getcwd()

    # Connecting (i.e., establishing) with the above name database to file
    connection = sqlite3.connect(sqliteFile)
    cursor = connection.cursor()

    # Set the path - this should be the matches?
    path = os.path.join(transportDatabaseDir, fileName)

    # Read CSV into dataframe
    reader = pd.DataFrame.from_csv(path, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                   tupleize_cols=False, infer_datetime_format=False)

    # From dataframe, create database
    DataReader = reader.to_sql(name='NTP', con=connection, flavor='sqlite', if_exists='replace', index=True,
                               index_label=None)

    # From database, perform query type, and re-import into dataframe.
    databaseName = pd.read_sql('SELECT * FROM NTP', con=connection)

    # Commit connection, close database cursor and connection
    connection.commit()
    cursor.close()
    connection.close()
    print(path)

    # drop the list of features specified
    databaseName.drop(listOfFeatures, axis=1, inplace=True)

    # Do not include any experiments which have null values.
    databaseName = databaseName.dropna()

    # Categorize the text names into integers using factorize. The [0] is used to specify just the array
    # NMId
    # Salt type
    # ObsRPShape

    databaseName.NMId = databaseName.NMId.factorize()[0]
    databaseName.SaltType = databaseName.SaltType.factorize()[0]
    databaseName.ObsRPShape = databaseName.ObsRPShape.factorize()[0]
    databaseName.TypeNOM = databaseName.TypeNOM.factorize()[0]
    databaseName.Coating = databaseName.Coating.factorize()[0]

    return (sqliteFile, fileName, databaseName)


def importNtpDatabaseWithUniques(sqliteFile, fileName, listOfFeatures):
    '''
    This function imports the nanotransport database. Note that there are some values which are factorized
    and the uniques are extracted and exported:
        ['NMId']
        ['SaltType']
        ['ObsRPShape']
        ['TypeNOM']
        ['Coating']


    :param sqliteFile: Desired SQLITE File Name
    :param fileName: CSV Database File Name
    :param listOfFeatures: Include list of features by #'ing out (i.e,. #'d out are included) - see example.
    :return:
    :example:

    # CSV Database File Name
    fileName = 'SQLiteImportData_v1.csv'

    # Desired SQLITE File Name
    sqliteFile = 'NanoTransportParameters1.sqlite'

    # Initialize dataframe to store database
    ntpDataset = pd.DataFrame()

    # Include list of features by #'ing out (i.e,. #'d out are included).
    listOfFeatures = ['PublicationTitle',
                      'ProfileID',
                      'Hamaker',
                      'Dispersivity',
                      'NanoSize',
                      'pHIEPDist',
                      'PartCollSizeRatio',
                      'ColumnLWRatio',
                      'mbEffluent',
                      'mbRetained',
                      'mbRetained_norm',
                      'mbEffluent_norm',
                      'Notes1',
                      'Notes2',
                      'Notes3',
                      'Notes4',
                      'Notes5']

    [sqliteFile, fileName, ntpDataset] = import_ntp_database(sqliteFile=sqliteFile, fileName=fileName,
                                                            listOfFeatures=listOfFeatures)
    '''
    import sqlite3
    import pandas as pd
    import os
    from os import getcwd, walk
    import fnmatch

    # Look through project directory and locate transport database (filename)
    matches = []  # initialize array to hold matches
    for root, dirnames, filenames in os.walk('..'):
        for filename in fnmatch.filter(filenames, fileName):
            matches.append(os.path.join(root, filename))

    print matches[0]
    # Move to the working directory that contains the transport database.
    os.chdir(os.path.dirname(matches[0]))

    # Label the current working directory
    transportDatabaseDir = getcwd()

    # Connecting (i.e., establishing) with the above name database to file
    connection = sqlite3.connect(sqliteFile)
    cursor = connection.cursor()

    # Set the path - this should be the matches?
    path = os.path.join(transportDatabaseDir, fileName)

    # Read CSV into dataframe
    reader = pd.DataFrame.from_csv(path, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                   tupleize_cols=False, infer_datetime_format=False)

    # From dataframe, create database
    DataReader = reader.to_sql(name='NTP', con=connection, flavor='sqlite', if_exists='replace', index=True,
                               index_label=None)

    # From database, perform query type, and re-import into dataframe.
    databaseName = pd.read_sql('SELECT * FROM NTP', con=connection)

    # Commit connection, close database cursor and connection
    connection.commit()
    cursor.close()
    connection.close()
    print(path)

    # drop the list of features specified
    databaseName.drop(listOfFeatures, axis=1, inplace=True)

    # Do not include any experiments which have null values.
    databaseName = databaseName.dropna()

    # Categorize the text names into integers using factorize. The [0] is used to specify just the array
    # NMId
    # Salt type
    # ObsRPShape

    NMIdNames = list(set(databaseName['NMId'].unique()))
    SaltTypeNames = list(set(databaseName['SaltType'].unique()))
    ObsRPShapeNames = list(set(databaseName['ObsRPShape'].unique()))
    TypeNOMNames = list(set(databaseName['TypeNOM'].unique()))
    CoatingNames = list(set(databaseName['Coating'].unique()))

    databaseName.NMId = databaseName.NMId.factorize()[0]
    databaseName.SaltType = databaseName.SaltType.factorize()[0]
    databaseName.ObsRPShape = databaseName.ObsRPShape.factorize()[0]
    databaseName.TypeNOM = databaseName.TypeNOM.factorize()[0]
    databaseName.Coating = databaseName.Coating.factorize()[0]

    return (sqliteFile, fileName, databaseName, NMIdNames, SaltTypeNames, ObsRPShapeNames, TypeNOMNames, CoatingNames)


def countClasses(target_feature):
    '''
    This is a simple function to count the number in each class for classification result.
    * I found a more robust implementation for this old function. What I should do is bring it into pandas
    and then use the groupby function, like df.groupby(XX].count(), where XX is a list of factorized counts. I should
    really try to keep this consistent...

    :param target_feature:
    :return:
    '''
    import pandas as pd
    target_feature = list(target_feature)
    # HE
    HE_count = target_feature.count(0)
    # IRwd
    IRwD_count = target_feature.count(1)
    # LD
    LD_count = target_feature.count(2)
    # SIR
    SIR_count = target_feature.count(3)
    # IRwd
    EXP_count = target_feature.count(4)

    return (HE_count, IRwD_count, LD_count, SIR_count, EXP_count)


def multi2binRefactor(target_feature):
    import numpy as np
    bin_class = []
    refactor_bin_class = []
    for targets in target_feature:
        # if targets == (0 or 4):
        #     bin_class = 0
        # elif targets == (1 or 2 or 3):
        #     bin_class = 1
        if targets == 0:
            bin_class = 0
        elif targets == 1:
            bin_class = 1
        elif targets == 2:
            bin_class = 1
        elif targets == 3:
            bin_class = 1
        elif targets == 4:
            bin_class = 0
        refactor_bin_class.append(bin_class)
    # refactor_bin_class = np.array(refactor_bin_class)
    return (refactor_bin_class)


def countBinClasses(target_feature):
    '''
    This is a simple function to count the number in each binary class for classification result.
    My common use for this is to count the number in class [0], which are HE and EXP, and
    to cound the number in class [1], which are SIR, LD, and IRwd.

    :param target_feature: this is an array that contain only two classes 0 or 1.

    :return: returns an array where item [0] is a count of the 0s and [1] is a count of the 1s.
    '''

    target_feature = list(target_feature)
    binClass1Count = target_feature.count(0)
    binClass2Count = target_feature.count(1)

    return (binClass1Count, binClass2Count)


def exportTreeJson(decision_tree, out_file=None, feature_names=None):
    '''Export a decision tree in JSON format.

    This function generates a JSON representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to JSON.

    out : file object or string, optional (default=None)
        Handle or name of the output file.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    Returns
    -------
    out_file : file object
        The file object to which the tree was exported.  The user is
        expected to `close()` this object when done with it.

    '''

    import numpy as np
    from sklearn.tree import _tree
    def arr_to_py(arr):
        arr = arr.ravel()
        wrapper = float
        if np.issubdtype(arr.dtype, np.int):
            wrapper = int
        return map(wrapper, arr.tolist())

    def node_to_str(tree, node_id):
        node_repr = '"error": %.4f, "samples": %d, "value": %s' \
                    % (tree.init_error[node_id],
                       tree.n_samples[node_id],
                       arr_to_py(tree.value[node_id]))
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X[%s]" % tree.feature[node_id]

            label = '"label": "%s <= %.2f"' % (feature,
                                               tree.threshold[node_id])
            node_type = '"type": "split"'
        else:
            node_type = '"type": "leaf"'
            label = '"label": "Leaf - %d"' % node_id
        node_repr = ", ".join((node_repr, label, node_type))
        return node_repr

    def recurse(tree, node_id, parent=None):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Open node with description
        out_file.write('{%s' % node_to_str(tree, node_id))

        # write children
        if left_child != _tree.TREE_LEAF:  # and right_child != _tree.TREE_LEAF
            out_file.write(', "children": [')
            recurse(tree, left_child, node_id)
            out_file.write(', ')
            recurse(tree, right_child, node_id)
            out_file.write(']')

        # close node
        out_file.write('}')

    if out_file is None:
        out_file = open("tree.json", "w")
    elif isinstance(out_file, basestring):
        out_file = open(out_file, "w")

    if isinstance(decision_tree, _tree.Tree):
        recurse(decision_tree, 0)
    else:
        recurse(decision_tree.tree_, 0)

    return out_file


def import_agg_rfecv_outputs(fileName1, fileName2, fileName3, fileName4, fileName5):
    '''

    :param fileName1:
    :param fileName2:
    :param fileName3:
    :param fileName4:
    :param fileName5:
    :return:
    '''
    import pandas as pd
    import os
    from os import getcwd

    # This is the script working directory. It is where the file is located and where things start.
    scriptDir = getcwd()
    print(scriptDir)

    # Move up one directory and check working directory.
    os.chdir("..")
    gitMasterDir = getcwd()
    print(gitMasterDir)

    os.chdir('transport_database')
    transportDatabaseDir = getcwd()
    print(transportDatabaseDir)

    path1 = os.path.join(transportDatabaseDir, fileName1)
    path2 = os.path.join(transportDatabaseDir, fileName2)
    path3 = os.path.join(transportDatabaseDir, fileName3)
    path4 = os.path.join(transportDatabaseDir, fileName4)
    path5 = os.path.join(transportDatabaseDir, fileName5)

    reader_f1 = pd.DataFrame.from_csv(path1, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                      tupleize_cols=False, infer_datetime_format=False)

    reader_f2 = pd.DataFrame.from_csv(path2, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                      tupleize_cols=False, infer_datetime_format=False)

    reader_f3 = pd.DataFrame.from_csv(path3, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                      tupleize_cols=False, infer_datetime_format=False)

    reader_f4 = pd.DataFrame.from_csv(path4, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                      tupleize_cols=False, infer_datetime_format=False)

    reader_f5 = pd.DataFrame.from_csv(path5, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                      tupleize_cols=False, infer_datetime_format=False)


def singleCSVImport(path1):
    '''

    :param path1:
    :param fileName1:
    :return:
    '''
    import pandas as pd

    return pd.DataFrame.from_csv(path1, header=0, sep=',', index_col=0, parse_dates=True, encoding=None,
                                 tupleize_cols=False, infer_datetime_format=False)


def getTreeLineage(tree, feature_names):
    '''
    :param tree: An sklearn decision tree object (alternatively, a single tree from a random forest).
    :param feature_names: This is a list of feature names ['featureName1','featureName2', etc..]
    :return: returns a textual description of the decision tree logic.

    '''
    import numpy as np
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
        lineage.append((parent, split, threshold[parent], features[parent]))
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    for child in idx:
        for node in recurse(left, right, child):
            print node


def relPermittivity(row):
    '''

    :param row: a dataframe with permittivities matching the following list (dataframes that contain other materials
                must be added to the list):
                Index == [name = permittivity value]
                0 == [C60= 4.4]
                1 == [TiO2= 110]
                2 == [ZnO= 2]
                3 == [CuO= 1]
                4 == [MWCNTs = 1328]
                5 == [Ag = 2.65] # (ref1)
                6 == [CeO2 = 26]
                7 == [Fe = 14.2]
                8 == [SiO2 = 3.9]
                9 == [nHAP = 7]
                10 == [nBiochar = 2.9]
                * Note that the dataframe.factorize must put these in the EXACT order
                ** If something is messed up it will display 9999

                The dataframe column must contain the following columns:
                ['NMId']


    :return:
    :examples

    :references:
    ref1: Silver Metallization, Adams et al., 2008 (book)
    '''
    if (row['NMId'] == 0):
        relPermVal = 4.4
    elif (row['NMId'] == 1):  # TiO2
        relPermVal = 110
    elif (row['NMId'] == 2):  # ZnO
        relPermVal = 2
    elif (row['NMId'] == 3):  # CuO
        relPermVal = 1
    elif (row['NMId'] == 4):  # MWCNTs
        relPermVal = 1328
    elif (row['NMId'] == 5):  # Ag
        relPermVal = 2.65
    elif (row['NMId'] == 6):  # CeO2
        relPermVal = 26
    elif (row['NMId'] == 7):  # Fe
        relPermVal = 14.2
    elif (row['NMId'] == 8):  # SiO2
        relPermVal = 3.9
    elif (row['NMId'] == 9):  # nHAP
        relPermVal = 7
    elif (row['NMId'] == 10):  # nBiochar
        relPermVal = 2.9
    else:
        relPermVal = 9999

    return float(relPermVal)


def debyeLength(row):
    '''

    :param row: a dataframe with parameters required to calculate Debye Length:
                ['SaltType']
                ['pH']
                ['relPermValue']
                ['tempKelvin']
    :return: Debye length
            if it's a monovalent, symettric electrolyte (e.g., NaCl, KCl), no buffer, or in or CaCl2

    :references 'Electrostatic Double Layer Force: Part II', Dr. Pallab Ghosh, DCE IIT Guwahati, India
    :example
    dataf = {'relPermValue' : [4, 5, 24, 110],
         'tempKelvin': [298, 298, 298, 298],
         'IonStr': [10, 0.1, 1, 0],
         'SaltType':[1, 1, 3,5 ],
         'pH': [5, 6, 7, 8]}

    # create new column containing debeye lengths
    dataf['debyeLength'] = df.apply(debyeLength,axis=1)


    '''
    permFreeSpace = float(8.854 * 10 ** -12)
    bolzConstant = float(1.3806504 * 10 ** -23)
    elecCharge = float(1.602176466 * 10 ** -19)
    numAvagadro = float(6.02 * 10 ** 23)
    if (row['IonStr'] == 0) or (row['SaltType'] == 3):
        ionStr1 = float(1 * 10 ** (float(row['pH']) - 14))
        ionStr2 = float(1 * 10 ** (-1 * float(row['pH'])))
        ZiCi = float(1 ** 2 * ionStr1 + 1 ** 2 * ionStr2)
        return float((1 / (float((numAvagadro * elecCharge ** 2 / (
        permFreeSpace * float(row['relPermValue']) * bolzConstant * float(row['tempKelvin'])) * ZiCi) ** 0.5))))
    elif (row['SaltType'] == 1):
        ZiCi = float(2 ** 2 * float(row['IonStr']) + 1 ** 2 * 2 * float(row['IonStr']))
        return float((1 / (float((numAvagadro * elecCharge ** 2 / (
        permFreeSpace * float(row['relPermValue']) * bolzConstant * float(row['tempKelvin'])) * ZiCi) ** 0.5))))
    else:
        ZiCi = float(1 ** 2 * float(row['IonStr']) + 1 ** 2 * float(row['IonStr']))
        return float((1 / (float((numAvagadro * elecCharge ** 2 / (
        permFreeSpace * float(row['relPermValue']) * bolzConstant * float(row['tempKelvin'])) * ZiCi) ** 0.5))))


def epmToZeta(row):
    '''
    This employs the Helmholtz-Smulochowski equation to relate the electrophoretic mobility to the zeta potential.
    :param row: a dataframe with parameters required to turn electrophoretic mobility into zeta potential:
                ['tempKelvin']
                ['epm']
    :return:
    :references 'Dielectric Constant of Water from 0 to 100 C', Malmberg and Maryott 1956
    '''

    dialConstWater25 = 7.083e-12  # CV**(-1/cm) @ 25C
    tempC = float(row['tempKelvin']) - 273.15
    dialConstWater = 87.740 - 0.40008 * tempC + 9.398 * (10 ** -4) * (tempC ** 2) - 1.410 * (10 ** -6) * (tempC ** 3)
    return float(dialConstWater)


def absViscTempWater(row):
    '''
    This equation determines the absolute viscosity as a function of temperature for water.
    Effective range is from T = 273.15 (freezing, or 0C) to 373.15 (boiling, phase change or 100C)
    Absolute viscosity (u) is the same as dynamic viscosity. Kinematic viscosity,v, is u/p, where p is the fluid density.

    :param row: a dataframe with parameters required to determine the absolute viscosity of water as a function of temp:
                ['tempKelvin']
    :return: absolute viscosity in Pa.s
    :references: R.F. Crouch and A. Cameron, Viscosity-Temperature Equations for Lubricants,
                 Journal of the Institute of Petroleum, Vol. 47, 1961, pp. 307-313
    :examples:
    dataf = {'relPermValue' : [4, 5, 24, 110],
         'tempKelvin': [298, 298, 298, 298],
         'IonStr': [10, 0.1, 1, 0],
         'SaltType':[1, 1, 3,5 ],
         'pH': [5, 6, 7, 8]}

    # create new column containing absolute viscosities
    dataf['viscWater'] = df.apply(absViscTempWater,axis=1)
    '''
    import math
    A = float(-3.7188)
    B = float(578.919)
    C = float(-137.546)
    return float(math.exp(A + (B / (C + float(row['tempKelvin'])))))


def densityWaterTemp(row):
    '''
    This equation gives the density as a function of temperature for water.
    Effective range is from T = 273.15 (freezing, or 0C) to 373.15 (boiling, phase change or 100C)
    It should be noted that the volumetric temperature coefficient, beta, changes for water as a function of temperature
    I am using it for water near 20c, which is a decent approximation.

    l_density = ;   % [kg/m^3]
    :param row: a dataframe with parameters required to determine the density of water as a function of temp:
                ['tempKelvin']
    :return: water density kg/m**3
    :examples:
    dataf = {'relPermValue' : [4, 5, 24, 110],
         'tempKelvin': [298, 298, 298, 298],
         'IonStr': [10, 0.1, 1, 0],
         'SaltType':[1, 1, 3,5 ],
         'pH': [5, 6, 7, 8]}

    # create new column containing water densitiess
    dataf['densWater'] = df.apply(densityWaterTemp,axis=1)

    '''

    return float(998.2071 / (1 + 0.000207 * (float(row['tempKelvin']) - 293.15)))


def chordInputData(DataSet, corrVarName1, corrVarName2, csvName, indexNames, NMIdNames):
    '''
    :param DataSet: This is a dataframe containing the data with uniques (i.e., from importNtpDatabaseWithUniques)
                        The dataset should have headers.
    :param corrVarName1: This is the first correlation variable name that you want to compare (e.g., 'NMId')
    :param corrVarName2: This is the second correlation variable name you want to compare (e.g., 'ObsRPShape')
    :param csvName: ["C60", "TiO2", "ZnO", "CuO", "MWCNT", "Ag", "CeO2", "Fe", "SiO2", "nHAP", "nBiochar"]
    :param indexNames: [ "HE", "IRwD", "LD", "SiR", "Exp"]
    :return: prints, saves, an returns the dataframe suitable for import into chord diagram
             into a csv with name 'csvName'
    :Example:
    NMIdNames = ["C60", "TiO2", "ZnO", "CuO", "MWCNT", "Ag", "CeO2", "Fe", "SiO2", "nHAP", "nBiochar"]
    indexNames = [ "HE", "IRwD", "LD", "SiR", "Exp"]
    # SaltTypeNames = [u'CaCl2', u'NaHCO3', u'None', u'KCl', u'NaCl', u'KNO3']
    # TypeNOMNames = [u'None', u'HA', u'Oxalic', u'Formic', u'Alg', u'Citric', u'FA', u'TRIZMA', u'SRHA']
    chordInputData(ntpDataset,'TypeNOM','ObsRPShape','data.csv',indexNames,TypeNOMNames)

    '''
    import numpy as np
    import pandas as pd
    correlationVariable1 = np.array(DataSet[corrVarName1])
    correlationVariable2 = np.array(DataSet[corrVarName2])
    out = pd.DataFrame()
    out[corrVarName1] = correlationVariable1
    out[corrVarName2] = correlationVariable2
    out = out.groupby([corrVarName2, corrVarName1], squeeze=True).size()
    numberGroups = max(correlationVariable2) + 1
    a = []
    for i in range(0, numberGroups):
        b = out[i]
        a.append(b)
    df = pd.DataFrame(a).fillna(0, inplace=False)
    df.columns = NMIdNames
    df.index = indexNames
    df.to_csv(csvName)
    print df
    return df


def linBinValueDataForChord(row, binVariable, noBins):
    '''
    :param row:  a dataframe
    :param binVariable: the variable that you want to bin
    :param noBins: the number of containers
    :return: a mask of the same length of the dataframe that assigns a bin class to each value.
            Also returns the bin declaration.
    :example:
    ntpDataset['debyeLengthBin'] = binValueDataForChord(ntpDataset,'debyeLength',3)
    '''
    # binVariable = 'debyeLength'
    # df = row.sort(binVariable)
    import numpy as np
    ntpMin = min(row[binVariable])
    ntpMax = max(row[binVariable])
    bins = np.linspace(ntpMin, ntpMax, noBins)
    mask = np.digitize(row[binVariable], bins)
    return mask, bins


def logBinValueDataForChord(row, binVariable, noBins):
    '''
    :param row:  a dataframe
    :param binVariable: the variable that you want to bin
    :param noBins: the number of containers
    :return: a mask of the same length of the dataframe that assigns a bin class to each value.
            Also returns the bin declaration.
    :example:
    ntpDataset['debyeLengthBin'] = binValueDataForChord(ntpDataset,'debyeLength',3)
    '''
    # binVariable = 'debyeLength'
    # df = row.sort(binVariable)
    import numpy as np
    ntpMin = min(row[binVariable])
    ntpMax = max(row[binVariable])
    bins = np.logspace(start=np.log10(ntpMin), stop=np.log10(ntpMax), num=noBins, endpoint=True)
    mask = np.digitize(row[binVariable], bins)
    return mask, bins


def pecletNumber(row):
    '''

    :param row: a dataframe with at the following columns:
        ['tempKelvin'] ['PartDiam']['Darcy']['CollecDiam']
    :return:
    '''
    import math

    # Stokes-einstein diffusion correction factor when bounded by solid walls and/or fluid-fluid interfaces
    # Rajagopalan and Tien, 1976
    bolzConstant = float(1.3806504 * 10 ** -23)
    A = float(-3.7188)
    B = float(578.919)
    C = float(-137.546)
    absVisc = float(math.exp(A + (B / (C + float(row['tempKelvin']))))) / 1000
    densWater = float(998.2071 / (1 + 0.000207 * (float(row['tempKelvin']) - 293.15)))
    kin_visc = absVisc / densWater
    stokesEinsteinDiff = float(bolzConstant * float(row['tempKelvin'])) / (
    6 * math.pi * absVisc * float(row['PartDiam'] / 2))

    return float(row['Darcy'] * row['CollecDiam'] / stokesEinsteinDiff)


def plot_corr(df, size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np

    corr = df.corr()
    label = df.corr()
    mask = np.tri(corr.shape[0], k=-1)
    corr = np.ma.array(corr, mask=mask)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    cmap = cm.get_cmap('jet', 10)
    cmap.set_bad('w')

    plt.xticks(range(len(label.columns)), label.columns, rotation=90)
    plt.yticks(range(len(label.columns)), label.columns)
    ax.imshow(corr, interpolation='nearest', cmap=cmap)
    plt.show()


def stratShuffleSplitRFECVRandomForestClassification(nEstimators,
                                                     iterator1,
                                                     minSamplesSplit,
                                                     maxFeatures,
                                                     maxDepth,
                                                     nFolds,
                                                     targetDataMatrix,
                                                     trainingData,
                                                     trainingDataMatrix,
                                                     SEED):
    '''

    :param nEstimators: This is the number of trees in the forest (typically 500-1000 or so)
    :param iterator1: This is the number of model iterations. For a breakdown of model structure, see the wiki
                      (it's clearly marked...somewhere)
    :param minSamplesSplit: this is the minimum number of samples to split. 2 is a bit small...less is typically more.
    :param maxFeatures:
    :param nFolds:
    :param targetDataMatrix:
    :param trainingData:
    :param trainingDataMatrix:
    :param SEED:
    :return:
    '''
    import multiprocessing
    import numpy as np
    multiprocessing.cpu_count()
    # from helperFunctions import *
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    from sklearn import cross_validation
    from sklearn.feature_selection import RFECV
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import StratifiedShuffleSplit

    # rfecv pre-allocation tables, seeding
    X_train = []
    X_holdout = []
    y_train = []
    y_holdout = []
    rfecvGridScoresAll = []
    optimumLengthAll = []
    # feature_names = []
    a = []
    rfc_all_f1 = []
    nameListAll = pd.DataFrame()
    optimumLengthAll = pd.DataFrame()
    classScoreAll = pd.DataFrame()
    classScoreAll2 = pd.DataFrame()
    classScoreAll3 = pd.DataFrame()
    featureImportancesAll = pd.DataFrame()
    rfecvGridScoresAll = pd.DataFrame()


    # Re-definition of the RFC to employ feature importance as a proxy for weighting to employ RFECV.
    class RandomForestClassifierWithCoef(RandomForestClassifier):
        def fit(self, *args, **kwargs):
            super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
            self.coef_ = self.feature_importances_

    ## Re-creation of the RFC object with ranking proxy coefficients
    rfc = RandomForestClassifierWithCoef(n_estimators=nEstimators,
                                         min_samples_split=minSamplesSplit,
                                         bootstrap=True,
                                         n_jobs=-1,
                                         max_features=maxFeatures,
                                         oob_score=True,
                                         max_depth=maxDepth)

    ## Employ Recursive feature elimination with automatic tuning of the number of features selected with CV (RFECV)
    #
    for kk in range(0, iterator1):
        print "iteration no: ", kk + 1
        # Shuffle and split the dataset using a stratified approach to minimize the influence of class imbalance.
        SSS = StratifiedShuffleSplit(targetDataMatrix, n_iter=1, test_size=0.10, random_state=SEED * kk)
        for train_index, test_index in SSS:
            X_train, X_holdout = trainingDataMatrix[train_index], trainingDataMatrix[test_index]
            y_train, y_holdout = targetDataMatrix[train_index], targetDataMatrix[test_index]

        # Call the RFECV function. Additional splitting is done by stratification shuffling and splitting. 5 folds. 5 times,
        # with a random seed controlling the split.

        rfecv = RFECV(estimator=rfc, step=1,
                      cv=StratifiedKFold(y_train, n_folds=nFolds, shuffle=True, random_state=SEED * kk),
                      scoring='accuracy')  # Can  use 'accuracy' or 'f1' f1_weighted, f1_macro, f1_samples

        # First, the recursive feature elimination model is trained. This fits to the optimum model and begins recursion.
        rfecv = rfecv.fit(X_train, y_train)

        # Second, the cross-validation scores are calculated such that grid_scores_[i] corresponds to the CV score
        # of the i-th subset of features. In other words, from all the features to a single feature, the cross validation
        # score is recorded.
        rfecvGridScoresAll = rfecvGridScoresAll.append([rfecv.grid_scores_])

        # Third, the .support_ attribute reports whether the feature remains after RFECV or not. The possible parameters are
        # inspected by their ranking. Low ranking features are removed.
        supPort = rfecv.support_  # True/False values, where true is a parameter of importance identified by recursive alg.
        possParams = rfecv.ranking_
        min_feature_params = rfecv.get_params(deep=True)
        optimumLengthAll = optimumLengthAll.append([rfecv.n_features_])
        featureSetIDs = list(supPort)
        featureSetIDs = list(featureSetIDs)
        # print feature_names
        feature_names = list(trainingData.columns.values)
        namedFeatures = list(trainingData.columns.values)
        namedFeatures = np.array(namedFeatures)

        # Loop over each item in the list of true/false values, if true, pull out the corresponding feature name and store
        # it in the appended namelist. This namelist is rewritten each time, but the information is retained.
        nameList = []  # Initialize a blank array to accept the list of names for features identified as 'True',
        # or important.
        # print featureSetIDs
        # print len(featureSetIDs)
        for i in range(0, len(featureSetIDs)):
            if featureSetIDs[i]:
                nameList.append(feature_names[i])
            else:
                a = 1
                # print("didn't make it")
                # print(feature_names[i])
        nameList = pd.DataFrame(nameList)
        nameListAll = nameListAll.append(nameList)  # append the name list
        nameList = list(nameList)
        nameList = np.array(nameList)

        # Fourth, the training process begins anew, with the objective to trim to the optimum feature and retrain the model
        # without cross validation i.e., test the holdout set. The new training test set size for the holdout validation
        # should be the entire 90% of the training set (X_trimTrainSet). The holdout test set also needs to be
        # trimmed. The same transformation is performed on the holdout set (X_trimHoldoutSet).
        X_trimTrainSet = rfecv.transform(X_train)
        X_trimHoldoutSet = rfecv.transform(X_holdout)


        # Fifth, no recursive feature elimination is needed (it has already been done and the poor features removed).
        # Here the model is trained against the trimmed training set X's and corresponding Y's.
        rfc.fit(X_trimTrainSet, y_train)

        # Holdout test results are generated here.
        preds = rfc.predict(
            X_trimHoldoutSet)  # Predict the class from the holdout dataset. Previous call: rfecv.predict(X_holdout)
        print preds
        print y_holdout
        rfc_all_f1 = metrics.f1_score(y_holdout, preds, average='weighted')  # determine the F1
        rfc_all_f2 = metrics.r2_score(y_holdout, preds)  # determine the R^2 Score
        rfc_all_f3 = metrics.mean_absolute_error(y_holdout,
                                                 preds)  # determine the MAE - Do this because we want to determine sign.

        # append the previous scores for aggregated analysis
        classScoreAll = classScoreAll.append([rfc_all_f1])  # append the previous scores for aggregated analysis.
        classScoreAll2 = classScoreAll2.append([rfc_all_f2])
        classScoreAll3 = classScoreAll3.append([rfc_all_f3])
        refinedFeatureImportances = rfc.feature_importances_  # determine the feature importances for aggregated analysis.
        featureImportancesAll = featureImportancesAll.append([refinedFeatureImportances])


    # Output file creation
    print("List of Important Features Identified by Recursive Selection Method:")
    print(nameListAll)
    nameListAll.to_csv('./outputFiles/class_IFIRS.csv')
    nameListAll.count()

    print("f1 weighted score for all runs:")
    print(classScoreAll)
    classScoreAll.to_csv('./outputFiles/f1_score_all.csv')

    print("R^2 score for all runs:")
    print(classScoreAll2)
    classScoreAll2.to_csv('./outputFiles/class_Rsq_score_all.csv')

    print("MAE score for all runs:")
    print(classScoreAll3)
    classScoreAll3.to_csv('./outputFiles/class_MAE_score_all.csv')

    print("Optimal number of features:")
    print(optimumLengthAll)
    optimumLengthAll.to_csv('./outputFiles/class_optimum_length.csv')

    print("Selected Feature Importances:")
    print(featureImportancesAll)
    featureImportancesAll.to_csv('./outputFiles/class_sel_feature_importances.csv')

    print("mean_squared_error Grid Score for Increasing Features")
    print(rfecvGridScoresAll)
    rfecvGridScoresAll.to_csv('./outputFiles/class_rfecv_grid_scores.csv')


