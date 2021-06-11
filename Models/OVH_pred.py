# Importing checkModels
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Importing Models
import numpy as np
import scipy.interpolate
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import sklearn
from sklearn.metrics import r2_score
import string
import formatter
from numpy import linalg
from random import shuffle
import random

# PCA
from time import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
#from sklearn.cross_validation import train_test_split
#from sklearn.cross_validation import KFold

# RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


def readFile(filename):
    df = pd.read_csv(filename, delimiter='\t')
    return df


def getAllUPIs_2(dfDVH, dfOVH, source, source2, structure, groupnumber):
    records_DVH = (dfDVH.loc[dfDVH[u'structure'] == structure])
    records_OVH = dfOVH.loc[dfOVH[u'Source'].isin([source, source2]) & dfOVH[u'Target'].isin([structure])]
    records_clasDF = clasDF.loc[clasDF[u'group'] == groupnumber]  # .isin([groupnumber])]
    upiSets = [set(records_DVH[u'upi_2'].values), set(records_OVH[u'upi_2'].values),
               set(records_clasDF[u'upi_2'].values)]
    upis = list(set.intersection(*upiSets))
    return upis


def DVH_comparison_measure(Inputdata, DVH_prediction, Volumenormalisation):
    Volumenorm = Volumenormalisation

    numpat = Inputdata.shape[0]
    numbins = Inputdata.shape[1]
    gammas = np.zeros([numpat, numbins])
    gamavg = np.zeros(numpat)

    for pat in np.arange(numpat):

        Vparam = np.zeros([len(dose_grid), len(dose_grid)])
        Dparam = np.zeros([len(dose_grid), len(dose_grid)])

        for i in np.arange(len(dose_grid)):
            for j in np.arange(len(dose_grid)):
                Vparam[i, j] = (abs(DVH_prediction[pat, :][i] - Inputdata[pat][j])) ** 2  # /(Volumenorm)**2
                Dparam[i, j] = (abs(dose_grid[i] - dose_grid[j])) ** 2  # /(0.03*max(dose_grid))**2

        temp = np.zeros([len(dose_grid), len(dose_grid)])

        for i in np.arange(len(dose_grid)):
            for j in np.arange(len(dose_grid)):
                temp[i, j] = math.sqrt(Vparam[i, j] + Dparam[i, j])

        for i in np.arange(len(dose_grid)):
            temp[i, :] = min(temp[i, :])

        for i in np.arange((len(dose_grid) - 1)):
            temp = np.delete(temp, 0, 1)

        gammas[pat, :] = np.ravel(temp)
        gamavg[pat] = sum(gammas[pat, :]) / numpat

    return gamavg, gammas


def PCA_Function(Inputdata, numcomp):
    pca = PCA(numcomp, svd_solver='auto', whiten=False).fit(Inputdata)
    eigenvect_norm = pca.components_
    eigenval_norm = pca.transform(Inputdata)
    what = pca.explained_variance_ratio_
    DVH_mean = Inputdata.mean(axis=0)
    EVR = np.cumsum(what * 100)

    if sum(eigenvect_norm[0] / len(eigenvect_norm[0])) < 0:
        for j in np.arange(numcomp):
            eigenvect_norm[j] = -1 * eigenvect_norm[j]
            eigenval_norm[j, :] = -1 * eigenval_norm[j]

    return EVR, eigenvect_norm, DVH_mean, eigenval_norm


def Construct_DVH(Eigenvalues_query, Eigenvectors, DVH_mean):
    numpat = Eigenvalues_query.shape[0]
    numbins = Eigenvectors.shape[1]

    DVH_pred = np.zeros((numpat, numbins))

    for pat in np.arange(numpat):
        DVH_pred[pat, :] = DVH_mean + Eigenvalues_query[pat, 0] * Eigenvectors[0] + Eigenvalues_query[pat, 1] * \
                           Eigenvectors[1]

    return DVH_pred


def Dose_weight_function():
    n = [0.5, 0.8, 1, 2, 3, 4, 5]
    bins = np.arange(0, 80, 2)
    dmax = max(bins);

    wf = np.zeros([len(n), bins.shape[0]]);

    for i in np.arange(len(n)):
        for dose in np.arange(bins.shape[0]):
            wf[i, dose] = (bins[dose] / dmax) ** (n[i]);

    return wf


def Dissimilarity(TrueDVH, DVH_prediction, Dissimilarity_measure):
    numpat = TrueDVH.shape[0]
    numbins = TrueDVH.shape[1]
    discre = np.zeros((numpat, numbins))
    totdiscr = np.zeros(numpat)

    if Dissimilarity_measure == 1:
        for pat in np.arange(numpat):
            for i in np.arange(numbins):
                discre[pat, i] = abs(
                    TrueDVH[pat, i] - DVH_prediction[pat, i])  # math.sqrt(abs(TrueDVH[pat,i] - DVH_prediction[pat,i]))
                totdiscr[pat] = sum(discre[pat, :])
        Dissimilarity_measure = str('Volumetric difference')
    elif Dissimilarity_measure == 2:
        for pat in np.arange(numpat):
            for i in np.arange(numbins):
                discre[pat, i] = math.sqrt(abs(TrueDVH[pat, i] ** 2 - DVH_prediction[pat, i] ** 2))
                #   discre[pat,i] = discre[pat,i] * Dose_weight_function()[5,i]**2 # n = 3, this is important
                totdiscr[pat] = sum(discre[pat, :])
        Dissimilarity_measure = str('Squared volumetric difference')
    elif Dissimilarity_measure == 3:
        for pat in np.arange(numpat):
            for i in np.arange(numbins):
                discre[pat, i] = math.sqrt(abs(TrueDVH[pat, i] ** 2 - DVH_prediction[pat, i] ** 2))
                discre[pat, i] = discre[pat, i] * Dose_weight_function()[5, i] ** 2  # n = 3, this is important
                totdiscr[pat] = sum(discre[pat, :])
        Dissimilarity_measure = str('Squared weighted volumetric difference')
    elif Dissimilarity_measure == 4:
        totdiscr = DVH_comparison_measure(TrueDVH, DVH_prediction, 0.9)
        discre = totdiscr[1]
        totdiscr = totdiscr[0]
        Dissimilarity_measure = str('Minimal distance between graphs')

    return totdiscr, discre, Dissimilarity_measure


def calceud(dosegrid, inputdvh, n):
    product = np.zeros(len(dosegrid))

    for k in np.arange(len(dosegrid)):
        product[k] = dosegrid[k] ** n * np.append((-1) * np.diff(inputdvh), 0)[k] / 100
    EUD = sum(product) ** (1 / n)
    return EUD


def getAllUPIs_2(dfDVH, dfOVH, source, structure, groupnumber):
    records_DVH = (dfDVH.loc[dfDVH[u'struct'] == structure])
    records_OVH = dfOVH.loc[dfOVH[u'source'].isin([source]) & dfOVH[u'target'].isin([structure])]
    #     records_clasDF = clasDF.loc[clasDF[u'group'] == groupnumber] #.isin([groupnumber])]
    patSets = [set(records_DVH[u'patID'].values), set(records_OVH[u'patID'].values)]
    pats = list(set.intersection(*patSets))
    return pats


def tmp_DVH_OVH(patlist, structure, source, dfDVH, dfOVH):
    tmp_DVH = dfDVH[(dfDVH['patID'].isin(patlist)) & (dfDVH['struct'] == structure)]
    tmp_DVH = tmp_DVH[
        ['patID', 'struct', '0', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000', '1100', '1200',
         '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300', '2400', '2500', '2600',
         '2700', '2800', '2900', '3000', '3100', '3200', '3300', '3400', '3500', '3600', '3700', '3800', '3900', '4000',
         '4100', '4200', '4300', '4400', '4500', '4600', '4700', '4800', '4900', '5000', '5100', '5200', '5300', '5400',
         '5500', '5600', '5700', '5800', '5900', '6000']]
    tmp_OVH = dfOVH[(dfOVH['patID'].isin(patlist)) & dfOVH['target'].isin([structure]) & dfOVH['source'].isin([source])]
    Output_DVH = np.array(tmp_DVH[[x for x in tmp_DVH.columns[2:]]])
    Output_OVH = np.array(tmp_OVH[[x for x in tmp_OVH.columns[5:]]])

    return Output_DVH, Output_OVH, tmp_OVH, tmp_DVH


def OVHpred(patID):
    """Script that used the earlier produced OVH prediction function within the other
    parts of this script and outputs the prediction.

    :params patID: patient ID to run OVH model on
    :return DVH_pred: DVH prediction values
    :return dose_grid: standard dose values for the DVH points
    """
    dfOVH = pd.read_csv(r"C:\Users\t.meerbothe\Desktop\Model Karen/ovh_array.txt")

    dfOVH = dfOVH.rename(columns={'patientID':'patID'}).drop_duplicates(subset = ['patID','target'])

    dvhFile=r'C:\Users\t.meerbothe\Desktop\Model Karen/DVHs_uit_dicom.csv'

    dfDVH = pd.read_csv(dvhFile)

    dfDVH = dfDVH.drop_duplicates(subset=['patID','struct'])
    dfOVH.head()

    structure = 'RECTUM'
    structure2 = 'ANAL_SPH'
    source = 'PTVpros+vs'

    patlist = list(dfDVH.patID)
    with open(r'C:\Users\t.meerbothe\Desktop\WS0102\train_pat.txt') as f:
        trainpat = [line.rstrip() for line in f]
    trainpat = np.delete(trainpat, 28)
    trainpat = np.delete(trainpat, 11)
    newpatlist = np.zeros(len(patlist))
    for i in range(len(patlist)):
        newpatlist[i] = patlist[i] in trainpat
    patlist = np.array(patlist)
    patlist = patlist[newpatlist > 0]

    patlist = list(patlist)

    Output_DVHOVH = tmp_DVH_OVH(patlist, structure, source, dfDVH, dfOVH)#.Output_DVH
    Output_DVH = Output_DVHOVH[0]
    Output_OVH = Output_DVHOVH[1]

    dose_grid = np.arange(0,6100,100)# dose_grid[np.arange(40)]

    OVH_grid = (np.arange(Output_OVH.shape[1])-200)

    DVH_PCA = PCA_Function(Output_DVH,10)
    DVH_eigenvect = DVH_PCA[1]
    DVH_mean = DVH_PCA[2]

    d95bin = 57#math.floor(52.5/1.5) #[math.floor(70/2), math.floor(0.95*70/2)]
    aEUD = 1
    V0avg = calceud(dose_grid,DVH_mean,aEUD)
    V095 = DVH_mean[d95bin]

    V1avg = calceud(dose_grid,DVH_eigenvect[0],aEUD)
    V195 = DVH_eigenvect[0,d95bin] # V1-95%

    V2avg = calceud(dose_grid,DVH_eigenvect[1],aEUD)
    V295 = DVH_eigenvect[1,d95bin]

    a_norm = [[V195, V295], [V1avg, V2avg]]

    patlist = list(dfDVH.patID)
    with open(r'C:\Users\t.meerbothe\Desktop\WS0102\test_pat.txt') as f:
        trainpat = [line.rstrip() for line in f]
    newpatlist = np.zeros(len(patlist))
    for i in range(len(patlist)):
        newpatlist[i] = patlist[i] in trainpat
    patlist = np.array(patlist)
    patlist = patlist[newpatlist > 0]

    tup = np.where(patlist == patID)[0]
    patNr = int(np.floor(tup[0] / 2))
    patlist = patlist[patNr]
    patlist = list([patlist])

    Output_DVHOVH = tmp_DVH_OVH(patlist, structure, source, dfDVH, dfOVH)
    Output_OVH = Output_DVHOVH[1]

    DVH_pred = np.zeros((Output_OVH.shape[0], len(dose_grid)))

    DVH_pred_95 = np.zeros(Output_OVH.shape[0])
    DVH_pred_avg = np.zeros(Output_OVH.shape[0])

    for i in np.arange(Output_OVH.shape[0]):
        OVH_95 = (1 - (Output_OVH[i][np.where(OVH_grid == 0)[0][0]])) * 100
        OVH_avg = (1 - (Output_OVH[i][np.where(OVH_grid == 10)[0][0]])) * 100.0
        # DVH_pred_95[i] = OVH_95 * m1 + b1
        # DVH_pred_avg[i] = OVH_avg * m2 + b2
        DVH_pred_95[i] = OVH_95 * -0.838856444003 + 86.153240649767 # Rectum
        DVH_pred_avg[i] = OVH_avg * -51.196199230256 + 5592.288126157915 # Rectum
        # DVH_pred_95[i] = OVH_95 * -1.2210938722975888 + 122.09158099551941 # Anal sphincter
        # DVH_pred_avg[i] = OVH_avg * -40.547775986886606 + 4440.099325158458 # Anal sphincter

        a = [DVH_pred_95[i] - V095, DVH_pred_avg[i] - V0avg]

        DVH_pred[i, :] = DVH_mean + np.linalg.solve(a_norm, a)[0] * DVH_eigenvect[0] + np.linalg.solve(a_norm, a)[1] * \
                         DVH_eigenvect[1]

    return DVH_pred[0,:], dose_grid