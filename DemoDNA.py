from naive_feature_selection import *
from numpy import genfromtxt
from sklearn.svm import LinearSVC
from sklearn import metrics
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
import os

#%% data downloader


import gzip
import wget
import os
import tarfile
import zipfile
import shutil


def extract_file(path, to_directory='./data/'):
    try:
        os.stat(extract_path)
    except:
        os.mkdir(extract_path)
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else: 
        raise (ValueError, "Could not extract `%s` as no appropriate extractor is found" % path)
    #
    os.chdir(to_directory)
    #
    try:
        file = opener(path, mode)
        try: file.extractall()
        finally: file.close()
    finally:
        os.chdir(cwd)

def download_file(url, out_directory = './data/'):
    # Download archive
    try:
        file = wget(url)
        cwd = os.getcwd()
        # Read the file inside the .gz archive located at url
        extract_all(cwd+'/'+file, out_directory)
        return 0
    except Exception as e:
        print(e)
        return 1


    
#%% Test on UCI gene expression cancer RNA-Seq Data Set 
print("Importing RNA-Seq data...")

cwd = os.getcwd()
data_path = cwd+'/data/TCGA-PANCAN-HiSeq-801x20531/data.csv'
label_path = cwd+'/data/TCGA-PANCAN-HiSeq-801x20531/labels.csv'
if not os.path.exists(data_path):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz'
    download_file(url)


X_data = genfromtxt(data_path, delimiter=',',skip_header=1)
X_data=X_data[:,1:]
X_labels=read_csv(data_path,nrows=1)
X_labels=X_labels.columns[1:]
labels=read_csv(label_path,header=0)
y_data=labels['Class']=='BRCA' # Check for BRCA labels
y_data=1.0*y_data.to_numpy()

# Split training / test set
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

# Test Naive Feature Selection, followed by l2 SVM
k=100 # Target number of features
nfs = NaiveFeatureSelection(k=k, alpha=1e-4)

X_train_nfs=nfs.fit_transform(X_train,y_train)
clfsv = LinearSVC(random_state=0, tol=1e-5)
clfsv.fit(X_train_nfs, y_train)
X_test_nfs = nfs.transform(X_test)
y_pred_NFS = clfsv.predict(X_test_nfs)
score_nfs = metrics.accuracy_score(y_test==1, y_pred_NFS)
print("NFS accuracy:\t%0.3f" % score_nfs)

print('Positive genes:')
print([X_labels[i] for i in range(100) if clfsv.coef_[0][i]>=0])


#%% Plot sparsity / likelihood tradeoff
import matplotlib.pyplot as plt
kmax=5000
xvals=range(5,kmax,100)
resv=np.zeros(len(xvals))
for i in range(0,len(xvals)):
    k=xvals[i]
    nfs = NaiveFeatureSelection(k=k, alpha=1e-4)
    nfs.fit(X_train,y_train)
    resv[i]=nfs.res_nfs['objv']

plt.style.use('seaborn-white')
plt.plot(xvals, resv,'-b',linewidth=4)
plt.xlabel('Sparsity k',fontsize=14)
plt.ylabel('Likelihood',fontsize=14)
plt.savefig('tradeoff.pdf')
plt.show()


#%% Plot curve
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.io as pio

trace = go.Scatter(
    x=list(xvals),
    y=resv,
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4)
)
data = [trace]
layout = go.Layout(
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='<b>Sparsity k',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#5f5f5f',
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='<b>Likelihood<br><br>',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#5f5f5f'
            )
        )
    )
)
fig = go.Figure(data=data, layout=layout)
pio.write_image(fig, 'tradeoff.pdf')
#py.iplot(data, layout=layout, filename='basic-line')
#py.plot(data, filename = 'basic-line', auto_open=True)

#%%
