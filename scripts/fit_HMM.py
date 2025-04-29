import pandas as pd
import numpy as np
from joblib import Memory

from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker,NiftiMapsMasker,NiftiSpheresMasker,NiftiMasker
from pathlib import Path
from sklearn.cluster import KMeans
from hmmlearn import hmm
from scipy import stats
import timeit
import pickle
import os


TR=1.3

datafolder='path_to_postprocessed_BIDS_folder'
outputfolder=os.getcwd()+'/output/'
cachelocation = 'path_to_cache'

tasktype=['rest','ta1','ta2']


nstate=8
fitHMM=True
inferStates=False
seed=45

excludehead=10
excludetail=-6

sublist=['01', '02', '03', '09', '15', '17', '19', '20','25', '27', '28', '30',
                 '31', '32', '33', '34', '36', '37', '38', '39', '40',
                 '41', '42', '44', '45', '46', '47', '51','52', '53', '54', '55', '56', '57', '58',
                 '63', '64','66', '67', '68','71', '72','74', '77', '78', '79']

if seed!=None:
    randomstate='_seed'+str(seed)
else:
    randomstate=''

yeo_atlas = datasets.fetch_atlas_yeo_2011()
masker = NiftiLabelsMasker(labels_img=yeo_atlas.thick_17, standardize="zscore_sample",
                           memory=cachelocation,memory_level=1,t_r=TR,
                           high_pass=0.01,detrend=True)
masker.fit()

def makeNiftyFileName(root,subid,task):
    runexceptions=[('19','rest'),('33','ta1')] #these data are in run-02
    runindex='01'
    if (subid,task) in runexceptions:
        runindex='02'
    return root+'sub-'+subid+'/func/sub-'+subid+'_task-'+task+'_run-'+runindex+'_space-MNI152NLin6Asym_res-2_desc-denoisedSmoothed_bold.nii.gz'


dataBySubject={}
datainfo=[]
#zscore the data for each subid then concatenate
print('compute masked timecourses')
for subid in sublist:
    print(subid)
    tmpData=[]
    for task in tasktype:
        filename = makeNiftyFileName(datafolder,subid,task)
        if Path(filename).is_file():
            tmpData.append(masker.transform(filename)[excludehead:excludetail])
            datainfo.append([subid, task, len(tmpData[-1]),filename])
        else:
            print('File not found: '+filename)
    tmpData = np.vstack(tmpData)
    dataBySubject[subid] = tmpData
pickle.dump(dataBySubject,
            open(outputfolder + ''.join(tasktype)+'_Yeo17_hmmDataBySubject.pkl', 'wb'))
resortedDF=pd.DataFrame(datainfo,columns=['subid','task','n','file'])
resortedDF.to_csv(outputfolder+''.join(tasktype) + '_datainfo.csv',index=None)

if fitHMM:
    nlist = []
    hmmData = []
    for subid in resortedDF['subid'].unique():
        if len(hmmData) > 0:
            hmmData = np.vstack((hmmData, dataBySubject[subid]))
        else:
            hmmData = dataBySubject[subid]
        nlist = nlist + list(resortedDF.loc[resortedDF['subid']==subid,'n'].values)
    del dataBySubject
    # # # Initialize with k-means clustering
    start = timeit.default_timer()
    print('Doing k-means clustering, # of states: '+str(nstate))
    kmeans_train = KMeans(n_clusters=nstate, init='k-means++', n_init=50, max_iter=200, tol=0.001,random_state=seed).fit(hmmData)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    # # # HMM fit
    print(np.shape(hmmData))
    global_cov = np.cov(hmmData.T)
    print(np.shape(global_cov))
    start = timeit.default_timer()
    print('Fitting HMM')
    if seed!=None:
        hmmmodel = hmm.GaussianHMM(n_components=nstate, covariance_type='full',verbose=True,init_params='t',
                            n_iter=2000,tol=0,random_state=seed)
        hmmmodel.covars_ = np.array([global_cov] * nstate)
        hmmmodel.means_ = kmeans_train.cluster_centers_
        hmmmodel.startprob_ = [1 / nstate] * nstate

    else:
        hmmmodel = hmm.GaussianHMM(n_components=nstate, covariance_type='full',verbose=True,init_params='t',
                            n_iter=2000,tol=0)
        hmmmodel.covars_=np.array([global_cov]*nstate)
        hmmmodel.means=kmeans_train.cluster_centers_
        hmmmodel.startprob_ = [1 / nstate] * nstate
    hmmmodel.fit(hmmData,lengths=nlist)
    print('HMM fitting finished at ' + str(hmmmodel.monitor_.iter) + ' iterations')
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    print('Saving Data')
    pickle.dump(hmmmodel,
                open(outputfolder + ''.join(tasktype)+'_Yeo17_hmm'+str(nstate)+randomstate+'.pkl', 'wb'))
    log_prob,state_seq=hmmmodel.decode(hmmData,lengths=nlist)
    proba=hmmmodel.predict_proba(hmmData,lengths=nlist)

    sublist = []
    tasklist = []
    for i,rows in resortedDF.iterrows():
        sublist = sublist + [rows['subid']] * rows['n']
        tasklist = tasklist + [rows['task']] * rows['n']
    df = pd.concat([pd.DataFrame({'subid': sublist, 'task': tasklist, 'state': state_seq}),
                    pd.DataFrame(np.round(proba, 3), columns=range(nstate))], axis=1)
    df.to_csv(outputfolder + ''.join(tasktype) +'_Yeo17_hmm'+str(nstate)+randomstate+'.csv', index=None)

