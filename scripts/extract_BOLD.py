import pandas as pd
import numpy as np
from pathlib import Path
import os
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker,NiftiMasker
from nilearn.image import math_img


TR=1.3

cachelocation = 'path_to_cache'
outputfolderroot='path_to_output' #with gsr

extractSubcortical=False
extract400=False
extract17=False
extractHippo=True

resortedDF = pd.read_csv(os.getcwd()+'/output/' +'nifty_filelist_for_conditions', index_col=None,dtype=str)
resortedDF['n']=resortedDF['n'].astype(int)

if extract17:
    outputfolder = outputfolderroot + 'Yeo17/'
    yeo_atlas = datasets.fetch_atlas_yeo_2011()
    brain_masker = NiftiLabelsMasker(labels_img=yeo_atlas.thick_17, standardize="zscore_sample",
                                     memory_level=1, memory=cachelocation,t_r=TR,low_pass=0.1, high_pass=0.01,
                                     detrend=True)
    brain_masker.fit()
    for sub,x in resortedDF.groupby('subid'):
        subject_data=[]
        for task,g in x.groupby('task'):
            subject_file=resortedDF[(resortedDF['subid']==sub)&(resortedDF['task']==task)]['file'].values[0]
            pd.DataFrame(brain_masker.transform(subject_file),columns=range(17)).to_csv(
                outputfolder+sub+'_'+task+'.csv',index=False)


if extract400:
    outputfolder = outputfolderroot + 'schaefer400/'
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    atlas_filename = atlas.maps
    brain_masker = NiftiLabelsMasker(labels_img=atlas_filename, smoothing_fwhm=6, detrend=True,
                                     low_pass=0.1, high_pass=0.01, t_r=TR, standardize="zscore_sample",
                                     memory_level=1, memory=cachelocation)

    brain_masker.fit()
    for sub,x in resortedDF.groupby('subid'):
        subject_data=[]
        for task,g in x.groupby('task'):
            subject_file=resortedDF[(resortedDF['subid']==sub)&(resortedDF['task']==task)]['file'].values[0]
            pd.DataFrame(brain_masker.transform(subject_file),columns=range(400)).to_csv(
                outputfolder+sub+'_'+task+'.csv',index=False)




if extractSubcortical:
    outputfolder=outputfolderroot+'subcortical/'
    atlas_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-1mm')

    masker = NiftiLabelsMasker(labels_img=atlas_ho.maps, standardize="zscore_sample",low_pass=0.1, high_pass=0.01,
                               memory=cachelocation, memory_level=1, detrend=True, t_r=TR)
    labels = atlas_ho.labels[1:]
    masker.fit()
    for sub,x in resortedDF.groupby('subid'):
        subject_data=[]
        for task,g in x.groupby('task'):
            subject_file=resortedDF[(resortedDF['subid']==sub)&(resortedDF['task']==task)]['file'].values[0]
            pd.DataFrame(masker.transform(subject_file),columns=labels).to_csv(
                outputfolder+sub+'_'+task+'.csv',index=False)

if extractHippo:
    outputfolder=outputfolderroot+'hippo/'
    atlas_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')

    left_hippo = math_img("img==9", img=atlas_ho.maps)
    right_hippo = math_img("img==19", img=atlas_ho.maps)

    left_masker = NiftiMasker(standardize="zscore_sample", low_pass=0.1, high_pass=0.01,
                              memory=cachelocation, memory_level=1, detrend=True, t_r=TR, mask_img=left_hippo)
    left_masker.fit()
    right_masker = NiftiMasker(standardize="zscore_sample", low_pass=0.1, high_pass=0.01,
                               memory=cachelocation, memory_level=1, detrend=True, t_r=TR, mask_img=right_hippo)
    right_masker.fit()
    voxel_indices_left = np.where(left_hippo.get_fdata() > 0)
    voxel_indices_right = np.where(right_hippo.get_fdata() > 0)
    mni_coords_left = np.vstack(
        [left_hippo.affine[:3, :3] @ np.array([i, j, k]) + left_hippo.affine[:3, 3] for i, j, k in
         zip(*voxel_indices_left)])
    mni_coords_right = np.vstack(
        [right_hippo.affine[:3, :3] @ np.array([i, j, k]) + right_hippo.affine[:3, 3] for i, j, k in
         zip(*voxel_indices_right)])
    for sub,x in resortedDF.groupby('subid'):
        subject_data=[]
        for task,g in x.groupby('task'):
            subject_file=resortedDF[(resortedDF['subid']==sub)&(resortedDF['task']==task)]['file'].values[0]
            voxel_timeseries_left = left_masker.transform(subject_file)
            voxel_timeseries_right = right_masker.transform(subject_file)
            tmp = pd.DataFrame({'x': mni_coords_left[:, 0], 'y': mni_coords_left[:, 1], 'z': mni_coords_left[:, 2],
                                'signal': np.mean(voxel_timeseries_left, axis=0)})
            tmp=pd.concat([tmp,pd.DataFrame(voxel_timeseries_left.T,columns=range(len(voxel_timeseries_left)))],axis=1)

            tmp = tmp.groupby('y').mean()[range(len(voxel_timeseries_left))]

            leftDF=pd.DataFrame(tmp.values.T,columns=['left_'+str(x) for x in tmp.index])

            tmp = pd.DataFrame({'x': mni_coords_right[:, 0], 'y': mni_coords_right[:, 1], 'z': mni_coords_right[:, 2],
                                'signal': np.mean(voxel_timeseries_right, axis=0)})
            tmp = pd.concat([tmp, pd.DataFrame(voxel_timeseries_right.T, columns=range(len(voxel_timeseries_right)))],
                            axis=1)

            tmp = tmp.groupby('y').mean()[range(len(voxel_timeseries_right))]
            rightDF = pd.DataFrame(tmp.values.T, columns=['right_' + str(x) for x in tmp.index])

            pd.concat([leftDF, rightDF],axis=1).to_csv(outputfolder+sub+'_'+task+'.csv',index=False)

