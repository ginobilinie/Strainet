'''
Target: Transpose (permute) the order of dimensions of a image  
Created on Jan, 22th 2016
Author: Dong Nie 

reference from: http://simpleitk-prototype.readthedocs.io/en/latest/user_guide/plot_image.html
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio

path='/shenlab/lab_stor5/dongnie/challengeData/'
def main():
    ids=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    ids = range(0,30)
    for id in ids:
#         datafn = os.path.join(path,'Case%02d.mhd'%id)
#         outdatafn = os.path.join(path,'Case%02d.nii.gz'%id)
#         
#         dataOrg = sitk.ReadImage(datafn)
#         dataMat = sitk.GetArrayFromImage(dataOrg)
#         #gtMat=np.transpose(gtMat,(2,1,0))
#         dataVol = sitk.GetImageFromArray(dataMat)
#         sitk.WriteImage(dataVol,outdatafn)
        
        
        datafn = os.path.join(path,'TestData_mhd/Case%02d.mhd'%id)
        dataOrg = sitk.ReadImage(datafn)
        spacing = dataOrg.GetSpacing()
        origin = dataOrg.GetOrigin()
        direction = dataOrg.GetDirection()
        dataMat = sitk.GetArrayFromImage(dataOrg)
        
        gtfn = os.path.join(path,'submission_niigz/preTestCha_model0110_iter14w_sub%02d.nii.gz'%id)
        gtOrg = sitk.ReadImage(gtfn)
        gtMat = sitk.GetArrayFromImage(gtOrg)
        #gtMat=np.transpose(gtMat,(2,1,0))
        
        outgtfn = os.path.join(path,'submission_mhd/Case%02d_segmentation.mhd'%id)
        gtVol = sitk.GetImageFromArray(gtMat)
        gtVol.SetSpacing(spacing)
        gtVol.SetOrigin(origin)
        gtVol.SetDirection(direction)
        sitk.WriteImage(gtVol,outgtfn)
#         
#         prefn='preSub%d_as32_v12.nii'%id
#         preOrg=sitk.ReadImage(prefn)
#         preMat=sitk.GetArrayFromImage(preOrg)
#         preMat=np.transpose(preMat,(2,1,0))
#         preVol=sitk.GetImageFromArra(preMat)
#         sitk.WriteImage(preVol,prefn)
 
        
if __name__ == '__main__':     
    main()
