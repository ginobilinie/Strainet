import os
import h5py
import SimpleITK as sitk
import cv2
import numpy as np
import scipy.ndimage as nd

'''
obtain 0/1 organ maps for the specified organID
Input:
    img: the original image map
    organID: the specified organID
Output:
    the 0/1 feature map for the specified organID
'''
def obtainSingleOrganMap(img, organID):
    i,j,k = np.where(img==organID)
    img1 = np.zeros(img.shape,dtype=int)
    img1[i,j,k] = 1
    return img1


'''
 Return array with completely isolated single cells removed
:param array: Array with completely isolated single cells
:param struct: Structure array for generating unique regions
:return: Array with minimum region size > 1
'''
def filter_isolated_cells(array, struct):

    filtered_array = np.copy(array)
    #id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_regions, num_ids = nd.measurements.label(filtered_array, structure=struct)
    #print 'id_region shape is ',id_regions.shape
    #print 'num_ids is ',num_ids
    #id_regions:unique label for unique features
    #num_features: how many objects are found
    #id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    id_sizes = np.array(nd.measurements.sum(array, id_regions, range(num_ids + 1))) #number of pixels for this region (id)
    #An array of the sums of values of input inside the regions defined by labels with the same shape as index. If 'index' is None or scalar, a scalar is returned.
    #print 'id_sizes shape is ',id_sizes.shape
    #print 'id_sizes is ', id_sizes
    maxV=np.amax(id_sizes) 
    for v in id_sizes:
        if v==maxV:
            continue
        area_mask = (id_sizes == v)
        #print 'area_mask.shape is ', area_mask.shape
        filtered_array[area_mask[id_regions]] = 0
    return filtered_array

'''
denoise Images for each unique intensity, we remove the isolated regions with given struct (kernels)
'''
def denoiseImg_isolation(array, struct):
    uniqueVs=np.unique(array)
    denoised_array=np.zeros(array.shape)
    for v in uniqueVs:
        temp_array=np.zeros(array.shape)
        vMask=(array==v)
        temp_array[vMask]=v
        #print 'vMask shape, ',vMask.shape
        #print 'arrayV shape, ',arrayV.shape
        filtered_array = filter_isolated_cells(temp_array,struct)
        denoised_array[(filtered_array==v)]=v
    return denoised_array 



