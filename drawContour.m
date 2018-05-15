%%% draw the spatial factor's impact
function drawContour()
    Slice=53;
    lw=0.01;
    path='./';
    imgfn='img1.mhd';
    gtfn='gt1.nii.gz';
    sruacmfn='res_v5_acm2_preSub1.nii.gz';
    srufn='res_v5_preSub1.nii.gz';
    ssaefn='res_v5_acm_preSub1.nii.gz';
    advfn='res_v5_adversarial_preSub1.nii.gz';
    
    malffn = 'preSub_wdice_wce_1012_104000.nii.gz';
    unetfn = 'denseCrf3dSegmMap_pelvic.nii.gz';
%     vnetfn = 'preSub_wce_wdice_adImpo_viewExp_1106_sub1.nii.gz';
    vnetfn = 'preSub1_clean_166000.nii.gz';
    asdnetfn = 'preSub1_74000_clean_mt.nii.gz';
    resunetfn = 'res_v5_preSub1.nii.gz';
    info = mha_read_header([path,imgfn]);
    mrimg = single(mha_read_volume(info));
    [gt gthead]=rest_ReadNiftiImage([path,gtfn]);
    [sruacm gthead]=rest_ReadNiftiImage([path,sruacmfn]);
    [sru gthead]=rest_ReadNiftiImage([path,srufn]);
    [ssae gthead]=rest_ReadNiftiImage([path,ssaefn]);
    [adv gthead]=rest_ReadNiftiImage([path,advfn]);
    [malf gthead]=rest_ReadNiftiImage([path,malffn]);
    [unet gthead]=rest_ReadNiftiImage([path,unetfn]);
    [vnet gthead]=rest_ReadNiftiImage([path,vnetfn]);
    [asdnet gthead]=rest_ReadNiftiImage([path,asdnetfn]);
    [resunet gthead]=rest_ReadNiftiImage([path,resunetfn]);
    imagesc(mrimg(:,:,Slice),[0,800]);
    axis off;
    axis image;
    colormap(gray);
    %% for ground truth
    hold on;
    [c_gt,h_gt]=drawContour4Organ(gt(:,:,Slice),1);
    orange = [0.824 0.706 0.549];
    h_gt.LineWidth = lw;
    h_gt.LineColor=orange;
    hold on;
    [c_gt,h_gt]=drawContour4Organ(gt(:,:,Slice),2);
    silver = [1.0 0.98 0.98];
    h_gt.LineWidth =lw;
    h_gt.LineColor=silver;
    hold on;
    [c_gt,h_gt]=drawContour4Organ(gt(:,:,Slice),3);
    silver = [0.737 0.561 0.561];
    h_gt.LineWidth = lw;
    h_gt.LineColor=silver;
 
    %% for others
    other=adv;
    hold on;
    [c_other,h_other]=drawContour4Organ(other(:,:,Slice),1);
    yellow = [1 1 0];
    h_other.LineWidth = lw;
    h_other.LineColor=yellow;
    hold on;
    [c_other,h_other]=drawContour4Organ(other(:,:,Slice),2);
    red = [1.0 0 0];
    h_other.LineWidth = lw;
    h_other.LineColor=red;
    hold on;
    [c_other,h_other]=drawContour4Organ(other(:,:,Slice),3);
    cyan = [0 1 1];
    h_other.LineWidth = lw;
    h_other.LineColor=cyan;
return

function [c,h]=drawContour4Organ(mat,organID)
    tmp=zeros(size(mat));
    tmp(find(mat==organID))=organID;
    [c,h]=contour(tmp);
return