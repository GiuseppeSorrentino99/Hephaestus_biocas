# /******************************************
# *MIT License
# *
# *Copyright (c) [2021] [Eleonora D'Arnese, Emanuele Del Sozzo, Davide Conficconi,  Marco Domenico Santambrogio]
# *
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# *of this software and associated documentation files (the "Software"), to deal
# *in the Software without restriction, including without limitation the rights
# *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# *copies of the Software, and to permit persons to whom the Software is
# *furnished to do so, subject to the following conditions:
# *
# *The above copyright notice and this permission notice shall be included in all
# *copies or substantial portions of the Software.
# *
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# *SOFTWARE.
# ******************************************/

import os
import pydicom
import cv2
import numpy as np
import math
import glob
import re
import time
import pandas as pd
from torch.multiprocessing import Pool, Process, set_start_method
import struct
import statistics
import argparse
import kornia
import torch
import gc

m_Gaussfaze=1
m_Gausssave=np.zeros((1,8*128))
m_GScale=1.0/30000000.0

compute_metric = None
precompute_metric = None
device = "cuda"

#torch.cuda.empty_cache()
ref_vals = None
move_data = None
torch.cuda.empty_cache()

def no_transfer(input_data):
    return input_data

def to_cuda(input_data):
    return input_data.cuda(non_blocking=True)

def my_squared_hist2d_t(sample, bins, smin, smax):
    D, N = sample.shape
    edges = torch.linspace(smin, smax, bins + 1, device=device)
    nbin = edges.shape[0] + 1
    
    # Compute the bin number each sample falls into.
    Ncount = D*[None]
    for i in range(D):
        Ncount[i] = torch.searchsorted(edges, sample[i, :], right=True)
    
    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[i, :] == edges[-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1
    
    # Compute the sample indices in the flattened histogram matrix.
    xy = Ncount[0]*nbin+Ncount[1]
           

    # Compute the number of repetitions in xy and assign it to the
    hist = torch.bincount(xy, None, minlength=nbin*nbin)
    
    # Shape into a proper matrix
    hist = hist.reshape((nbin, nbin))

    hist = hist.float()
    
    # Remove outliers (indices 0 and -1 for each dimension).
    hist = hist[1:-1,1:-1]
    
    return hist

def precompute_mutual_information(Ref_uint8):
    
    Ref_uint8_ravel = Ref_uint8.ravel().double()
    href = torch.histc(Ref_uint8_ravel, bins=256)
    href /= Ref_uint8_ravel.numel()
    href=href[href>0.000000000000001]
    eref=(torch.sum(href*(torch.log2(href))))*-1
    
    return eref


def mutual_information(Ref_uint8_ravel, Flt_uint8_ravel, eref):
    
    if(device == "cuda"):
        ref_vals = torch.ones(Ref_uint8_ravel.numel(), dtype=torch.int, device=device)
        idx_joint = torch.stack((Ref_uint8_ravel, Flt_uint8_ravel)).long()
        j_h_init = torch.sparse.IntTensor(idx_joint, ref_vals, torch.Size([hist_dim, hist_dim])).to_dense()/Ref_uint8_ravel.numel()
    else:
        idx_joint = torch.stack((Ref_uint8_ravel, Flt_uint8_ravel))
        j_h_init = my_squared_hist2d_t(idx_joint, hist_dim, 0, 255)/Ref_uint8_ravel.numel()
    j_h = j_h_init[j_h_init>0.000000000000001]
    entropy=(torch.sum(j_h*(torch.log2(j_h))))*-1
    
    hflt=torch.sum(j_h_init,axis=0) 
    hflt=hflt[hflt>0.000000000000001]
    eflt=(torch.sum(hflt*(torch.log2(hflt))))*-1
    
    mutualinfo=eref+eflt-entropy
    
    return(mutualinfo)

# def precompute_cross_correlation(Ref_uint8_ravel):

#     return torch.sum(Ref_uint8_ravel * Ref_uint8_ravel)

    
# def cross_correlation(Ref_uint8_ravel, Flt_uint8_ravel, cc_ref):
    
#     cc_ref_flt = torch.sum(Ref_uint8_ravel * Flt_uint8_ravel)
#     cc_flt = torch.sum(Flt_uint8_ravel * Flt_uint8_ravel)
#     return - cc_ref_flt/torch.sqrt(cc_ref*cc_flt)

# def precompute_mean_squared_error(Ref_uint8_ravel):
#     pass

# def mean_squared_error(Ref_uint8_ravel, Flt_uint8_ravel, mse_ref):
    
#     return torch.sum((Ref_uint8_ravel - Flt_uint8_ravel)**2)

# def to_matrix_blocked(vector_params):
#     mat_params=torch.empty((2,3))
#     mat_params[0][2]=vector_params[0]
#     mat_params[1][2]=vector_params[1]    
#     if vector_params[2] > 1 or vector_params[2] < -1:
#         mat_params[0][0]=1 #cos_teta
#         mat_params[1][1]=1 #cos_teta
#         mat_params[0][1]=0
#         mat_params[1][0]=0
#     else:
#         mat_params[0][0]=vector_params[2] #cos_teta
#         mat_params[1][1]=vector_params[2] #cos_teta
#         mat_params[0][1]=torch.sqrt(1-(vector_params[2]**2))
#         mat_params[1][0]=-mat_params[0][1]
#     return (mat_params)

# def compute_moments(img):
#     moments = torch.empty(6, device=device)
#     l = torch.arange(img.shape[0], device=device)
#     moments[0] = torch.sum(img) # m00
#     moments[1] = torch.sum(img * l) # m10
#     moments[2] = torch.sum(img * (l**2)) # m20
#     moments[3] = torch.sum(img * l.reshape((img.shape[0], 1)) ) # m01
#     moments[4] = torch.sum(img * (l.reshape((img.shape[0], 1)))**2 ) # m02
#     moments[5] = torch.sum(img * l * l.reshape((img.shape[0], 1))) # m11
#     return moments

# def estimate_initial(Ref_uint8, Flt_uint8, params):
#     #print("ESTIMATE INITIAL")
#     #ref_mom =cv2.moments(Ref_uint8.numpy())
#     #flt_mom = cv2.moments(Flt_uint8.numpy())
    
#     ref_mom = compute_moments(Ref_uint8)
#     flt_mom = compute_moments(Flt_uint8)
        
#     flt_avg_10 = flt_mom[1]/flt_mom[0]
#     flt_avg_01 = flt_mom[3]/flt_mom[0]
#     flt_mu_20 = (flt_mom[2]/flt_mom[0]*1.0)-(flt_avg_10*flt_avg_10)
#     flt_mu_02 = (flt_mom[4]/flt_mom[0]*1.0)-(flt_avg_01*flt_avg_01)
#     flt_mu_11 = (flt_mom[5]/flt_mom[0]*1.0)-(flt_avg_01*flt_avg_10)

#     ref_avg_10 = ref_mom[1]/ref_mom[0]
#     ref_avg_01 = ref_mom[3]/ref_mom[0]
#     ref_mu_20 = (ref_mom[2]/ref_mom[0]*1.0)-(ref_avg_10*ref_avg_10)
#     ref_mu_02 = (ref_mom[4]/ref_mom[0]*1.0)-(ref_avg_01*ref_avg_01)
#     ref_mu_11 = (ref_mom[5]/ref_mom[0]*1.0)-(ref_avg_01*ref_avg_10)
    
#     params[0][2] = ref_mom[1]/ref_mom[0]-flt_mom[1]/flt_mom[0]
#     params[1][2] = ref_mom[3]/ref_mom[0] - flt_mom[3]/flt_mom[0]
    
#     rho_flt=0.5*torch.atan((2.0*flt_mu_11)/(flt_mu_20-flt_mu_02))
#     rho_ref=0.5*torch.atan((2.0*ref_mu_11)/(ref_mu_20-ref_mu_02))
#     delta_rho=rho_ref-rho_flt
    
#     roundness=(flt_mom[2]/flt_mom[0]) / (flt_mom[4]/flt_mom[0])
#     if math.fabs(roundness-1.0)>=0.3:
#         params[0][0]= torch.cos(delta_rho)
#         params[0][1] = -torch.sin(delta_rho)
#         params[1][0] = torch.sin(delta_rho)
#         params[1][1] = torch.cos(delta_rho)
#     else:
#         params[0][0]= 1.0
#         params[0][1] = 0.0
#         params[1][0] = 0.0
#         params[1][1] = 1.0
#     return (params)

def NormalVariateGenerator():

    global m_Gaussfaze,m_Gausssave,m_GScale
    m_Gaussfaze = m_Gaussfaze-1
    if (m_Gaussfaze):
        return m_GScale * m_Gausssave[m_Gaussfaze];
    else:
        return FastNorm();

def SignedShiftXOR(x):
    uirs = np.uint32(x);
    c=np.int32((uirs << 1) ^ 333556017) if np.int32(x <= 0) else np.int32(uirs << 1)
    return c

def FastNorm():
    m_Scale = 30000000.0
    m_Rscale = 1.0 / m_Scale
    m_Rcons = 1.0 / (2.0 * 1024.0 * 1024.0 * 1024.0)
    m_ELEN = 7  #LEN must be 2 ** ELEN  
    m_LEN = 128
    m_LMASK = (4 * (m_LEN - 1))
    m_TLEN = (8 * m_LEN)
    m_Vec1 = np.zeros(m_TLEN)
    m_Lseed = 12345
    m_Irs = 12345
    m_GScale = m_Rscale
    fake = 1.0 + 0.125 / m_TLEN
    m_Chic2 = np.sqrt(2.0 * m_TLEN - fake * fake) / fake
    m_Chic1 = fake * np.sqrt(0.5 / m_TLEN)
    m_ActualRSD = 0.0
    inc = 0
    mask = 0
    m_Nslew = 0
    if (not(m_Nslew & 0xFF)):
        if (m_Nslew & 0xFFFF):
            pass
        else:
            ts = 0.0
            p = 0
            while(True):
                while(True):
                    m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
                    m_Irs = np.int64(SignedShiftXOR(m_Irs))
                    r = np.int32((m_Irs)+ np.int64(m_Lseed))
                    tx = m_Rcons * r
                    m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
                    m_Irs = np.int64(SignedShiftXOR(m_Irs))
                    r = np.int32((m_Irs) + np.int64(m_Lseed))
                    ty = m_Rcons * r
                    tr = tx * tx + ty * ty
                    if ((tr <= 1.0) and (tr >= 0.1)):
                        break
                m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
                m_Irs = np.int64(SignedShiftXOR(m_Irs))
                r = np.int32((m_Irs) + np.int64(m_Lseed))
                if (r < 0):
                    r = ~r
                tz = -2.0 * np.log((r + 0.5) * m_Rcons)
                ts += tz;
                tz = np.sqrt(tz / tr);
                m_Vec1[p] = (int)(m_Scale * tx * tz);
                p=p+1
                m_Vec1[p] = (int)(m_Scale * ty * tz);
                p=p+1
                if (p >= m_TLEN):
                    break
            ts = m_TLEN / ts
            tr = np.sqrt(ts)
            for p in range(0, m_TLEN):
                tx = m_Vec1[p] * tr
                m_Vec1[p]= int(tx - 0.5) if int(tx < 0.0) else int(tx + 0.5)
            ts = 0.0
            for p in range(0,m_TLEN):
                tx = m_Vec1[p]
                ts += (tx * tx)
            ts = np.sqrt(ts / (m_Scale * m_Scale * m_TLEN))
            m_ActualRSD = 1.0 / ts
            m_Nslew=m_Nslew+1
            global m_Gaussfaze
            m_Gaussfaze = m_TLEN - 1
            m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
            m_Irs = np.int64(SignedShiftXOR(m_Irs))
            t = np.int32((m_Irs) + np.int64(m_Lseed))
            if (t < 0):
                t = ~t
            t = t >> (29 - 2 * m_ELEN)
            skew = (m_LEN - 1) & t
            t = t >> m_ELEN
            skew = 4 * skew
            stride = int((m_LEN / 2 - 1)) & t
            t = t >> (m_ELEN - 1)
            stride = 8 * stride + 4
            mtype = t & 3
            stype = m_Nslew & 3
            if(stype==1):
                inc = 1
                mask = m_LMASK
                pa = m_Vec1[4 * m_LEN]
                pa_idx = 4 * m_LEN
                pb = m_Vec1[4 * m_LEN + m_LEN]
                pb_idx = 4 * m_LEN + m_LEN
                pc = m_Vec1[4 * m_LEN + 2 * m_LEN]
                pc_idx = 4 * m_LEN + 2 * m_LEN
                pd = m_Vec1[4 * m_LEN + 3 * m_LEN]
                pd_idx = 4 * m_LEN + 3 * m_LEN
                p0 = m_Vec1[0]
                p0_idx = 0
                global m_Gausssave
                m_Gausssave = m_Vec1
                i = m_LEN
                pb = m_Vec1[4 * m_LEN + m_LEN + (inc * (m_LEN - 1))]
                pb_idx = 4 * m_LEN + m_LEN + (inc * (m_LEN - 1))
                while(True):
                    skew = (skew + stride) & mask
                    pe = m_Vec1[skew]
                    pe_idx = skew
                    p = -m_Vec1[pa_idx]
                    q = m_Vec1[pb_idx]
                    r = m_Vec1[pc_idx]
                    s = -m_Vec1[pd_idx]
                    t = int(p + q + r + s) >> 1
                    p = t - p
                    q = t - q
                    r = t - r
                    s = t - s
  
                    t = m_Vec1[pe_idx]
                    m_Vec1[pe_idx] = p
                    pe = m_Vec1[skew+inc]
                    pe_idx = skew+inc
                    p = -m_Vec1[pe_idx]
                    m_Vec1[pe_idx] = q
                    pe = m_Vec1[skew + 2 * inc]
                    pe_idx = skew + 2 * inc
                    q = -m_Vec1[pe_idx]
                    m_Vec1[pe_idx] = r
                    pe = m_Vec1[skew + 3 * inc]
                    pe_idx = skew + 3 * inc
                    r = m_Vec1[pe_idx]
                    m_Vec1[pe_idx] = s
                    s = int(p + q + r + t) >> 1
                    m_Vec1[pa_idx] = s - p
                    pa = m_Vec1[pa_idx + inc]
                    pa_idx = pa_idx + inc
                    m_Vec1[pb_idx] = s - t
                    pb = m_Vec1[pb_idx - inc]
                    pb_idx = pb_idx - inc
                    m_Vec1[pc_idx] = s - q
                    pc = m_Vec1[pc_idx + inc]
                    pc_idx = pc_idx + inc
                    m_Vec1[pd_idx] = s - r
                    if(i==1):
                        break
                    else:
                        pd = m_Vec1[pd_idx + inc] 
                        pd_idx = pd_idx + inc
                    i=i-1
                    if (i==0):
                        break
                ts = m_Chic1 * (m_Chic2 + m_GScale * m_Vec1[m_TLEN - 1])
                m_GScale = m_Rscale * ts * m_ActualRSD
                return (m_GScale * m_Vec1[0])
  
            else:
                print("Error")
    else:
        return 10

def to_matrix_complete(vector_params):
    """
        vector_params contains tx, ty, tz for translation on x, y and z axes respectively
        and cosine of phi, theta, psi for rotations around x, y, and z axes respectively.
    """
    mat_params=torch.empty((3,4), device = device)
    mat_params[0][3]=vector_params[0] 
    mat_params[1][3]=vector_params[1] 
    mat_params[2][3]=vector_params[2]
    cos_phi = vector_params[3]
    cos_theta = vector_params[4]
    cos_psi = vector_params[5]


    if cos_phi > 1 or cos_phi < -1:
        cos_phi = 1

    if cos_theta > 1 or cos_theta < -1:
        cos_theta = 1
    if cos_psi > 1 or cos_psi < -1:
        cos_psi = 1
    sin_phi = -torch.sqrt(torch.Tensor([1-(cos_phi**2)]))
    sin_theta = -torch.sqrt(torch.Tensor([1-(cos_theta**2)]))
    sin_psi = -torch.sqrt(torch.Tensor([1-(cos_psi**2)]))
    sin_theta_sin_psi = sin_theta * sin_psi
    sin_theta_cos_psi = sin_theta * cos_psi
    cos_theta_cos_psi = cos_theta * cos_psi
    cos_theta_sin_psi = cos_theta * sin_psi
    cos_phi_cos_psi = cos_phi * cos_psi
    cos_phi_sin_psi = cos_phi * sin_psi
    sin_phi_cos_theta = sin_phi * cos_theta
    sin_phi_sin_psi = sin_phi * sin_psi
    cos_phi_cos_theta = cos_phi * cos_theta
    sin_phi_cos_psi = sin_phi * cos_psi
    mat_params[0][0] = cos_theta_cos_psi
    mat_params[1][0] = cos_theta_sin_psi
    mat_params[2][0] = -sin_theta
    mat_params[0][1] = -cos_phi_sin_psi + sin_phi * sin_theta_cos_psi
    mat_params[1][1] = cos_phi_cos_psi + sin_phi * sin_theta_sin_psi
    mat_params[2][1] = sin_phi_cos_theta
    mat_params[0][2] = sin_phi_sin_psi + cos_phi * sin_theta_cos_psi
    mat_params[1][2] = -sin_phi_cos_psi + cos_phi * sin_theta_sin_psi
    mat_params[2][2] = cos_phi_cos_theta
    #print(mat_params)
    return (mat_params)


def compute_moments(img):
    moments = torch.empty(6, device=device)
    l = torch.arange(img.shape[0], device=device)
    moments[0] = torch.sum(img) # m00
    moments[1] = torch.sum(img * l) # m10
    moments[2] = torch.sum(img * (l**2)) # m20
    moments[3] = torch.sum(img * l.reshape((img.shape[0], 1)) ) # m01
    moments[4] = torch.sum(img * (l.reshape((img.shape[0], 1)))**2 ) # m02
    moments[5] = torch.sum(img * l * l.reshape((img.shape[0], 1))) # m11
    return moments

def estimate_initial(Ref_uint8s,Flt_uint8s, params, volume):
    
    tot_flt_avg_10 = 0
    tot_flt_avg_01 = 0
    tot_flt_mu_20 = 0
    tot_flt_mu_02 = 0
    tot_flt_mu_11 = 0
    tot_ref_avg_10 = 0
    tot_ref_avg_01 = 0
    tot_ref_mu_20 = 0
    tot_ref_mu_02 = 0
    tot_ref_mu_11 = 0
    tot_params1 = 0
    tot_params2 = 0
    tot_roundness = 0
    for i in range(0, volume):
        Ref_uint8 = Ref_uint8s[i, :, :]
        Flt_uint8 = Flt_uint8s[i, :, :]
        try:
            ref_mom = compute_moments(Ref_uint8)
            flt_mom = compute_moments(Flt_uint8)
        except:
             continue
        flt_avg_10 = flt_mom[1]/flt_mom[0]
        flt_avg_01 = flt_mom[3]/flt_mom[0]
        flt_mu_20 = (flt_mom[2]/flt_mom[0]*1.0)-(flt_avg_10*flt_avg_10)
        flt_mu_02 = (flt_mom[4]/flt_mom[0]*1.0)-(flt_avg_01*flt_avg_01)
        flt_mu_11 = (flt_mom[5]/flt_mom[0]*1.0)-(flt_avg_01*flt_avg_10)
        ref_avg_10 = ref_mom[1]/ref_mom[0]
        ref_avg_01 = ref_mom[3]/ref_mom[0]
        ref_mu_20 = (ref_mom[2]/ref_mom[0]*1.0)-(ref_avg_10*ref_avg_10)
        ref_mu_02 = (ref_mom[4]/ref_mom[0]*1.0)-(ref_avg_01*ref_avg_01)
        ref_mu_11 = (ref_mom[5]/ref_mom[0]*1.0)-(ref_avg_01*ref_avg_10)
        params1 = ref_mom[1]/ref_mom[0]-flt_mom[1]/flt_mom[0]
        params2 = ref_mom[3]/ref_mom[0] - flt_mom[3]/flt_mom[0]
        roundness=(flt_mom[2]/flt_mom[0]) / (flt_mom[4]/flt_mom[0])
        tot_flt_avg_10 += flt_avg_10
        tot_flt_avg_01 += flt_avg_01
        tot_flt_mu_20 += flt_mu_20
        tot_flt_mu_02 += flt_mu_02
        tot_flt_mu_11 += flt_mu_11
        tot_ref_avg_10 += ref_avg_10
        tot_ref_avg_01 += ref_avg_01
        tot_ref_mu_20 += ref_mu_20
        tot_ref_mu_02 += ref_mu_02
        tot_ref_mu_11 += ref_mu_11
        tot_params1 += params1
        tot_params2 += params2
        tot_roundness += roundness
    tot_flt_avg_10 = tot_flt_avg_10/volume
    tot_flt_avg_01 = tot_flt_avg_01/volume
    tot_flt_mu_20 = tot_flt_mu_20/volume
    tot_flt_mu_02 = tot_flt_mu_02/volume
    tot_flt_mu_11 = tot_flt_mu_11/volume
    tot_ref_avg_10 = tot_ref_avg_10/volume
    tot_ref_avg_01 = tot_ref_avg_01/volume
    tot_ref_mu_20 = tot_ref_mu_20/volume
    tot_ref_mu_02 = tot_ref_mu_02/volume
    tot_ref_mu_11 = tot_ref_mu_11/volume
    tot_params1 = tot_params1/volume
    tot_params2 = tot_params2/volume
    tot_roundness = tot_roundness/volume
    # try:
    #     rho_flt=0.5*torch.atan((2.0*flt_mu_11)/(flt_mu_20-flt_mu_02))
    #     rho_ref=0.5*torch.atan((2.0*ref_mu_11)/(ref_mu_20-ref_mu_02))
    #     delta_rho=rho_ref-rho_flt
    #     if math.fabs(tot_roundness-1.0)<0.3:
    #         delta_rho = 0
    # except Exception:
    #     delta_rho = 0
#since the matrix we want to create is an affine matrix, the initial parameters have been prepared as a "particular" affine, the similarity matrix.
    params[0][3] = tot_params1
    params[1][3] = tot_params2
    rho_flt=0.5*math.atan((2.0*tot_flt_mu_11)/(tot_flt_mu_20-tot_flt_mu_02))
    rho_ref=0.5*math.atan((2.0*tot_ref_mu_11)/(tot_ref_mu_20-tot_ref_mu_02))
    delta_rho=rho_ref-rho_flt
    if math.fabs(tot_roundness-1.0)>=0.3:
        params[0][0]= torch.cos(torch.tensor([delta_rho]))
        params[0][1] = -torch.sin(torch.tensor([delta_rho]))
        params[1][0] = torch.sin(torch.tensor([delta_rho]))
        params[1][1] = torch.cos(torch.tensor([delta_rho]))
    else:
        params[0][0]= torch.Tensor([1.0])
        params[0][1] = torch.Tensor([0.0])
        params[1][0] = torch.Tensor([0.0])
        params[1][1] = torch.Tensor([1.0])
    params[2][2] = torch.Tensor([1.0])
    params[0][2] = params[0][3] = torch.Tensor([0.0])
    params[2][0] = params[2][1] = torch.Tensor([0.0])
    #params[5] = torch.cos(torch.tensor([delta_rho]))
    #print("before: ", params)
    return params

def transform(image, par, volume):
    tmp_img = image.reshape((1, 1, *image.shape)).float()
    t_par = torch.unsqueeze(par, dim=0)
    img_warped = kornia.geometry.warp_affine3d(tmp_img, t_par, dsize=(volume, tmp_img.shape[3], tmp_img.shape[4]), align_corners = True)
    return img_warped 

def compute_mi(ref_img, flt_img, t_mat, eref, volume):
    flt_warped = transform(flt_img, to_cuda(t_mat), volume)
    mi = mutual_information(ref_img, flt_warped.ravel(), eref)
    mi = -(mi.cpu())
    #print(mi)
    return mi

# def compute_cc(ref_img, flt_img, t_mat, cc_ref):
#     flt_warped = transform(flt_img, t_mat)
#     cc = cross_correlation(ref_img, flt_warped.ravel(), cc_ref)
#     return cc.cpu()

# def compute_mse(ref_img, flt_img, t_mat, mse_ref):
#     flt_warped = transform(flt_img, t_mat)
#     mse = mean_squared_error(ref_img, flt_warped.ravel(), mse_ref)
#     return mse.cpu()

def estimate_rho(Ref_uint8s,Flt_uint8s, params, volume):
    tot_flt_avg_10 = 0
    tot_flt_avg_01 = 0
    tot_flt_mu_20 = 0
    tot_flt_mu_02 = 0
    tot_flt_mu_11 = 0
    tot_ref_avg_10 = 0
    tot_ref_avg_01 = 0
    tot_ref_mu_20 = 0
    tot_ref_mu_02 = 0
    tot_ref_mu_11 = 0
    tot_params1 = 0
    tot_params2 = 0
    tot_roundness = 0
    for i in range(0, volume):
        Ref_uint8 = Ref_uint8s[i, :, :]
        Flt_uint8 = Flt_uint8s[i, :, :]
        
        try:
            ref_mom = compute_moments(Ref_uint8)
            #print(ref_mom)
            flt_mom = compute_moments(Flt_uint8)
            #print(flt_mom)

            flt_avg_10 = flt_mom[1]/flt_mom[0]
            flt_avg_01 = flt_mom[3]/flt_mom[0]
            flt_mu_20 = (flt_mom[2]/flt_mom[0]*1.0)-(flt_avg_10*flt_avg_10)
            flt_mu_02 = (flt_mom[4]/flt_mom[0]*1.0)-(flt_avg_01*flt_avg_01)
            flt_mu_11 = (flt_mom[5]/flt_mom[0]*1.0)-(flt_avg_01*flt_avg_10)
            ref_avg_10 = ref_mom[1]/ref_mom[0]
            ref_avg_01 = ref_mom[3]/ref_mom[0]
            ref_mu_20 = (ref_mom[2]/ref_mom[0]*1.0)-(ref_avg_10*ref_avg_10)
            ref_mu_02 = (ref_mom[4]/ref_mom[0]*1.0)-(ref_avg_01*ref_avg_01)
            ref_mu_11 = (ref_mom[5]/ref_mom[0]*1.0)-(ref_avg_01*ref_avg_10)
            
            params[0] = ref_mom[1]/ref_mom[0] - flt_mom[1]/flt_mom[0]
            params[1] = ref_mom[3]/ref_mom[0] - flt_mom[3]/flt_mom[0]

            roundness=(flt_mom[2]/flt_mom[0]) / (flt_mom[4]/flt_mom[0])
            tot_flt_avg_10 += flt_avg_10
            tot_flt_avg_01 += flt_avg_01
            tot_flt_mu_20 += flt_mu_20
            tot_flt_mu_02 += flt_mu_02
            tot_flt_mu_11 += flt_mu_11
            tot_ref_avg_10 += ref_avg_10
            tot_ref_avg_01 += ref_avg_01
            tot_ref_mu_20 += ref_mu_20
            tot_ref_mu_02 += ref_mu_02
            tot_ref_mu_11 += ref_mu_11
            tot_params1 += params[0]
            tot_params2 += params[1]
            tot_roundness += roundness
            #print(i)
        except:
             continue
        
    tot_flt_avg_10 = tot_flt_avg_10/volume
    tot_flt_avg_01 = tot_flt_avg_01/volume
    tot_flt_mu_20 = tot_flt_mu_20/volume
    tot_flt_mu_02 = tot_flt_mu_02/volume
    tot_flt_mu_11 = tot_flt_mu_11/volume
    tot_ref_avg_10 = tot_ref_avg_10/volume
    tot_ref_avg_01 = tot_ref_avg_01/volume
    tot_ref_mu_20 = tot_ref_mu_20/volume
    tot_ref_mu_02 = tot_ref_mu_02/volume
    tot_ref_mu_11 = tot_ref_mu_11/volume
    tot_params1 = tot_params1/volume
    tot_params2 = tot_params2/volume
    tot_roundness = tot_roundness/volume
    try: 
        rho_flt=0.5*torch.atan((2.0*flt_mu_11)/(flt_mu_20-flt_mu_02))
        rho_ref=0.5*torch.atan((2.0*ref_mu_11)/(ref_mu_20-ref_mu_02))
        delta_rho=rho_ref-rho_flt
        #since the matrix we want to create is an affine matrix, the initial parameters have been prepared as a "particular" affine, the similarity matrix.
        if math.fabs(tot_roundness-1.0)<0.3:
            delta_rho = 0
    except Exception as e:
        delta_rho = 0
    
    #print(tot_params1, tot_params2)
    return torch.Tensor([delta_rho]), torch.Tensor([tot_params1]), torch.Tensor([tot_params2])

def estimate_initial3D(Ref_uint8s,Flt_uint8s, params, volume):
    params = torch.zeros((6,))
    psi, tx, ty = estimate_rho(Ref_uint8s, Flt_uint8s, params, volume)
    rot_ref_phi = torch.rot90(Ref_uint8s, dims=[0,2])
    rot_flt_phi = torch.rot90(Flt_uint8s, dims=[0,2])
    phi, _, _  = estimate_rho(rot_ref_phi, rot_flt_phi, params, volume)
    rot_ref_theta = torch.rot90(Ref_uint8s, dims=[1,2])
    rot_flt_theta = torch.rot90(Flt_uint8s, dims=[1,2])
    theta, _, _ = estimate_rho(rot_ref_theta, rot_flt_theta, params, volume)
    params[0] = tx
    params[1] = ty
    params[2] = torch.Tensor([0])
    params[3] = torch.cos(phi)
    params[4] = torch.cos(theta)
    params[5] = torch.cos(psi)
    #params = [tx, ty, 0, torch.cos(phi), torch.cos(theta), torch.cos(psi)]
    return params

def OnePlusOne(Ref_uint8, Flt_uint8, volume, eref):
    
    #parent_cpu = parent.cpu() 
    m_CatchGetValueException = False
    m_MetricWorstPossibleValue = 0

    m_Maximize = False
    m_Epsilon = 1.5e-4

    m_Initialized = False
    m_GrowthFactor = 1.05
    m_ShrinkFactor = torch.pow(m_GrowthFactor, torch.tensor(-0.25))
    m_InitialRadius = 1.01
    m_MaximumIteration = 1000
    m_Stop = False
    m_CurrentCost = 0
    m_CurrentIteration = 0
    m_FrobeniusNorm = 0.0

    spaceDimension = 6 

    A = torch.eye(spaceDimension)*m_InitialRadius
    f_norm = torch.zeros(spaceDimension)
    child = torch.empty(spaceDimension)
    delta = torch.empty(spaceDimension)
    
    #parentPosition = estimate_initial3D(Ref_uint8, Flt_uint8, volume)
    parent = torch.zeros((3,4), device = device)

    estimate_initial(Ref_uint8, Flt_uint8, parent, volume)
    parent = parent.cpu()
    parentPosition = torch.empty((6,))
    parentPosition=torch.Tensor([parent[0][3],parent[1][3],0, 1.0, 1.0, parent[0][0]])
    #print("initial params", parentPosition)
    #print("after", parentPosition)
    Ref_uint8_ravel = Ref_uint8.ravel()#.double()
    #parent = to_matrix_complete(parentPosition)
    pvalue = compute_mi(Ref_uint8_ravel, Flt_uint8, move_data(parent), eref, volume)
    childPosition = torch.empty(spaceDimension)

    m_CurrentIteration = 0
    
    for i in range (0,m_MaximumIteration):
        m_CurrentIteration=m_CurrentIteration+1
    
        for j in range (0, spaceDimension):
            f_norm[j]= NormalVariateGenerator() 
    
        delta = A.matmul(f_norm)#A * f_norm
        child = torch.Tensor(parentPosition) + delta
        childPosition = to_matrix_complete(child)
        #print(parentPosition)
        cvalue = compute_mi(Ref_uint8_ravel, Flt_uint8, move_data(childPosition), eref, volume)

        adjust = m_ShrinkFactor
    
        if(m_Maximize):
            if(cvalue > pvalue):
                pvalue = cvalue
                child, parentPosition = parentPosition, child 
                adjust = m_GrowthFactor
            else:
                pass
        else:
            if(cvalue < pvalue):
                pvalue = cvalue
                child, parentPosition = parentPosition, child 
                adjust = m_GrowthFactor
            else:
                pass
                
            
        m_CurrentCost = pvalue
        m_FrobeniusNorm = torch.norm(A) # 'fro'
    
        if(m_FrobeniusNorm <= m_Epsilon):
            break
    
        alpha = ((adjust - 1.0) / torch.dot(f_norm, f_norm))
    
        for c in range(0, spaceDimension):
            for r in range(0,spaceDimension):
                A[r][c] += alpha * delta[r] * f_norm[c];
    #print("final params", parentPosition)
    return (parentPosition) 

def save_data(OUT_STAK, name, res_path, volume):
    OUT_STAK = torch.reshape(OUT_STAK,(volume,1,512, 512))
    # print(OUT_STAK.shape)
    # print(OUT_STAK.size())
    for i in range(len(OUT_STAK)):
        b=name[i].split('/')
        c=b.pop()
        d=c.split('.')
        cv2.imwrite(os.path.join(res_path, d[0][0:2]+str(int(d[0][2:5])+1)+'.png'), kornia.tensor_to_image(OUT_STAK[i].cpu().byte())) #Creare cartelle 



def register_images(filename, Ref_uint8, Flt_uint8, volume):
    global precompute_metric
    #start_single_sw = time.time()
    eref = precompute_mutual_information(Ref_uint8)
    flt_u = torch.unsqueeze(Flt_uint8, dim=0).float()
    flt_stack = torch.stack((flt_u, flt_u))
    optimal_params = OnePlusOne(Ref_uint8, Flt_uint8, volume, eref)
    #print("optimal: ", optimal_params) 
    params_trans=to_matrix_complete(optimal_params)
    flt_transform = transform(Flt_uint8, to_cuda(params_trans),volume)
    #end_single_sw = time.time()
    # print('Final time: ', end_single_sw - start_single_sw)
    #with open(filename, 'a') as file2:
    #            file2.write("%s\n" % (end_single_sw - start_single_sw))
 
    return (params_trans)

def compute(CT, PET, name, curr_res, t_id, patient_id, filename,volume):
    #for (lef,rig)in reversed([(80, 160)]):
        #
        #print("iteration ", lef, rig)
        # print("Nuova Iterazione")
        final_img=[]
        times=[]
        t = 0.0
        it_time = 0.0
        hist_dim = 256
        dim = 512
        
        global move_data
        move_data = no_transfer if device=='cpu' else to_cuda
        left = 80 #int(volume/2 - subvolume/2)
        right = 160 #int(volume/2 + subvolume/2)
        #print("left", left)
        #print("right", right)
        global ref_vals
        ref_vals = torch.ones(dim*dim*len(CT[left:right]), dtype=torch.int, device=device)
        refs = []
        flts = []
        couples = 0
        for c,ij in enumerate(zip(CT, PET)):
            i = ij[0]
            j = ij[1]  
            ref = pydicom.dcmread(i)
            Ref_img = torch.tensor(ref.pixel_array.astype(np.int16), dtype=torch.int16, device=device)
            Ref_img[Ref_img==-2000]=1
            
            flt = pydicom.dcmread(j)
            Flt_img = torch.tensor(flt.pixel_array.astype(np.int16), dtype=torch.int16, device=device)

            Ref_img = (Ref_img - Ref_img.min())/(Ref_img.max() - Ref_img.min())*255
            Ref_uint8 = Ref_img.round().type(torch.uint8)
                    
            Flt_img = (Flt_img - Flt_img.min())/(Flt_img.max() - Flt_img.min())*255
            Flt_uint8 = Flt_img.round().type(torch.uint8)
        
            refs.append(Ref_uint8)
            flts.append(Flt_uint8)
            couples = couples + 1
            if couples >= len(CT):
                break
            
        refs3D = torch.cat(refs)
        flts3D = torch.cat(flts)
        refs3D = torch.reshape(refs3D,(len(CT),512,512))
        flts3D = torch.reshape(flts3D,(len(CT),512,512))
        #print(flts3D.shape)
        start_time = time.time()
        transform_matrix=(register_images(filename, refs3D[left:right], flts3D[left:right], len(CT[left:right])))
        #print(transform_matrix.shape)
        flt_transforms = []
        N = 4
        for index in range(N):
            
            #flts3D = torch.reshape(flts3D,(len(CT[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))]),512,512))
            subvolume = flts3D[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))]
            #print(subvolume.shape)
            flt_transform = transform(subvolume, to_cuda(transform_matrix), len(PET[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))]))
            flt_transforms.append(flt_transform.cpu())
        end_time = time.time()
        with open(filename, 'a') as file2:
                file2.write("%s\n" % (end_time - start_time))
        for index in range(N):
            save_data(flt_transforms[index], PET[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))], curr_res, len(PET[int(index*volume/N):int(np.minimum(int((index+1)*volume/N), volume))]))
        """
        final_img = f_img.cpu()
        end_time= time.time()
        it_time = (end_time - start_time)
        times.append(it_time)
        t=t+it_time
        df = pd.DataFrame([t, np.mean(times), np.std(times)],columns=['Test'+str(patient_id)])#+str(config)accel_id.get_config())])
        times_df = pd.DataFrame(times,columns=['Test'+str(patient_id)])#+str(config)accel_id.get_config())])
        df_path = os.path.join(curr_res,'Time_powll_%02d.csv' % (t_id))
        times_df_path = os.path.join(curr_res,'Img_powll_%02d.csv' % (t_id))
        df.to_csv(df_path, index=False)
        times_df.to_csv(times_df_path, index=False)
        save_data(final_img,PET,curr_res,volume)
        """

def compute_wrapper(args, num_threads=1):
    config=args.config
    
    for k in range(args.offset, args.patient):
        pool = []
        #curr_prefix = args.prefix+str(k)
        curr_prefix = args.prefix
        curr_ct = os.path.join(curr_prefix,args.ct_path)
        curr_pet = os.path.join(curr_prefix,args.pet_path)
        curr_res = os.path.join("",args.res_path)
        os.makedirs(curr_res,exist_ok=True)
        CT=glob.glob(curr_ct+'/*dcm')
        PET=glob.glob(curr_pet+'/*dcm')
        #print(curr_ct)
        #print(curr_pet)
        PET.sort(key=lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
        CT.sort(key=lambda var:[int(y) if y.isdigit() else y for y in re.findall(r'[^0-9]|[0-9]+',var)])
        assert len(CT) == len(PET)
        images_per_thread = len(CT) // num_threads
        # print(images_per_thread)
        for i in range(num_threads):
            start = images_per_thread * i
            end = images_per_thread * (i + 1) if i < num_threads - 1 else len(CT)
            name = "t%02d" % (i)
            set_start_method('spawn')
            pool.append(Process(target=compute, args=(CT[start:end], PET[start:end], name, curr_res, i, k, args.filename,args.volume)))
        for t in pool:
            t.start()
        for t in pool:
            t.join()

hist_dim = 256
dim = 512

def main():

    parser = argparse.ArgumentParser(description='Iron software for IR onto a python env')
    parser.add_argument("-pt", "--patient", nargs='?', help='Number of the patient to analyze', default=1, type=int)
    parser.add_argument("-o", "--offset", nargs='?', help='Starting patient to analyze', default=0, type=int)
    parser.add_argument("-cp", "--ct_path", nargs='?', help='Path of the CT Images', default='./')
    parser.add_argument("-pp", "--pet_path", nargs='?', help='Path of the PET Images', default='./')
    parser.add_argument("-rp", "--res_path", nargs='?', help='Path of the Results', default='./')
    parser.add_argument("-t", "--thread_number", nargs='?', help='Number of // threads', default=1, type=int)
    parser.add_argument("-px", "--prefix", nargs='?', help='prefix Path of patients folder', default='./')
    parser.add_argument("-im", "--image_dimension", nargs='?', help='Target images dimensions', default=512, type=int)
    parser.add_argument("-c", "--config", nargs='?', help='prefix Path of patients folder', default='./')
    parser.add_argument("-dvc", "--device", nargs='?', help='Target device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument("-vol", "--volume", nargs='?', help='Volume',type = int, default=512)
    parser.add_argument("-f", "--filename", nargs='?', help='Filename', default="test.csv")

    
    args = parser.parse_args()
    num_threads=args.thread_number

    patient_number=args.patient
   
    #print(args.config)
    #print(args)

    global compute_metric
    compute_metric = compute_mi
    global precompute_metric
    precompute_metric = precompute_mutual_information


    compute_wrapper(args, num_threads)
        
    #print("Faber (1+1) python is at the end :)")



if __name__== "__main__":
    main()

















