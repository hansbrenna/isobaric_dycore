#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:08:48 2018
filter functions for the isobaric gcm
@author: hanbre
"""
import numpy as np
from numba import jit

#filter functions
def polar_fourier_filter(psi):
    return psi


def filter_4dx_components(psi):
    #filter meridional components
    psi_trans = np.fft.rfft(psi,axis=1)
    if len(psi.shape) == 2:
        psi_trans[:,11:] = 0.0+0.0j
    else:    
        psi_trans[:,11:,:] = 0.0+0.0j

    psi = np.fft.irfft(psi_trans,axis=1)
    psi_complex = psi.copy()
    psi = psi.real
    return psi

@jit
def fourth_order_shapiro_filter(psi):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    #4th order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        elif i == nlambda-3:
            i=-3
        elif i == nlambda-4:
            i=-4
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered_lambda[i,j,k] = ((1./256)*(186.*psi[i,j,k]+56.*(psi[i-1,j,k]
                           +psi[i+1,j,k])-28*(psi[i-2,j,k]+psi[i+2,j,k])
                           +8*(psi[i-3,j,k]+psi[i+3,j,k])-(psi[i-4,j,k]+psi[i+4,j,k])))
           
    #4th order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            elif j in (nphi-3,nphi-4,2,3):
                order = 2
            else:
                order = 4
            for k in range(nP):  
                if order == 4:
                    psi_filtered[i,j,k] = ((1./256.)*(186.*psi_filtered_lambda[i,j,k]+56.*(psi_filtered_lambda[i,j-1,k]
                               +psi_filtered_lambda[i,j+1,k])-28*(psi_filtered_lambda[i,j-2,k]+psi_filtered_lambda[i,j+2,k])
                               +8*(psi_filtered_lambda[i,j-3,k]+psi_filtered_lambda[i,j+3,k])-(psi_filtered_lambda[i,j-4,k]+psi_filtered_lambda[i,j+4,k])))
                elif order == 2:
                    psi_filtered[i,j,k] = ((1./16.)*(-psi_filtered_lambda[i,j-2,k]+4*psi_filtered_lambda[i,j-1,k]
                               +10*psi_filtered_lambda[i,j,k]+4*psi_filtered_lambda[i,j+1,k]-psi_filtered_lambda[i,j+2,k])) 
                elif order == 1:
                    psi_filtered[i,j,k] = ((0.25)*(psi_filtered_lambda[i,j-1,k]+2*psi_filtered_lambda[i,j,k]+psi_filtered_lambda[i,j+1,k]))
    return psi_filtered

@jit
def fourth_order_shapiro_filter_2D(psi):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    #4th order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        elif i == nlambda-3:
            i=-3
        elif i == nlambda-4:
            i=-4
        for j in range(1,nphi-1):
            psi_filtered_lambda[i,j] = ((1./256)*(186.*psi[i,j]+56.*(psi[i-1,j]
                       +psi[i+1,j])-28*(psi[i-2,j]+psi[i+2,j])
                       +8*(psi[i-3,j]+psi[i+3,j])-(psi[i-4,j]+psi[i+4,j])))
           
    #4th order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            elif j in (nphi-3,nphi-4,2,3):
                order = 2
            else:
                order = 4
            for k in range(nP):  
                if order == 4:
                    psi_filtered[i,j] = ((1./256.)*(186.*psi_filtered_lambda[i,j]+56.*(psi_filtered_lambda[i,j-1]
                               +psi_filtered_lambda[i,j+1])-28*(psi_filtered_lambda[i,j-2]+psi_filtered_lambda[i,j+2])
                               +8*(psi_filtered_lambda[i,j-3]+psi_filtered_lambda[i,j+3])-(psi_filtered_lambda[i,j-4]+psi_filtered_lambda[i,j+4])))
                elif order == 2:
                    psi_filtered[i,j] = ((1./16.)*(-psi_filtered_lambda[i,j-2]+4*psi_filtered_lambda[i,j-1]
                               +10*psi_filtered_lambda[i,j]+4*psi_filtered_lambda[i,j+1]-psi_filtered_lambda[i,j+2])) 
                elif order == 1:
                    psi_filtered[i,j] = ((0.25)*(psi_filtered_lambda[i,j-1]+2*psi_filtered_lambda[i,j]+psi_filtered_lambda[i,j+1]))
    return psi_filtered

def Robert_Asselin_filter(psi_f,psi_n,psi_p):
    filter_parameter = 0.1
    psi_n_filtered = psi_n + 0.5*filter_parameter*(psi_p-2*psi_n+psi_f)
    return psi_n_filtered