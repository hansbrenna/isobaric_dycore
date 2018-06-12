#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:05:48 2018

@author: Hans Brenna
@license: MIT
This program implements a hydrostatic primitive equations GCM on an isobaric 
vertical coordinate. The numerics will be 2nd order centered differences in 
space and time on a uniform lat, lon horizontal grid. Subgrid scale diffusion 
is neglected and energy will be removed from the grid scale by a 4th order
Shapiro filter in both horizontal directions. Forcing will be adapted from 
Held & Suarez 1994 with newtonian relaxation to radiative equilibrium and 
Reyleigh damping of the horizontal velocities in the lower layers. To control 
the 2dt computational mode I will apply a Robert-Asselin filter to the 
prognostic variables.

Prognostic varriables: u,v,T,Ps
Diagnostic variables: z,omega,omega_s,Q,theta,Fl,Fphi

A speedup of the code was acheived using the numba just-in-time compiler @jit
on most of the functions doing for loops.

This early version uses a low time step (~50-100 s) to ensure linear stability. I will add
polar filtering using fft to control the short waves close to the poles to 
increase the time step to ~300 s

Current issues: Does not conserve total mass. drift is small ~0.1% over 10 days
considering potential fixes
"""
from __future__ import print_function
import sys
from numba import jit
import numpy as np
import pdb
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import time
#import timing

np.seterr(all='ignore')
np.seterr(divide='raise',invalid='raise') #Set numpy to raise exceptions on 
#invalid operations and divide by zero

#placeholdet functions

@jit
def prognostic_u(u_f,u_n,u_p,v_n,us,omega_n,z_n,Ps_n,Fl):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                #advection of u in u-direction
                adv_uu = ((u_n[i,j,k]/(a*cosphi[j]))*((u_n[i+1,j,k]
                      -u_n[i-1,j,k])/(2*dlambda)))
                #advection of u in v-direction
                adv_vu = (v_n[i,j,k]/(a)*(u_n[i,j+1,k]
                       -u_n[i,j-1,k])/(2*dphi))
                #advection of u in omega-direction
                if k == 0:
                    adv_omegau = 0.0
                elif k == nP-1:
                    adv_omegau = (omega_n[i,j,k]*(us[i,j]-u_n[i,j,k-1])/((Ps_n[i,j]-P[nP-1])+dP))
                else:
                    adv_omegau = (omega_n[i,j,k]*(u_n[i,j,k+1]-u_n[i,j,k-1])/(2*dP))
                #coriolis term
                cor_u = 2*OMEGA*np.sin(phi[j])*v_n[i,j,k]
                #gradient of geopotential height
                gdz_u = ((g)/(a*np.cos(phi[j]))*(z_n[i+1,j,k]-z_n[i-1,j,k])
                      /(2*dlambda))
                #curvature term
                curv_u = ((u_n[i,j,k]*v_n[i,j,k]*np.tan(phi[j]))/a)
                
                u_f[i,j,k] = (u_p[i,j,k]+2*dt*(-(adv_uu+adv_vu+adv_omegau)+cor_u
                           -gdz_u+curv_u-Fl[i,j,k]))
    #handle poles            
    u_f[:,0,:] = 0.0
    u_f[:,nphi-1,:] = 0.0
                
    return u_f

@jit
def prognostic_v(v_f,v_n,v_p,u_n,vs,omega_n,z_n,Ps_n,Fphi):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                #advection of v in u-direction
                adv_uv = ((u_n[i,j,k]/(a*np.cos(phi[j])))*((v_n[i+1,j,k]
                -v_n[i-1,j,k])/(2*dlambda)))
                #advection of v in v-direction
                adv_vv = (((v_n[i,j,k]/(a))*((v_n[i,j+1,k]
                       -v_n[i,j-1,k])/(2*dphi))))
                if k == 0:
                    adv_omegav = 0.0
                elif k == nP-1:
                    adv_omegav = (omega_n[i,j,k]*(vs[i,j]-v_n[i,j,k-1])/((Ps_n[i,j]-P[nP-1])+dP))
                else:
                    adv_omegav = (omega_n[i,j,k]*(v_n[i,j,k+1]-v_n[i,j,k-1])/(2*dP))
                cor_v = 2*OMEGA*np.sin(phi[j])*u_n[i,j,k]
                gdz_v = g/a*(z_n[i,j+1,k]-z_n[i,j-1,k])/(2*dphi)
                curv_v = (u_n[i,j,k]*u_n[i,j,k]*np.tan(phi[j]))/a
                
                v_f[i,j,k] = (v_p[i,j,k]+2*dt*(-(adv_uv+adv_vv+adv_omegav)-cor_v
                   -gdz_v-curv_v-Fphi[i,j,k]))
    
    #handle poles            
    v_f[:,0,:] = 0.0
    v_f[:,nphi-1,:] = 0.0
             
    return v_f

@jit
def prognostic_T(T_f,T_n,T_p,u_n,v_n,omega_n,theta_n,Ps_n,theta_s,Q_n):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                adv_uT = ((u_n[i,j,k]/(a*np.cos(phi[j])))*((T_n[i+1,j,k]
                       -T_n[i-1,j,k])/(2*dlambda)))
                adv_vT = (v_n[i,j,k]/(a))*((T_n[i,j+1,k]-T_n[i,j-1,k])/(2*dphi))
                if k == 0:
                    adv_omegaT = 0.0
                elif k == nP-1:
                    adv_omegaT = (((omega_n[i,j,k]*T_n[i,j,k])/theta_n[i,j,k])
                       *((theta_s[i,j]-theta_n[i,j,k-1])/((Ps_n[i,j]-P[nP-1])+dP)))
                else:
                    adv_omegaT = (((omega_n[i,j,k]*T_n[i,j,k])/theta_n[i,j,k])
                       *((theta_n[i,j,k+1]-theta_n[i,j,k-1])/(2*dP)))
                T_f[i,j,k] = (T_p[i,j,k]+2*dt*(-(adv_uT+adv_vT+adv_omegaT)+Q_n[i,j,k]))
    
    #pole handling
    for k in range(nP):
        T_f[:,0,k] = T_f[:,1,k].mean()
        T_f[:,nphi-1,k] = T_f[:,nphi-2,k].mean()
    return T_f

@jit
def prognostic_Ps(Ps_f,Ps_n,Ps_p,omegas_n,lnPs):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            adv_P = ((us[i,j])/(a*np.cos(phi[j]))*(Ps_n[i+1,j]-Ps_n[i-1,j])
                  /(2*dlambda)+(vs[i,j]/(a))*(Ps_n[i,j+1]-Ps_n[i,j-1])/(2*dphi))
            Ps_f[i,j] = Ps_p[i,j]+2*dt*(-adv_P+omegas_n[i,j])
    #pole handling
    Ps_f[:,0] = Ps_f[:,1].mean()
    Ps_f[:,nphi-1] = Ps_f[:,nphi-2].mean()
    lnPs = np.log(Ps_f)
#    
#    if Ps_f.min() < 95500.:
#        Ps_f[np.where(Ps_f < 95500.)] = 95500.
    return Ps_f,lnPs

@jit
def diag_omega(omega_n,u_n,v_n):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
             for k in range(nP):
                 int_div_u = 0.0
                 int_div_v = 0.0
                 if k == 0:
                     omega_n[i,j,k] = 0.0
                 else:
                     for m in range(0,k+1):
                         if m == 0:
                             dP_m = dP/2.
                         else:
                             dP_m = dP
                         int_div_u += (u_n[i+1,j,m]-u_n[i-1,j,m])/(2*dlambda)*dP_m
                         int_div_v += (cosphi[j+1]*v_n[i,j+1,m]-cosphi[j-1]*v_n[i,j-1,m])/(2*dphi)*dP_m
                     omega_n[i,j,k] = -(1./(a*np.cos(phi[j]))*int_div_u+1./(a*cosphi[j])*int_div_v)
    return omega_n

@jit
def diag_omegas(omegas_n,omega_n):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            omegas_n[i,j] = (omega_n[i,j,nP-1]+(-(1./(a*np.cos(phi[j])))
                          *(us[i+1,j]-us[i-1,j])/(2*dlambda)+(1./(a*cosphi[j]))*(cosphi[j+1]*vs[i,j+1]
                          -cosphi[j-1]*vs[i,j-1])/(2*dphi))*(Ps_n[i,j]-95000))
    return omegas_n

@jit
def diag_z(z_n,T_m,dlnP):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                int_Tdlnp = 0.0
                for m in range(k,nP):
                    int_Tdlnp += T_m[i,j,m]*dlnP[i,j,m] #Tm_n is average temp of layer between two levels.
                z_n[i,j,k] = (R/g)*int_Tdlnp
                
    #handle poles:
    for k in range(nP):
        z_n[:,0,k] = z_n[:,1,k].mean()
        z_n[:,nphi-1,k] = z_n[:,nphi-2,k].mean()
    
#    if z_n.min() < 100.:
#        z_n[np.where(z_n < 100.)] = 100.
    if z_n.min() < 0.0:
        print('Negative values of z_n. Layer are crossing. Exiting...')
        sys.exit()
    return z_n

@jit
def helper_Tm(T_m,T_n,Ts):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP-1):
                T_m[i,j,k] = (T_n[i,j,k+1]+T_n[i,j,k])*0.5
            T_m[i,j,nP-1] = (Ts[i,j]+T_n[i,j,nP-1])*0.5
    return T_m

@jit
def helper_dlnP(dlnP,lnPs,P):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP-1):
                dlnP[i,j,k] = lnP[k+1]-lnP[k]
            dlnP[i,j,nP-1] = lnPs[i,j] - lnP[nP-1]
    return dlnP

@jit
def diag_Q(Q_n,T_n,Ps_n,T_eq):
    sigmab = 0.7
    kT = np.zeros(shape=[nlambda,nphi,nP])
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                function = ((P[k]/Ps_n[i,j])-sigmab)/(1.-sigmab)
                if function > 0.:
                    kT[i,j,k] = ka+(ks-ka)*function*np.cos(phi[j])**4
                else:
                    kT[i,j,k] = ka
                    #kT[i,j,k] = ka+(ks-ka)*np.max([0,function])*np.cos(phi[j])**4
                
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                Q_n[i,j,k] = -kT[i,j,k]*(T_n[i,j,k]-T_eq[i,j,k])
    return Q_n

@jit
def diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                theta_n[i,j,k] = T_n[i,j,k]*(P0/P[k])**(R/cp)  
        
        for i in range(nlambda):
            if i == nlambda-1:
                i=-1
            for j in range(1,nphi-1):
                theta_s[i,j] = Ts[i,j]*(P0/Ps_n[i,j])**(R/cp)
        #handle poles:
    for k in range(nP):
        theta_n[:,0,k] = theta_n[:,1,k].mean()
        theta_n[:,nphi-1,k] = theta_n[:,nphi-2,k].mean()
    return theta_n,theta_s

#surface functions
@jit
def diag_surface_wind(us,vs,Vs,u_n,v_n,Ps_p,A):

    A = (kv_surf_w/(kv_surf_w+cv*Vs*(Ps_p-P[nP-1]))) #Vs and Ps used at previous time step
    us = A*u_n[:,:,nP-1]
    vs = A*v_n[:,:,nP-1]
    Vs = np.sqrt(us*us+vs*vs)
    return us,vs,Vs 
    

def H_S_equilibrium_temperature(T_eq,T_eqs,Ps_n):
    dTy = 60 #K
    dthtz = 10 #K
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                HS_T_func = ((315-dTy*np.sin(phi[j])*np.sin(phi[j])-dthtz
                             *np.log(P[k]/P0)*(np.cos(phi[j]))**2)*(P[k]/P0)**kappa)
                T_eq[i,j,k] = np.max([200.,HS_T_func])
    #handle poles
    for k in range(nP):
        T_eq[:,0,k] = T_eq[:,1,k].mean()
        T_eq[:,nphi-1,k] = T_eq[:,nphi-2,k].mean()
        
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            HS_T_func = ((315-dTy*np.sin(phi[j])*np.sin(phi[j])-dthtz
                        *np.log(Ps_n[i,j]/P0)*(np.cos(phi[j]))**2)*(Ps_n[i,j]/P0)**kappa)
            T_eqs[i,j] = np.max([200.,HS_T_func])
    #handle poles
    T_eqs[:,0] = T_eqs[:,1].mean()
    T_eqs[:,nphi-1] = T_eqs[:,nphi-2].mean()
    return T_eq, T_eqs

@jit(nopython=True)
def H_S_friction(Fl,Fphi,u_n,v_n,Ps_n,kv,sigmab):
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                function = ((P[k]/Ps_n[i,j])-sigmab)/(1.-sigmab)
                if function > 0:
                    kv[i,j,k] = kf*function
                else:
                    kv[i,j,k] = 0
    
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                Fl[i,j,k] = kv[i,j,k]*u_n[i,j,k]
                Fphi[i,j,k] = kv[i,j,k]*v_n[i,j,k] 
    return Fl,Fphi

#filter functions
def polar_fourier_filter(psi):
    #filter zonal wavenumbers in polar regions
    #south pole
    psi_trans = np.fft.rfft(psi,axis=0)
    psi_trans[5:,1,:] = 0.0
    psi_trans[8:,2,:] = 0.0
    psi_trans[10:,3,:] = 0.0
    psi_trans[12:,4,:] = 0.0
    psi_trans[15:,5,:] = 0.0
    psi_trans[18:,6,:] = 0.0
    psi_trans[20:,7,:] = 0.0
    #north pole
    psi_trans[5:,nphi-2,:] = 0.0
    psi_trans[8:,nphi-3,:] = 0.0
    psi_trans[10:,nphi-4,:] = 0.0
    psi_trans[12:,nphi-5,:] = 0.0
    psi_trans[15:,nphi-6,:] = 0.0
    psi_trans[18:,nphi-7,:] = 0.0
    psi_trans[20:,nphi-8,:] = 0.0
    psi = np.fft.irfft(psi_trans,axis=0)
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
def first_order_shapiro_filter(psi):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered_lambda[i,j,k] = (0.25*(psi[i-1,j,k]+2*psi[i,j,k]+psi[i+1,j,k]))
                
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered[i,j,k] = (0.25*(psi_filtered_lambda[i,j-1,k]+2
                            *psi_filtered_lambda[i,j,k]+psi_filtered_lambda[i,j+1,k]))
    return psi_filtered


@jit
def first_order_shapiro_filter_2D(psi):
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            psi_filtered_lambda[i,j] = (0.25*(psi[i-1,j]+2*psi[i,j]+psi[i+1,j]))
                
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            psi_filtered[i,j] = (0.25*(psi_filtered_lambda[i,j-1]+2
                        *psi_filtered_lambda[i,j]+psi_filtered_lambda[i,j+1]))
    return psi_filtered


@jit
def second_order_shapiro_filter(psi):
    order = 2
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    #2nd order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        for j in range(1,nphi-1):
            for k in range(nP):
                psi_filtered_lambda[i,j,k] = ((1./16.)*(-psi[i-2,j,k]+4*psi[i-1,j,k]
                               +10*psi[i,j,k]+4*psi[i+1,j,k]-psi[i+2,j,k])) 
           
    #2nd order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            for k in range(nP):  
                if order == 2:
                    psi_filtered[i,j,k] = ((1./16.)*(-psi_filtered_lambda[i,j-2,k]+4*psi_filtered_lambda[i,j-1,k]
                               +10*psi_filtered_lambda[i,j,k]+4*psi_filtered_lambda[i,j+1,k]-psi_filtered_lambda[i,j+2,k])) 
                elif order == 1:
                    psi_filtered[i,j,k] = ((0.25)*(psi_filtered_lambda[i,j-1,k]
                    +2*psi_filtered_lambda[i,j,k]+psi_filtered_lambda[i,j+1,k]))
    return psi_filtered

@jit
def second_order_shapiro_filter_2D(psi):
    order = 2
    psi_filtered_lambda = psi.copy()
    psi_filtered = psi.copy()
    #2nd order shapiro in lambda direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        elif i == nlambda-2:
            i=-2
        for j in range(1,nphi-1):
            psi_filtered_lambda[i,j] = ((1./16.)*(-psi[i-2,j]+4*psi[i-1,j]
                               +10*psi[i,j]+4*psi[i+1,j]-psi[i+2,j])) 
           
    #2nd order shapiro filter in phi direction
    for i in range(nlambda):
        if i == nlambda-1:
            i=-1
        for j in range(1,nphi-1):
            if j in (nphi-2,1):
                order = 1
            if order == 2:
                psi_filtered[i,j] = ((1./16.)*(-psi_filtered_lambda[i,j-2]+4*psi_filtered_lambda[i,j-1]
                           +10*psi_filtered_lambda[i,j]+4*psi_filtered_lambda[i,j+1]-psi_filtered_lambda[i,j+2])) 
            elif order == 1:
                psi_filtered[i,j] = ((0.25)*(psi_filtered_lambda[i,j-1]
                +2*psi_filtered_lambda[i,j]+psi_filtered_lambda[i,j+1]))
    return psi_filtered
                
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

#output functions
def plotter_help(psi,func=1,flip=0):
    absmax = np.max(np.abs(psi))
    if np.min(psi) < 0.:
        cmap = 'coolwarm'
        vmin = -absmax
        vmax = absmax
    else:
        cmap = 'viridis'
        vmin = psi.min()
        vmax = psi.max()
    
    if func == 1:
        CM = plt.pcolormesh(psi.transpose(),vmin=vmin,vmax=vmax,cmap=cmap)
    elif func == 0:
        CM = plt.contourf(psi.transpose(),vmin=vmin,vmax=vmax,cmap=cmap)
    elif func == 2:
        CM = plt.contourf(psi.transpose(),1000,vmin=vmin,vmax=vmax,cmap=cmap)
    if flip:
        plt.gca().invert_yaxis()
        
    plt.colorbar(CM)
    
def compare_filters(psi):
    psif4 = fourth_order_shapiro_filter(psi)
    psif2 = second_order_shapiro_filter(psi)
    psif1 = first_order_shapiro_filter(psi)
    psiff = filter_4dx_components(psi)
    plt.plot(phi,psi[30,:,8],phi,psif4[30,:,8],phi,psif2[30,:,8],phi,psif1[30,:,8],phi,psiff[30,:,8])
    plt.legend(['UF','S4','S2','S1','FF'])
    plt.figure()
    plt.plot(lambd,psi[:,8,8],lambd,psif4[:,8,8],lambd,psif2[:,8,8],lambd,psif1[:,8,8],lambd,psiff[:,8,8])
    plt.legend(['UF','S4','S2','S1','FF'])
    

def print_max_min_of_field(field,name):
    maxf = np.max(field)
    minf = np.min(field)
    #posmax = np.where(field == maxf)
    #posmin = np.where(field == minf)
    print('{}: Max = {}; Min = {}'.format(name,maxf,minf))
    
def threedarray2DataArray(arr):
    da = xr.DataArray(arr,coords = [lon,lat,P],dims = ['lon','lat','lev'])
    return da

def twodarray2DataArray(arr):
    da = xr.DataArray(arr,coords = [lon,lat],dims = ['lon','lat'])
    return da

def add_fields_to_history(u_h,u_n,v_h,v_n,Ps_h,Ps_n,T_h,T_n,omega_h,omega_n
                          ,omegas_h,omegas_n,Q_h,Q_n,theta_h,theta_n):
    u_h += u_n
    v_h += v_n
    Ps_h += Ps_n
    T_h += T_n
    omega_h += omega_n
    omegas_h += omegas_n
    Q_h += Q_n
    theta_h += theta_n
    if t % 86400 == 0:
        u_h /= steps_per_day
        v_h /= steps_per_day
        T_h /= steps_per_day
        Ps_h /= steps_per_day
        omega_h /= steps_per_day
        omegas_h /= steps_per_day
        Q_h /= steps_per_day
        theta_h /= steps_per_day
    return u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h

start = time.time()
#define constants
pi = np.pi
nphi = 46
nlambda = 72    
nP = 9 
g = 9.81
cp = 1004.
kappa = 2./7
R = kappa*cp
a = 6.371e+6
OMEGA = 7.292e-5
kf = 1./86400.
ka = 1/40.*kf
ks = 1/4.*kf
dP = 100*100. #Pa
P0 = 100000. #Pa
kv_surf_w = 24
cv = 0.01
sigmab = 0.7
dlambda_deg = 5.
dlambda = np.deg2rad(dlambda_deg)
dphi_deg = 4.
dphi = np.deg2rad(dphi_deg)
dt = 100 #seconds
tstop = 86400*1
#tstop = 40000
t = 0
steps_per_day = 86400/dt
#define fields
phi = np.array([-(pi/2)+(j)*dphi for j in range(nphi)])
cosphi = np.cos(phi)
lambd = (np.array([-(pi)+(i)*dlambda for i in range(nlambda)]))
P = np.array([100,200,300,400,500,600,700,800,900],dtype=float)*100 #Pa
lon = np.rad2deg(lambd)
lat = np.rad2deg(phi)
lnP = np.log(P)
init = False
if init:
    #prognostic fields
    Ps_f = np.zeros(shape=[nlambda,nphi])
    Ps_n = np.zeros(shape=[nlambda,nphi])
    Ps_p = np.zeros(shape=[nlambda,nphi])
    Ps_h = np.zeros(shape=[nlambda,nphi])
    
    u_f = np.zeros(shape=[nlambda,nphi,nP])
    u_n = np.zeros(shape=[nlambda,nphi,nP])
    u_p = np.zeros(shape=[nlambda,nphi,nP])
    u_h = np.zeros(shape=[nlambda,nphi,nP])
    
    v_f = np.zeros(shape=[nlambda,nphi,nP])
    v_n = np.zeros(shape=[nlambda,nphi,nP])
    v_p = np.zeros(shape=[nlambda,nphi,nP])
    v_h = np.zeros(shape=[nlambda,nphi,nP])
    
    T_f = np.zeros(shape=[nlambda,nphi,nP])
    T_n = np.zeros(shape=[nlambda,nphi,nP])
    T_p = np.zeros(shape=[nlambda,nphi,nP])
    T_h = np.zeros(shape=[nlambda,nphi,nP])
    
    #diagnostic fields
    z_n = np.zeros(shape=[nlambda,nphi,nP])
    z_p = np.zeros(shape=[nlambda,nphi,nP])
    z_h = np.zeros(shape=[nlambda,nphi,nP])
    
    omega_n = np.zeros(shape=[nlambda,nphi,nP])
    omega_p = np.zeros(shape=[nlambda,nphi,nP])
    omega_h = np.zeros(shape=[nlambda,nphi,nP])
    
    omegas_n = np.zeros(shape=[nlambda,nphi])
    omegas_p = np.zeros(shape=[nlambda,nphi])
    omegas_h = np.zeros(shape=[nlambda,nphi])
    
    Q_n = np.zeros(shape=[nlambda,nphi,nP])
    Q_p = np.zeros(shape=[nlambda,nphi,nP])
    Q_h = np.zeros(shape=[nlambda,nphi,nP])
    
    theta_n = np.zeros(shape=[nlambda,nphi,nP])
    theta_p = np.zeros(shape=[nlambda,nphi,nP])
    theta_s = np.zeros(shape=[nlambda,nphi])
    theta_h = np.zeros(shape=[nlambda,nphi,nP])
    
    Fl = np.zeros(shape=[nlambda,nphi,nP])
    Fphi = np.zeros(shape=[nlambda,nphi,nP])
    
    #surface fields
    Ts = np.zeros(shape=[nlambda,nphi])
    us = np.zeros(shape=[nlambda,nphi])
    vs = np.zeros(shape=[nlambda,nphi])
    A = np.zeros(shape=[nlambda,nphi])
    Vs = np.zeros(shape=[nlambda,nphi])
    
    #helper fields
    dlnP = np.zeros(shape=[nlambda,nphi,nP])
    T_m = np.zeros(shape=[nlambda,nphi,nP])
    kv = np.zeros(shape=[nlambda,nphi,nP])
    Ps_mean = []
    
    #Equilibrium temperature field
    T_eq = np.zeros(shape=[nlambda,nphi,nP])
    T_eqs = np.zeros(shape=[nlambda,nphi])
    #initialize
    Ps_n[:,:] = 100000.
    Ps_p[:,:] = 100000.
    T_eq,T_eqs = H_S_equilibrium_temperature(T_eq,T_eqs,Ps_n)
    T_n = T_eq + np.random.random(size=[nlambda,nphi,nP])/1e3 #Random perturbations to break hemispheric and zonal symmetry
    T_p = T_eq
    Ts = T_eqs
    T_m = helper_Tm(T_m,T_n,Ts)
    theta_n,theta_s = diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n)
    z_n = diag_z(z_n,T_m,dlnP)
    lnPs = np.log(Ps_n)
    
#time stepping loop
n = 0
day = 1
while t<tstop:
    n += 1
    #print('Begin timestep: {}'.format(n))
    Fl,Fphi = H_S_friction(Fl,Fphi,u_n,v_n,Ps_n,kv,sigmab)
    u_f = prognostic_u(u_f,u_n,u_p,v_n,us,omega_n,z_n,Ps_n,Fl)
    v_f = prognostic_v(v_f,v_n,v_p,u_n,vs,omega_n,z_n,Ps_n,Fphi)
    T_f = prognostic_T(T_f,T_n,T_p,u_n,v_n,omega_n,theta_n,Ps_n,theta_s,Q_n)
    Ps_f,lnPs = prognostic_Ps(Ps_f,Ps_n,Ps_p,omegas_n,lnPs)
    #polar filter
#    u_f = polar_fourier_filter(u_f)
#    v_f = polar_fourier_filter(v_f)
#    T_f = polar_fourier_filter(T_f)
    #Ps_f = polar_fourier_filter(Ps_f)
    # Apply shapiro filters of order 2 and 4 to the solution.
#    if n % 31 == 0:
#        u_f = second_order_shapiro_filter(u_f)
#        v_f = second_order_shapiro_filter(v_f)
#        T_f = second_order_shapiro_filter(T_f)
#        #Ps_f = second_order_shapiro_filter_2D(Ps_f)
    u_f = fourth_order_shapiro_filter(u_f)
    v_f = fourth_order_shapiro_filter(v_f)
    T_f = fourth_order_shapiro_filter(T_f)
    Ps_f = fourth_order_shapiro_filter_2D(Ps_f)
    #if T_f.max()>320.:
    #    sys.exit()
    #Ps_f = filter_4dx_components(Ps_f)
    #Apply Robert-Asselin Frequency filter to prognostic variables at time n
    u_n = Robert_Asselin_filter(u_f,u_n,u_p)
    v_n = Robert_Asselin_filter(v_f,v_n,v_p)
    T_n = Robert_Asselin_filter(T_f,T_n,T_p)
    Ps_n = Robert_Asselin_filter(Ps_f,Ps_n,Ps_p)
    #print('Finished prognostic equations')
    t = t+dt
    #print('Advance time to: t={}'.format(t))
    u_p = u_n.copy()
    u_n = u_f.copy()    
    v_p = v_n.copy()
    v_n = v_f.copy()
    T_p = T_n.copy()
    T_n = T_f.copy()
    Ps_p = Ps_n.copy()
    Ps_n = Ps_f.copy()
    #print('Diagnostic equations')
    omega_n = diag_omega(omega_n,u_n,v_n)
    omegas_n = diag_omegas(omegas_n,omega_n)
    T_m = helper_Tm(T_m,T_n,Ts)
    dlnP = helper_dlnP(dlnP,lnPs,P)
    z_n = diag_z(z_n,T_m,dlnP)
    theta_n,theta_s = diag_theta(theta_n,theta_s,T_n,Ts,P,Ps_n)
    Q_n = diag_Q(Q_n,T_n,Ps_n,T_eq)
    us,vs,Vs = diag_surface_wind(us,vs,Vs,u_n,v_n,Ps_p,A)
    if n%100 == 0:
        print('Values for time step {}'.format(n))
        print_max_min_of_field(u_n,'U')
        print_max_min_of_field(v_n,'V')
        print_max_min_of_field(T_n,'T')
        print_max_min_of_field(Ps_n,'Ps')
        print('Ps.mean: {}'.format(np.average(Ps_n[:,1:nphi-1],axis=1,weights=np.cos(phi[1:nphi-1])).mean()))
        Ps_mean.append(np.average(Ps_n[:,1:nphi-1],axis=1,weights=np.cos(phi[1:nphi-1])).mean())
        print_max_min_of_field(omega_n,'omega')
        print_max_min_of_field(z_n,'z')

    
#handle output    
    u_h,v_h,Ps_h,T_h,omega_h,omegas_h,Q_h,theta_h = (add_fields_to_history(u_h,
                                                u_n,v_h,v_n,Ps_h,Ps_n,T_h,T_n
                                                ,omega_h,omega_n,omegas_h
                                                ,omegas_n,Q_h,Q_n,theta_h,theta_n))
    if t % 86400 == 0:
        print('writing history file')
        ds = xr.Dataset()
        ds['U'] = threedarray2DataArray(u_h)
        ds['V'] = threedarray2DataArray(v_h)
        ds['Ps'] = twodarray2DataArray(Ps_h)
        ds['T'] = threedarray2DataArray(T_h)
        ds['OMEGA'] = threedarray2DataArray(omega_h)
        ds['OMEGAS'] = twodarray2DataArray(omegas_h)
        ds['Q'] = threedarray2DataArray(Q_h)
        ds['THETA'] = threedarray2DataArray(theta_h)
        ds.to_netcdf('history_out_day{}.nc'.format(day))
        day += 1
        del ds
        u_h[:] = 0;v_h[:] = 0;Ps_h[:] = 0;T_h[:] = 0;omega_h[:] = 0;omegas_h[:] = 0;
        Q_h[:] = 0;theta_h[:] = 0;
    
#handle errors and exceptions

end = time.time()
print(end - start)