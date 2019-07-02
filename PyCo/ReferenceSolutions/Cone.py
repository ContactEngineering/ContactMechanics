#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from PyCo.ContactMechanics import HardWall
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace
from PyCo.Topography import Topography
from PyCo.System import make_system

def Load_and_Mean_pressure(R,angle,v):
    """
    Given contact radius, angle and v

    Parameters
    ----------
    R : float
        Contact radius
    angle : float
        half of Cone_angle
    v : float
        poisson ratio
    """
    E=Es/(1-v**2)
    beta = np.pi / 2 - angle
    external_load=np.pi*Contact_R**2*E*np.tan(beta)/2
    mean_Pressure=exter_load/(np.pi*Contact_R**2)
    return external_load,mean_Pressure

def contact_R_and_Area(penetration,angle):
    """
    Given penetration and angle

    Parameters
    ----------
    Penetration : float
        object penetration depth
    angle : float
        half of Cone_angle
    """
    beta = np.pi / 2 - angle
    contact_radius=2*contact_radius/(np.pi*np.tan(beta))
    area=np.pi*contact_radius**2
    return contact_radius,area

def deformation(penetration,contact_radius,angle):
    """
    Given penetration, contact radius and angle

    Parameters
    ----------
    Penetration : float
        object penetration depth
    contact_radius : float

    angle : float
        half of Cone angle
    """
    Deformation=np.zeros_like(X)
    r=penetration*np.tan(angle)
    beta=np.pi/2-angle
    R=np.sqrt(x**2 + y**2 )
    if contact_radius==0:
        Deformation[:,:]=0
    else:
        R_scale_0=(R<=contact_radius)
        Deformation[R_scale_0]=(np.max(R[R_scale_0])-R[R_scale_0])*np.tan(beta)+penetration*(1-2/np.pi)
        
        R_scale_1=(R==contact_radius)
        Deformation[R_scale_1]=penetration*(1-2/np.pi)

        R_scale_2=(R>=contact_radius)
        Deformation[R_scale_2]=2*penetration*(np.arcsin(contact_radius/R[R_scale_2])-R[R_scale_2]/contact_radius+np.sqrt((R[R_scale_2]/contact_radius)**2-1))/np.pi
    
    return Deformation

def pressure(penetration,mean_pressure,contact_radius,angle):
    """
    Given penetration, mean pressure, contact radius and angle

    Parameters
    ----------
    Penetration : float
        object penetration depth
    mean pressure : float
         mean pressure = external load / working area
    contact radius : float
    angle : float
         half of cone angle
    """
    pressure=np.zeros_like(X)
    R=np.sqrt(x**2 + y**2 )
    r=penetration*np.tan(angle)
    R_scale=(R<=contact_radius)
    pressure[R_scale]=mean_pressure*np.arccosh(contact_radius/R[R_scale])
    return pressure
