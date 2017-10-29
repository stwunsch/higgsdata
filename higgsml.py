#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

__doc__="""
ATLAS Higgs Machine Learning Challenge 2014
Read CERN Open Data Portal Dataset http://opendata.cern.ch/record/328
and manipulate it
 - KaggleWeight and KaggleSet are removed
  - Label is changd from charcter to integer 0 or 1
 - DetailLabel is introduced indicating subpopulations
 - systematics effect are simulated
     - bkg_weight_norm : manipulates the background weight
     - tau_energy_scale : manipulates PRI_tau_pt and recompute other quantities accordingly
             Some WARNING : variable DER_mass_MMC is not properly manipulated (modification is linearised), 
             and I advocate to NOT use DER_mass_MMC when doSystTauEnergyScale is enabled
             There is a threshold in the original HiggsML file at 20GeV on PRI_tau_energy. 
             This threshold is moved when changing sysTauEnergyScale which is unphysicsal. 
             So if you're going to play with sysTauEnergyScale (within 0.9-1.1), 
             I suggest you remove events below say 22 GeV *after* the manipulation
             applying doSystTauEnergyScale with sysTauENergyScale=1. does NOT yield identical results as not applyield 
             doSystTauEnergyScale, this is because of rounding error and zero mass approximation.
             doSysTauEnerbyScale impacts PRI_tau_pt as well as PRI_met and PRI_met_phi
    - so overall I suggest that when playing with doSystTauEnergyScale, the reference is
          - not using DER_mass_MMC
          - applying *after* this manipulation PRI_tau_pt>22
          - run with sysTauENergyScale=1. to have the reference
          
Author D. Rousseau LAL, Nov 2016

Modification Dec 2016 (V. Estrade):
- Wrap everything into separated functions.
- V4 class now handle 1D-vector values (to improve computation efficiency).
- Fix compatibility with both python 2 and 3.
- Use pandas.DataFrame to ease computation along columns
- Loading function for the base HiggsML dataset (fetch it on the internet if needed)

Refactor March 2017 (V. Estrade):
- Split load function (cleaner)

July 06 2017 (V. Estrade):
- Add normalization_weight function
"""
__version__ = "3.6"
__author__ = "David Rousseau, and Victor Estrade "

import sys
import os
import gzip
import copy
import pandas as pd
import numpy as np

from download import maybe_download
from download import get_data_dir

def load_higgs():
    url = "http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz"
    filename = os.path.join(get_data_dir(), "atlas-higgs-challenge-2014-v2.csv.gz")
    maybe_download(filename, url)
    data = pd.read_csv(filename)
    return data

def normalize_weight(W, y, background_luminosity=410999.84732187376, signal_luminosity=691.9886077135781):
    """Normalize the given weight to assert that the luminosity is the same as the nominal.
    Returns the normalized weight vector/Series
    """
    background_weight_sum = W[y==0].sum()
    signal_weight_sum = W[y==1].sum()
    W_new = W.copy()
    W_new[y==0] = W[y==0] * ( background_luminosity / background_weight_sum )
    W_new[y==1] = W[y==1] * ( signal_luminosity / signal_weight_sum )
    return W_new
# ==================================================================================
#  V4 Class and physic computations
# ==================================================================================

class V4:
    """
    A simple 4-vector class to ease calculation
    """
    px=0
    py=0
    pz=0
    e=0
    def __init__(self,apx=0., apy=0., apz=0., ae=0.):
        """
        Constructor with 4 coordinates
        """
        self.px = apx
        self.py = apy
        self.pz = apz
        self.e = ae
        if self.e + 1e-3 < self.p():
            raise ValueError("Energy is too small! Energy: {}, p: {}".format(self.e, self.p()))

    def copy(self):
        return copy.deepcopy(self)
    
    def p2(self):
        return self.px**2 + self.py**2 + self.pz**2
    
    def p(self):
        return np.sqrt(self.p2())
    
    def pt2(self):
        return self.px**2 + self.py**2
    
    def pt(self):
        return np.sqrt(self.pt2())
    
    def m(self):
        return np.sqrt( np.abs( self.e**2 - self.p2() ) ) # abs is needed for protection
    
    def eta(self):
        return np.arcsinh( self.pz/self.pt() )
    
    def phi(self):
        return np.arctan2(self.py, self.px)
    
    def deltaPhi(self, v):
        """delta phi with another v"""
        return (self.phi() - v.phi() + 3*np.pi) % (2*np.pi) - np.pi
    
    def deltaEta(self,v):
        """delta eta with another v"""
        return self.eta()-v.eta()
    
    def deltaR(self,v):
        """delta R with another v"""
        return np.sqrt(self.deltaPhi(v)**2+self.deltaEta(v)**2 )

    def eWithM(self,m=0.):
        """recompute e given m"""
        return np.sqrt(self.p2()+m**2)

    # FIXME this gives ugly prints with 1D-arrays
    def __str__(self):
        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (self.px, self.py,self.pz,self.e,self.pt(),self.eta(),self.phi(),self.m())

    def scale(self,factor=1.): # scale
        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = np.abs( factor*self.e )
    
    def scaleFixedM(self,factor=1.): 
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)
    
    def setPtEtaPhiM(self, pt=0., eta=0., phi=0., m=0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt*np.cos(phi)
        self.py = pt*np.sin(phi)
        self.pz = pt*np.sinh(eta)
        self.e = self.eWithM(m)
    
    def sum(self, v):
        """Add another V4 into self"""
        self.px += v.px
        self.py += v.py
        self.pz += v.pz
        self.e += v.e
    
    def __iadd__(self, other):
        """Add another V4 into self"""
        try:
            self.px += other.px
            self.py += other.py
            self.pz += other.pz
            self.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return self
    
    def __add__(self, other):
        """Add 2 V4 vectors : v3 = v1 + v2 = v1.__add__(v2)"""
        copy = self.copy()
        try:
            copy.px += other.px
            copy.py += other.py
            copy.pz += other.pz
            copy.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy

# magic variable
# FIXME : does it really returns sqrt(2) if in dead center ?
def METphi_centrality(aPhi, bPhi, cPhi):
    """
    Calculate the phi centrality score for an object to be between two other objects in phi
    Returns sqrt(2) if in dead center
    Returns smaller than 1 if an object is not between
    a and b are the bounds, c is the vector to be tested
    """
    # Safely compute and set to zeros results of zero divisions
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        
        A = np.true_divide(np.sin(cPhi - aPhi), np.sin(bPhi - aPhi))
        A[A == np.inf] = 0
        A = np.nan_to_num(A)
        
        B = np.true_divide(np.sin(bPhi - cPhi), np.sin(bPhi - aPhi))
        B[B == np.inf] = 0
        B = np.nan_to_num(B)
        
        res = (A+B)/np.sqrt(A**2 + B**2)
        res[res == np.inf] = 0
        res = np.nan_to_num(res)
    return res


# another magic variable
def eta_centrality(eta, etaJ1, etaJ2):
    """
    Calculate the eta centrality score for an object to be between two other objects in eta
    Returns 1 if in dead center
    Returns value smaller than 1/e if object is not between
    """
    center = (etaJ1 + etaJ2) / 2.
    
    # Safely compute and set to zeros results of zero divisions
    with np.errstate(divide='ignore', invalid='ignore'):
        width  = 1. / (etaJ1 - center)**2
        width[width == np.inf] = 0
        width = np.nan_to_num(width)
        
    return np.exp(-width * (eta - center)**2)


# ==================================================================================
#  Now we enter in the manipulation procedures (everything works on data inplace)
# ==================================================================================

def label_to_float(data):
    """
    Transform the string labels to float values.
    s -> 1.0
    b -> 0.0

    Works inplace on the given data !

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
    """
    if data['Label'].dtype == object:
        #copy entry in human usable form
        data["Label"] = (data["Label"] == 's').astype("float")
    else:
        pass

# ==================================================================================
def getDetailLabel(origWeight, Label, num=True):
    """
    Given original weight and label, 
    return more precise label specifying the original simulation type.
    
    Args
    ----
        origWeight: the original weight of the event
        Label : the label of the event (can be {"b", "s"} or {0,1})
        num: (default=True) if True use the numeric detail labels
                else use the string detail labels. You should prefer numeric labels.

    Return
    ------
        detailLabel: the corresponding detail label ("W" is the default if not found)

    Note : Could be better optimized but this is fast enough.
    """
    # prefer numeric detail label
    detail_label_num={
        57207:0, # Signal
        4613:1,
        8145:2,
        4610:3,
        917703: 105, #Z
        5127399:111,
        4435976:112,
        4187604:113,
        2407146:114,
        1307751:115,
        944596:122,
        936590:123,
        1093224:124,
        225326:132,
        217575:133,
        195328:134,
        254338:135,
        2268701:300 #T
        }
    # complementary for W detaillabeldict=200
    #previous alphanumeric detail label    
    detail_lable_str={
       57207:"S0",
       4613:"S1",
       8145:"S2",
       4610:"S3",
       917703:"Z05",
       5127399:"Z11",
       4435976:"Z12",
       4187604:"Z13",
       2407146:"Z14",
       1307751:"Z15",
       944596:"Z22",
       936590:"Z23",
       1093224:"Z24",
       225326:"Z32",
       217575:"Z33",
       195328:"Z34",
       254338:"Z35",
       2268701:"T"
    }

    if num:
        detailLabelDict = detail_label_num
    else:
        detailLabelDict = detail_label_str

    iWeight=int(1e7*origWeight+0.5)
    detailLabel = detailLabelDict.get(iWeight, "W") # "W" is the default value if not found
    if detailLabel == "W" and (Label != 0 and Label != 'b') :
        raise ValueError("ERROR! if not in detailLabelDict sould have Label==1 ({}, {})".format(iWeight,Label))
    return detailLabel

def add_detail_label(data, num=True):
    """
    Add a 'detailLabel' column with the detailed labels.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        num: (default=True) if True use the numeric detail labels
                else use the string detail labels. You should prefer numeric labels.
    """
    if "origWeight" in data.columns:
        detailLabel = [getDetailLabel(w, label, num=num) for w, label in zip(data["origWeight"], data["Label"])]
    else:
        detailLabel = [getDetailLabel(w, label, num=num) for w, label in zip(data["Weight"], data["Label"])]
    data["detailLabel"] = detailLabel

# ==================================================================================

def bkg_weight_norm(data, systBkgNorm):
    """
    Apply a scaling to the weight.
    Keeps the previous weights in the 'origWeight' columns

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    TODO maybe explain why it scales only the background.
    """
    # only a weight manipulation
    data["origWeight"] = data["Weight"]
    if not "detailLabel" in data.columns:
        add_detail_label(data)
    # scale the weight, arbitrary but reasonable value
    data["Weight"] = ( data["Weight"]*systBkgNorm ).where(data["detailLabel"] == "W", other=data["origWeight"])

# ==================================================================================
# TES : Tau Energy Scale
# ==================================================================================
def tau_energy_scale(data, systTauEnergyScale, missing_value=-999.0):
    """
    Manipulate one primary input : the PRI_tau_pt and recompute the others values accordingly.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        systTauEnergyScale : the factor applied : PRI_tau_pt <-- PRI_tau_pt * systTauEnergyScale
        missing_value : (default=-999.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.

    Notes :
    -------
        Add 'ORIG_mass_MMC' and 'ORIG_sum_pt' columns.
        Recompute :
            - PRI_tau_pt
            - PRI_met
            - PRI_met_phi
            - DER_deltaeta_jet_jet
            - DER_mass_jet_jet
            - DER_prodeta_jet_jet
            - DER_lep_eta_centrality
            - DER_mass_transverse_met_lep
            - DER_mass_vis
            - DER_pt_h
            - DER_deltar_tau_lep
            - DER_pt_tot
            - DER_sum_pt
            - DER_pt_ratio_lep_tau
            - DER_met_phi_centrality
            - DER_mass_MMC
        Round up to 3 decimals.

    """
    # Keep some old values for some later computations
    data["ORIG_mass_MMC"] = data["DER_mass_MMC"]
    data["ORIG_sum_pt"] = data["DER_sum_pt"]

    # scale tau energy scale, arbitrary but reasonable value
    data["PRI_tau_pt"] *= systTauEnergyScale 

    # now recompute the DER quantities which are affected

    # first built 4-vectors
    vtau = V4() # tau 4-vector
    vtau.setPtEtaPhiM(data["PRI_tau_pt"], data["PRI_tau_eta"], data["PRI_tau_phi"], 0.8) # tau mass 0.8 like in original

    vlep = V4() # lepton 4-vector
    vlep.setPtEtaPhiM(data["PRI_lep_pt"], data["PRI_lep_eta"], data["PRI_lep_phi"], 0.) # lep mass 0 (either 0.106 or 0.0005 but info is lost)

    vmet = V4() # met 4-vector
    vmet.setPtEtaPhiM(data["PRI_met"], 0., data["PRI_met_phi"], 0.) # met mass zero,

    # fix MET according to tau pt change
    vtauDeltaMinus = vtau.copy()
    vtauDeltaMinus.scaleFixedM( (1.-systTauEnergyScale)/systTauEnergyScale )
    vmet += vtauDeltaMinus
    vmet.pz = 0.
    vmet.e = vmet.eWithM(0.)
    data["PRI_met"] = vmet.pt()
    data["PRI_met_phi"] = vmet.phi()

    # first jet if it exists
    vj1 = V4()
    vj1.setPtEtaPhiM(data["PRI_jet_leading_pt"].where( data["PRI_jet_num"] > 0, other=0 ),
                         data["PRI_jet_leading_eta"].where( data["PRI_jet_num"] > 0, other=0 ),
                         data["PRI_jet_leading_phi"].where( data["PRI_jet_num"] > 0, other=0 ),
                         0.) # zero mass

    # second jet if it exists
    vj2=V4()
    vj2.setPtEtaPhiM(data["PRI_jet_subleading_pt"].where( data["PRI_jet_num"] > 1, other=0 ),
                     data["PRI_jet_subleading_eta"].where( data["PRI_jet_num"] > 1, other=0 ),
                     data["PRI_jet_subleading_phi"].where( data["PRI_jet_num"] > 1, other=0 ),
                     0.) # zero mass

    vjsum = vj1 + vj2

    data["DER_deltaeta_jet_jet"] = vj1.deltaEta(vj2).where(data["PRI_jet_num"] > 1, other=missing_value)
    data["DER_mass_jet_jet"] = vjsum.m().where(data["PRI_jet_num"] > 1, other=missing_value)
    data["DER_prodeta_jet_jet"] = ( vj1.eta() * vj2.eta() ).where(data["PRI_jet_num"] > 1, other=missing_value)

    eta_centrality_tmp = eta_centrality(data["PRI_lep_eta"],
                                        data["PRI_jet_leading_eta"],
                                        data["PRI_jet_subleading_eta"])

    data["DER_lep_eta_centrality"] = eta_centrality_tmp.where(data["PRI_jet_num"] > 1, other=missing_value)

    # compute many vector sum
    vtransverse = V4()
    vtransverse.setPtEtaPhiM(vlep.pt(), 0., vlep.phi(), 0.) # just the transverse component of the lepton
    vtransverse += vmet
    data["DER_mass_transverse_met_lep"] = vtransverse.m()

    vltau = vlep + vtau # lep + tau
    data["DER_mass_vis"] = vltau.m()

    vlmet = vlep + vmet # lep + met # Seems to be unused ?
    vltaumet = vltau + vmet # lep + tau + met

    data["DER_pt_h"] = vltaumet.pt()

    data["DER_deltar_tau_lep"] = vtau.deltaR(vlep)

    vtot = vltaumet + vjsum
    data["DER_pt_tot"] = vtot.pt()

    data["DER_sum_pt"] = vlep.pt() + vtau.pt() + data["PRI_jet_all_pt"] # sum_pt is the scalar sum
    data["DER_pt_ratio_lep_tau"] = vlep.pt()/vtau.pt()


    data["DER_met_phi_centrality"] = METphi_centrality(data["PRI_lep_phi"], data["PRI_tau_phi"], data["PRI_met_phi"])

    # FIXME do not really recompute MMC, apply a simple scaling, better than nothing (but not MET dependence)
    rescaled_mass_MMC = data["ORIG_mass_MMC"] * data["DER_sum_pt"] / data["ORIG_sum_pt"]
    data["DER_mass_MMC"] = data["ORIG_mass_MMC"].where(data["ORIG_mass_MMC"] < 0, other=rescaled_mass_MMC)

    # delete non trivial objects to save memory (useful?)
    # del vtau, vlep, vmet, vlmet, vltau, vltaumet

    # Fix precision to 3 decimals
    DECIMALS = 3

    data["DER_mass_MMC"] = data["DER_mass_MMC"].round(decimals=DECIMALS)
    data["DER_mass_transverse_met_lep"] = data["DER_mass_transverse_met_lep"].round(decimals=DECIMALS)
    data["DER_mass_vis"] = data["DER_mass_vis"].round(decimals=DECIMALS)
    data["DER_pt_h"] = data["DER_pt_h"].round(decimals=DECIMALS)
    data["DER_deltaeta_jet_jet"] = data["DER_deltaeta_jet_jet"].round(decimals=DECIMALS)
    data["DER_mass_jet_jet"] = data["DER_mass_jet_jet"].round(decimals=DECIMALS)
    data["DER_prodeta_jet_jet"] = data["DER_prodeta_jet_jet"].round(decimals=DECIMALS)
    data["DER_deltar_tau_lep"] = data["DER_deltar_tau_lep"].round(decimals=DECIMALS)
    data["DER_pt_tot"] = data["DER_pt_tot"].round(decimals=DECIMALS)
    data["DER_sum_pt"] = data["DER_sum_pt"].round(decimals=DECIMALS)
    data["DER_pt_ratio_lep_tau"] = data["DER_pt_ratio_lep_tau"].round(decimals=DECIMALS)
    data["DER_met_phi_centrality"] = data["DER_met_phi_centrality"].round(decimals=DECIMALS)
    data["DER_lep_eta_centrality"] = data["DER_lep_eta_centrality"].round(decimals=DECIMALS)
    data["PRI_tau_pt"] = data["PRI_tau_pt"].round(decimals=DECIMALS)
    data["PRI_tau_eta"] = data["PRI_tau_eta"].round(decimals=DECIMALS)
    data["PRI_tau_phi"] = data["PRI_tau_phi"].round(decimals=DECIMALS)
    data["PRI_lep_pt"] = data["PRI_lep_pt"].round(decimals=DECIMALS)
    data["PRI_lep_eta"] = data["PRI_lep_eta"].round(decimals=DECIMALS)
    data["PRI_lep_phi"] = data["PRI_lep_phi"].round(decimals=DECIMALS)
    data["PRI_met"] = data["PRI_met"].round(decimals=DECIMALS)
    data["PRI_met_phi"] = data["PRI_met_phi"].round(decimals=DECIMALS)
    data["PRI_met_sumet"] = data["PRI_met_sumet"].round(decimals=DECIMALS)
    data["PRI_jet_leading_pt"] = data["PRI_jet_leading_pt"].round(decimals=DECIMALS)
    data["PRI_jet_leading_eta"] = data["PRI_jet_leading_eta"].round(decimals=DECIMALS)
    data["PRI_jet_leading_phi"] = data["PRI_jet_leading_phi"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_pt"] = data["PRI_jet_subleading_pt"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_eta"] = data["PRI_jet_subleading_eta"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_phi"] = data["PRI_jet_subleading_phi"].round(decimals=DECIMALS)
    data["PRI_jet_all_pt"] = data["PRI_jet_all_pt"].round(decimals=DECIMALS)


# ==================================================================================
#  NEW FEATURES : 
# ==================================================================================

def add_radian_1(data, inplace=True):
    """
    $ \min(\phi_{tau} - \phi_{lep}, \phi_{tau} - \phi_{met}, \phi_{lep} - \phi_{met}) $ (radian 1)
    """
    if not inplace:
        data = data.copy()
    tau_lep = (data['PRI_tau_phi'] - data['PRI_lep_phi'] + np.pi) % (2*np.pi) - np.pi
    tau_met = (data['PRI_tau_phi'] - data['PRI_met_phi'] + np.pi) % (2*np.pi) - np.pi
    lep_met = (data['PRI_lep_phi'] - data['PRI_met_phi'] + np.pi) % (2*np.pi) - np.pi
    all_ = pd.concat([tau_lep, tau_met, lep_met], axis=1)
    radian_1 = all_.min(axis=1)
    data['radian_1'] = radian_1
    return data

def add_radian_2(data, inplace=True):
    """
    $ \min(\phi_{tau} - \phi_{met}, \phi_{lep} - \phi_{met}) $ (radian 2)
    """
    if not inplace:
        data = data.copy()
    tau_met = (data['PRI_tau_phi'] - data['PRI_met_phi'] + np.pi) % (2*np.pi) - np.pi
    lep_met = (data['PRI_lep_phi'] - data['PRI_met_phi'] + np.pi) % (2*np.pi) - np.pi
    all_ = pd.concat([tau_met, lep_met], axis=1)
    radian_2 = all_.min(axis=1)
    data['radian_2'] = radian_2
    return data

def add_radian_3(data, inplace=True):
    """
    $ \min(\phi_{tau} - \phi_{lep}, \phi_{tau} - \phi_{met}) $ (radian 3)
    """
    if not inplace:
        data = data.copy()
    tau_lep = (data['PRI_tau_phi'] - data['PRI_lep_phi'] + np.pi) % (2*np.pi) - np.pi
    tau_met = (data['PRI_tau_phi'] - data['PRI_met_phi'] + np.pi) % (2*np.pi) - np.pi
    all_ = pd.concat([tau_lep, tau_met], axis=1)
    radian_3 = all_.min(axis=1)
    data['radian_3'] = radian_3
    return data

def add_radian_4(data, inplace=True):
    """
    $ \min(\phi_{tau} - \phi_{lep}, \phi_{lep} - \phi_{met}) $ (radian 4)
    """
    if not inplace:
        data = data.copy()
    tau_lep = (data['PRI_tau_phi'] - data['PRI_lep_phi'] + np.pi) % (2*np.pi) - np.pi
    lep_met = (data['PRI_lep_phi'] - data['PRI_met_phi'] + np.pi) % (2*np.pi) - np.pi
    all_ = pd.concat([tau_lep, lep_met], axis=1)
    radian_1 = all_.min(axis=1)
    data['radian_1'] = radian_1
    return data

def add_radians(data, inplace=True):
    """
    $ \min(\phi_{tau} - \phi_{lep}, \phi_{tau} - \phi_{met}, \phi_{lep} - \phi_{met}) $ (radian 1)
    $ \min(\phi_{tau} - \phi_{met}, \phi_{lep} - \phi_{met}) $ (radian 2)
    $ \min(\phi_{tau} - \phi_{lep}, \phi_{tau} - \phi_{met}) $ (radian 3)
    $ \min(\phi_{tau} - \phi_{lep}, \phi_{lep} - \phi_{met}) $ (radian 4)
    """
    if not inplace:
        data = data.copy()
    tau_lep = (data['PRI_tau_phi'] - data['PRI_lep_phi'] + np.pi) % (2*np.pi) - np.pi
    tau_met = (data['PRI_tau_phi'] - data['PRI_met_phi'] + np.pi) % (2*np.pi) - np.pi
    lep_met = (data['PRI_lep_phi'] - data['PRI_met_phi'] + np.pi) % (2*np.pi) - np.pi
    data['radian_1'] = pd.concat([tau_lep, tau_met, lep_met], axis=1).min(axis=1)
    data['radian_2'] = pd.concat([tau_met, lep_met], axis=1).min(axis=1)
    data['radian_3'] = pd.concat([tau_lep, tau_met], axis=1).min(axis=1)
    data['radian_4'] = pd.concat([tau_lep, lep_met], axis=1).min(axis=1)
    return data

# ==================================================================================
#  MAIN : here is defined the behaviour of this module as a main script
# ==================================================================================
import argparse

def parse_args():
    """
    ArgumentParser.

    Return
    ------
        args: the parsed arguments.
    """
    
    # First create a parser with a short description of the program.
    # The parser will automatically handle the usual stuff like the --help messages.
    parser = argparse.ArgumentParser(
        description="Higgs manipulation script. Can be used to produce new dataset with skewed features."
                    "Handle : Backgroung Weight norm (--Wnorm) and Tau energy scaling (--tes).\n"
                    "To get the behaviour of the previous version use :\n"
                    "--Wnorm 1 --tes 1.05 -r --csv -o atlas-higgs-challenge-2014-v2_v2.5_manip.csv")

    # ====================================================================================
    # Real things here
    # ====================================================================================

    parser.add_argument("--quiet", "-q", help="Verbosity level", action="store_true", dest='quiet')
    parser.add_argument("-r", "--restricted", 
        help="Option flag to write only the needed columns instead of keeping all the information."
        "Same behavior as previous version.",
        action="store_true", dest='restricted')
    parser.add_argument("--csv", help="Option flag to prevent compression into gzip file.",
        action="store_true", dest='csv')
    parser.add_argument("--float-label", help="Option flag to convert labels to float.",
        action="store_true", dest='float_label')
    parser.add_argument('--Wnorm', help='Backgroung Weight norm scale factor', type=float, dest='w_scale')
    parser.add_argument('--tes', help='Tau energy scale factor. Reasonable value [0.9, 1.1]', type=float, dest='tes')
    parser.add_argument('-i', help='the name of the input file', dest="in_file")
    parser.add_argument('-o', default="data/Higgs_output.csv", help='the name of the output file', dest="out_file")

    # Now do your job and parse my command line !
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    quiet = args.quiet # quiet flag
    restricted = args.restricted # restricted flag
    float_label = args.float_label # label to float flag
    csv = args.csv # csv flag
    w_scale = args.w_scale # Weight scaling factor
    tes = args.tes # Tau energy scale factor
    in_file = args.in_file # input file
    out_file = args.out_file # output file


    # Restrict the column output (for compatibility with previous version)
    columns = [ "EventId",
                "DER_mass_MMC",
                "DER_mass_transverse_met_lep",
                "DER_mass_vis",
                "DER_pt_h",
                "DER_deltaeta_jet_jet",
                "DER_mass_jet_jet",
                "DER_prodeta_jet_jet",
                "DER_deltar_tau_lep",
                "DER_pt_tot",
                "DER_sum_pt",
                "DER_pt_ratio_lep_tau",
                "DER_met_phi_centrality",
                "DER_lep_eta_centrality",
                "PRI_tau_pt",
                "PRI_tau_eta",
                "PRI_tau_phi",
                "PRI_lep_pt",
                "PRI_lep_eta",
                "PRI_lep_phi",
                "PRI_met",
                "PRI_met_phi",
                "PRI_met_sumet",
                "PRI_jet_num",
                "PRI_jet_leading_pt",
                "PRI_jet_leading_eta",
                "PRI_jet_leading_phi",
                "PRI_jet_subleading_pt",
                "PRI_jet_subleading_eta",
                "PRI_jet_subleading_phi",
                "PRI_jet_all_pt",
                "Weight",
                "Label",
                "origWeight",
                "detailLabel",
                ] if restricted else None

    if not quiet:
        print("Loading the dataset")

    data = load_higgs(in_file)

    if float_label:
        if not quiet:
            print("Label convertion to float")
        label_to_float(data)
    if w_scale is not None:
        if not quiet:
            print("Weight rescaling :", w_scale)
        bkg_weight_norm(data, w_scale)
    if tes is not None:
        if not quiet:
            print("Tau energy rescaling :", tes)
        tau_energy_scale(data, tes)

    compression = None if csv else "gzip"
    _, ext = os.path.splitext(out_file)
    if csv and ext != ".csv":
        out_file += ".csv"
    elif not csv and ext != ".gz":
        out_file += ".gz"
    if not quiet:
        print("Writting results to :", out_file)
    data.to_csv(out_file, index=False, compression=compression, columns=columns)
    print("Done.")
