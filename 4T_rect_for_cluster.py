#!/usr/bin/env python
# coding: utf-8

from __future__ import division  # so that 1/2 == 0.5, and not 0
import warnings
warnings.filterwarnings('ignore')
import kwant
import tinyarray
from math import pi, sqrt, tanh 
import numpy as np
import scipy.sparse.linalg as sla 
from matplotlib import pyplot
from scipy import signal
import time
from IPython.display import Audio
sound_file = 'button.wav'
import math
from cmath import exp
from kwant.digest import gauss




#initialize the parameter space
class SimpleNamespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


#define the onsite and hopping terms
def hopping(sitei, sitej, phi, salt):
    xi, yi = sitei.pos
    xj, yj = sitej.pos
    return -exp(-0.5j * phi * (xi - xj) * (yi + yj))

def onsite(site, phi, salt):
    (x, y) = site.pos     
    # potential well

    t = np.linspace(-50, 50, 1000, endpoint=True)       
    square_wave = signal.square(0.06*t + 1.6)

    if square_wave[int((x+50)*10)] < 0:
        return 0
    else:
        return 0.3

# make a system including define shape, adding sites, adding leads
def make_system(L=50):
    def central_region(pos):
        x, y = pos
        return -L < x < L and -L/2 < y < L/2

    lat = kwant.lattice.honeycomb()
    a,b = lat.sublattices
    sys = kwant.Builder()

    sys[lat.shape(central_region, (0, 0))] = onsite
    sys[lat.neighbors()] = hopping

    pv1, pv2 = lat.prim_vecs
    symmetry = kwant.TranslationalSymmetry( -pv2)
    symmetry_neg = kwant.TranslationalSymmetry( +pv2)
    symmetry_pv1 = kwant.TranslationalSymmetry(pv1)
    symmetry_n_pv1 = kwant.TranslationalSymmetry(-pv1)

    # make a four lead system for hall measurement
    def create_lead_bot_left(symmetry):
        axis=(-5, 8)
        lead = kwant.Builder(symmetry)
        lead[lat.wire(axis, 10)] = 0
        lead[lat.neighbors()] = hopping
        return lead

    def create_lead_bot_right(symmetry):
        axis=(40, 2)
        lead = kwant.Builder(symmetry)
        lead[lat.wire(axis, 10)] = 0
        lead[lat.neighbors()] = hopping
        return lead

    def create_lead_top_left(symmetry_neg):
        axis=(-35, 8)
        lead = kwant.Builder(symmetry_neg)
        lead[lat.wire(axis, 10)] = 0
        lead[lat.neighbors()] = hopping
        return lead

    def create_lead_top_right(symmetry_neg):
        axis=(10, 2)
        lead = kwant.Builder(symmetry_neg)
        lead[lat.wire(axis, 10)] = 0
        lead[lat.neighbors()] = hopping
        return lead

    lead0= create_lead_bot_left(symmetry)
    sys.attach_lead(lead0)
    
    lead1= create_lead_bot_right(symmetry)
    sys.attach_lead(lead1)

    lead2= create_lead_top_left(symmetry_neg)
    sys.attach_lead(lead2)
    
    lead3= create_lead_top_right(symmetry_neg)
    sys.attach_lead(lead3)
    
    sym = kwant.TranslationalSymmetry((1, 0))
    sym.add_site_family(lat.sublattices[0], other_vectors=[(-1, 2)])
    sym.add_site_family(lat.sublattices[1], other_vectors=[(-1, 2)])

    lead = kwant.Builder(sym)

    def lead_shape(pos):
        x, y = pos
        return -L/2 < y < L/2

    lead[lat.shape(lead_shape, (0,0))] = 0
    lead[lat.neighbors()] = hopping
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())

    kwant.plot(sys)
    return sys.finalized()




#build the sys
sys = make_system()
kwant.plotter.bands(sys.leads[0], args=[1/40.0, ""])




# calculate the conductance from one lead to the other if there are more than two leads.
def calculate_sigmas(G):

    r = np.linalg.pinv(G)

    V = r.dot(np.array([-1, 1, 0, 0, 0, 0]))

    E_x_top = V[2] - V[3]
    E_x_left = V[4] - V[5]
    return E_x_top,E_x_left




# conductance as a function of magnetic field and energy. test for a smaller range
ts = time.time()

energy = -0.4
Bs = np.linspace(-0.04, .04, 200)

E_xx_top = []
E_xx_left = []
for B in Bs:
    s = kwant.smatrix(sys, energy,args=[B, ""])
    G = np.array([[s.transmission(i, j) for i in range(len(sys.leads))]
                  for j in range(len(sys.leads))])
    G -= np.diag(np.sum(G, 0))
    sigmas = calculate_sigmas(G)
    E_x_top = sigmas[0]
    E_x_left = sigmas[1]
    E_xx_top.append(E_x_top)
    E_xx_left.append(E_x_left)

pyplot.plot(Bs,E_xx_top)
pyplot.show()

pyplot.plot(Bs,E_xx_left)
pyplot.show() 
time.time() - ts 




# calculate for real system, conductance as a function of magnetic field and energy
energies = np.linspace(-0.3, -0.1, 400)
Bs = np.linspace(0, 1, 300)

conductances_B_E_top = []
conductances_B_E_left = []

for B in Bs:
    conductances_E_top = []
    conductances_E_left = []
    for en in energies:
        smatrix = kwant.smatrix(sys, en, args=[B, ""])
        G = np.array([[smatrix.transmission(i, j) for i in range(len(sys.leads))]
                      for j in range(len(sys.leads))])
        G -= np.diag(np.sum(G, 0))
        sigmas = calculate_sigmas(G)
        E_x_top = sigmas[0]
        E_x_left = sigmas[1]
        conductances_E_top.append(E_x_top)
        conductances_E_left.append(E_x_left)
    conductances_B_E_top.append(conductances_E_top)
    conductances_B_E_left.append(conductances_E_left)
conductances_B_E_top = np.array(conductances_B_E_top)
conductances_B_E_left = np.array(conductances_B_E_left)
print(conductances_B_E.shape)

x = energies
print(x.shape)
y = Bs
print(y.shape)
xv,yv = np.meshgrid(x,y)
pyplot.pcolormesh(x,y,conductances_B_E)
pyplot.show()
np.savetxt("conductances_E_B_4T_006well_0dot3P_Ndot3toNdot1_0to1_small", conductances_B_E, delimiter=",")







