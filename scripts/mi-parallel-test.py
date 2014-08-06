#!/usr/bin/env python 


import sys
sys.path.append("../src/")
import numpy as np 
import bmu
import utils
import matplotlib.pylab as plt
import mi
import feast
import pickle

biom_fp = "../data/american-gut-mf.biom"
map_fp = "../data/american-gut-mf.txt"
n_select = 25


data, samples, features = bmu.load_biom(biom_fp)
map_data = bmu.load_map(map_fp)
labels, label_map = utils.label_formatting(map_data, samples, "SEX", 
    signed=False)
samples = np.array(samples)
features = np.array(features)
data = utils.normalize(data+1.)

mutual_info = mi.calc_mi(data=data, labels=labels)
si = np.array(sorted(range(len(mutual_info)), 
  key=lambda k: mutual_info[k])[::-1][:1000])

print "Running CMI"
cmi_mat = mi.cmi_matrix(data[:,si], labels, par=True, cpus=10)
output = {"cmi_mat":cmi_mat, "si":si, "mutual_info":mutual_info}
pickle.dump(output, open( "../files/cmi-mat.pkl", "wb" ) )

print "Running MI"
mi_mat = mi.mi_matrix(data[:,si], par=True, cpus=10)
output = {"mi_mat":mi_mat, "si":si, "mutual_info":mutual_info}
pickle.dump(output, open( "../files/mi-mat.pkl", "wb" ) )



