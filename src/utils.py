#!/usr/bin/env python 
import numpy as np 

__authors__ = "Gregory Ditzler" 
__copyright__ = "Copyright 2014, EESI Laboratory (Drexel University)"
__license__ = "GPL"
__maintainer__ = "Gregory Ditzler"
__email__ = "gregory.ditzler@gmail.com"

def label_formatting(map_data, sample_ids, label_field, signed=False):
  """
  @map_data: meta-data from biom utils 
  @sample_ids: sample ids from the biom file 
  @label_field: field in the map file to use as the label
  @signed: boolean [+1,-1]

  @labels_numeric: array of labels in numeric format
  @label_map: map from integers to the original labels (dictionary) 
  """
  lng_labels = []
  for key in map_data.keys():
    lng_labels.append(map_data[key][label_field])
  lng_labels = np.array(lng_labels)

  unique_labels = np.unique(lng_labels)
  label_map = {}
  if signed is True:
    if len(unique_labels) != 2:
      ValueError("Labels must only contain exactly two classes")
    for u_lab, u_int in map(None, unique_labels, [1.,-1.]):
      label_map[u_lab] = u_int
  else:
    for u_lab,u_int in map(None, unique_labels, range(len(unique_labels))):
      label_map[u_lab] = u_int

  labels_numeric = []
  for sid in sample_ids:
    labels_numeric.append( label_map[map_data[sid][label_field]] )
  labels_numeric = np.array(labels_numeric)
  return labels_numeric, label_map 

def normalize(data, scale=None):
  """
  to be filled in later
  """
  if scale == "log":
    data = numpy.ceil(np.log(data/np.min(data)))
  else:
    data = numpy.ceil(data/np.min(data))
  return None


def count2abun(count_matrix):
  """
  Convert X into a relative abundance matrix
  """
  scale_factor = count_matrix.sum(axis=1)
  return count_matrix/numpy.tile(scale_factor,
      [count_matrix.shape[1],1]).transpose()


