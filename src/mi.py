#!/usr/bin/env python 
import numpy as np
import ctypes as c

__authors__ = "Gregory Ditzler" 
__copyright__ = "Copyright 2014, EESI Laboratory (Drexel University)"
__license__ = "GPL"
__maintainer__ = "Gregory Ditzler"
__email__ = "gregory.ditzler@gmail.com"


try:
  libMIToolbox = c.CDLL("libMIToolbox.so"); 
except:
  raise Exception("Error: could not load libMIToolbox.so")


def calc_mi(data, labels):
  """
  Calculate the Mutual Information 
  @data
  @labels

  @mutual_information
  """
  data, labels = __check_data__(data, labels)
  n_observations, n_features = data.shape
  c_n_observations = c.c_int(n_observations)
  output = []

  for c_data in data.transpose():
    libMIToolbox.calculateMutualInformation.restype = c.c_double 
    result = libMIToolbox.calculateMutualInformation(
        c_data.ctypes.data_as(c.POINTER(c.c_double)),
        labels.ctypes.data_as(c.POINTER(c.c_double)),
        c_n_observations)
    output.append(result.real)
  return np.array(output)

def calc_cmi(X, Y, Z):
  """
  Calculate Mutual Information: I(X;Y|Z) = H(X|Z) - H(X|YZ)
  @X
  @Y
  @Z

  @CMI
  """
  c_n_observations = c.c_int(len(X))
  libMIToolbox.calculateMutualInformation.restype = c.c_double
  result = libMIToolbox.calculateConditionalMutualInformation(
        X.ctypes.data_as(c.POINTER(c.c_double)),
        Y.ctypes.data_as(c.POINTER(c.c_double)),
        Z.ctypes.data_as(c.POINTER(c.c_double)),
        c_n_observations)
  return result.real

def mim(data, labels, n_select):
  """
  Mutual Information Maximization (MIM)
  @data
  @labels
  @n_select

  @selected_indices
  """
  return np.argsort(calc_mi(data, labels))[:n_select] 

def __check_data__(data, labels):
  """
    Check dimensions of the data and the labels.  Raise and exception
    if there is a problem.

    Data and Labels are automatically cast as doubles before calling the 
    feature selection functions

    @param data: the data 
    @param labels: the labels
    @return (data, labels): ndarray of floats
    @rtype: tuple
  """
  if isinstance(data, np.ndarray) is False:
    raise Exception("data must be an numpy ndarray.")
  if isinstance(labels, np.ndarray) is False:
    raise Exception("labels must be an numpy ndarray.")
  if len(data) != len(labels):
    raise Exception("data and labels must be the same length")
  return 1.0*data, 1.0*labels

