#!/usr/bin/env python

'''
Usage based on https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb#scrollTo=si-w2NPretDg

A model trained to predict chromatin tates and gene expression directly from a DNA sequence as the only input.
The input sequence length is 393,216. The predicted output corresponds to 128 base pair windows (rows) for the center 114,688 base pairs.
The input sequence is one hot encoded using the order of indices corresponding to 'ACGT' with N values being all zeros.

Paper: https://www.nature.com/articles/s41592-021-01252-x

Check prediction tasks of interest in targets_mouse.txt and targets_human.txt

Requires around 10Gb of memmory.
Run inside Enformer conda environment created.

'''

######
### Load arguments
######

import sys, getopt

def main(argv):
   features = ''
   output = 'New'
   table = 0
   try:
      opts, args = getopt.getopt(argv,"hd:f:o:t:",["seq=","features=","output=","table="])
   except getopt.GetoptError:
      print('Run_Enformer_single_sequence.py -d <fasta seq file> -f <feature IDs from human and mouse table indexes to get predictions for [0,23,809]> -o <output> -t <save full table of predictions [0/1]>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Run_Enformer_single_sequence.py -d <fasta seq file> -f <feature IDs from human and mouse table indexes to get predictions for [0,23,809]> -o <output> -t <save full table of predictions [0/1]>')
         sys.exit()
      elif opt in ("-d", "--seq"):
         fasta_file = arg
      elif opt in ("-f", "--features"):
         features = arg
      elif opt in ("-o", "--output"):
         output = arg
      elif opt in ("-t", "--table"):
         table = arg
   if fasta_file=='': sys.exit("Input FASTA file not found")
   print('Input FASTA file is ', fasta_file)
   print('Feature IDs are ', features)
   print('Output is ', output)
   print('Save table: ', table)
   return fasta_file, features, output, table

if __name__ == "__main__":
   fasta_file, features, output, table = main(sys.argv[1:])

######
### Load libraries
######

import Enformer_functions
from Enformer_functions import *

import pyBigWig
import re
import os
import math
import subprocess

######
### Download model
######

print("\nDownload model ...\n")

transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
model = Enformer(model_path)

######
### Download target features
######

print("\nDownload target features ...\n")

# Cite: Kelley et al Cross-species regulatory sequence activity prediction. PLoS Comput. Biol. 16, e1008050 (2020).
human_targets = pd.read_csv('https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt', sep='\t') # human
mouse_targets = pd.read_csv('https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_mouse.txt', sep='\t') # mouse

df_targets = pd.concat([human_targets, mouse_targets])

######
### Parse input sequence
######

print("\nParse DNA sequences ...\n")

from Bio import SeqIO

sequence_one_hot = []
sequence_names = []
for record in SeqIO.parse(fasta_file, "fasta"):
    fasta_string = str(record.seq)
    
    # pad with NAs if sequence is smaller than Enformer's input (393216bp)
    if len(fasta_string) != SEQUENCE_LENGTH:
        print("\nSequence samller than Enformer input --> padding with Ns ...\n")
        add = (SEQUENCE_LENGTH - len(fasta_string)) / 2
        addL = math.ceil(add)
        addR = math.floor(add)
        fasta_string = "N"*addL + fasta_string + "N"*addR

    one_hot = one_hot_encode(fasta_string.upper())
    if one_hot.shape[0] != SEQUENCE_LENGTH: sys.exit("Sequence in input FASTA file not 393216bp (" + str(one_hot.shape[0]) + "bp)")

    # save
    sequence_one_hot.append(one_hot)
    sequence_names.append(str(record.id))

print(np.array(sequence_one_hot).shape)

######
### Make predictions
######

print("\nMake predictions ...\n")

# make predictions for specific sequences for all features
predictions_all = model.predict_on_batch(np.array(sequence_one_hot))

predictions = np.concatenate((predictions_all['human'], predictions_all['mouse']), axis=2)
print(predictions.shape)

### combine tables per sequence
df_list = []
for i in range(predictions.shape[0]):
    row_names = [sequence_names[i] + '_bin' + str(b) for b in range(predictions.shape[1])]
    tmp = pd.DataFrame(predictions[i,:,:], columns=df_targets.description, index=row_names)
    df_list.append(tmp)

### combine tables into final prediction table
df = pd.concat(df_list)
print(df.shape)

######
### Save predictions of requested features
######

print("\nSave predictions ...\n")

if features != '':
   # Extract IDs for features based on index
   IDs = [df_targets.loc[df_targets['index'] == int(i)].index.values[0] for i in features.split(",")]

   df_subset = df.iloc[:,IDs]
   df_subset.to_csv(output + '_features.csv', index=True)

######
### Save predictions of full table
######

if table == "1":
  df.to_csv(output + '_all_features.csv', index=True)
