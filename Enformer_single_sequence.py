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
   interval='NA'
   output='New'
   table=0
   pltTracks=0
   bigwigs=0
   try:
      opts, args = getopt.getopt(argv,"hd:s:f:i:o:t:p:b:",["seq=","species=","features=","interval=","output=","table=","pltTracks=","bigwigs="])
   except getopt.GetoptError:
      print('Run_Enformer_single_sequence.py -d <fasta seq file [>=393216bp]> -s <species features [human/mouse]> -f <feature IDs from table indexes to get predictions for [0,23,809]> -i <interval [chr11:35082742-35197430, will be resized]> -o <output> -t <save full table of predictions [0/1]> -p <plot tracks [0/1]> -b <make bigwigs [0/1]>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Run_Enformer_single_sequence.py -d <fasta seq file [>=393216bp]> -s <species features [human/mouse]> -f <feature IDs from table indexes to get predictions for [0,23,809]> -i <interval [chr11:35082742-35197430, will be resized]> -o <output> -t <save full table of predictions [0/1]> -p <plot tracks [0/1]> -b <make bigwigs [0/1]>')
         sys.exit()
      elif opt in ("-d", "--seq"):
         fasta_file = arg
      elif opt in ("-s", "--species"):
         species = arg
      elif opt in ("-f", "--features"):
         features = arg
      elif opt in ("-i", "--interval"):
         interval = arg
      elif opt in ("-o", "--output"):
         output = arg
      elif opt in ("-t", "--table"):
         table = arg
      elif opt in ("-p", "--pltTracks"):
         pltTracks = arg
      elif opt in ("-b", "--bigwigs"):
         bigwigs = arg
   if fasta_file=='': sys.exit("Input FASTA file not found")
   if species=='': sys.exit("Species features not found")
   if features=='': sys.exit("Feature IDs not found")
   print('Input FASTA file is ', fasta_file)
   print('Species features is ', species)
   print('Feature IDs are ', features)
   print('Interval is ', interval)
   print('Output is ', output)
   print('Save table: ', table)
   print('Plot tracks: ', pltTracks)
   print('Bigwigs: ', bigwigs)
   return fasta_file, species, features, interval, output, table, pltTracks, bigwigs

if __name__ == "__main__":
   fasta_file, species, features, interval, output, table, pltTracks, bigwigs = main(sys.argv[1:])

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
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_' + species + '.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')

######
### If interval not provided, make predictions for input sequence - get sequence of input fasta file if 393216bp
######

if interval == 'NA':
    print("\nMake predictions for input sequence...\n")

    with open(fasta_file, "r") as f:
      fasta_content = f.read()

    fasta_string = "".join(fasta_content.split("\n")[1:]).upper()
    
    # pad with NAs if sequence is smaller than Enformer's input (393216bp)
    if len(fasta_string) != SEQUENCE_LENGTH:
      print("\nSequence samller than Enformer input --> padding with Ns ...\n")
      add = (SEQUENCE_LENGTH - len(fasta_string)) / 2
      addL = math.ceil(add)
      addR = math.floor(add)
      fasta_string = "N"*addL + fasta_string + "N"*addR

    sequence_one_hot = one_hot_encode(fasta_string)
    if sequence_one_hot.shape[0] != SEQUENCE_LENGTH: sys.exit("Input FASTA file not 393216bp (" + str(sequence_one_hot.shape[0]) + "bp)")

######
### Get sequence of genomic interval of the fasta file
######

if interval != 'NA':

    print("\nMake predictions for a genomic interval of the fasta file ...\n")

    chr, start, end = re.split('[:-]', interval)
    fasta_extractor = FastaStringExtractor(fasta_file)
    target_interval = kipoiseq.Interval(chr, int(start), int(end))
    target_interval_ext = target_interval.resize(SEQUENCE_LENGTH)

    sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval_ext))
    print(sequence_one_hot.shape)

######
### Make predictions
######

print("\nMake predictions ...\n")

# make predictions for specific sequence for all features
predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])[species][0] # 0 assuming it's only the first sequence
predictions.shape

# Extract IDs for features based on index
features_split = [int(i) for i in features.split(",")]
IDs = df_targets.loc[df_targets['index'].isin(features_split)].index
tracks = {}
for i in IDs:
    tracks[df_targets.description[i]] = predictions[:, i]
    if 'CAGE' in df_targets.description[i]: tracks[df_targets.description[i]] = np.log10(1 + predictions[:, i]) # log for CAGE

######
### Subset predictions of centred region?
######


######
### Save predictions of requested features
######

print("\nSave predictions ...\n")

# rownames
final_target_interval = target_interval.resize(114688)
chr_list = [chr] * predictions.shape[0]
start_list = [final_target_interval.start + 128 * i for i in range(predictions.shape[0])]
end_list = [i + 127 for i in start_list]
row_names = [str(chr_list[i] + '_' + str(start_list[i]) + '_' + str(end_list[i])) for i in range(len(chr_list))]

df = pd.DataFrame(predictions[:,IDs], columns=df_targets.description[IDs], index=row_names)
df.to_csv(output + '_' + species + '_features.csv', index=True)

######
### Save predictions of full table
######

if table == "1":
  df = pd.DataFrame(predictions, columns=df_targets.description, index=row_names)
  df.to_csv(output + '_' + species + '_all_features.csv', index=True)

######
### Plot tracks
######

# print("\nPlot tracks ...\n")

# if pltTracks == "1":
#     if interval == 'NA':
#       plot_tracks(tracks, start=-57344, end=57344, xtitle="") # predictions are for middle 114688 bp
#     if interval != 'NA':
#       target_interval_pred = target_interval.resize(114688)
#       plot_tracks(tracks, start=target_interval_pred.start, end=target_interval_pred.end, xtitle=target_interval)

#     plt.savefig(output + '_' + species + "_predicted_tracks.pdf", format='pdf')

######
### Contribution scores for every CAGE and plot tracks
######

# only if needed tracks or bigwigs
if pltTracks == "1" or bigwigs == "1":
   tracks = {}
   for ID in IDs:
      if 'CAGE' in df_targets.description[ID]:
         print("\nContribution scores for " + df_targets.description[ID] + " ... (takes some minutes)\n")
         
         target_mask = np.zeros_like(predictions)
         for idx in [447, 448, 449]: # these are the centered bins, about which we want to find the regions attentind to
            target_mask[idx, ID] = 1 # task of interest, usually CAGE
         
         # This will take some time since tf.function needs to get compiled.
         contribution_scores = model.contribution_input_grad(sequence_one_hot.astype(np.float32), target_mask, output_head=species).numpy()
         pooled_contribution_scores = tf.nn.avg_pool1d(np.abs(contribution_scores)[np.newaxis, :, np.newaxis], 128, 128, 'VALID')[0, :, 0].numpy()[1088:-1088]
         
         tracks[df_targets.description[ID]] = np.log10(1 + predictions[:, ID])
         tracks[df_targets.description[ID] + ' gradient*input'] = pooled_contribution_scores
         tracks[df_targets.description[ID] + ' gradient*input - y-axis trimmed'] = np.minimum(pooled_contribution_scores, 1)
      else:
         tracks[df_targets.description[ID]] = predictions[:, ID]

if pltTracks == "1":
    print("\nPlot tracks ...\n")
    
    if interval == 'NA':
      plot_tracks(tracks, start=-57344, end=57344, xtitle="Input sequence") # predictions are for middle 114688 bp
    if interval != 'NA':
      target_interval_pred = target_interval.resize(114688)
      plot_tracks(tracks, start=target_interval_pred.start, end=target_interval_pred.end, xtitle=target_interval)

    plt.savefig(output + '_' + species + "_predicted_and_contr_scores.pdf", format='pdf')

######
### Create bigwig files
######

if bigwigs == "1":

    print("\nCreating bigwigs ...\n")
    
    # Create directory for tracks if not exists
    directory = output + '_' + species + "_tracks/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # coordinates
    final_target_interval = target_interval.resize(114688)
    chr_list = [chr] * predictions.shape[0]
    start_list = [final_target_interval.start + 128 * i for i in range(predictions.shape[0])] # shapes for all tracks should be the same
    end_list = [i + 127 for i in start_list]

    for ID in tracks.keys():

        # scores
        values = tracks[ID]

        # save file
        bw = pyBigWig.open(str(directory + re.sub('[^A-Za-z0-9]+', '_', ID) + '.bw'), 'w')
        bw.addHeader([(chr, target_interval.end)])
        bw.addEntries(chr_list, start_list, ends=end_list, values=values)
        bw.close()

        print(ID, 'done')
