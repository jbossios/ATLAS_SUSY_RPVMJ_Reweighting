#######################################################################
#                                                                     #
# Purpose: ROOT file -> H5 for dijet samples for reweighiting studies #
#                                                                     #
# Authour: Jona Bossio (jbossios@cern.ch)                             #
# Date:    24 Jan 2022                                                #
#                                                                     #
#######################################################################

import copy
import resource
import ROOT
import h5py
import os
import sys
import argparse
import logging
import numpy as np
import uproot

ROOT.gInterpreter.GenerateDictionary("vector<vector<float> >", "vector")
ROOT.gInterpreter.GenerateDictionary("vector<vector<int> >", "vector")

#############################
# Helpful functions
#############################

# Just-in-time compile custom helper functions
ROOT.gInterpreter.Declare("""
#include "HelperClasses.h"

using namespace ROOT::VecOps;
using Vec  = std::vector<iJet>;
using VecF = const RVec<float>&;
using VecI = const RVec<int>&;
using RVecf = RVec<float>;
using Vecf = std::vector<float>;
using Veci = std::vector<int>;

Vec SelectJets(VecF jetPt, VecF jetEta, VecF jetPhi, VecF jetE, VecI jetSig, VecI jetOR, VecI jetID, float minPt, float maxNjets){
 Vec Jets;
 unsigned int jetCounter = 0;
 for(auto ijet = 0; ijet < jetPt.size(); ijet++) {
   if(jetPt[ijet] >= minPt && jetSig[ijet] && jetOR[ijet] && jetCounter < maxNjets) {
     iJet TLVjet = iJet();
     TLVjet.SetPtEtaPhiE(jetPt[ijet],jetEta[ijet],jetPhi[ijet],jetE[ijet]);
     TLVjet.partonTruthLabelID = jetID[ijet];
     Jets.push_back(TLVjet);
     jetCounter++;
   }
 }
 if(jetCounter < maxNjets){ // add fake jets
   for(auto ijet = 0; ijet < (maxNjets-jetCounter); ijet++){
     iJet TLVjet = iJet();
     TLVjet.SetPtEtaPhiE(0,0,0,0);
     TLVjet.partonTruthLabelID = -1;
     Jets.push_back(TLVjet);
   }
 }
 return Jets;
}

Vec SelectQuarkJets(Vec jets){
  Vec Jets;
  for(auto ijet = 0; ijet < jets.size(); ijet++) {
    if(jets[ijet].partonTruthLabelID != -1 && jets[ijet].partonTruthLabelID != 21) {
      Jets.push_back(jets[ijet]);
    }
  }
  return Jets;
}

Veci GetMasks(Vec Jets){
  Veci JetMasks;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    if(Jets[ijet].Pt() == 0){
      JetMasks.push_back(0);
    } else {
      JetMasks.push_back(1);
    }
  }
  return JetMasks;
}

RVecf GetPtsRVec(Vec Jets){
  RVecf JetPts;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetPts.push_back(Jets[ijet].Pt());
  }
  return JetPts;
}
Vecf GetPts(Vec Jets){
  Vecf JetPts;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetPts.push_back(Jets[ijet].Pt());
  }
  return JetPts;
}
Vecf GetEtas(Vec Jets){
  Vecf JetEtas;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetEtas.push_back(Jets[ijet].Eta());
  }
  return JetEtas;
}
Vecf GetPhis(Vec Jets){
  Vecf JetPhis;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetPhis.push_back(Jets[ijet].Phi());
  }
  return JetPhis;
}
Vecf GetMasses(Vec Jets){
  Vecf JetMasses;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetMasses.push_back(Jets[ijet].M());
  }
  return JetMasses;
}
""")

def main(**kargs):

  ################################################################################
  # DO NOT MODIFY (below this line)
  ################################################################################

  # Global settings
  TreeName              = 'trees_SRRPV_'
  ApplyEventSelections  = True
  Debug                 = False
  MinNjets              = 1
  maxNjets              = 8
  minJetPt              = 20 # to be safe but there seems to be no jet below 20GeV

  # Create file with selected options
  Config = open('Options.txt','w')
  Config.write('ApplyEventSelections = {}\n'.format(ApplyEventSelections))
  Config.write('MinNjets             = {}\n'.format(MinNjets))
  Config.write('maxNjets             = {}\n'.format(maxNjets))
  Config.write('minJetPt             = {}\n'.format(minJetPt))
  Config.close()

  # Logger
  logging.basicConfig(format='%(levelname)s: %(message)s', level='INFO')
  log = logging.getLogger('')
  if Debug: log.setLevel("DEBUG")

  # Enable multithreading
  nthreads = 6
  if not kargs['debug']: ROOT.EnableImplicitMT(nthreads)

  ##############################################################################################
  # Find out how many events pass the event selections
  ##############################################################################################

  # Create TChain with input TTrees
  input_file = ROOT.TFile.Open(kargs['inputFile'])
  tree = input_file.Get(TreeName)
  log.info('{} events will be processed'.format(tree.GetEntries()))

  # Create RDataFrame
  DF = ROOT.RDataFrame(tree)
  if not DF:
    log.fatal('RDataFrame can not be created, exiting')
    sys.exit(1)
  if kargs['debug']: DF = DF.Range(100)

  # Select jets and discard events w/o enough passing jets
  DF = DF.Define('GoodJets', f'SelectJets(jet_pt, jet_eta, jet_phi, jet_e, jet_isSig, jet_passOR, jet_PartonTruthLabelID, {minJetPt}, {maxNjets})').Filter(f"GoodJets.size() >= {MinNjets}")
  DF = DF.Define('GoodQuarkJets', f'SelectQuarkJets(GoodJets)')
  DF = DF.Define('nQuarkJets', f'int(GoodQuarkJets.size())')
  DF = DF.Define('JetPts', f'GetPts(GoodJets)')
  DF = DF.Define('JetPtsRVec', f'GetPtsRVec(GoodJets)')
  DF = DF.Define('JetEtas', f'GetEtas(GoodJets)')
  DF = DF.Define('JetPhis', f'GetPhis(GoodJets)')
  DF = DF.Define('JetMasses', f'GetMasses(GoodJets)')
  DF = DF.Define('JetMasks', f'GetMasks(GoodJets)')
  DF = DF.Define('djmass', f'(GoodJets[0]+GoodJets[1]).M()')
  DF = DF.Define('deta', f'GoodJets[0].Eta()-GoodJets[1].Eta()')
  DF = DF.Define('HT', 'ROOT::VecOps::Sum(JetPtsRVec)')

  log.info('Get number of selected events')
  nPassingEvents = DF.Count().GetValue() # get number of selected events
  log.info(f'nPassingEvents = {nPassingEvents}')

  # Write a temporary ROOT file with per-jet info
  log.info('Writing temporary ROOT file ProcessedTTree_JetLevel.root...')
  columns = ['JetPts', 'JetEtas', 'JetPhis', 'JetMasses', 'JetMasks']
  DF.Snapshot('ProcessedTTree', 'ProcessedTTree_JetLevel.root', columns)

  # Write a temporary ROOT file with event-level info
  log.info('Writing temporary ROOT file ProcessedTTree_EventLevel.root...')
  columns = ['normweight', 'HT', 'nQuarkJets', 'djmass', 'deta']
  DF.Snapshot('ProcessedTTree', 'ProcessedTTree_EventLevel.root', columns)
  del tree
  input_file.Close()

  log.info('Reading ProcessedTTree_EventLevel.root...')
  input_file = uproot.open('ProcessedTTree_EventLevel.root')
  log.info('Getting TTree...')
  tree = input_file["ProcessedTTree"]
  log.info('Creating dataframe...')
  df = tree.pandas.df()
  log.info('Writing H5 file...')
  outFileName = '{}_{}_for_reweighting.h5'.format(kargs['sample'], kargs['dsid'])
  df['HT'].to_hdf(outFileName, key='HT', mode='w')
  for column in ['normweight', 'nQuarkJets', 'djmass', 'deta']:
    df[column].to_hdf(outFileName, key=column, mode='a')

  log.info('Reading ProcessedTTree_JetLevel.root...')
  input_file = uproot.open('ProcessedTTree_JetLevel.root')
  log.info('Getting TTree...')
  tree = input_file["ProcessedTTree"]
  log.debug(f'tree ={tree}')
  log.info('Creating dataframe...')
  df = tree.pandas.df()
  log.debug(f'df ={df}')
  for column in ['JetPts', 'JetEtas', 'JetPhis', 'JetMasses', 'JetMasks']:
    log.info(f'Writing {column} into the H5 file...')
    array = df[column].to_numpy()
    final_array = array.reshape(int(len(array)/maxNjets), maxNjets)
    with h5py.File(outFileName, 'a') as hf:
      Group = hf.create_group(column)
      Group.create_dataset(column, data = final_array)

  # Remove temporary ROOT files
  os.system('rm ProcessedTTree_JetLevel.root ProcessedTTree_EventLevel.root')

  # Close input file
  log.info('>>> ALL DONE <<<')

if __name__ == '__main__':
  datasets = { # dsid : path
    'JZAll' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/workshop/FT/mc16a/qcd_cleaned.root',
  }
  for dsid, input_file in datasets.items():
    main(sample = 'mc16a_dijets', inputFile = input_file, dsid = dsid, debug = False)

