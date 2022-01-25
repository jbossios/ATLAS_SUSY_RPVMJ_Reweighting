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
using Vecf = RVec<float>;

Vec SelectJets(VecF jetPt, VecF jetEta, VecF jetPhi, VecF jetE, VecI jetSig, VecI jetOR, VecI jetID, float minPt){
 Vec Jets;
 for(auto ijet = 0; ijet < jetPt.size(); ijet++) {
   if(jetPt[ijet] >= minPt && jetSig[ijet] && jetOR[ijet]) {
     iJet TLVjet = iJet();
     TLVjet.SetPtEtaPhiE(jetPt[ijet],jetEta[ijet],jetPhi[ijet],jetE[ijet]);
     TLVjet.partonTruthLabelID = jetID[ijet];
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

//Vec SelectQuarkJets(VecF jetPt, VecF jetEta, VecF jetPhi, VecF jetE, VecI jetID){
// Vec Jets;
// for(auto ijet = 0; ijet < jetID.size(); ijet++) {
//   if(jetID[ijet] != -1 && jetID[ijet] != 21) {
//     iJet TLVjet = iJet();
//     TLVjet.SetPtEtaPhiE(jetPt[ijet],jetEta[ijet],jetPhi[ijet],jetE[ijet]);
//     TLVjet.partonTruthLabelID = jetID[ijet];
//     Jets.push_back(TLVjet);
//   }
// }
// return Jets;
//}

Vecf GetPts(Vec Jets){
  Vecf JetPts;
  for(auto ijet = 0; ijet < Jets.size(); ijet++) {
    JetPts.push_back(Jets[ijet].Pt());
  }
  return JetPts;
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
  maxNjets              = 20
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
  DF = DF.Define('GoodJets', f'SelectJets(jet_pt, jet_eta, jet_phi, jet_e, jet_isSig, jet_passOR, jet_PartonTruthLabelID, {minJetPt})').Filter(f"GoodJets.size() >= {MinNjets}")
  DF = DF.Define('GoodQuarkJets', f'SelectQuarkJets(GoodJets)')
  DF = DF.Define('ZeroQuarkJetsFlag', 'int(GoodQuarkJets.size() == 0)')
  DF = DF.Define('JetPts', f'GetPts(GoodJets)')
  DF = DF.Define('HT', 'ROOT::VecOps::Sum(JetPts)')
  log.info('Get number of selected events')
  nPassingEvents = DF.Count().GetValue() # get number of selected events
  log.info(f'{nPassingEvents = }')

  ##############################################################################################
  # Create output H5 file
  ##############################################################################################

  outFileName = '{}_{}_for_reweighting.h5'.format(kargs['sample'], kargs['dsid'])
  columns = ['normweight', 'HT', 'ZeroQuarkJetsFlag']
  data = DF.AsNumpy(columns = columns)
  with h5py.File(outFileName, 'w') as hf:
    Group  = hf.create_group('data')
    for column in columns:
      Group.create_dataset(column, data = data[column])

  # Close input file
  input_file.Close()
  log.info('>>> ALL DONE <<<')

if __name__ == '__main__':
  datasets = { # dsid : path
    'JZAll' : '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/workshop/FT/mc16a/qcd_cleaned.root',
  }
  for dsid, input_file in datasets.items():
    main(sample = 'mc16a_dijets', inputFile = input_file, dsid = dsid, debug = False)

