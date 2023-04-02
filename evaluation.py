"""Many helper methods that are used in conjunction with evaluate.py. This sets out each experiment, each with different conditions"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from os.path import join
import resultsHandler as rh
import dataHandler as dh
import random
import torch
import numpy as np
import OPD_modifications
import mPower_modifications
import generalisation_modifications
from evaluate import evaluate
from evaluate import evaluate_generalisation

def unmodified(dataset, samples, participants, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, seeds, repetitions=1, verbose_odds = 0.05, n_splits=10, training_split=0.7):
    """This can be used to evaluate both data sets with all three models for the base line performance. That is, when all inflationary effects are present."""
    def kf_method(X, y, samples, participants):
        return list(KFold(n_splits=n_splits, shuffle=True).split(X))
        
    test_size = 1.0 - training_split
    def split_method(X, y, samples, participants):
        return train_test_split(X, y, test_size=test_size)
    
    return evaluate(dataset, kf_method, split_method, samples, participants, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, repetitions, verbose_odds, seeds)


def modified_OPD(samples, participants, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, seeds, repetitions=1, verbose_odds = 0.05, n_splits=10, training_split=0.7):
    """This is used to evaluate the ``Parkinsons Data Set'' with all three models when the inflationary effects are removed. It works by using OPD_modifications.py to partition the data in such a way that the inflationary effects are removed, through the twinning method described in the manuscript"""
    
    dataset = 'OPD'
    
    def kf_method(X, y, samples, participants):
        return OPD_modifications.kf(samples,participants)
        
    def split_method(X, y, samples, participants):
        train_index, test_index = OPD_modifications.split(samples,participants,training_split=training_split)
        return X[train_index],X[test_index],y[train_index],y[test_index]
    
    return evaluate(dataset, kf_method, split_method, samples, participants, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, repetitions, verbose_odds, seeds)
    
def modified_mPower(samples, participants, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, seeds, repetitions=1, verbose_odds = 0.05, n_splits=10, training_split=0.7,age_range=3):
    """This is used to evaluate the ``mPower Data Set'' with all three models when the inflationary effects are removed. It works by using mPower_modifications.py to partition the data in such a way that the inflationary effects are removed, through the twinning method described in the manuscript"""

    dataset = 'mPower'

    def kf_method(X, y, samples, participants):
        return mPower_modifications.kf(samples,participants,age_range)
        
    def split_method(X, y, samples, participants):
        train_index, test_index = mPower_modifications.split(samples,participants,age_range)
        return X[train_index],X[test_index],y[train_index],y[test_index]
    
    return evaluate(dataset, kf_method, split_method, samples, participants, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, repetitions, verbose_odds, seeds)
    
    


def generalisationPerformance_twinned(OPD_samples,OPD_participants,mPower_samples, test_size, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, test_seeds, validation_repetitions=1, test_repetitions=10, verbose_odds = 0.05, n_splits=5, age_range=3):
    """This evalutes the generalisation performance using twinned data, that is, when all inflationary effects are removed. Note that it is not split into data sets because generalisation performance is evaluated with the union of both data sets"""
    dataset = 'mPower'

    def validation_method(X, y, samples, participants):
        return mPower_modifications.kf(samples,participants,age_range)
    
    def test_method(X, y, samples, participants):
        indices = list(range(len(samples)))
        train_index = indices[:-test_size]
        test_index = indices[-test_size:]
        s = samples.iloc[train_index]
        p = s.drop_duplicates(subset=['healthCode'])[['healthCode', 'age', 'diagnosis-year','gender','onset-year','professional-diagnosis']]
        matches,remaining_samples = mPower_modifications.create_twin_dataset(s,p,age_range)
            
        train_index = [samples.index.get_loc(ind) for ind in matches.index] 
        
        random.shuffle(train_index)
        random.shuffle(test_index)
        
        train_genders = samples.iloc[train_index]['gender']
        test_genders = samples.iloc[test_index]['gender']
#        print('male ratio: train',sum(train_genders=='Male'),len(train_genders),sum(train_genders=='Male')/len(train_genders))
#        print('male ratio: test',sum(test_genders=='Male'),len(test_genders),sum(test_genders=='Male')/len(test_genders))
        
        return X[train_index],X[test_index],y[train_index],y[test_index]
        
    repeated_test_results = rh.results_DataFrame()
    ozkan_test_results = rh.multiple_runs_DataFrame()
    caliskan_test_results = rh.multiple_runs_DataFrame()
    ulhaq_test_results = rh.multiple_runs_DataFrame()
    
    return evaluate_generalisation(dataset, OPD_samples, OPD_participants, mPower_samples, test_size, validation_method, test_method, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, test_seeds,validation_repetitions,test_repetitions, verbose_odds)




def generalisationPerformance_unmodified(OPD_samples,OPD_participants,mPower_samples, test_size, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, test_seeds, validation_repetitions=1, test_repetitions=10, verbose_odds = 0.05, n_splits=5):
    """This evalutes the generalisation performance using unmodified training data, that is, when all inflationary effects are present. Note that it is not split into data sets because generalisation performance is evaluated with the union of both data sets"""

    dataset = 'mPower'
        
    def validation_method(X, y, samples, participants):
        return list(KFold(n_splits=n_splits, shuffle=True).split(X))

    def test_method(X, y, samples, participants):
        indices = list(range(len(samples)))
        train_index = indices[:-test_size]
        test_index = indices[-test_size:]
        random.shuffle(train_index)
        random.shuffle(test_index)
        return X[train_index],X[test_index],y[train_index],y[test_index]
                
    return evaluate_generalisation(dataset, OPD_samples, OPD_participants, mPower_samples, test_size, validation_method, test_method, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, test_seeds,validation_repetitions,test_repetitions, verbose_odds)



    
