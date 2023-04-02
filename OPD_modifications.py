"""Helper methods to create partitions of the ``Parkinsons Data Set'', such as the twinned method, to remove inflationary effects"""

from copy import copy
from itertools import chain
import random
import dataHandler as dh
import numpy as np


def OPD_match_female_participants():
    """Matching female participants by hand as there is not enough. This is only used when inflationary effects are removed"""
    #hand crafting female matches and then removing them. This is because there are so few female participants
    temp1 = ['S42','S50']
    random.shuffle(temp1)
    participant_folds = [['S10','S08'], ['S07','S26'], ['S17','S06'], [temp1[0],'S05'], [temp1[1],random.choice(['S34','S21'])]]

    return participant_folds
    
    
def OPD_match_male_participants(OPD_participants):
    participant_folds = []
    for _,row in dh.df_retrieve(OPD_participants,{'status':0,'Gender':'M'}).iterrows():

        age = row['Age']
        min_age = age - 3
        max_age = age + 3

        possible_choices = dh.df_retrieve(OPD_participants,{'Gender':'M','status':1})
        cond = (possible_choices['Age'] >= min_age) & (possible_choices['Age'] <= max_age)
        possible_choices = possible_choices[cond]
        match = possible_choices.sample(1)
        OPD_participants.drop(match.index,inplace=True)

        participant_folds.append([row['participant'],match.iloc[0]['participant']])
        
    return participant_folds
    
def kf(OPD_samples,OPD_participants):
    """After finding twins, partition them off in a k-fold CV fashion"""

    working_OPD_participants = copy(OPD_participants)
    
    female_participant_folds = OPD_match_female_participants()
    male_participant_folds = OPD_match_male_participants(working_OPD_participants)

    participant_folds = female_participant_folds + male_participant_folds 
    folds = []
        
    iter_i = np.random.permutation(len(participant_folds))
    
    for i in iter_i:
        test_participants = participant_folds[i]
        train_participants = [participant_folds[j] for j in iter_i[iter_i != i]]

        train_participants = list(chain.from_iterable(train_participants))

        train_index = np.where(OPD_samples['participant'].isin(train_participants))
        test_index = np.where(OPD_samples['participant'].isin(test_participants))
        
        folds.append((train_index,test_index))

    return folds
    
    
def split(OPD_samples,OPD_participants,training_split=0.7):
    """After finding twins, partition them off in a train-test fashion"""
        
    working_OPD_participants = copy(OPD_participants)
    female_participant_folds = OPD_match_female_participants()
    male_participant_folds = OPD_match_male_participants(working_OPD_participants)
    participant_folds = female_participant_folds + male_participant_folds 
    
    how_many = int(np.round(training_split*len(participant_folds)))
    iter_i = np.random.permutation(len(participant_folds))
    
    test_participants = [participant_folds[i] for i in iter_i[how_many:]]
    train_participants = [participant_folds[i] for i in iter_i[:how_many]]
    
    test_participants = list(chain.from_iterable(test_participants))
    train_participants = list(chain.from_iterable(train_participants))
    
    test_index = np.where(OPD_samples['participant'].isin(test_participants))
    train_index = np.where(OPD_samples['participant'].isin(train_participants))
    
    return train_index,test_index
