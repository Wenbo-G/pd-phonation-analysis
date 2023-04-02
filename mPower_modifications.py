"""Helper methods to perform the twinning method on the ``mPower Data Set'', so that inflationary effects can be removed."""

import itertools
import plotly.figure_factory as ff
import math
import collections
from copy import copy
import pandas as pd
import numpy as np
import dataHandler as dh
import random

def kf(mPower_samples, mPower_participants, age_range=3):
    """This is one of the few functions you will interact with. This simply asks to create a twinned data set from the given mPower dataset, then create 5 paritions of it that are all similarly distributed"""
    
    matches,remaining_samples = create_twin_dataset(mPower_samples,mPower_participants,age_range)
    
    pdm_folds, cm_folds, pdf_folds, cf_folds = create_gender_class_folds(matches,k=5)
    
    best_combination_submissions, best_combination_indices = find_best_fold_combination(pdm_folds,cm_folds,pdf_folds,cf_folds)
    
    fold_indices = get_indices(mPower_samples,matches,best_combination_indices,best_combination_submissions,pdm_folds,cm_folds,pdf_folds,cf_folds)
    
    for f in fold_indices:
        train_genders = mPower_samples.iloc[f[0]]['gender']
        test_genders = mPower_samples.iloc[f[1]]['gender']
    
    return fold_indices
    
    
def split(mPower_samples, mPower_participants, age_range=3):
    """This uses the kfold method and creates 7 splits, then finds a combination of 5 of the splits that roughly equals 70% of the total submissions."""
    
    matches,remaining_samples = create_twin_dataset(mPower_samples,mPower_participants,age_range)
    
    pdm_folds, cm_folds, pdf_folds, cf_folds = create_gender_class_folds(matches,k=7)

    best_combination_indices, best_combination_submissions, solution = create_splits(pdm_folds,cm_folds,pdf_folds,cf_folds)
    fold_indices = get_indices(mPower_samples,matches,best_combination_indices,best_combination_submissions,pdm_folds,cm_folds,pdf_folds,cf_folds,split_folds=solution)
    train_index, test_index = fold_indices
    return train_index, test_index

def create_gender_class_folds(matches,k):
    pdm = dh.df_retrieve(matches,{'gender':'Male','professional-diagnosis':True})
    cm = dh.df_retrieve(matches,{'gender':'Male','professional-diagnosis':False})
    pdf = dh.df_retrieve(matches,{'gender':'Female','professional-diagnosis':True})
    cf = dh.df_retrieve(matches,{'gender':'Female','professional-diagnosis':False})
    pdm_folds = create_k_folds(pdm,k=k)
    cm_folds = create_k_folds(cm,k=k)
    pdf_folds = create_k_folds(pdf,k=k)
    cf_folds = create_k_folds(cf,k=k)
    
    return pdm_folds, cm_folds, pdf_folds, cf_folds
    

def create_twin_dataset(mPower_samples,mPower_participants,age_range):
    mPower_participants.set_index(mPower_participants['healthCode'],inplace=True)
    mPower_participants.index.name = 'Index'

    mPower_participants = restrict_by_age(mPower_participants,age_range)
    mPower_samples = restrict_by_age(mPower_samples,age_range)
    
    healthCodes = mPower_samples['healthCode'].unique() 
    working_samples = copy(mPower_samples)
    mapping = {}

    for healthCode in healthCodes: 
        mapping[healthCode] = sum(mPower_samples['healthCode'] == healthCode)
        
    working_samples['number of submissions'] = working_samples['healthCode'].map(mapping)   

    matches4, working_samples = findMatches(working_samples,mPower_participants,age_range,'Female',False, True)
    matches3, working_samples = findMatches(working_samples,mPower_participants,age_range,'Female',True, True)
    matches2, working_samples = findMatches(working_samples,mPower_participants,age_range,'Male',False, True)
    matches1, working_samples = findMatches(working_samples,mPower_participants,age_range,'Male',True, True)

    matches = pd.DataFrame(columns = matches4.columns)
    matches = matches.append(matches4)
    matches = matches.append(matches3)
    matches = matches.append(matches2)
    matches = matches.append(matches1)
    
    return matches, working_samples


def findMatches(samples, participants, age_range, gender, professional_diagnosis, sort):
    
    matches = pd.DataFrame(columns=samples.columns)
        
    sample_subset = dh.df_retrieve(samples,{'gender':gender,'professional-diagnosis':professional_diagnosis})
    sample_subset = sample_subset.sample(frac=1)
    if sort: sample_subset = sample_subset.sort_values(by='age',ascending=True)
        
        
    healthCodes = sample_subset['healthCode'].unique()

    for healthCode in healthCodes:
        age = participants.loc[healthCode,'age']
        number_of_submissions = sum(sample_subset['healthCode'] == healthCode)
        
        matches, samples = pickMatches(healthCode,age,gender,professional_diagnosis,number_of_submissions,matches,samples,age_range)
        
    return matches, samples

    
def pickMatches(healthCode,age,gender,professional_diagnosis,number_of_submissions,matches,samples,age_range):#,twin_candidates,how_many):

    for age_diff in range(age_range+1):
        candidates = dh.df_retrieve(samples,{'gender':gender,'professional-diagnosis':not professional_diagnosis})
        age_condition = (candidates['age'] <= age + age_diff) & (candidates['age'] >= age - age_diff)
        twin_candidates = candidates[age_condition]
        
        if len(twin_candidates) > 0:
            
            number_of_candidates = len(twin_candidates)
            
            if number_of_candidates < number_of_submissions:
                #Not enough candidates, take all candidates and reduce number of submissions by corresponding amount
                #Will then expand the age condition and look again.
                
                matched_samples = twin_candidates
                participant_submissions = dh.df_retrieve(samples,{'healthCode':healthCode}).sample(n=number_of_candidates)
                number_of_submissions -= number_of_candidates
                
                matches = matches.append(matched_samples)
                matches = matches.append(participant_submissions)
                
                samples.drop(participant_submissions.index,inplace=True)
                samples.drop(matched_samples.index,inplace=True)
                
            else:
                #More than enough candidates, or exactly the same candidates and submissions. 
                #Either way, samples from candidates and match with all submissions, then removes both sets
                
                sampling_weights = 1/twin_candidates['number of submissions']
                matched_samples = twin_candidates.sample(n=number_of_submissions,weights=sampling_weights)
                participant_submissions = dh.df_retrieve(samples,{'healthCode':healthCode})
                
                matches = matches.append(matched_samples)
                matches = matches.append(participant_submissions)
                
                samples.drop(participant_submissions.index,inplace=True)
                samples.drop(matched_samples.index,inplace=True)
                
                return matches,samples
    return matches,samples
            

def restrict_by_age(df,age_range):
    """Restricts the dataframe by age, removing all submissions from those too old or too young to have a PD/Control twin, given an age_range
    """    
    df = copy(df) #Prevents inplace changes to mPower data set
    cf = dh.df_retrieve(df,{'gender':'Female','professional-diagnosis':False})
    pdf = dh.df_retrieve(df,{'gender':'Female','professional-diagnosis':True})
    
    max_age_cf = cf['age'].max()
    min_age_cf = cf['age'].min()
    max_age_pdf = pdf['age'].max()
    min_age_pdf = pdf['age'].min()

    min_age_f = max(min_age_cf,min_age_pdf) - age_range
    max_age_f = min(max_age_cf,max_age_pdf) + age_range

    cond1 = (df['gender'] == 'Female') & (df['age'] < min_age_f)
    cond2 = (df['gender'] == 'Female') & (df['age'] > max_age_f)
    to_remove = df[cond1 | cond2]
    df.drop(to_remove.index,inplace=True)

    
    cm = dh.df_retrieve(df,{'gender':'Male','professional-diagnosis':False})
    pdm = dh.df_retrieve(df,{'gender':'Male','professional-diagnosis':True})

    max_age_cm = cm['age'].max()
    min_age_cm = cm['age'].min()
    max_age_pdm = pdf['age'].max()
    min_age_pdm = pdf['age'].min()

    min_age_m = max(min_age_cm,min_age_pdm) - age_range
    max_age_m = min(max_age_cm,max_age_pdm) + age_range

    cond1 = (df['gender'] == 'Male') & (df['age'] < min_age_m)
    cond2 = (df['gender'] == 'Male') & (df['age'] > max_age_m)
    to_remove = df[cond1 | cond2]
    df.drop(to_remove.index,inplace=True)
    
    return df




#######################



def create_k_folds(samples, k, verbose=False):
    
    working_participants = pd.DataFrame(columns = ['age','number of submissions'])
    working_participants.index.name = 'healthCode'

    healthCodes = samples['healthCode'].unique()
    for healthCode in healthCodes:
        cond = samples['healthCode']==healthCode
        number_of_submissions = sum(cond)
        age = samples[cond]['age'].iloc[0] #this is a series of many ages, but it should all be the same anyways
        working_participants.loc[healthCode] = {'age': age, 'number of submissions': number_of_submissions}

    participant_folds, working_participants = initialise_participant_folds(k, working_participants)

    target_submissions = math.floor(len(samples)/k)

    temp = []
    for index,row in working_participants.iterrows():
        number_of_submissions = row['number of submissions']
        healthCode = index
        temp.append({'number of submissions': number_of_submissions,'healthCode': healthCode})

        if len(temp) >= k:
            weights = np.array(participant_folds['total submissions'])
            weights = target_submissions-weights
            weights = weights.clip(min=0)
            weights = weights**2
            
            
            temp = weighted_reorder(temp,weights)
            participant_folds = increment_participant_folds(participant_folds,temp)
            temp = []

    if len(temp) > 0:
        temp = sorted(temp, key=lambda k: k['number of submissions'],reverse=True) 
        for t in temp:
            idx_min = participant_folds['total submissions'].idxmin()
            participant_folds.at[idx_min,'total submissions'] += t['number of submissions']
            participant_folds.at[idx_min,'participants'].append(t['healthCode'])
            
          
    if verbose:    
        all_ages = []
        
        for i in range(k):
            ages = []
            for healthCode in participant_folds.iloc[i]['participants']: ages += samples[samples['healthCode'] == healthCode]['age'].tolist()
            all_ages.append(ages)
            
        group_labels = list(range(1,len(all_ages)+1))
        fig = ff.create_distplot(all_ages, group_labels, bin_size=1,show_hist=False)
        fig.show()
        
    return participant_folds
    
def initialise_participant_folds(k,participants):
    """This method is simply because there are some participants that make the whole process harder. 
    
    To address this, we pre-populate the participant folds so that these participants are handled first and so other participants can be allocated accordingly"""
    
    participant_folds = pd.DataFrame(columns=['total submissions','participants'])
    for i in range(k): #initialise participant_folds
        participant_folds.at[i,'total submissions'] = 0
        participant_folds.at[i,'participants'] = []
    participant_folds['total submissions'] = participant_folds['total submissions'].astype(int)
    
    
    problem_healthCodes = ['bae1bf32-94bf-42a7-96d0-ee23fd98245e',
                           '0373e041-80ad-41cf-ba5f-e3b4a1d27e54',]
    
    for problem_healthCode in problem_healthCodes:
        if problem_healthCode in participants.index:
            inds = participant_folds[participant_folds['total submissions'] == 0].index #can only add to folds with no submissions already
            ind = random.choice(inds)
            participant_folds.at[ind,'total submissions'] += participants.loc[problem_healthCode,'number of submissions']
            participant_folds.at[ind,'participants'].append(problem_healthCode)
            participants.drop(problem_healthCode,inplace=True)
            
    return participant_folds, participants
    
def weighted_take(series,weights):
    indices = list(range(len(series)))
    index = random.choices(indices,weights=weights)[0]
    choice = series[index]
    series.pop(index)
    weights = np.delete(weights,index)
    
    return choice,series,weights
    
def weighted_reorder(series,weights):
    temp_series = sorted(series, key=lambda k: k['number of submissions'],reverse=True) 
    ind_choices = list(range(len(temp_series)))
    for temp in temp_series:   
        ind, ind_choices, weights = weighted_take(ind_choices,weights)
        series[ind] = temp

    return series
    
def increment_participant_folds(participant_folds, series):
    
    for i in participant_folds.index:
        participant_folds.at[i,'total submissions'] += series[i]['number of submissions']
        participant_folds.at[i,'participants'].append(series[i]['healthCode'])
    return participant_folds





####################



def create_splits(folds1,folds2,folds3,folds4):
    
    total_submissions = sum(folds1['total submissions']) + sum(folds2['total submissions']) + sum(folds3['total submissions']) + sum(folds4['total submissions'])
    target = math.floor(total_submissions*0.7)
    
    combination_indices, combination_submissions = get_all_fold_combinations(folds1,folds2,folds3,folds4)

    split_combination_submissions = []
    for submissions in combination_submissions:
        training_combination,_ = solve(submissions,target,r=5)
        split_combination_submissions.append(training_combination)
        
    distance = (np.array(split_combination_submissions) - target) ** 2
    minidx = np.argmin(distance)
        
    best_combination_indices = combination_indices[minidx]
    best_combination_submissions = combination_submissions[minidx]
    _,solution = solve(best_combination_submissions,target,r=5)
    solution = [best_combination_submissions.index(s) for s in solution] 
    
    return best_combination_indices, best_combination_submissions, solution
    

def solve(series,target,r):
    combinations = np.array(list(itertools.combinations(series,r=r)))
    sums = combinations.sum(axis=1)
    distance = (sums-target)**2
    minidx = distance.argmin()
    return sums[minidx],combinations[minidx]
    
    
def find_best_fold_combination(folds1,folds2,folds3,folds4):
    """Best here is defined as the combination of folds that results in folds that are very similar in total number of samples"""
    
    combination_indices, combination_submissions = get_all_fold_combinations(folds1,folds2,folds3,folds4)

    stds = np.array(combination_submissions).std(axis=1)
    minidx = stds.argmin()
    best_combination_submissions = combination_submissions[minidx]
    best_combination_indices = combination_indices[minidx]
    
    return best_combination_submissions, best_combination_indices


def get_all_fold_combinations(folds1,folds2,folds3,folds4):

    index1 = collections.deque(folds1.index)
    index2 = collections.deque(folds2.index)
    index3 = collections.deque(folds3.index)
    index4 = collections.deque(folds4.index)

    combination_indices = []
    combination_submissions = []

    for _ in range(len(index1)):
        for _ in range(len(index2)):
            for _ in range(len(index3)):
                for _ in range(len(index4)):

                    combination_index = list(zip(index1,index2,index3,index4))
                    combination_indices.append(combination_index)
                    submissions = []
                    for index in combination_index:
                        total_submissions = 0
                        total_submissions += folds1.loc[index[0],'total submissions'] 
                        total_submissions += folds2.loc[index[1],'total submissions'] 
                        total_submissions += folds3.loc[index[2],'total submissions'] 
                        total_submissions += folds4.loc[index[3],'total submissions']
                        submissions.append(total_submissions)
                    combination_submissions.append(submissions)                    
                    index4.rotate(1)
                index3.rotate(1)
            index2.rotate(1)
        index1.rotate(1)
        
    return combination_indices, combination_submissions
    

####################
def get_indices(mPower_samples,matches,best_combination_indices,best_combination_submissions,pdm_folds,cm_folds,pdf_folds,cf_folds,split_folds=None):
    """This method takes in the best combination of folds, and the folds themselves which consist of the participants, and retrieves the indices of the corresponding submissions from those participants, as they are indexed in mPower_samples
    """
    samples = copy(mPower_samples[['healthCode','recordId','age','professional-diagnosis']]) #anything else?
    
    mapping = {}
    temp = pd.DataFrame(columns=['fold'],index=samples['healthCode'].unique()) #a dataframe with all the healthcodes and what fold they belong to 
    temp.index.name = 'healthCode'
        
    for i, indices in enumerate(best_combination_indices):
        if split_folds is None: which_fold = i
        else: which_fold = 0 if i in split_folds else 1
            
        mapping.update(dict.fromkeys(pdm_folds.loc[indices[0],'participants'],which_fold))
        mapping.update(dict.fromkeys(cm_folds.loc[indices[1],'participants'],which_fold))
        mapping.update(dict.fromkeys(pdf_folds.loc[indices[2],'participants'],which_fold))
        mapping.update(dict.fromkeys(cf_folds.loc[indices[3],'participants'],which_fold))
    
    
    matches['fold'] = matches['healthCode'].map(mapping)
    mapping2 = {key:val for key,val in list(zip(matches['recordId'],matches['fold']))}
    samples['fold'] = samples['recordId'].map(mapping2)
    
    fold_indices = []
    
    for which_fold in matches['fold'].unique():
        fold_index = np.where(samples['fold']==which_fold)[0]
        fold_indices.append(fold_index)
        
    if split_folds is not None: return fold_indices
    
    iter_i = np.random.permutation(len(fold_indices))
    
    folds = []
    for i in iter_i:
        test_indices = fold_indices[i]
        train_indices = [fold_indices[j] for j in iter_i[iter_i != i]]
        train_indices = list(itertools.chain.from_iterable(train_indices))
        folds.append((train_indices,test_indices))
    
    
    return folds




    


    


    
