import dataHandler as dh
import pandas as pd
import random
import numpy as np
import math

def sample_test_set(OPD_samples,OPD_participants,mPower_samples,OPD_PD_sample_size,OPD_C_sample_size,mPower_PD_sample_size,mPower_C_sample_size,seed):
    """This is the helper method to combine the ``Parkinsons Data Set'' and the ``mPower Data Set'' and get it ready for testing generalisation performance. It returns the training samples and the testing samples"""
    random.seed(seed)
    np.random.seed(seed)

    OPD_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 
                'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
                'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    mPower_features = ['Mean pitch (Hz)','Minimum pitch (Hz)','Maximum pitch (Hz)','Jitter (local) (%)','Jitter (local, absolute)','Jitter (rap) (%)','Jitter (ppq5) (%)','Jitter (ddp) (%)',
                'Shimmer (local) (%)','Shimmer (local, dB) (dB)','Shimmer (apq3) (%)','Shimmer (apq5) (%)','Shimmer (apq11) (%)','Shimmer (dda) (%)',
                'Mean noise-to-harmonics ratio','Mean harmonics-to-noise ratio (dB)','spread1 (negative entropy of F0)','spread2 (standard error of F0)','PPE','DFA','RPDE','d2']
        
    mapping = {key:value for key,value in zip(OPD_features,mPower_features)}
    mapping['status'] = 'professional-diagnosis'
    mapping['sample'] = 'recordId'
    mapping['participant'] = 'healthCode'
    mapping['Gender'] = 'gender'
    mapping['Age'] = 'age'
    mapping['Years after diagnosis'] = 'diagnosis-year'


    samples = OPD_samples.merge(OPD_participants[['participant','Gender','Age','Years after diagnosis']],on='participant')
    samples['sample'] = [ '%s_%i'%(p,s) for p,s in zip(samples['participant'],samples['sample'])]
    samples['status'] = [bool(a) for a in samples['status']]
    samples['Gender'] = ['Male' if g =='M' else 'Female' for g in samples['Gender']]
    
    
    samples.columns = samples.columns.map(mapping)
    samples = samples[samples.columns.dropna()]
    samples = mPower_samples.append(samples,ignore_index=True)

    participants = samples.drop_duplicates(subset=['healthCode'])[['healthCode', 'age', 'diagnosis-year','gender','onset-year','professional-diagnosis']]

    #OPD sample sizes are assumed to be even
    OPD_PD_male_sample_size = OPD_PD_female_sample_size = int(OPD_PD_sample_size/2) #Ideally would use a ratio of 1.5 to 1 of male to female, but too few samples to do
    OPD_C_male_sample_size = OPD_C_female_sample_size = int(OPD_C_sample_size/2)

    
    OPD_PD_male_participants = dh.df_retrieve(participants.iloc[-31:],{'professional-diagnosis':True,'gender':'Male'}) #The last 32 participants are from OPD
    OPD_C_male_participants = dh.df_retrieve(participants.iloc[-31:],{'professional-diagnosis':False,'gender':'Male'}) #The last 32 participants are from OPD
    OPD_PD_female_participants = dh.df_retrieve(participants.iloc[-31:],{'professional-diagnosis':True,'gender':'Female'}) #The last 32 participants are from OPD
    OPD_C_female_participants = dh.df_retrieve(participants.iloc[-31:],{'professional-diagnosis':False,'gender':'Female'}) #The last 32 participants are from OPD
    
    OPD_PD_test_male_participants = OPD_PD_male_participants.sample(OPD_PD_male_sample_size,weights=OPD_PD_male_participants['age'])
    OPD_C_test_male_participants = OPD_C_male_participants.sample(OPD_C_male_sample_size,weights=OPD_C_male_participants['age'])
    OPD_PD_test_female_participants = OPD_PD_female_participants.sample(OPD_PD_female_sample_size,weights=OPD_PD_female_participants['age'])
    OPD_C_test_female_participants = OPD_C_female_participants.sample(OPD_C_female_sample_size,weights=OPD_C_female_participants['age'])
    
    OPD_PD_test_participants = OPD_PD_test_male_participants.append(OPD_PD_test_female_participants)
    OPD_C_test_participants = OPD_C_test_male_participants.append(OPD_C_test_female_participants)
    OPD_test_participants = OPD_PD_test_participants.append(OPD_C_test_participants)


#    mPower_PD_male_sample_size = mPower_PD_female_sample_size = int(mPower_PD_sample_size/2)
#    mPower_C_male_sample_size = mPower_C_female_sample_size = int(mPower_C_sample_size/2)
    mPower_PD_female_sample_size = math.ceil(mPower_PD_sample_size*0.4)
    mPower_PD_male_sample_size = mPower_PD_sample_size - mPower_PD_female_sample_size #This should make the ratio roughly 1.5 to 1 (or 0.6/0.4)
    mPower_C_male_sample_size = mPower_C_female_sample_size = int(mPower_C_sample_size/2)
    
    mPower_PD_male_participants = dh.df_retrieve(participants.iloc[:-31],{'professional-diagnosis':True,'gender':'Male'})
    mPower_C_male_participants = dh.df_retrieve(participants.iloc[:-31],{'professional-diagnosis':False,'gender':'Male'})
    mPower_PD_female_participants = dh.df_retrieve(participants.iloc[:-31],{'professional-diagnosis':True,'gender':'Female'})
    mPower_C_female_participants = dh.df_retrieve(participants.iloc[:-31],{'professional-diagnosis':False,'gender':'Female'})
    
    mPower_PD_test_male_participants = mPower_PD_male_participants.sample(mPower_PD_male_sample_size,weights=mPower_PD_male_participants['age']**0.5) #many patients are older
    mPower_C_test_male_participants = mPower_C_male_participants.sample(mPower_C_male_sample_size,weights=mPower_C_male_participants['age']**2) #many controls are younger
    mPower_PD_test_female_participants = mPower_PD_female_participants.sample(mPower_PD_female_sample_size,weights=mPower_PD_female_participants['age']**0.5)
    mPower_C_test_female_participants = mPower_C_female_participants.sample(mPower_C_female_sample_size,weights=mPower_C_female_participants['age']**2)
    
    mPower_PD_test_participants = mPower_PD_test_male_participants.append(mPower_PD_test_female_participants)
    mPower_C_test_participants = mPower_C_test_male_participants.append(mPower_C_test_female_participants)
    mPower_test_participants = mPower_PD_test_participants.append(mPower_C_test_participants)
    
    test_participants = OPD_test_participants.append(mPower_test_participants)
    test_samples = pd.DataFrame()
    for healthCode in test_participants['healthCode']:
        cond = samples['healthCode'] == healthCode
        participant_samples = samples[cond]
        participant_sample = participant_samples.sample(1)
        test_samples = test_samples.append(participant_sample)
        samples = samples.loc[~cond]
    participants = participants[~participants['healthCode'].isin(test_participants['healthCode'])]
    
    
    return samples,participants,test_samples,test_participants

