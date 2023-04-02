"""Generic helper functions to handle the data"""

def df_retrieve(data,cols_vals):
    """
    Retrieves a subset of a dataframe based on equivalence matches
    
    Parameters:
        data (pandas.DataFrame): the dataframe from which a subset is to be created from
        cols_vals (dict): a dict with the key as the column name and the val as the equivalence val
    """
    for col,val in cols_vals.items():
        data = data[data[col] == val]
    
    return data
    
def OPD_summary(samples,participants):
    """OPD refers to the "Parkinsons Data Set". It stands for Oxford Parkinsons Data, as the data was processed in Oxford. It is a public data set, so nothing in this is private"""

    samples = samples.merge(participants[['participant','Age','Gender']], on='participant', how='left')
    
    print('There are %i samples:' % (len(samples)))
    PD = df_retrieve(samples,{'status':1})
    Control = df_retrieve(samples,{'status':0})
    Male = df_retrieve(samples,{'Gender':'M'})
    Female = df_retrieve(samples,{'Gender':'F'})
    
    PD_male = df_retrieve(samples,{'Gender':'M','status':1})
    PD_female = df_retrieve(samples,{'Gender':'F','status':1})
    Control_male = df_retrieve(samples,{'Gender':'M','status':0})
    Control_female = df_retrieve(samples,{'Gender':'F','status':0})
    
    print('\tPD: %i, age: %.2f +- %.2f' % (len(PD),PD['Age'].mean(),PD['Age'].std()))
    print('\tControl: %i, age: %.2f +- %.2f' % (len(Control),Control['Age'].mean(),Control['Age'].std()))
    print('\n\tMale: %i, age: %.2f +- %.2f' % (len(Male),Male['Age'].mean(),Male['Age'].std()))
    print('\tFemale: %i, age: %.2f +- %.2f' % (len(Female),Female['Age'].mean(),Female['Age'].std()))      
    
    print('\n\tPD and male: %i, age: %.2f +- %.2f' % (len(PD_male),PD_male['Age'].mean(),PD_male['Age'].std()))
    print('\tPD and female: %i, age: %.2f +- %.2f' % (len(PD_female),PD_female['Age'].mean(),PD_female['Age'].std()))
    print('\tControl and male: %i, age: %.2f +- %.2f' % (len(Control_male),Control_male['Age'].mean(),Control_male['Age'].std()))
    print('\tControl and female: %i, age: %.2f +- %.2f' % (len(Control_female),Control_female['Age'].mean(),Control_female['Age'].std()))


    print('\nThere are %i participants:' % (len(participants)))
    PD = df_retrieve(participants,{'status':1})
    Control = df_retrieve(participants,{'status':0})
    Male = df_retrieve(participants,{'Gender':'M'})
    Female = df_retrieve(participants,{'Gender':'F'})
    
    PD_male = df_retrieve(participants,{'Gender':'M','status':1})
    PD_female = df_retrieve(participants,{'Gender':'F','status':1})
    Control_male = df_retrieve(participants,{'Gender':'M','status':0})
    Control_female = df_retrieve(participants,{'Gender':'F','status':0})
    
    print('\tPD: %i, age: %.2f +- %.2f' % (len(PD),PD['Age'].mean(),PD['Age'].std()))
    print('\tControl: %i, age: %.2f +- %.2f' % (len(Control),Control['Age'].mean(),Control['Age'].std()))
    print('\n\tMale: %i, age: %.2f +- %.2f' % (len(Male),Male['Age'].mean(),Male['Age'].std()))
    print('\tFemale: %i, age: %.2f +- %.2f' % (len(Female),Female['Age'].mean(),Female['Age'].std()))      
    
    print('\n\tPD and male: %i, age: %.2f +- %.2f' % (len(PD_male),PD_male['Age'].mean(),PD_male['Age'].std()))
    print('\tPD and female: %i, age: %.2f +- %.2f' % (len(PD_female),PD_female['Age'].mean(),PD_female['Age'].std()))
    print('\tControl and male: %i, age: %.2f +- %.2f' % (len(Control_male),Control_male['Age'].mean(),Control_male['Age'].std()))
    print('\tControl and female: %i, age: %.2f +- %.2f' % (len(Control_female),Control_female['Age'].mean(),Control_female['Age'].std()))
    


def mPower_summary(samples,participants):
    """Print summary of the mPower data set. Although the mPower data set is private, this method has been reviewed and does not contain anything that may reveal details of the data which cannot be found online anyway"""
    
    print('There are %i samples:' % (len(samples)))
    PD = df_retrieve(samples,{'professional-diagnosis':True})
    Control = df_retrieve(samples,{'professional-diagnosis':False})
    Male = df_retrieve(samples,{'gender':'Male'})
    Female = df_retrieve(samples,{'gender':'Female'})
    
    PD_male = df_retrieve(samples,{'gender':'Male','professional-diagnosis':True})
    PD_female = df_retrieve(samples,{'gender':'Female','professional-diagnosis':True})
    Control_male = df_retrieve(samples,{'gender':'Male','professional-diagnosis':False})
    Control_female = df_retrieve(samples,{'gender':'Female','professional-diagnosis':False})
        
    print('\tPD: %i, age: %.2f +- %.2f' % (len(PD),PD['age'].mean(),PD['age'].std()))
    print('\tControl: %i, age: %.2f +- %.2f' % (len(Control),Control['age'].mean(),Control['age'].std()))
    print('\n\tMale: %i, age: %.2f +- %.2f' % (len(Male),Male['age'].mean(),Male['age'].std()))
    print('\tFemale: %i, age: %.2f +- %.2f' % (len(Female),Female['age'].mean(),Female['age'].std()))        
    
    print('\n\tPD and male: %i, age: %.2f +- %.2f' % (len(PD_male),PD_male['age'].mean(),PD_male['age'].std()))
    print('\tPD and female: %i, age: %.2f +- %.2f' % (len(PD_female),PD_female['age'].mean(),PD_female['age'].std()))
    print('\tControl and male: %i, age: %.2f +- %.2f' % (len(Control_male),Control_male['age'].mean(),Control_male['age'].std()))
    print('\tControl and female: %i, age: %.2f +- %.2f' % (len(Control_female),Control_female['age'].mean(),Control_female['age'].std()))


    print('\nThere are %i participants:' % (len(participants)))
    PD = df_retrieve(participants,{'professional-diagnosis':True})
    Control = df_retrieve(participants,{'professional-diagnosis':False})
    Male = df_retrieve(participants,{'gender':'Male'})
    Female = df_retrieve(participants,{'gender':'Female'})
    
    PD_male = df_retrieve(participants,{'gender':'Male','professional-diagnosis':True})
    PD_female = df_retrieve(participants,{'gender':'Female','professional-diagnosis':True})
    Control_male = df_retrieve(participants,{'gender':'Male','professional-diagnosis':False})
    Control_female = df_retrieve(participants,{'gender':'Female','professional-diagnosis':False})
    print('\tPD: %i, age: %.2f +- %.2f' % (len(PD),PD['age'].mean(),PD['age'].std()))
    print('\tControl: %i, age: %.2f +- %.2f' % (len(Control),Control['age'].mean(),Control['age'].std()))
    print('\n\tMale: %i, age: %.2f +- %.2f' % (len(Male),Male['age'].mean(),Male['age'].std()))
    print('\tFemale: %i, age: %.2f +- %.2f' % (len(Female),Female['age'].mean(),Female['age'].std()))  
    
    print('\n\tPD and male: %i, age: %.2f +- %.2f' % (len(PD_male),PD_male['age'].mean(),PD_male['age'].std()))
    print('\tPD and female: %i, age: %.2f +- %.2f' % (len(PD_female),PD_female['age'].mean(),PD_female['age'].std()))
    print('\tControl and male: %i, age: %.2f +- %.2f' % (len(Control_male),Control_male['age'].mean(),Control_male['age'].std()))
    print('\tControl and female: %i, age: %.2f +- %.2f' % (len(Control_female),Control_female['age'].mean(),Control_female['age'].std()))
