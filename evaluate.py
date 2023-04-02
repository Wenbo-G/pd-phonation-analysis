"""A big helper method to evaluate the models with repeated k-fold cross validation and a train-test split, given certain settings. This is used in conjunction with evaluation.py"""


from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from os.path import join
import resultsHandler as rh
import random
import torch
import numpy as np
import OPD_modifications
import mPower_modifications
import generalisation_modifications


def evaluate(dataset, kf_method, split_method, samples, participants, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, repetitions, verbose_odds, seeds, skip_kf=False, skip_traintest_split=False):

    
    import warnings
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    
    
    repeated_kfold_results = rh.results_DataFrame() #holds avg results for all repetitions of kfold cv, indexed by hyperparameters of gridsearch
    repeated_traintest_results = rh.results_DataFrame() #holds avg results for all repetitions of traintest split, indexed by hyperparameters of gridsearch
    verbose, verbose2 = False, False

    iter_num = 0
          
    total_iters = len(global_settings)*len(ozkan_settings)+len(global_settings)*len(caliskan_settings)+len(global_settings)*len(ulhaq_settings)
    print('There are a total of %i global settings, %i ozkan settings, %i caliskan settings, and %i ulhaq settings, creating a total of %i iterations' 
      % (len(global_settings),len(ozkan_settings),len(caliskan_settings),len(ulhaq_settings),total_iters))
      
    for global_setting in global_settings:
    
        preprocessor, preprocessing_method = global_setting
        
        X,y = to_numpy(samples)
        
        transformer = preprocessor()
            
        for model_type, model_settings in {'ozkan':ozkan_settings,'caliskan':caliskan_settings,'ulhaq':ulhaq_settings}.items():
    
            for model_setting in model_settings:
                iter_num += 1
                if model_type == 'ozkan': component,k = model_setting
                elif model_type == 'caliskan': lrs,epochs,rhos,lams,Bs,activations,latent_size = model_setting
                elif model_type == 'ulhaq': kernel,gamma,C,num_features = model_setting
                    
                kfold_results = rh.results_DataFrame() #hold results of kfold cv, indexed by which repetition
                traintest_results = rh.multiple_runs_DataFrame() #hold results of traintest split, indexed by which repetition
                
                for seed,repetition in zip(seeds,range(repetitions)):

                    print('Iteration %i of %i, repetition: %i for %s model' % (iter_num,total_iters,repetition,model_type), end='\r')
                    
                    if not skip_kf:
                        ################## K FOLD CV ##################                    
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        
                        fold = 0
                        fold_results = rh.multiple_runs_DataFrame() #holds result of each fold, indexed by folds

                        kf = kf_method(X,y,samples,participants)
                        for train_index, test_index in kf:
                            X_train,y_train,X_test,y_test = X[train_index],y[train_index],X[test_index],y[test_index]

                            if preprocessing_method == 'both': transformer.fit(X)
                            else: transformer.fit(X_train)
                            
                            X_train, X_test = transformer.transform(X_train), transformer.transform(X_test)
                            
                            verbose = True if random.random() < verbose_odds else False
                            if model_type == 'ozkan': fold_results.loc[fold] = ozkan_method(k,component,preprocessing_method,X_train,y_train,X_test,y_test)
                            elif model_type == 'caliskan': fold_results.loc[fold] = caliskan_method(X_train,y_train,X_test,y_test,epochs,lrs,activations,lams,rhos,Bs,latent_size,verbose=verbose)
                            elif model_type == 'ulhaq': fold_results.loc[fold] = ulhaq_method(kernel,gamma,C,num_features,dataset,X_train,y_train,X_test,y_test)
                            
                            fold += 1
                        
                        kfold_results.loc[repetition] = rh.process_multiple_runs(fold_results)
                        ################################################

                    if not skip_traintest_split:
                        ################## TRAIN TEST SPLIT ##################                    
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        X_train, X_test, y_train, y_test = split_method(X,y,samples,participants)
                        if preprocessing_method == 'both': transformer.fit(X)
                        else: transformer.fit(X_train)
                        X_train, X_test = transformer.transform(X_train), transformer.transform(X_test)
                        
                        verbose2 = True if random.random() < verbose_odds else False
                        if model_type == 'ozkan': traintest_results.loc[repetition] = ozkan_method(k,component,preprocessing_method,X_train,y_train,X_test,y_test)
                        elif model_type == 'caliskan': traintest_results.loc[repetition] = caliskan_method(X_train,y_train,X_test,y_test,epochs,lrs,activations,lams,rhos,Bs,latent_size,verbose=verbose2)
                        elif model_type == 'ulhaq': traintest_results.loc[repetition] = ulhaq_method(kernel,gamma,C,num_features,dataset,X_train,y_train,X_test,y_test)
                        ######################################################
        
                if model_type == 'ozkan': model_name = model_type + ' PCA_%i k_%i' % (component, k)
                elif model_type =='caliskan': 
                    model_name = model_type
                    for a in activations: model_name += ' %s' % a.__name__
                    model_name += ' latent:%i, epochs:%i, lr:%0.4f' % (latent_size,epochs[0],lrs[0])
                elif model_type == 'ulhaq': 
                    if isinstance(gamma,str): model_name = model_type + ' %s gamma:%s C:%i num_features:%i' % (kernel,gamma,C,num_features)
                    else: model_name = model_type + ' %s gamma:%.4f C:%i num_features:%i' % (kernel,gamma,C,num_features)
                
                model_name += ' %s %s' % (preprocessor.__name__,preprocessing_method)
                
                if not skip_kf: repeated_kfold_results.loc[model_name] = rh.process_multiple_kfold_runs(kfold_results)
                if not skip_traintest_split: repeated_traintest_results.at[model_name] = rh.process_multiple_runs(traintest_results)
                
                if not skip_kf: temp_save_csv(repeated_kfold_results,'repeated kfold results')
                if not skip_traintest_split: temp_save_csv(repeated_traintest_results,'repeated split results')
                
    if skip_traintest_split: return repeated_kfold_results 
    elif skip_kf: return repeated_traintest_results
    else: return repeated_kfold_results,repeated_traintest_results
    
    
def evaluate_generalisation(dataset, OPD_samples, OPD_participants, mPower_samples, test_size, validation_method, test_method, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, test_seeds,validation_repetitions,test_repetitions, verbose_odds):
    
    import torch.nn as nn
    
    
    repeated_test_results = rh.results_DataFrame()
    ozkan_test_results = rh.multiple_runs_DataFrame()
    caliskan_test_results = rh.multiple_runs_DataFrame()
    ulhaq_test_results = rh.multiple_runs_DataFrame()
        
    PD_ratio = 0.1
    OPD_C_sample_size = 2 #Theres only 8 controls from OPD, so we'll just take a few
    OPD_PD_sample_size = min(4,max(2,int(test_size*PD_ratio*0.1//2)))
    mPower_PD_sample_size = int(test_size*PD_ratio) - OPD_PD_sample_size
    mPower_C_sample_size = test_size - mPower_PD_sample_size - OPD_C_sample_size - OPD_PD_sample_size
        
    for rep,test_seed in zip(range(test_repetitions),test_seeds):
        #if rep==0: continue
        train_validate_samples, train_validate_participants, test_samples, test_participants = generalisation_modifications.sample_test_set(OPD_samples, OPD_participants, mPower_samples,OPD_PD_sample_size,OPD_C_sample_size,mPower_PD_sample_size,mPower_C_sample_size,test_seed)
    
        ########## GRID SEARCH WITH TRAIN VALIDATION ##########
        print('\nGRID SEARCH AND VALIDATION PHASE:\n', end='\t')
#        print(sum(train_validate_participants['professional-diagnosis'] == True),len(train_validate_participants),sum(train_validate_participants['professional-diagnosis'] == True)/len(train_validate_participants))
        validation_seeds = [rep*100 + s for s in list(range(validation_repetitions))] #makes the validation seed different depending on which test repetition it is on 
        if (len(ozkan_settings) == 1) and (len(caliskan_settings) == 1) and (len(ulhaq_settings) == 1):
            print('Warning: Skipping validation step')
            ozkan_test_settings = ozkan_settings
            ulhaq_test_settings = ulhaq_settings
            caliskan_test_settings = caliskan_settings
        else:
            validation_results = evaluate(dataset, validation_method, None, train_validate_samples, train_validate_participants, to_numpy, global_settings, ozkan_settings, caliskan_settings, ulhaq_settings, ozkan_method, caliskan_method, ulhaq_method, validation_repetitions, verbose_odds, validation_seeds, skip_traintest_split=True)

            best_ozkan_result,best_caliskan_result,best_ulhaq_result = rh.get_best_result(validation_results,by='average accuracy')
            best_ozkan_parameters,best_caliskan_parameters,best_ulhaq_parameters = rh.get_best_parameters(best_ozkan_result.name,best_caliskan_result.name,best_ulhaq_result.name)
            ozkan_test_settings = [best_ozkan_parameters]
            ulhaq_test_settings = [best_ulhaq_parameters]
            best_caliskan_activations = [nn.ReLU,nn.Sigmoid] if best_caliskan_parameters[0] == 'Sigmoid' else [nn.ReLU,nn.ReLU]
            caliskan_test_settings = [([best_caliskan_parameters[3]]*4,[best_caliskan_parameters[2]]*4,[0.15,0.25],[0.03,0.03],[2,2],best_caliskan_activations,best_caliskan_parameters[1])]
        ########################################################
        
        #################### TEST ####################
        print('\nTEST PHASE:\n', end='\t')
#        print(sum(test_participants['professional-diagnosis'] == True),len(test_participants),sum(test_participants['professional-diagnosis'] == True)/len(test_participants))
        samples = train_validate_samples.append(test_samples)
        participants = train_validate_participants.append(test_participants) #not required
        test_results = evaluate(dataset, None, test_method, samples, participants, to_numpy, global_settings, ozkan_test_settings, caliskan_test_settings, ulhaq_test_settings, ozkan_method, caliskan_method, ulhaq_method, 1, verbose_odds, [test_seed], skip_kf=True)
        
        ozkan_result, caliskan_result, ulhaq_result = rh.separate_results(test_results)
        ozkan_test_results.loc[rep] = ozkan_result['run results'].loc[0]
        caliskan_test_results.loc[rep] = caliskan_result['run results'].loc[0]
        ulhaq_test_results.loc[rep] = ulhaq_result['run results'].loc[0]
        ##############################################    
        
    #Compile all test results
    repeated_test_results.at['ozkan'] = rh.process_multiple_runs(ozkan_test_results)
    repeated_test_results.at['caliskan'] = rh.process_multiple_runs(caliskan_test_results)
    repeated_test_results.at['ulhaq'] = rh.process_multiple_runs(ulhaq_test_results)

    return repeated_test_results
    
    
def temp_save_csv(df,saveName):
    
    cols = ['average accuracy', 'std accuracy', 'average mcc' ,'std mcc' ,'average auroc', 'std auroc', 'best accuracy']
    savePath = join('Results','temp',saveName+'.csv')
    df[cols].to_csv(savePath)
    
    
