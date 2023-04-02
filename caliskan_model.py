#The code here is based on the paper: Diagnosis of the Parkinson disease by using deep neural network classifier, Caliskan et al., 2017
#https://www.semanticscholar.org/paper/DIAGNOSIS-OF-THE-PARKINSON-DISEASE-BY-USING-DEEP-Caliskan-Badem/64925ae2d69735f2e2c86d0e4763e26bf79d1ea8
#It uses a stacked auto encoder framework to train and classify patients with PD


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import resultsHandler as rh
import time


class AutoEncoder(nn.Module):
    """Generic autoencoder class, to be stacked"""
    def __init__(self,input_size,latent_size,activation):
        super(AutoEncoder, self).__init__()
        
        activation1, activation2 = activation
        
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, latent_size, bias=True),
            nn.BatchNorm1d(latent_size),
            activation1()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(latent_size, input_size, bias=True),
            activation2()
        )
        
    def forward(self, x):
        encoded = self.hidden_layer(x)
        decoded = self.output_layer(encoded)
        return encoded,decoded
        

class ClassifierLayer(nn.Module):
    """Generic classification layer to transform encoded inputs into a prediction"""
    def __init__(self,latent_size):
        super(ClassifierLayer, self).__init__()
        self.output_layer = nn.Sequential(
            # z_dim x 1 x 1 -> z_dim x 1 x 1
            nn.Linear(latent_size, 2, bias=True),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.output_layer(x)
        return x
    
    
class caliskan_model(nn.Module):
    """The model mentioned in the paper. The weights of this model will be copied over from the trained autoEncoders and the classifier layer"""
    def __init__(self,input_size,latent_size,activation,AE1,AE2,clf):
        super(caliskan_model, self).__init__()
        
        
        activation1, activation2 = activation
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_size, latent_size, bias=True),
            nn.BatchNorm1d(latent_size),
            activation1()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(latent_size, latent_size, bias=True),
            nn.BatchNorm1d(latent_size),
            activation2()
        )
        self.output_layer = nn.Sequential(
            # z_dim x 1 x 1 -> z_dim x 1 x 1
            nn.Linear(latent_size, 2, bias=True),
            nn.Softmax(dim=1)
        )
        
        dict_AE1 = dict(AE1.named_parameters())
        dict_AE2 = dict(AE2.named_parameters())
        dict_clf = dict(clf.named_parameters())
        dict_model = dict(self.named_parameters())

        with torch.no_grad():
            dict_model['hidden_layer1.0.weight'].copy_(dict_AE1['hidden_layer.0.weight'])
            dict_model['hidden_layer1.0.bias'].copy_(dict_AE1['hidden_layer.0.bias'])
            dict_model['hidden_layer2.0.weight'].copy_(dict_AE2['hidden_layer.0.weight'])
            dict_model['hidden_layer2.0.bias'].copy_(dict_AE2['hidden_layer.0.bias'])
            dict_model['output_layer.0.weight'].copy_(dict_clf['output_layer.0.weight'])
            dict_model['output_layer.0.bias'].copy_(dict_clf['output_layer.0.bias'])

        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.output_layer(x)
        return x  
        
        

class phonation_dataset(Dataset):
    """Dataset class for the data"""
    def __init__(self,X,y,for_autoencoder=False):
        self.X = X#torch.from_numpy(X).float()#.type(torch.FloatTensor)

        if for_autoencoder: self.y = X#torch.from_numpy(X).float()
        else: self.y = y#torch.from_numpy(np.array([[0.,1.] if z == 1 else [1.,0.] for z in y])).float()#.type(torch.FloatTensor)
        
        self.len = self.X.shape[0]

        if not (self.y.shape[0] == self.X.shape[0]):
            raise ValueError('Lengths of inputs do not match, len(x) is %i, len(y) is %i' % (x.shape[0],y.shape[0]))

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
  
    def __len__(self):
        return self.len
        


def getEncodings(model,X):
    """Helper function to get the encodings from the autoencoders"""
    model.eval()
    with torch.no_grad():
        encodings,_ = model(X)
    return encodings


def train_autoencoders(model,X_train,X_test,batch_size,epochs,lr,rho,B,lam,verbose=False):
    """Method to train the autoencoders. The loss used by the authors is a combination of 3 parts, one is the standard MSE_loss, the other two are defined below"""
    
    drop_last = True if len(X_train)%batch_size == 1 else False
    train_dataset = phonation_dataset(X_train,X_train,for_autoencoder=True)
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,drop_last=drop_last)
    
    drop_last = True if len(X_test)%(batch_size*8) == 1 else False
    test_dataset = phonation_dataset(X_test,X_test,for_autoencoder=True)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size*8,drop_last=drop_last)    
    
    optimizer = optim.Adam(model.parameters(), lr)
    
    train_losses = []
    test_losses = []

    def KLD(rho, activations):
        """One part of the loss function"""
        rho_hat = torch.sum(activations, dim=0)
        [S,_] = activations.shape
        rho_hat = rho_hat/S
        KL = torch.sum((rho * torch.log(rho / rho_hat)) + ((1-rho) * torch.log((1-rho)/(1-rho_hat))))
        return KL

    def regLoss(lam,weights):
        """Another part of the loss function"""
        loss = lam/2*torch.norm(weights)**2
        return loss
    
    def loss_F(output,target,activations,weights,lam,rho,B):
        """The combined loss function"""
        loss = F.mse_loss(output,target)
        loss += B*KLD(rho,activations)
        loss += regLoss(lam,weights)
        return loss

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i,(x,target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            activations,y_pred = model(x)
            weights = dict(model.named_parameters())['hidden_layer.0.weight'].data
            loss = loss_F(y_pred,target,activations,weights,lam,rho,B)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_losses.append(epoch_loss/len(train_dataloader))
        
        ########## test loss ##########
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for _,(x_test,target_test) in enumerate(test_dataloader):
                activations_test,y_pred_test = model(x_test)
                test_loss += loss_F(y_pred_test,target_test,activations_test,weights,lam,rho,B).item()
            test_losses.append(test_loss/len(test_dataloader))
        model.train()
        ###############################
        
        if verbose: print('Epoch: %i, loss: %.4f' % (epoch, epoch_loss), end='\r')
        
    return train_losses,test_losses
    
    
def train_classifiers(model,X_train,y_train,X_test,y_test,batch_size,epochs,lr,verbose=False):
    """Method to train the classifiers (including the final stacked auto encoder model)"""
    
    drop_last = True if len(X_train)%batch_size == 1 else False
    train_dataset = phonation_dataset(X_train,y_train)
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,drop_last=drop_last)    
    
    drop_last = True if len(X_test)%(batch_size*8) == 1 else False
    test_dataset = phonation_dataset(X_test,y_test)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size*8,drop_last=drop_last)    
    
    optimizer = optim.Adam(model.parameters(), lr)
        
    train_losses = []
    test_losses = []
    
    loss_F = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i,(x,target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_F(y_pred,target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_losses.append(epoch_loss/len(train_dataloader))       
        
        ########## test loss ##########
        model.eval()
        with torch.no_grad():    
            test_loss = 0
            for _,(x_test,target_test) in enumerate(test_dataloader):
                y_pred_test = model(x_test)
                test_loss += loss_F(y_pred_test,target_test).item()
            test_losses.append(test_loss/len(test_dataloader))
        model.train()
        ###############################
        
        if verbose: print('Epoch: %i, loss: %.4f' % (epoch, epoch_loss), end='\r')

    return train_losses,test_losses
        
        


def caliskan(X_train,y_train,X_test,y_test,epochs,lrs,activations,lams,rhos,Bs,latent_size,verbose=False):
    """The model from the paper mentioned above. Each individual part (2 auto encoders and 1 classifier layer) is trained separately. All parts are then combined and trained together as a whole"""
    start = time.time()
    
    epochs1, epochs2, epochs3, epochs4 = epochs
    lr1, lr2, lr3, lr4 = lrs
    lam1, lam2 = lams
    rho1,rho2 = rhos
    B1, B2 = Bs
    input_size = 22
    
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(np.array([[0.,1.] if y == 1 else [1.,0.] for y in y_train])).float()
    y_test_tensor = torch.from_numpy(np.array([[0.,1.] if y == 1 else [1.,0.] for y in y_test])).float()

    #initialise models AE1 AE2 clf 
    AE1 = AutoEncoder(input_size,latent_size,activations)
    AE2 = AutoEncoder(latent_size,latent_size,activations)
    clf = ClassifierLayer(latent_size)
    
    #train AE1 with X_train, get encodings 
    AE1_train_loss,AE1_test_loss = train_autoencoders(AE1,X_train,X_test,batch_size=64,epochs=epochs1,lr=lr1,lam=lam1,rho=rho1,B=B1,verbose=verbose)
    encodings = getEncodings(AE1,X_train)
    test_encodings = getEncodings(AE1,X_test)
    
    #train AE2 with encodings, get encodings2
    AE2_train_loss,AE2_test_loss = train_autoencoders(AE2,encodings,test_encodings,epochs=epochs2,batch_size=64,lr=lr2,lam=lam2,rho=rho2,B=B2,verbose=verbose)
    encodings = getEncodings(AE2,encodings)
    test_encodings = getEncodings(AE2,test_encodings)
    
    #train clf with encodings2 on y_train
    clf_train_loss,clf_test_loss = train_classifiers(clf,encodings,y_train,test_encodings,y_test_tensor,batch_size=64,epochs=epochs3,lr=lr3,verbose=verbose)
    
    #initialise caliskan model and copy weights from AE1, AE2, clf
    model = caliskan_model(input_size,latent_size,activations,AE1,AE2,clf)
    #train model with X_train and y_train
    model_train_loss,model_test_loss = train_classifiers(model,X_train,y_train,X_test,y_test_tensor,batch_size=64,epochs=epochs4,lr=lr4,verbose=verbose)

    #inference time
    #get predicted Y and probs
    model.eval()
    with torch.no_grad():
        prob_predicted_Y = model(X_test)
    predicted_Y = torch.argmax(prob_predicted_Y,dim=1)

    if verbose:
        print('Training completed in %.1f seconds' % (time.time()-start))
        ########################
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,sharey=False,figsize=[20,5])
        plt.subplots_adjust(wspace=0.25)

        ax1.set_title('AE1')
        ax1.plot(AE1_train_loss, label='training losses')#,color='tab:blue')
        ax1.plot(AE1_test_loss, label='testing losses')#,color='tab:blue')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epochs')
        ax1.legend()
        #ax1.tick_params(axis='y')

        ax2.set_title('AE2')
        ax2.plot(AE2_train_loss, label='training losses')#,color='tab:blue')
        ax2.plot(AE2_test_loss, label='testing losses')#,color='tab:blue')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epochs')
        ax2.legend()

        ax3.set_title('clf')
        ax3.plot(clf_train_loss, label='training losses')#,color='tab:blue')
        ax3.plot(clf_test_loss, label='testing losses')#,color='tab:blue')
        ax3.set_ylabel('loss')
        ax3.set_xlabel('epochs')
        ax3.legend()

        ax4.set_title('model')
        ax4.plot(model_train_loss, label='training losses')#,color='tab:blue')
        ax4.plot(model_test_loss, label='testing losses')#,color='tab:blue')
        ax4.set_ylabel('loss')
        ax4.set_xlabel('epochs')
        ax4.legend()

        plt.show()
        ########################
    
    return rh.getPerformanceMetrics(y_test,predicted_Y,prob_predicted_Y[:, 1])
        
        
        
