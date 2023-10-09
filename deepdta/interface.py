import os
import pandas as pd
import keras
from keras.callbacks import EarlyStopping

from .network import build_combined_categorical
from .dataset import Dataset




class DeepDTA:
    def __init__(self, 
                 max_seqlen_smiles, max_seqlen_protein,
                 vocab_size_smiles, vocab_size_protein,
                 num_filters, filter_size_smiles, filter_size_protein):
        self.params = {
            'max_smi_len':max_seqlen_smiles,
            'max_seq_len':max_seqlen_protein,
            'charsmiset_size':vocab_size_smiles,
            'charseqset_size':vocab_size_protein,
        }
        self.num_filters = num_filters
        self.filter_size_smiles = filter_size_smiles
        self.filter_size_protein = filter_size_protein
    
        ## Model build
        self.model = build_combined_categorical(self.params, self.num_filters, self.filter_size_smiles, self.filter_size_protein)


    def evaluate(self, xd, xt, batch_size=256):
        ## Dataset
        dataset = Dataset(xd, xt)
        
        ## Predict
        return self.model.predict([dataset.XD, dataset.XT])
        
        
    def save(self, filepath):
        self.model.save(filepath)
        print(f'[INFO] Keras model is successfully saved at {filepath}')
        
        
    def load(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f'[INFO] The pretrained model is successfully loaded from {filepath}')

  
  
        
class Trainer:
    def __init__(self, model):
        self.model = model
        
    
    def train(self, train_data, test_data, valid_data=None,
              batch_size=256, n_epochs=100, log_interval=20,
              ckpt_dir=os.path.join('.', 'ckpt'), ckpt_filename='deepdta_tdc',
              verbose=2):

        ## valid is given?
        if valid_data is None:
            valid_data = test_data
        
        ## fold
        XD_train, XT_train, Y_train = train_data.XD, train_data.XT, train_data.Y
        XD_valid, XT_valid, Y_valid = valid_data.XD, valid_data.XT, valid_data.Y
        XD_test,  XT_test,  Y_test  = test_data.XD,  test_data.XT,  test_data.Y
        
        ## callback
        callbacks = [EarlyStopping(monitor='val_loss', patience=15), ]
        
        ## fit
        history = self.model.model.fit(
            [XD_train, XT_train], Y_train,
            batch_size=batch_size, epochs=n_epochs, shuffle=True,
            validation_data=([XD_valid, XT_valid], Y_valid),
            callbacks = callbacks,
            verbose=verbose,
        )
        
        ## save
        self.model.save(os.path.join(ckpt_dir, f'{ckpt_filename}.h5'))
        pd.DataFrame(history.history).to_csv(os.path.join(ckpt_dir, f'{ckpt_filename}_history.csv'))
        
        return history

        
