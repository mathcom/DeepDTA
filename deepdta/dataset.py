import numpy as np


## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64



## ######################## ##
#
#  DATASET Class
#
## ######################## ## 

class Dataset:
    def __init__(self, xd, xt, y=None, max_seqlen_smiles=100, max_seqlen_protein=1000):
        ## assert
        if y is None:
            assert (len(xd) == len(xt))
        else:
            assert (len(xd) == len(xt)) and (len(xd) == len(y))
        
        ## params
        self.max_seqlen_smiles  = max_seqlen_smiles
        self.max_seqlen_protein = max_seqlen_protein
        
        ## constant
        self.char2idx_smiles    = CHARISOSMISET
        self.vocab_size_smiles  = CHARISOSMILEN
        self.char2idx_protein   = CHARPROTSET
        self.vocab_size_protein = CHARPROTLEN
        
        ## raw data
        self.xd = xd
        self.xt = xt
        self.y  = y
        
        ## preprocess
        self.XD = [self.label_smiles(smi) for smi in self.xd]
        self.XT = [self.label_sequence(seq) for seq in self.xt]
        
        ## numpy
        self.XD = np.array(self.XD)
        self.XT = np.array(self.XT)
        self.Y  = np.array(self.y) if self.y is not None else np.array(self.y) 


    def label_smiles(self, smi):
        X = np.zeros(self.max_seqlen_smiles)
        for i, ch in enumerate(smi[:self.max_seqlen_smiles]):
            X[i] = self.char2idx_smiles[ch]
        return X


    def label_sequence(self, seq):
        X = np.zeros(self.max_seqlen_protein)
        for i, ch in enumerate(seq[:self.max_seqlen_protein]):
            X[i] = self.char2idx_protein[ch]
        return X


