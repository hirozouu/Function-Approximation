import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils

MY_BATCH_SIZE = 100

class MultiFunction():

    data_train = None
    train_loader = None
    data_test = None
    test_loader = None

    def __init__(self) -> None:
        # prepare the data from the csv files
        dftarget = pd.read_csv("target.csv", header=None, dtype=np.float32)
        dftarget = dftarget.T
        dfinput = pd.read_csv("input.csv", header=None, dtype=np.float32)

        X_train, X_test, Y_train, Y_test = train_test_split(dfinput.values, dftarget.values,test_size=0.2)

        # dataloader for training
        self.data_train = data_utils.TensorDataset(
            torch.tensor(X_train), torch.tensor(Y_train)
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.data_train,batch_size=MY_BATCH_SIZE,shuffle=True
        )

        # dataloader for test
        self.data_test = data_utils.TensorDataset(
            torch.tensor(X_test), torch.tensor(Y_test)
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.data_test,batch_size=MY_BATCH_SIZE,shuffle=False
        )

def main():
    datset = MultiFunction()

if __name__ == '__main__':
    main()