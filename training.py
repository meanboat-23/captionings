from text_process_and_make_dataset import *
from s2vt_model import S2VT
import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # parse the arguments and adds value
    parser.add_argument('--annotation_path', help='File path to annotation', default='/home/minbae/Desktop/sequence/annotations.txt')
    parser.add_argument('--train_data_dir', help='Directory path to training annotation', default='/home/minbae/Desktop/sequence/output/train')
    parser.add_argument('--val_data_dir', help='Directory path to training images',default='/home/minbae/Desktop/sequence/output/train_val')
    parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
    parser.add_argument('--batch_size', help='Batch size for training', default=5, type=int)
    parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
    parser.add_argument('--learning_rate', help='Learning rate for training', default=1e-4, type=float)
    args = parser.parse_args()
    params = vars(args)  # return into dict format

train_dataset = preprocessdata(params['train_data_dir'], params['annotation_path'])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=params['batch_size'])

validation_dataset = preprocessdata(params['val_data_dir'], params['annotation_path'])
validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=params['batch_size'])

model = S2VT(vocab_size=train_dataset.vocabsize).to(device)

optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
loss = torch.nn.CrossEntropyLoss()

epochs = params['epoch']

for epoch in range(epochs + 1):
    model.train()
    for batch_index, (X_train, y_train) in enumerate(train_dataloader):     # y_train.shape=(batch, 80), X_train.shape=(batch,80,4096)
        optimizer.zero_grad()
        hypothesis = model(X_train, y_train).type(torch.FloatTensor)
        cost = loss(hypothesis, y_train.type(torch.FloatTensor))
        
        cost.requires_grad_(True)
        cost.backward()
        optimizer.step()

for epoch in range(epochs + 1):
    model.eval()
    for batch_index, (X_train, y_train) in enumerate(validation_dataloader):
        batch, _, _ = X_train.shape
        predict = model(X_train)
        prediction = predict.cpu().detach().numpy()
        y_trains = y_train.cpu().detach().numpy()
        #print(prediction)
        for i in range (batch):
            answer = [sentencefromindex(y_trains[i], train_dataset.index2word)]
            print(answer)
            generated_words = [sentencefromindex(prediction[i], train_dataset.index2word)]
            print(generated_words)
        
