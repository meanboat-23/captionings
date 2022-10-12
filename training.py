from text_process_and_make_dataset import *
from s2vt_model import S2VT
import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/home/minbae/Desktop/sequence/logging')

import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.translate.meteor_score import single_meteor_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # parse the arguments and adds value
    parser.add_argument('--train_annotation_path', help='File path to annotation',
                        default='/home/minbae/Desktop/sequence/annotations.txt')
    parser.add_argument('--test_annotation_path', help='File path to annotation',
                        default='/home/minbae/Desktop/sequence/test.txt')
    parser.add_argument('--val_annotation_path', help='File path to annotation',
                        default='/home/minbae/Desktop/sequence/val.txt')
    parser.add_argument('--train_data_dir', help='Directory path to training annotation',
                        default='/home/minbae/Desktop/sequence/output/train')
    parser.add_argument('--val_data_dir', help='Directory path to training images',
                        default='/home/minbae/Desktop/sequence/output/validation')
    parser.add_argument('--test_data_dir', help='Directory path to training images',
                        default='/home/minbae/Desktop/sequence/output/test')
    parser.add_argument('--timestep', help='Total timestep', default=80, type=int)
    parser.add_argument('--batch_size', help='Batch size for training', default=5, type=int)
    parser.add_argument('--epoch', help='Total epoch', default=15, type=int)
    parser.add_argument('--learning_rate', help='Learning rate for training', default=1e-4, type=float)
    args = parser.parse_args()
    params = vars(args)  # return into dict format

train_dataset = preprocessdata(params['train_data_dir'], params['train_annotation_path'])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=params['batch_size'])

validation_dataset = preprocessdata(params['val_data_dir'], params['val_annotation_path'])
validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=params['batch_size'])

test_dataset = preprocessdata(params['test_data_dir'], params['test_annotation_path'])
test_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=params['batch_size'])

model = S2VT(vocab_size=train_dataset.vocabsize).to(device)

optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
loss = torch.nn.CrossEntropyLoss()

f = open('/home/minbae/Desktop/sequence/captions.txt', 'w')

epochs = params['epoch']

for epoch in range(epochs + 1):
    model.train()
    for batch_index, (X_train, y_train) in enumerate(
            train_dataloader):  # y_train.shape=(5,80), X_train.shape=(5,80,4096)
        optimizer.zero_grad()
        hypothesis = model(X_train, y_train)
        hypothesis = hypothesis.view(-1, train_dataset.vocabsize)
        y_train = y_train[:, 1:].contiguous().view(-1)

        cost = loss(hypothesis, y_train)

        writer.add_scalar("Loss/train", cost, epoch)
        cost.requires_grad_(True)
        cost.backward()
        optimizer.step()

    for batch_index, (X_Val, y_Val) in enumerate(
            validation_dataloader):
        hypothesis = model(X_Val, y_Val)
        hypothesis = hypothesis.view(-1, train_dataset.vocabsize)
        y_Val = y_Val[:, 1:].contiguous().view(-1)

        cost1 = loss(hypothesis, y_train)

        writer.add_scalar("Loss/val", cost1, epoch)
model.eval()
time = 0
for batch_index, (X_test, y_test) in enumerate(test_dataloader):
    batch, _, _ = X_test.shape
    predict = model(X_test)

    prediction = predict.cpu().detach().numpy()
    y_tests = y_test.cpu().detach().numpy()

    for i in range(batch):
        answer = sentencefromindex(y_tests[i], train_dataset.index2word)
        f.writelines(answer)
        generated_words = sentencefromindex(prediction[i], train_dataset.index2word)
        f.writelines(generated_words)
        prec = round(single_meteor_score(answer, generated_words), 4)
        writer.add_scalar("single_Meteor", prec, time)
        time = time + 1
writer.close()
f.close()
