import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def Model():
    model = torchvision.models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, 3)
    )
    return model.to(device)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_Data():
    
    train_dataset = datasets.ImageFolder(r'C:\Users\ra_saval\Desktop\fulhas\final_data\train', transform=transform)
    test_dataset = datasets.ImageFolder(r'C:\Users\ra_saval\Desktop\fulhas\final_data\test', transform=transform)

    # Create data loaders for the training and testing datasets
    #print(train_dataset.class_to_idx)
    #print(test_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader,test_loader

def val(model, test_loader, criterion, optimizer, epoch, step=5):
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            print(inputs.shape)
            print(labels.shape)
            labels = labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            print(predicted,type(predicted))
            print(labels,type(labels))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Test Accuracy: {accuracy:.2f}%')
            return accuracy
    
def train(model, train_loader, criterion, optimizer, epoch, step=5):
    model.train()
    
    for i, (images, labels) in enumerate(train_loader):
        #print(torch.cuda.is_available())
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        #print(predicted,type(predicted))
        #print(labels,type(labels))
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%step==0:
            print('EPOCH {} | ITER {} | AVG_LOSS {}'.format(epoch, i, loss))
        writer.add_scalar('TRAIN_LOSS', loss, epoch)
        
    return loss

def main():
    model = Model()
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model = model.cuda()
        print("using gpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100


    train_loader, val_loader = load_Data()
    
    history = open(r'.\checkpoints\history.csv','w')
    history.write('epochs , trainloss , valloss \n')

    for epoch in range(0,epochs):
        print(epoch)
        start = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        val_loss = val(model, val_loader, criterion, optimizer, epoch)
#         val_loss = train(model, val_loader, criterion, optimizer, epoch)
        print()
        print('-' * 50)
        print('EPOCH {} | LOSS {} | TIME {}'.format(epoch, train_loss, time.time() - start))
        print('-' * 50)
        print()


        history.write('{},{},{}\n'.format(epoch, train_loss, val_loss))
        save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),'loss' : train_loss}, r'.\checkpoints\checkpoint_{}.ckpt'.format(epoch))
    history.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
