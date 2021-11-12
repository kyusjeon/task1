import time
import cv2
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vit_unet import VitUNet16
from task1utils import pixel_accuracy, mIoU
from task1dataset import DroneDataset, create_df

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, patch=False):
    writer = SummaryWriter()
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            miou = mIoU(output, mask)
            pixel_acc = pixel_accuracy(output, mask)
            iou_score += miou
            accuracy += pixel_acc
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lr = get_lr(optimizer)
            lrs.append(lr)
            scheduler.step()

            running_loss += loss.item()
            
            writer.add_scalar("train/loss", loss, len(train_loader)*e+i)
            writer.add_scalar("train/miou", miou, len(train_loader)*e+i)
            writer.add_scalar("train/acc", pixel_acc, len(train_loader)*e+i)
            writer.add_scalar("train/lr", lr, len(train_loader)*e+i)

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    miou = mIoU(output, mask)
                    pixel_acc = pixel_accuracy(output, mask)
                    val_iou_score += miou
                    test_accuracy += pixel_acc
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()
                    
                    writer.add_scalar("val/loss", loss, len(val_loader)*e+i)
                    writer.add_scalar("val/miou", miou, len(val_loader)*e+i)
                    writer.add_scalar("val/acc", pixel_acc, len(val_loader)*e+i)

            # calculate mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                min_loss = (test_loss / len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model.state_dict(), './VitUNet_mIoU-{:.3f}.pt'.format(val_iou_score / len(val_loader)))

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                # if not_improve == 70:
                #     print('Loss not decrease for 70 times, Stop Training')
                #     break

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history

def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train', add_help=False)
    parser.add_argument('--data_dir', default='.', type=str)
    parser.add_argument('--save_dir', default='./VitUNet.pt', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--n_classes', default=23, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()
    
    IMAGE_PATH = args.data_dir + '/original_images/'
    MASK_PATH = args.data_dir + '/label_images_semantic/'

    df = create_df(IMAGE_PATH)
    
    X_trainval, X_test = train_test_split(df['id'].values, 
                                          test_size=0.1, 
                                          random_state=19)
    X_train, X_val = train_test_split(X_trainval, 
                                      test_size=0.15, 
                                      random_state=19)

    img = Image.open(IMAGE_PATH + df['id'][100] + '.jpg')
    mask = Image.open(MASK_PATH + df['id'][100] + '.png')
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    t_train = A.Compose([A.Resize(512, 768, interpolation=cv2.INTER_NEAREST), 
                         A.HorizontalFlip(), 
                         A.VerticalFlip(),
                         A.GridDistortion(p=0.2), 
                         A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                         A.GaussNoise()
                         ])
    t_val = A.Compose([A.Resize(512, 768, interpolation=cv2.INTER_NEAREST), 
                       A.HorizontalFlip(),
                       A.GridDistortion(p=0.2)
                       ])

    #datasets
    train_set = DroneDataset(IMAGE_PATH, 
                             MASK_PATH, 
                             X_train, 
                             mean, 
                             std, 
                             t_train, 
                             patch=False
                             )
    val_set = DroneDataset(IMAGE_PATH, 
                           MASK_PATH, 
                           X_val, 
                           mean, 
                           std, 
                           t_val, 
                           patch=False
                           )

    #dataloader
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              drop_last=True, 
                              num_workers=args.num_workers
                              )
    val_loader = DataLoader(val_set, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            drop_last=True, 
                            num_workers=args.num_workers
                            )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VitUNet16()
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.max_lr, 
                                  weight_decay=args.weight_decay
                                  )
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                args.max_lr, 
                                                epochs=args.epochs,
                                                steps_per_epoch=len(train_loader)
                                                )

    history = fit(args.epochs,
                  model, 
                  train_loader, 
                  val_loader, 
                  criterion, 
                  optimizer, 
                  sched, 
                  device
                  )

    torch.save(model.state_dict(), args.save_dir)

    plot_loss(history)
    plot_score(history)
    plot_acc(history)
