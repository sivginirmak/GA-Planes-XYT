from typing import Optional, Tuple, List, Union, Callable
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from tqdm import trange
import ipdb
import helpers

def nerf_forward(coords, target, nerf_model, encoding_fn, args):

  # Prepare batches.
  batches, batched_labels = helpers.prepare_chunks_sc(coords, target, encoding_function=encoding_fn, chunksize=args.chunksize)

  predictions = []
  loss = 0.

  for i, batch in enumerate(batches):

    batch = batch.unsqueeze(0)

    pred = nerf_model(batch)
    lbl = batched_labels[i]

    predictions.append(pred)
    loss += torch.nn.functional.mse_loss(pred.squeeze(), lbl)

  raw = torch.cat(predictions, dim=0).unsqueeze(-1)
  loss = loss / len(batches)

  return raw , loss


def normalize(img):
  vmax = img.max()
  vmin = img.min()
  return (img - vmin) / (vmax - vmin)

def iou(image1, image2):
    intersection = np.logical_and(image1>0, image2>0)
    union = np.logical_or(image1>0, image2>0)
    return np.sum(intersection) / np.sum(union) 

def test_threshold(thresh, gt_thresh, pred):
    # gt_thresh = gt >= thresh
    pred = normalize(pred)
    pred_thresh = pred >= thresh
    return iou(gt_thresh, pred_thresh)

def get_thr_iou(pred):
  # load gt video: (this consists of segmented frames)
  gt = np.load("data/gt_vid.npy")  # 96, 450, 350, binary valued 0 and 1
  gt_test = gt[::3,...]  # 32, 450, 350
  gt_train = list(gt)
  del gt_train[2::3]
  gt_train = np.array(gt_train)  # 64, 450, 350
  thresh_options = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] # [0.5] #

  # separate in to train and test frames
  pred_test = pred[::3,...]  # 32, 450, 350
  pred_train = list(pred)
  del pred_train[2::3]
  pred_train = np.array(pred_train)  # 64, 450, 350
  best_thresh = 0
  best_iou = 0
  for thresh in thresh_options:
      iou_val = test_threshold(thresh, gt_train, pred_train) 
      if iou_val > best_iou:
          best_iou = iou_val
          best_thresh = thresh
  test_iou = test_threshold(best_thresh, gt_test, pred_test)
  print(f' best thresh is {best_thresh} with train iou {best_iou} and test iou {test_iou}')
  return best_iou, test_iou, best_thresh # train , test ious

def train_seq(model, encode, optimizer, scheduler, args, ref_coords, all_images):
  
  # sequential training
  os.makedirs(args.savedir, exist_ok=True)


  train_psnrs_epoch = []
  train_losses_epoch = []
  val_psnrs_epoch = []
  val_losses_epoch = []

  train_ious_epoch = []
  val_ious_epoch = []

  device = args.device
  window_size = args.window_size
  stride = args.stride
  N = all_images.shape[0]

  num_windows = (N - window_size) // stride + 1

  

  for i in trange(args.n_epochs):

    pred_vid = []

    train_psnrs = []
    train_losses = []
    val_psnrs = []
    val_losses = []

    for j in range(num_windows):
      model.train()
      train_indices = np.arange(j * stride , j * stride + window_size)
      test_idx = train_indices[window_size // 2] # pick the one in the middle
      train_indices = np.delete(train_indices, window_size // 2)

      images = all_images[train_indices].to(device) # training window
      test_img = all_images[test_idx].to(device) # test

      coords = ref_coords[:, :, train_indices, :].to(device)

      height, width = test_img.shape[:2]
      test_coords = ref_coords[:, :, test_idx, :].to(device)

      # batching across time
      coords = coords.permute((2,0,1,3))

      target_img = images.reshape([-1])
      pred_img, imgLoss = nerf_forward(coords, target_img, model, encode, args)

      ### form prediction videos for each epoch
      ppp = pred_img.reshape([-1, height, width])[:2] # take the first two frames
      ppp = ppp.detach().cpu().numpy()
      pred_vid.append(ppp) 

      train_losses.append(imgLoss.item())

      imgLoss.backward()
      optimizer.step()
      if scheduler:
        scheduler.step()
      optimizer.zero_grad()
    
      psnr = -10. * torch.log10(imgLoss)
      train_psnrs.append(psnr.item())

      # testing
      model.eval()
      test_img = test_img.reshape([-1])
      pred_test, testLoss = nerf_forward(test_coords, test_img, model, encode, args)


      ppp = pred_test.reshape([1, height, width])
      ppp = ppp.detach().cpu().numpy()
      pred_vid.append(ppp) 

      

      val_losses.append(testLoss.item())
      val_psnr = -10. * torch.log10(testLoss)
      val_psnrs.append(val_psnr.item())
    
    train_losses_epoch.append(np.mean(train_losses))
    train_psnrs_epoch.append(np.mean(train_psnrs))
    val_losses_epoch.append(np.mean(val_losses))
    val_psnrs_epoch.append(np.mean(val_psnrs))
    
    pred_vid = np.concatenate(pred_vid, axis=0)
    iou_train, iou_test, thresh = get_thr_iou(pred_vid)
    train_ious_epoch.append(iou_train)
    val_ious_epoch.append(iou_test)
  
    print(f"Test Loss: {val_losses_epoch[-1]}, epoch:{i}")
    print(f"Test PSNR: {val_psnrs_epoch[-1]}, epoch:{i}")
    print(f"Test IoU: {iou_test}, epoch:{i}")
    

    # Plot example outputs
    pred_test = pred_test.reshape([height, width]).detach().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 1, 1]})
    ax[0].imshow(pred_test)
    ax[0].set_title(f'Iteration: {i}')
    ax[1].imshow(normalize(pred_test) > thresh) 
    ax[1].set_title(f'Threshold: {thresh}')
    ax[2].imshow(test_img.reshape([height, width]).detach().cpu().numpy())
    ax[2].set_title(f'Target')
    plt.savefig(args.savedir + "val_outs_iter_{}.png".format(i))
    plt.close()


  fig, ax = plt.subplots(1, 3, figsize=(12,4), gridspec_kw={'width_ratios': [1, 1, 1]})
  ax[0].plot(np.arange(args.n_epochs), train_psnrs_epoch, 'r')
  ax[0].plot(np.arange(args.n_epochs), val_psnrs_epoch, 'b')
  ax[0].set_title('PSNR (train=red, val=blue)')
  ax[1].plot(np.arange(args.n_epochs), train_losses_epoch, 'r')
  ax[1].plot(np.arange(args.n_epochs), val_losses_epoch, 'b')
  ax[1].set_title('Loss (train=red, val=blue)')
  ax[2].plot(np.arange(args.n_epochs), train_ious_epoch, 'r')
  ax[2].plot(np.arange(args.n_epochs), val_ious_epoch, 'b')
  ax[2].set_title('IoU (train=red, val=blue)')
  plt.savefig(args.savedir + f"loss_psnr_{args.seed}.png")
  plt.close()

  if args.save_on:

    np.save(args.savedir + f"val_ious_{args.seed}.npy", val_ious_epoch) 
    np.save(args.savedir + f"train_ious_{args.seed}.npy", train_ious_epoch)
    np.save(args.savedir + "pred_vid.npy", pred_vid)

  return val_psnrs_epoch, val_ious_epoch


