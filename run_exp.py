import torch
import numpy as np
from triplane_models import *
import parser
import helpers
import nerf_fcns

import ipdb
# init models (merged!)
def init_models(args):
  # Initialize models, encoders, and optimizer for NeRF training.
  device = args.device
  if args.pe:
    d_input = 3           # Number of input dimensions
    n_freqs = 10
    encoder = helpers.PositionalEncoder(d_input, n_freqs, log_space=True)
    encode = lambda x: encoder(x)
    in_features = encoder.d_output

  else:
    encode = None
    in_features = 3
  

  if args.model_type == "triplane":
    if args.feature == "add":
      if args.semi_convex:
        # print("semi convex")
        model = MiniTriplane(convex=True, addlines=False, Cp=args.Cp, Np=args.Np)
      elif args.convex: 
        # print("convex")
        model = ConvexTriplane(addlines=False, Cp=args.Cp, Np=args.Np)
      else:
        # print("nonconvex")
        model = MiniTriplane(convex=False, addlines=False, Cp=args.Cp, Np=args.Np)
    elif args.feature == "mult":
      model = Kplane(Cp=args.Cp, Np=args.Np)

  elif args.model_type == "GAplanes":
    if args.feature == "add":
      if args.semi_convex:
        model = MiniTriplane(convex=True, addlines=True, Cl=args.Cl, Cp=args.Cp, Cv=args.Cv, Nl=args.Nl, Np=args.Np, Nv=args.Nv) # Cl=Cl, Cp=Cp, Cv=Cv, Nv=Nv
      elif args.convex:
        model = ConvexTriplane(addlines=True, Cl=args.Cl, Cp=args.Cp, Cv=args.Cv, Nl=args.Nl, Np=args.Np, Nv=args.Nv)
      else:
        model = MiniTriplane(convex=False, addlines=True, Cl=args.Cl, Cp=args.Cp, Cv=args.Cv, Nl=args.Nl, Np=args.Np, Nv=args.Nv)
    elif args.feature == "mult":
      model = GAplane(convex=False, Cl=args.Cp, Cv=args.Cv, Nl=args.Nl, Np=args.Np, Nv=args.Nv) ## Cl = Cp
    
  print(model)
  model.to(device)

  return model, encode



def main():

    args = parser.get_parser()


    # set cuda device, random seed
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(device)
    args.device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ## needed for all nerf / sc / triplane etc.
    configs = {} # bundle all needed variables for training etc.

    images = np.load("data/person_masks.npy")
    images = images[:, :450, 350:700] # crop zeros

    # np.save("xyt_data/gt_vid.npy", images[:96]) # uncomment to form the gt for threshold optimization for iou score

    all_images = torch.from_numpy(images).to(device)
    all_images = all_images.to(torch.float32)
    T = images.shape[0]
    height, width = all_images.shape[1:]


    grid_x, grid_y = torch.linspace(-1, 1, width), torch.linspace(-1, 1, height)
    grid_t = torch.linspace(-1, 1, T)

    xx, yy, zz = torch.meshgrid(grid_x, grid_y, grid_t, indexing='xy')

    ref_coords = torch.stack([xx,yy,zz], -1).to(device) # torch.Size([H, W, T, 3])
    
    ### model selection (get from parser)
    model, encode = init_models(args)
    print(helpers.count_parameters(model))

    torch.manual_seed(args.seed)


    ## optimizer
    model_params = list(model.parameters())

    optimizer = torch.optim.AdamW(model_params, lr=args.lr)
    scheduler = None # torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9) ## uncomment if you want 

    val_psnrs, val_ious = nerf_fcns.train_seq(model, encode, optimizer, scheduler, args, ref_coords, all_images)

    # save the parser parameters
    with open(args.savedir + f'options.txt', 'w') as f:
      for arg in vars(args):
          f.write(f"{arg}: {getattr(args, arg)}\n")
      f.write(f"test iou: {val_ious[-1]}\n")
    





if __name__ == '__main__':
    main()