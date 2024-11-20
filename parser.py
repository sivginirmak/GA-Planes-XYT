import argparse

def get_parser():
    ''' parse arguments '''

    p = argparse.ArgumentParser()

    ########## required command line inputs ##########
    p.add_argument('--exp_name', type=str, required=True,
                   help='directory to save all output, checkpoints, etc')

    p.add_argument('--model_type', type=str, required=True, choices=['triplane', "GAplanes"])
    p.add_argument('--convex', type=int, required=True,
                    help='convex model indicator for triplane / ga planes')
    p.add_argument('--semi_convex', type=int, required=True,
                    help='semi-convex model indicator for triplane / ga planes')
      
    # Cl=args.Cl, Cp=args.Cp, Cv=args.Cv, Nv=args.Nv
    p.add_argument('--Cl', type=int, default=32)
    p.add_argument('--Cp', type=int, default=32)
    p.add_argument('--Cv', type=int, default=8)
    p.add_argument('--Nl', type=int, default=128)
    p.add_argument('--Np', type=int, default=128)
    p.add_argument('--Nv', type=int, default=64)

    ###### seq training
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--window_size', type=int, default=5)


    ########## general training ##########

    p.add_argument('--n_epochs', type=int, default=20,
                   help='number of training epochs')
    p.add_argument('--pe', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0, 
                   help='gpu id to use for training')
    p.add_argument('--lr', type=float, default=5e-4, 
                   help='learning rate')
    p.add_argument('--chunksize', type=int, default=2500, 
                   help='chunk size')
    p.add_argument('--display_rate', type=int, default=250, 
                   help='display rate')
    p.add_argument('--save_on', type=int, default=1, 
                   help='save models?')
    p.add_argument('--seed', type=int, default=0)
     
    ## add optional comments on the experiment
    p.add_argument("--comments", type=str, default=None)
    ## select feature merging
    p.add_argument("--feature", type=str, default="add") # add / multiply

    args = p.parse_args()
    

    args.savedir = f"Experiments/{args.exp_name}/"

    # convert ints to bools
    args.convex = bool(args.convex)
    args.semi_convex = bool(args.semi_convex)
    args.useSVM = bool(args.useSVM)
    args.pe = bool(args.pe)
    args.save_on = bool(args.save_on)

    
    return args

args = get_parser()
print(args)





