#!/bin/bash

exp_name=semiconvex-gaplane
feature=add # or mult
semi_convex=1
convex=0
lr=5e-4 
gpu=0
model_type=GAplanes #triplane GAplanes
pe=0
save_on=1
n_epochs=20
## gaplane sizes
Cl=32
Cp=16
Cv=8
Nl=128
Np=128
Nv=64

comments=first-experiment
chunksize=157500


python run_exp.py --exp_name $exp_name \
                  --model_type $model_type \
                  --gpu $gpu \
                  --pe $pe \
                  --convex $convex \
                  --semi_convex $semi_convex \
                  --save_on $save_on \
                  --n_epochs $n_epochs \
                  --Cp $Cp \
                  --Cl $Cl \
                  --Cv $Cv \
                  --Nv $Nv \
                  --Np $Np \
                  --Nl $Nl \
                  --feature $feature \
                  --comments $comments \
                  --chunksize $chunksize \
                  --lr $lr 






