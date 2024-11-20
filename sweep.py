import numpy as np
import subprocess

useGA = 1
# learning_rates = np.linspace(5e-4,1e-2,8) ##2e-3 gamma 0.9 ok
learning_rates = [5e-4] #, 0.003, 0.004]#[5e-4]
seeds =  [1,2,3] #[0]
if useGA:
    experiment_names = [
        # "gaplane-convex",
        "gaplane-semiconvex",
        "gaplane-nonconvex-concat",
        # "gaplane-nonconvex-mult"
    ] 
else:
    experiment_names = [
        "triplane-convex",
        "triplane-semiconvex",
        "triplane-nonconvex-concat",
        "triplane-nonconvex-mult"
    ]

# Loop through the learning rates
for exp_name in experiment_names:
    for seed in seeds:
        for lr in learning_rates:
            print(f"Running {exp_name} experiment with learning rate: {lr}")
            feature = "mult" if "mult" in exp_name else "add"
            if "semiconvex" in exp_name:
                semi_convex = 1
                convex = 0
            elif "nonconvex" in exp_name:
                semi_convex = 0
                convex = 0
            else: # convex
                semi_convex = 0
                convex = 1
            if useGA:
                subprocess.run([
                    "bash", "run.sh",
                    str(lr),               
                    exp_name,              
                    feature,
                    str(semi_convex),
                    str(convex),
                    str(seed)             
                ])
            else:
                subprocess.run([
                    "bash", "run_triplane.sh",
                    str(lr),               
                    exp_name,              
                    feature,
                    str(semi_convex),
                    str(convex)             
                ])
