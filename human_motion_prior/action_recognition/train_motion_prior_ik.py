import numpy as np
from human_motion_prior.action_recognition.motion_prior_ik import run_motion_prior_trainer

from configer import Configer


# ALL Setting is in *.ini
args = {}
ps = Configer(default_ps_fname='./motion_prior_ik_defaults.ini', **args) # This is the default configuration

# Make a message to describe the purpose of this experiment
expr_message = '\n[%s] %d H neurons, latentD=%d, batch_size=%d,  kl_coef = %.1e\n' \
               % (ps.expr_code, ps.num_neurons, ps.latentD, ps.batch_size, ps.kl_coef)
expr_message += '\n'
ps.expr_message = expr_message


run_motion_prior_trainer(ps)
