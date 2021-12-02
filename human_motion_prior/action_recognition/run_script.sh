export PYTHONPATH=../../
python -m torch.distributed.launch --nproc_per_node=$1 train_motion_prior_ik.py 

