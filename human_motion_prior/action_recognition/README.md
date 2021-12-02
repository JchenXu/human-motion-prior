## Exploring Versatile Prior for Human Motion via Motion Frequency Guidance

**This is the codebase for action recognition in [human-motion-prior](https://github.com/JchenXu/human-motion-prior).**

## Train MotionPriorIK
### Preparation
- You should first follow the [README.md](https://github.com/JchenXu/human-motion-prior/blob/main/README.md) to install related dependencies, and you should prepare the human motion prior trained by yourself or download our pretrained one [here](https://drive.google.com/file/d/12LAlvHJ34qNkOqCVawSDjBquQH6MI2Ma/view?usp=sharing).

- Put the pretrained motion prior in *human_motion_prior/models/pre_trained/motion_prior/pretrained_priorD.pth*


### Training
Run the commands below to start training:

```bash
sh run_script.sh 4
```


## Action Recognition
- Dowload the [BABEL](https://babel.is.tue.mpg.de/) dataset from [here](https://human-movement.is.tue.mpg.de/babel_feats_labels.tar.gz).
- Extract data and put them into the *./data/*
- Run the following code to convert action sequence data into prior embedding.
```
python motion_to_embedding.py
```
- Run the following code to train a three-layer MLP to achieve action recognition.
```
python ar_training.py
```


## Citation

```bibtex
@inproceedings{human_motion_prior,
  title = {Exploring Versatile Prior for Human Motion via Motion Frequency Guidance},
  author = {Jiachen Xu, Min Wang, Jingyu Gong, Wentao Liu, Chen Qian, Yuan Xie, Lizhuang Ma},
  booktitle = {2021 international conference on 3D vision (3DV)},
  year = {2021}
}
```
