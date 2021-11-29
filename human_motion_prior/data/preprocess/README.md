## Data Pre-process

1. Download the [AMASS dataset](https://amass.is.tue.mpg.de/download.php). Specifically, in this project, we use the following subsets
```
['Transitions_mocap', 'BMLmovi', 'MPI_mosh', 'MPI_Limits', 'SSM_synced', 'SFU', 'MPI_HDM05', 'TCD_handMocap', 
'HumanEva', 'TotalCapture', 'DFaust_67', 'ACCAD', 'CMU', 'EKUT', 'Eyes_Japan_Dataset', 'KIT']
```

2. Put the downloaded data in directory raw_data as follows:
```
raw_data
├── Transitions_mocap
├── BMLmovi
├── MPI_mosh
| ...
```

3. Extract each frame from the data with the rate of 30 fps
```bash
python frame_extraction.py
```

4. Save the necessary  parameters and merge into 128 frames per sequence. 
```bash
python to_sequence_data.py
```
The generated data file is *amass_smpl_30fps_128frame.npz*
