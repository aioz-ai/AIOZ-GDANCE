

# [Music-Driven Group Choreography (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Le_Music-Driven_Group_Choreography_CVPR_2023_paper.pdf)
### *[Nhat Le](https://minhnhatvt.github.io/), [Thang Pham](https://phamtrongthang123.github.io/), [Tuong Do](https://scholar.google.com/citations?user=qCcSKkMAAAAJ&hl=en), [Erman Tjiputra](https://sg.linkedin.com/in/erman-tjiputra), [Quang D. Tran](https://scholar.google.com/citations?user=DbAThEgAAAAJ&hl=en), [Anh Nguyen](https://cgi.csc.liv.ac.uk/~anguyen/)*
### [[Project Page](https://aioz-ai.github.io/AIOZ-GDANCE/)] [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Le_Music-Driven_Group_Choreography_CVPR_2023_paper.pdf)]


![](https://vision.aioz.io/f/1e065962a9b747b3a856/?dl=1)*<center> We demonstrate the AIOZ-GDANCE dataset with in-the-wild videos, music audio, and 3D group dance motion. </center>*

## Abstract
> Music-driven choreography is a challenging problem with a wide variety of industrial applications. Recently, many methods have been proposed to synthesize dance motions from music for a single dancer. However, generating dance motion for a group remains an open problem. In this paper, we present GDANCE, a new large-scale dataset for music-driven group dance generation. Unlike existing datasets that only support single dance, our new dataset contains group dance videos, hence supporting the study of group choreography. We propose a semi-autonomous labeling method with humans in the loop to obtain the 3D ground truth for our dataset. The proposed dataset consists of 16.7 hours of paired music and 3D motion from in-the-wild videos, covering 7 dance styles and 16 music genres. We show that naively applying single dance generation technique to creating group dance motion may lead to unsatisfactory results, such as inconsistent movements and collisions between dancers. Based on our new dataset, we propose a new method that takes an input music sequence and a set of 3D positions of dancers to efficiently produce multiple group-coherent choreographies. We propose new evaluation metrics for measuring group dance quality and perform intensive experiments to demonstrate the effectiveness of our method. Our code and dataset will be released to facilitate future research on group dance generation.

## Table of Contents
1. [AIOZ-GDANCE Dataset](#aioz-gdance-dataset)
2. [Visualizing](#visualizing)
3. [Prerequisites](#prerequisites)
4. [Usage](#usage)

## AIOZ-GDANCE Dataset
**[Download]** The dataset can be downloaded at [Link](https://vision.aioz.io/f/430eb9d90552480e8b4e/?dl=1)

The data directory is organized as follows:
- **split_sequence_names.txt**:
    -   a txt file containing seperate sequence names in the data (each sequence should have unique name or id)
- **musics**:
    -  contains raw music .wav file of each sequence with the corresponding name. The music frames are aligned with the motion frames.
- **motions_smpl**:
    -  contains the motion file of each sequence with the corresponding name, the motion is provided in .pkl file format.
    -  Each data dictionary mainly includes the following items:
        - `'smpl_poses': shape[num_persons x num_frames x 72]`: the motions contain 72-D vector pose sequences in SMPL pose format (24 joints).
        - ``'root_trans': shape[num_persons x num_frames x 3]``: sequences of root translation.

Here is an example python script to read the motion file
```python
import pickle
import numpy as np
data = pickle.load(open("sequence_name.pkl","rb"))
print(data.keys())

smpl_poses = data['smpl_poses']
smpl_trans = data['root_trans']

# ... may utilize the pose by using SMPL forward function: https://github.com/vchoutas/smplx
```
![Figure 4](https://github.com/aioz-ai/AIOZ-GDANCE/blob/main/4r.gif) 

![Figure 5](https://github.com/aioz-ai/AIOZ-GDANCE/blob/main/1r.gif)

## Visualizing


We provide demo code for loading and visualizing the motions. 

### Prerequisites
First, you need to download the [SMPL model](https://smpl.is.tue.mpg.de/) (v1.0.0) and rename the model files for visualization. The directory structure of the data is expected to be:

The directory structure of the data is expected to be:
```
<DATA_DIR>
├── motions_smpl/
├── musics/
└── split_sequence_names.txt

<SMPL_DIR>
├── SMPL_MALE.pkl
└── SMPL_FEMALE.pkl
```

Then run this to install the necessary packages
```
pip install scipy torch smplx chumpy vedo trimesh
pip install numpy==1.23.0
```

### Usage
#### Visualize the SMPL joints
The following command will first calculate the SMPL joint locations (joint rotations and root translation) and then plot on the 3D figure in realtime.
``` bash
python vis_smpl_kpt.py \
  --data_dir <DATA_DIR>/motions_smpl \
  --smpl_path <SMPL_DIR>/SMPL_FEMALE.PKL \
  --sequence_name sequence_name.pkl
```

#### Visualize the SMPL Mesh
The following command will calculate the SMPL meshes and visualize in 3D. 
``` bash
python vis_smpl_mesh.py \
  --data_dir <DATA_DIR>/motions_smpl \
  --smpl_path <SMPL_DIR>/SMPL_FEMALE.PKL \
  --sequence_name sequence_name.pkl
```


## TODO
- [x] ~~**Dataset**~~
- [ ] **Baseline model & evaluation code**: TBD
- [ ] **Training code**: TBD



## Citation
```
@inproceedings{aiozGdance,
    author    = {Le, Nhat and Pham, Thang and Do, Tuong and Tjiputra, Erman and Tran, Quang D. and Nguyen, Anh},
    title     = {Music-Driven Group Choreography},
    journal   = {CVPR},
    year      = {2023},
}		
```

## License
Software Copyright License for non-commercial scientific research purposes.
Please read carefully the following [terms and conditions](LICENSE) and any accompanying
documentation before you download and/or use AIOZ-GDANCE data, model and
software, (the "Data & Software"), including 3D meshes, images, videos,
textures, software, scripts, and animations. By downloading and/or using the
Data & Software (including downloading, cloning, installing, and any other use
of the corresponding github repository), you acknowledge that you have read
these [terms and conditions](LICENSE), understand them, and agree to be bound by them. If
you do not agree with these [terms and conditions](LICENSE), you must not download and/or
use the Data & Software. Any infringement of the terms of this agreement will
automatically terminate your rights under this [License](LICENSE).


## Acknowledgement
This repo used visualization code from [AIST++](https://github.com/google/aistplusplus_api/tree/main)

