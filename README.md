# A METHOD FOR X-RAY IMAGE LANDMARKS LOCALIZATION USING CYCLIC COORDINATE-GUIDED STRATEGY

Code of ICASSP 2024 paper: "A METHOD FOR X-RAY IMAGE LANDMARKS LOCALIZATION USING CYCLIC COORDINATE-GUIDED STRATEGY".

## Installation

1.  Clone this repository.

```Shell
git clone https://github.com/switch626/CCG-CL.git
```

2.  Create a new conda environment.

```Shell
conda create -n ccgcl python=3.8
conda activate ccgcl
```

3.  Install the dependencies in requirements.txt.

```Shell
pip install -r requirements.txt
```

4. Prepare the ISBI dataset or the Hand900 dataset.

5. Train and Test the model.

```Shell
python train.py
```

## Acknowledgments
Great thanks for these papers and their open-source codes：[FoveatedPyramid](https://github.com/logangilmour/FoveatedPyramid), [Maxim](https://github.com/google-research/maxim).
