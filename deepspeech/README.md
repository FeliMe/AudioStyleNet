# DeepSpeech Feature Extraction(VOCA: Voice Operated Character Animation)

This repository is adapted from VOCA: Voice Operated Character Animation by Cudeiro et al.
https://github.com/TimoBolkart/voca

VOCA is a simple and generic speech-driven facial animation framework that works across a range of identities. This codebase demonstrates how to synthesize realistic character animations given an arbitrary speech signal and a static character mesh. For details please see the scientific publication

Capture, Learning, and Synthesis of 3D Speaking Styles.<br/> 
D. Cudeiro*, T. Bolkart*, C. Laidlaw, A. Ranjan, M. J. Black<br/>
Computer Vision and Pattern Recognition (CVPR), 2019

A pre-print of the publication can be found on the [project website](https://voca.is.tue.mpg.de).


## Set-up

Set up virtual environment:
```
$ mkdir <your_home_dir>/.virtualenvs
$ virtualenv --no-site-packages <your_home_dir>/.virtualenvs/voca
```

Activate virtual environment:
```
$ cd voca
$ source <your_home_dir>/voca/bin/activate
```

The code uses Python 2.7 and it was tested on Tensorflow 1.12.0. The requirements (including tensorflow) can be installed using:
```
pip install -r requirements.txt
```


## Extract Features

```
python run_voca_feature_extraction.py --audiofiles <path to file or folder> --out_path <output dir>
```


## License

Free for non-commercial and scientific research purposes. By using this code, you acknowledge that you have read the license terms (https://voca.is.tue.mpg.de/license), understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not use the code.


## Referencing VOCA

When using this code, please cite VOCA. You find the most up to date bibliographic information at https://voca.is.tue.mpg.de.


## Acknowledgement

We thank Raffi Enficiaud and Ahmed Osman for pushing the release of psbody.mesh.









