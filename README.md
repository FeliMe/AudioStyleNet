# AudioStyleNet - Controlling StyleGAN through Audio

This repository contains the code for my master thesis on talking head generation by controlling the latent space of a pretrained StyleGAN model. The work was done at the [Visual Computing and Artificial Intelligence Group](https://niessnerlab.org) at the Technical University of Munich under the supervision of [Matthias Niessner](https://niessnerlab.org/members/matthias_niessner/profile.html) and [Justus Thies](https://niessnerlab.org/members/justus_thies/profile.html).

Link to thesis: [https://drive.google.com/file/d/1-p-aRpTGHM3Zz6LOIMPID34HvZOeeVJT/view?usp=sharing](https://drive.google.com/file/d/1-p-aRpTGHM3Zz6LOIMPID34HvZOeeVJT/view?usp=sharing)

<p align="center">
<img src="git_material/sample_video.gif">
</p>

## Video

See the demo video for more details and results.

[![AudioStyleNet](https://img.youtube.com/vi/yt7NSeYeNt0/0.jpg)](https://www.youtube.com/watch?v=yt7NSeYeNt0)

## Set-up

The code uses Python 3.7.5 and it was tested on PyTorch 1.4.0 with cuda 10.1.
(This project requires a GPU with cuda support.)

Clone the git project:

```
$ git clone https://github.com/FeliMe/AudioStyleNet.git
```

Create two virtual environments:

```
$ conda create -f environment.yml
$ conda create -n deepspeech python=3.6
```

Install requirements:

```
$ conda activate audiostylenet
$ pip install -r requirements.txt
$ conda activate deepspeech
$ pip install -r deepspeech/deepspeech_requirements.txt
```

Install ffmpeg

```
sudo apt install ffmpeg
```

## Demo

Download the pretrained AudioStyleNet model and the StyleGAN model from [Google Drive](https://drive.google.com/drive/folders/1EaxtIn_N_W8G1QYHakAdroxI3xpjhVub?usp=sharing) and place them in the `model/` folder.

run

```
$ python run_audiostylenet.py
```

A Google Colab Demo notebook can be found here: [https://colab.research.google.com/drive/17o1yFz9F6XmIrB6h99u1NUxbj8eFWFmR?usp=sharing](https://colab.research.google.com/drive/17o1yFz9F6XmIrB6h99u1NUxbj8eFWFmR?usp=sharing)

<!-- ## Use your own images
First, align your image or video:
```
$ python align_face.py --files <path to image or video> --out_dir data/images/
```

Project the aligned images into the latent space of StyleGAN using
```
$ python projector.py --input <path to image(s)> --output_dir data/images/
``` -->

## Use your own audio

To test the model with your own audio, first convert your audio to waveform and then run the following:

```
$ cd deepspeech
$ conda activate deepspeech
$ python run_voca_feature_extraction.py --audiofiles <path to .wav file> --out_dir ../data/audio/
$ conda deactivate
```

Then run `python run_audiostylenet.py` with adapted arguments.

<!-- ## Training

We provide code to train an AudioStyleNet model.
Additionally, prepare the training data using the helper functions in ```utils/data_helpers.py```<br/>

To start training, run
```
python train_audiostylenet.py
```

To visualize the training progress, run
```
tensorboard --logdir='./saves/audio_encoder/' --port 6006
```
This generates a [link](http://localhost:6006/) on the command line.  Open the link with a web browser to show the visualization. -->
