# "What happens if..." Learning to Predict the Effect of Forces in Images
This is the source code for a deep net that predicts the effect of applying a force to an object shown in a static image.

### Citation
If you find the code useful in your research, please consider citing:
```
@inproceedings{mottaghiECCV16,
    Author = {Roozbeh Mottaghi and Mohammad Rastegari and Abhinav Gupta and Ali Farhadi},
    Title = {``What happens if..." Learning to Predict the Effect of Forces in Images},
    Booktitle = {ECCV},
    Year = {2016}
}
```

### Requirements
- Python 3.5+
- Pytorch 0.4.0+
- Tensorboard

### Training

Prepare dataset: 
"""
./upload_dataset.sh

"""
To train the model, run 

'''
python train.py
'''

Check the argument list to set hyperparatmers and paths.

### Test
To test, run

'''
python test.py
'''

I am getting accracy of 20.2% with Alexnet as encoder. If you can experiment with Resnet and other variants, please send a pull request. 

This implementation is based on original lua implementation which can be found at [https://github.com/allenai/forces/](https://github.com/allenai/forces/)
