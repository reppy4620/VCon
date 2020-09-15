# VCon: Neural Voice-Conversion project
&nbsp; VCon is Voice-Conversion project which is for my graduation research.

# Usage

## Depending package

- numpy
- librosa
- resemblyzer : [github](https://github.com/resemble-ai/Resemblyzer)
- tqdm
- pytorch
- joblib

This project depends on those packages.  
If occur error because of missing package, please install that package.

## Preprocess
&nbsp; Training data is VCTK dataset.([link](http://www.udialogue.org/download/VCTK-Corpus.tar.gz))  
So you have to download this dataset before running preprocess.  

And then, please execute following command.

```
$ python preprocess.py --dataset_dir path/to/VCTK-Corpus --output_dir path/to/output_dir
```

When preprocess ends, .dat files per speaker are in output_dir.  
If your output_dir differ with output_dir property in configs/*.yaml, overwrite its property to path of your output_dir.

## Training

Quartz example
```
$ python main.py configs/quartz.yaml
```

## Inference
If you trained quartz model, set config_path to configs/quartz.yaml

Quartz example
```
$ python inference.py --src_path path/to/src.wav \
                      --tgt_path path/to/tgt.wav \
                      --config_path configs/quartz.yaml \
                      --ckpt_path path/to/model.ckpt \
                      --out_path path/to/output.wav
```

# How to add different model
1. Create model.py and pl_model.py file by using nn/autovc or nn/quartz as reference
2. Add model class to dict in utils/from_config.py


# Reference

1. AutoVC : [paper](https://arxiv.org/abs/1905.05879), [github](https://github.com/auspicious3000/autovc)
2. QuartzNet: [paper](https://arxiv.org/abs/1910.10261)
