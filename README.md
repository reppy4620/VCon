# VCon: Neural Voice-Conversion project
&nbsp; VCon is Voice-Conversion project which is for my graduation research.

Training process is now running with single GTX1080Ti, but good result may not be generated even if reconstruction in my situation.

# Usage

## Preprocess
&nbsp; Used dataset in this project is VCTK dataset.([link](http://www.udialogue.org/download/VCTK-Corpus.tar.gz))  
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
