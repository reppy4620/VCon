# VCon: Neural Voice-Conversion project
&nbsp; VCon is Voice-Conversion framework.  
You can freely add different model, and train with common procedure.

# Models
I've implemented several models based on the following paper,
but they have different parameters and architectures than the models described in the paper.

- AutoVC : [arXiv](https://arxiv.org/abs/1905.05879)
- VQVC+ : [arXiv](https://arxiv.org/abs/2006.04154)
- FragmentVC : [arXiv](https://arxiv.org/abs/2010.14150)
- AGAIN-VC: [arXiv](https://arxiv.org/abs/2011.00316)

# Usage

## Depending packages

- numpy
- scipy
- librosa
- resemblyzer : [github](https://github.com/resemble-ai/Resemblyzer)
- tqdm
- pytorch
- pytorch-lightning
- pyworld(Although I prepared WORLD dataset, I don't use in my model training)

This project depends on those packages.  
If occur error because of missing package, please install that package.

## Preprocess
&nbsp; I assume that training data is JVS dataset.([link](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus))  
So you have to download this dataset before running preprocess.  
If you wanna use another dataset, please rewrite preprocess.py to fit to the dataset

And then, please execute following command.

```
$ python preprocess.py --dataset_dir path/to/jvs \
                       --output_dir path/to/processed \
                       --data_type normal

--dataset_dir: path of jvs dataset which you downloaded from above link
--output_dir: "path/to/processed" is one of the examples. you can decide path freely.
--data_type: 'normal', 'world' or 'wav2vec' choose one to fit your dataset. 'normal' is default. 

'normal': load .wav file to np.ndarray and save it to '.pt' file.
'world': extract WORLD features(f0, sp, ap) and then save it.
'wav2vec': prepared for FragmentVC model but its training has failed.
```

When preprocess ends without --world, .dat files per speaker are in output_dir.

## Training

```
$ python main.py --config_path configs/<model_name>.yaml \
                 --data_dir path/to/processed \
                 --model_dir path/to/model_dir

--config_path or -c: path of config file
--data_dir or -d: path of preprocessed dir.
--model_dir or -m: you can decide path freely.
```

## Inference

```
$ python inference.py --src_path path/to/src.wav \
                      --tgt_path path/to/tgt.wav \
                      --config_path configs/<model_name>.yaml \
                      --ckpt_path path/to/model-checkpoint.ckpt \
                      --output_dir path/to/output_dir

--src_path: utterance of source speaker
--tgt_path: utterance of target speaker
--config_path: path of config file
--ckpt_path: path of checkpoint
--output_dir: you can decide path freely. if inference ends, output_dir will have src.wav, tgt.wav, gen.wav.
```

# How to add different model
1. Create model.py, pl_model.py by using modules/<model_name> as reference
2. Add pytorch model, lightning module and lightning data module class to dict in utils/from_config.py
3. Create a config file for new model.

# LICENSE
MIT License.