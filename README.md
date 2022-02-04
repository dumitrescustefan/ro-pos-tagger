# Romanian Part Of Speech Tagger

This repo contains code to train a POS tagger on Universal Dependencies' Ro RRT dataset.

It is used to eval transformer language models on [Romanian Transformers](https://github.com/dumitrescustefan/Romanian-Transformers) . Maybe it will develop into a usable model sometime in the future. In the mean time, if you want POS tagging, use NLP-Cube, Stanza or Spacy ;)

Here are the evals on the following models, with default params (except large models with batch_size 1 and grad_accumulation 8)

| Model                                          	| UPOS F1 	| XPOS F1 	|
|------------------------------------------------	|:-------:	|:-------:	|
| dumitrescustefan/bert-base-romanian-cased-v1   	|  0.9821 	|  0.9741 	|
| dumitrescustefan/bert-base-romanian-uncased-v1 	|  0.9826 	|  0.9728 	|
| racai/distilbert-base-romanian-cased           	|  0.9637 	|  0.9255 	|
| readerbench/RoGPT2-base                        	|  0.8982 	|  0.8015 	|
| readerbench/RoGPT2-medium                      	|  0.9092 	|  0.8201 	|
| xlm-roberta-base                               	|  0.9706 	|  0.9581 	|

