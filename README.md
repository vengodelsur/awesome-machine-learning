# Awesome Machine Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

*This version includes cherry-picked C++ and Python projects with some interesting issues (for celebrating Hacktoberfest in Open Data Science community).*

A curated list of awesome machine learning frameworks, libraries and software (by language). Inspired by `awesome-php`.

If you want to contribute to this list (please do), send me a pull request or contact me [@josephmisiti](https://twitter.com/josephmisiti).
Also, a listed repository should be deprecated if:

* Repository's owner explicitly say that "this library is not maintained".
* Not committed for long time (2~3 years).

Further resources:

* For a list of free machine learning books available for download, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md).

* For a list of (mostly) free machine learning courses available online, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/courses.md).

* For a list of blogs and newsletters on data science and machine learning, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/blogs.md).

* For a list of free-to-attend meetups and local events, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/meetups.md).

## Table of Contents

### Frameworks and Libraries
<!-- MarkdownTOC depth=4 -->

- [Awesome Machine Learning ![Awesome](https://github.com/sindresorhus/awesome)](#awesome-machine-learning-awesomehttpsgithubcomsindresorhusawesome)
  - [Table of Contents](#table-of-contents)
    - [Frameworks and Libraries](#frameworks-and-libraries)
    - [Tools](#tools)
  - [C++](#c)
      - [Computer Vision](#computer-vision-1)
      - [General-Purpose Machine Learning](#general-purpose-machine-learning-2)
      - [Natural Language Processing](#natural-language-processing)
  - [Python](#python)
      - [Computer Vision](#computer-vision-4)
      - [Natural Language Processing](#natural-language-processing-10)
      - [Recommender systems](#recommender-systems)
      - [Reinforcement Learning](#reinforcement-learning)
  - [TensorFlow](#tensorflow)
      - [General-Purpose Machine Learning](#general-purpose-machine-learning-28)
  - [Tools](#tools-1)
      - [Neural Networks](#neural-networks-2)
      - [Misc](#misc-2)
  - [Credits](#credits)



<a name="cpp"></a>
## C++

<a name="cpp-cv"></a>
#### Computer Vision

* [DLib](http://dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
Recommended issues tags: [help wanted](https://github.com/davisking/dlib/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
Some interesting issues:
  * [3D convolution and pooling](https://github.com/davisking/dlib/issues/989)
  * [Provide python wheels](https://github.com/davisking/dlib/issues/1527)
  * [Support for reading images in correct orientation using EXIF data](https://github.com/davisking/dlib/issues/1706)
  * [Add support of extended kernel recursive least squares algorithm](https://github.com/davisking/dlib/issues/515)
  * [find_optimal_rect_filter not interface in Python API](https://github.com/davisking/dlib/issues/1887)
* [OpenCV](https://opencv.org) - OpenCV has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.
Recommended issues tags: [needs investigation](https://github.com/opencv/opencv/issues?q=is%3Aissue+is%3Aopen+label%3A%22needs+investigation%22), [good first issue](https://github.com/opencv/opencv/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), [category: samples](https://github.com/opencv/opencv/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+samples%22), [category: documentation](https://github.com/opencv/opencv/issues?q=is%3Aissue+is%3Aopen+label%3A%22category%3A+documentation%22), [feature](https://github.com/opencv/opencv/issues?q=is%3Aissue+is%3Aopen+label%3Afeature)
Some interesting issues:
  * [Incomplete Python BRIEF Example](https://github.com/opencv/opencv/issues/14093)
  * [kalman.cpp sample (not module) seems to compare wrong entities](https://github.com/opencv/opencv/issues/5042)
  * [Crop and resize to work with blobFromImages](https://github.com/opencv/opencv/issues/15149)
  * [Python typing stub](https://github.com/opencv/opencv/issues/14590)

<a name="cpp-general-purpose"></a>
#### General-Purpose Machine Learning

* [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, contains fast inference implementation and supports CPU and GPU (even multi-GPU) computation.
Recommended issues tags: [documentation](https://github.com/catboost/catboost/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation),[need info](https://github.com/catboost/catboost/issues?q=is%3Aissue+is%3Aopen+label%3A%22need+info%22), [help wanted](https://github.com/catboost/catboost/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22), [good first issue](https://github.com/catboost/catboost/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
* [DLib](http://dlib.net/ml.html) - A suite of ML tools designed to be easy to imbed in other applications.
Recommended issues tags: [help wanted](https://github.com/davisking/dlib/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
Some interesting issues:
  * [3D convolution and pooling](https://github.com/davisking/dlib/issues/989)
  * [Provide python wheels](https://github.com/davisking/dlib/issues/1527)
  * [Support for reading images in correct orientation using EXIF data](https://github.com/davisking/dlib/issues/1706)
  * [Add support of extended kernel recursive least squares algorithm](https://github.com/davisking/dlib/issues/515)
  * [find_optimal_rect_filter not interface in Python API](https://github.com/davisking/dlib/issues/1887)
* [DyNet](https://github.com/clab/dynet) - A dynamic neural network library working well with networks that have dynamic structures that change for every training instance. Written in C++ with bindings in Python.
Some interesting issues:
  * [Simple C++ LSTM example for time-series prediction](https://github.com/clab/dynet/issues/1585)
* [Featuretools](https://github.com/featuretools/featuretools) - A library for automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning using reusable feature engineering "primitives". 
Recommended issues tags: [Good First Issue](https://github.com/featuretools/featuretools/issues?q=is%3Aissue+is%3Aopen+label%3A%22Good+First+Issue%22), [Feature Request](https://github.com/featuretools/featuretools/issues?q=is%3Aissue+is%3Aopen+label%3A%22Feature+Request%22)
* [igraph](http://igraph.org/) - General purpose graph library.
Recommended issues tags: [wishlist](https://github.com/igraph/igraph/issues?q=is%3Aissue+is%3Aopen+label%3Awishlist), [good first issue](https://github.com/igraph/igraph/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), [PR welcome](https://github.com/igraph/igraph/issues?q=is%3Aissue+is%3Aopen+label%3A%22PR+welcome%22), [help wanted](https://github.com/igraph/igraph/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
* [LightGBM](https://github.com/Microsoft/LightGBM) - Microsoft's fast, distributed, high performance gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.
[List of feature requests](https://github.com/microsoft/LightGBM/issues/2302), including new algorithms, new metric functions, GPU support and more.
* [mlpack](https://www.mlpack.org/) - A scalable C++ machine learning library.
Recommended issues tags: [t: feature request](https://github.com/mlpack/mlpack/issues?q=is%3Aissue+is%3Aopen+label%3A%22t%3A+feature+request%22), [good first issue](https://github.com/mlpack/mlpack/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), [help wanted](https://github.com/mlpack/mlpack/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
Some interesting issues:
  * [LMNN: don't recompute impostors during an SGD batch](https://github.com/mlpack/mlpack/issues/1490)
  * [MDL penalty for decision tree splits](https://github.com/mlpack/mlpack/issues/883)
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, Javascript and more.
Recommended issues tags: [Test](https://github.com/apache/incubator-mxnet/issues?q=is%3Aissue+is%3Aopen+label%3ATest), [Feature request](https://github.com/apache/incubator-mxnet/issues?q=is%3Aissue+is%3Aopen+label%3A%22Feature+request%22), [Good First Issue](https://github.com/apache/incubator-mxnet/issues?q=is%3Aissue+is%3Aopen+label%3A%22Good+First+Issue%22), [Doc](https://github.com/apache/incubator-mxnet/issues?q=is%3Aissue+is%3Aopen+label%3ADoc)
* [nmslib](https://github.com/nmslib/nmslib) - Non-Metric Space Library (NMSLIB): An efficient similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces.
Some interesting issues:
  * [Feature request: Hamming distance for non-binary vectors](https://github.com/nmslib/nmslib/issues/340)
  * [Compare against distance based hasing](https://github.com/nmslib/nmslib/issues/169)
* [Polyaxon](https://github.com/polyaxon/polyaxon) - A platform for reproducible and scalable machine learning and deep learning.
Recommended issues tags: [good first issue](https://github.com/polyaxon/polyaxon/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), [technical-debt](https://github.com/polyaxon/polyaxon/issues?q=is%3Aissue+is%3Aopen+label%3Atechnical-debt), [feature](https://github.com/polyaxon/polyaxon/issues?q=is%3Aissue+is%3Aopen+label%3Afeature), [enhancement](https://github.com/polyaxon/polyaxon/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement), [help wanted](https://github.com/polyaxon/polyaxon/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox.
Recommended issues tags: [good first issue](https://github.com/shogun-toolbox/shogun/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), [Tag: Meta Examples](https://github.com/shogun-toolbox/shogun/issues?q=is%3Aissue+is%3Aopen+label%3A%22Tag%3A+Meta+Examples%22), [Tag: Documentation](https://github.com/shogun-toolbox/shogun/issues?q=is%3Aissue+is%3Aopen+label%3A%22Tag%3A+Documentation%22), [Tag: Development Task](https://github.com/shogun-toolbox/shogun/issues?q=is%3Aissue+is%3Aopen+label%3A%22Tag%3A+Development+Task%22), [Tag: Testing](https://github.com/shogun-toolbox/shogun/issues?q=is%3Aissue+is%3Aopen+label%3A%22Tag%3A+Testing%22)
Some intersting issues:
  * [Implement Levenstein string distance algorithm](https://github.com/shogun-toolbox/shogun/issues/4639)
* [Stan](http://mc-stan.org/) - A probabilistic programming language implementing full Bayesian statistical inference with Hamiltonian Monte Carlo sampling.
Recommended issues tags: [code cleanup](https://github.com/stan-dev/stan/issues?q=is%3Aissue+is%3Aopen+label%3A%22code+cleanup%22), [feature](https://github.com/stan-dev/stan/issues?q=is%3Aissue+is%3Aopen+label%3Afeature), [good first issue](https://github.com/stan-dev/stan/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), [documentation](https://github.com/stan-dev/stan/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation), [algorithm](https://github.com/stan-dev/stan/issues?q=is%3Aissue+is%3Aopen+label%3Aalgorithm), [testing](https://github.com/stan-dev/stan/issues?q=is%3Aissue+is%3Aopen+label%3Atesting)
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) - A fast SVM library on GPUs and CPUs.
Recommended issues tags: [call for contribution](https://github.com/Xtra-Computing/thundersvm/issues?q=is%3Aissue+is%3Aopen+label%3A%22call+for+contribution%22)
* [Vowpal Wabbit (VW)](https://github.com/VowpalWabbit/vowpal_wabbit) - A fast out-of-core learning system.
Recommended issues tags: [Good First Issue](https://github.com/VowpalWabbit/vowpal_wabbit/issues?q=is%3Aissue+is%3Aopen+label%3A%22Good+First+Issue%22), [Technical debt](https://github.com/VowpalWabbit/vowpal_wabbit/issues?q=is%3Aissue+is%3Aopen+label%3A%22Technical+debt%22), [Feature Request](https://github.com/VowpalWabbit/vowpal_wabbit/issues?q=is%3Aissue+is%3Aopen+label%3A%22Feature+Request%22),
[Atomization](https://github.com/VowpalWabbit/vowpal_wabbit/issues?q=is%3Aissue+is%3Aopen+label%3AAtomization), [Help wanted](https://github.com/VowpalWabbit/vowpal_wabbit/issues?q=is%3Aissue+is%3Aopen+label%3A%22Help+wanted%22), [Test Issue](https://github.com/VowpalWabbit/vowpal_wabbit/issues?q=is%3Aissue+is%3Aopen+label%3A%22Test+Issue%22), [Documentation](https://github.com/VowpalWabbit/vowpal_wabbit/issues?q=is%3Aissue+is%3Aopen+label%3ADocumentation)
* [XGBoost](https://github.com/dmlc/xgboost) - A parallelized optimized general purpose gradient boosting library.
Some interesting issues:
  * [xgboost multiclass classification problem](https://github.com/dmlc/xgboost/issues/4840)

<a name="cpp-nlp"></a>
#### Natural Language Processing

* [BigARTM](https://github.com/bigartm/bigartm) - topic modelling platform.
* [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser).
Some interesting issues:
  * [parseIt exits when it sees an empty sentence](https://github.com/BLLIP/bllip-parser/issues/37)
  * [Tokenizer doesn't segment double/triple hyphens correctly](https://github.com/BLLIP/bllip-parser/issues/14)
* [fastText](https://github.com/facebookresearch/fastText) - Library for fast text representation and classification. 
Recommended issues tags: [Feature request](https://github.com/facebookresearch/fastText/issues?q=is%3Aissue+is%3Aopen+label%3A%22Feature+request%22)
* [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
Some interesting issues:
  * [Add universal pos](https://github.com/LanguageMachines/frog/issues/68)
* [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
Some interesting issues:
  * [add possibility to add extra user-defined rules on startup](https://github.com/LanguageMachines/ucto/issues/47)
  * [add tests for more languages](https://github.com/LanguageMachines/ucto/issues/56)
* [udpipe](https://github.com/ufal/udpipe) - UDPipe: Trainable pipeline for tokenizing, tagging, lemmatizing and parsing Universal Treebanks and other CoNLL-U files
Some interesting issues:  
  * [Tagging of words that end in a digit, e.g. Boeing777](https://github.com/ufal/udpipe/issues/101)
  * [Morphological dictionary and multi-word tokens](https://github.com/ufal/udpipe/issues/99)
 
<a name="python"></a>
## Python

<a name="python-cv"></a>
#### Computer Vision

* [albumentations](https://github.com/albu/albumentations) - А fast and framework agnostic image augmentation library that implements a diverse set of augmentation techniques. Supports classification, segmentation, detection out of the box. Was used to win a number of Deep Learning competitions at Kaggle, Topcoder and those that were a part of the CVPR workshops.
Some interesting issues:
  * [Add MaxConnectedComponent](https://github.com/albu/albumentations/issues/260)
  * [Unexpected behavior of transforms for multi-label masks](https://github.com/albu/albumentations/issues/280)
  * [Add RegionConfusion Transform](https://github.com/albu/albumentations/issues/275)
* [imgaug](https://github.com/aleju/imgaug) - Image augmentation for machine learning experiments.
Recommended issues tags: [TODO](https://github.com/aleju/imgaug/issues?q=is%3Aissue+is%3Aopen+label%3ATODO)
* [mmdetection](https://github.com/open-mmlab/mmdetection) - object detection toolbox based on PyTorch, part of the open-mmlab project.
Recommended issues tags: [community help wanted](https://github.com/open-mmlab/mmdetection/issues?q=is%3Aissue+is%3Aopen+label%3A%22community+help+wanted%22)
* [mtcnn](https://github.com/ipazc/mtcnn) - MTCNN face detection implementation for TensorFlow, as a PIP package.  
Some interesting issues:
  * [nipples being detected as eyes](https://github.com/ipazc/mtcnn/issues/52)
  * [Migrate to TF 2.0](https://github.com/ipazc/mtcnn/issues/56)
  * [Need face alignment feature](https://github.com/ipazc/mtcnn/issues/44)
* [Scikit-Image](https://github.com/scikit-image/scikit-image) - A collection of algorithms for image processing in Python.
Recommended issues tags: [type: enhancement](https://github.com/scikit-image/scikit-image/issues?q=is%3Aissue+is%3Aopen+label%3A%22type%3A+enhancement%22), [type: documentation](https://github.com/scikit-image/scikit-image/issues?q=is%3Aissue+is%3Aopen+label%3A%22type%3A+documentation%22), [type: performance](https://github.com/scikit-image/scikit-image/issues?q=is%3Aissue+is%3Aopen+label%3A%22type%3A+perfomance%22), [good first issue](https://github.com/scikit-image/scikit-image/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

<a name="python-nlp"></a>
#### Natural Language Processing

* [allennlp](https://github.com/allenai/allennlp) - An open-source NLP research library, built on PyTorch. 
Recommended issues tags: [Contributions welcome](https://github.com/allenai/allennlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22Contributions+welcome%22)
* [CLTK](https://github.com/cltk/cltk) - The Classical Language Toolkit.
* [Dedupe](https://github.com/dedupeio/dedupe) - A python library for accurate and scalable fuzzy matching, record deduplication and entity-resolution.
* [DeepPavlov](https://github.com/deepmipt/DeepPavlov/) - conversational AI library with many pretrained Russian NLP models.
* [DrQA](https://github.com/facebookresearch/DrQA) - Reading Wikipedia to answer open-domain questions.
* [editdistance](https://pypi.org/project/editdistance/) - fast implementation of edit distance.
* [FARM](https://github.com/deepset-ai/FARM) - Fast & easy transfer learning for NLP. Harvesting language models for the industry. 
Some interesting issues:
  * [Add ULMFiT](https://github.com/deepset-ai/FARM/issues/17)
  * [Add Conditional Random Fields as PredictionHead for Named Entity Recognition](https://github.com/deepset-ai/FARM/issues/40)
* [jellyfish](https://github.com/jamesturk/jellyfish) - a python library for doing approximate and phonetic matching of strings.
Some interesting issues:
  * [Adding QWERTY support to DL distance](https://github.com/jamesturk/jellyfish/issues/92)
* [natasha](https://github.com/natasha/natasha) - Rule-based named entity recognition library for russian language 
Some interesting issues:
  * [инструкция по обучению crf-теггера](https://github.com/natasha/natasha/issues/70)
  * [Фамилии в которых присутствует буква "Ё" неверно распознаются](https://github.com/natasha/natasha/issues/77)
  * [Не работает матчинг проспектов если вокруг адреса есть текст](https://github.com/natasha/natasha/issues/64)
* [NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER) - Named-entity recognition using neural networks providing state-of-the-art-results
* [NLTK](https://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
* [num2words](https://github.com/savoirfairelinux/num2words) - Modules to convert numbers to words. 42 --> forty-two 
* [Pattern](http://www.clips.ua.ac.be/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
* [Polyglot](https://github.com/aboSamoor/polyglot) - Multilingual text (NLP) processing toolkit.
* [Quepy](https://github.com/machinalis/quepy) - A python framework to transform natural language questions to queries in a database query language.
* [rasa_nlu](https://github.com/RasaHQ/rasa_nlu) - turn natural language into structured data.
* [Snips NLU](https://github.com/snipsco/snips-nlu) - Natural Language Understanding library for intent classification and entity extraction
* [spaCy](https://github.com/explosion/spaCy) - Industrial strength NLP with Python and Cython.
* [textacy](https://github.com/chartbeat-labs/textacy) - higher-level NLP built on Spacy.
* [transformers](https://github.com/huggingface/transformers) - Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.  
Some interesting issues:
  * [Instruction for Using XLM Text Generations](https://github.com/huggingface/transformers/issues/1414)
  * [add albert](https://github.com/huggingface/transformers/issues/1370)

<a name="python-recsys"></a>
#### Recommender systems
* [implicit](https://github.com/benfred/implicit) - Fast Python Collaborative Filtering for Implicit Feedback Datasets 
Some interesting issues:
  * [Lack of tutorials](https://github.com/benfred/implicit/issues/215)
* [Microsoft Recommenders](https://github.com/Microsoft/Recommenders): Examples and best practices for building recommendation systems, provided as Jupyter notebooks. The repo contains some of the latest state of the art algorithms from Microsoft Research as well as from other companies and institutions.
Recommended issues tags: [notebook](https://github.com/Microsoft/Recommenders/issues?q=is%3Aissue+is%3Aopen+label%3Anotebook), [test](https://github.com/Microsoft/Recommenders/issues?q=is%3Aissue+is%3Aopen+label%3Atest), [algorithm](https://github.com/Microsoft/Recommenders/issues?q=is%3Aissue+is%3Aopen+label%3Aalgorithm), [enhancement](https://github.com/Microsoft/Recommenders/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement), [help wanted](https://github.com/Microsoft/Recommenders/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22), [style improvement](https://github.com/Microsoft/Recommenders/issues?q=is%3Aissue+is%3Aopen+label%3A%22style+improvement%22)
* [Surprise](https://surpriselib.com) - A scikit for building and analyzing recommender systems.
Some interesting issues:
  * [Parallel Computation of Similarity Matrices](https://github.com/NicolasHug/Surprise/issues/169)
  * [Grid search classes should accept estimator instances instead of just classes](https://github.com/NicolasHug/Surprise/issues/213)


<a name="python-reinforcement-learning"></a>
#### Reinforcement Learning
* [Coach](https://github.com/NervanaSystems/coach) - Reinforcement Learning Coach by Intel® AI Lab enables easy experimentation with state of the art Reinforcement Learning algorithms
Recommended issues tags: [help wanted](https://github.com/NervanaSystems/coach/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22), [enhancement](https://github.com/NervanaSystems/coach/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
Some interesting issues:
  * [Model Zoo](https://github.com/NervanaSystems/coach/issues/406)
* [ViZDoom](https://github.com/mwydmuch/ViZDoom) - ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.
Recommended issues tags: [enhancement](https://github.com/mwydmuch/ViZDoom/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement), [feature request](https://github.com/mwydmuch/ViZDoom/issues?q=is%3Aissue+is%3Aopen+label%3A%22feature+request%22)

<a name="tensor"></a>
## TensorFlow

<a name="tensor-general-purpose"></a>
#### General-Purpose Machine Learning
* [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow) - A list of all things related to TensorFlow.
* [Golden TensorFlow](https://golden.com/wiki/TensorFlow) - A page of content on TensorFlow, including academic papers and links to related topics.

<a name="tools"></a>
## Tools

<a name="tools-neural-networks"></a>
#### Neural Networks
* [layer](https://github.com/cloudkj/layer) - Neural network inference from the command line

<a name="tools-misc"></a>
#### Misc
* [DVC](https://github.com/iterative/dvc) - Data Science Version Control is an open-source version control system for machine learning projects with pipelines support. It makes ML projects reproducible and shareable.
* [ML Workspace](https://github.com/ml-tooling/ml-workspace) - All-in-one web-based IDE for machine learning and data science. The workspace is deployed as a docker container and is preloaded with a variety of popular data science libraries (e.g., Tensorflow, PyTorch) and dev tools (e.g., Jupyter, VS Code).
* [Notebooks](https://github.com/rlan/notebooks) - A starter kit for Jupyter notebooks and machine learning. Companion docker images consist of all combinations of python versions, machine learning frameworks (Keras, PyTorch and Tensorflow) and CPU/CUDA versions.

<a name="credits"></a>
## Credits

* Some of the python libraries were cut-and-pasted from [vinta](https://github.com/vinta/awesome-python)
* References for Go were mostly cut-and-pasted from [gopherdata](https://github.com/gopherdata/resources/tree/master/tooling)
