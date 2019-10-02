# Awesome Machine Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

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
  - [C](#c)
      - [General-Purpose Machine Learning](#general-purpose-machine-learning-1)
      - [Computer Vision](#computer-vision)
  - [C++](#c)
      - [Computer Vision](#computer-vision-1)
      - [General-Purpose Machine Learning](#general-purpose-machine-learning-2)
      - [Natural Language Processing](#natural-language-processing)
      - [Speech Recognition](#speech-recognition)
      - [Sequence Analysis](#sequence-analysis)
      - [Gesture Detection](#gesture-detection)
  - [Python](#python)
      - [Computer Vision](#computer-vision-4)
      - [Natural Language Processing](#natural-language-processing-10)
      - [Recommender systems](#recommender-systems)
      - [General-Purpose Machine Learning](#general-purpose-machine-learning-21)
      - [Data Analysis / Data Visualization](#data-analysis--data-visualization-9)
      - [Misc Scripts / iPython Notebooks / Codebases](#misc-scripts--ipython-notebooks--codebases)
      - [Neural Networks](#neural-networks)
      - [Kaggle Competition Source Code](#kaggle-competition-source-code)
      - [Reinforcement Learning](#reinforcement-learning)
  - [TensorFlow](#tensorflow)
      - [General-Purpose Machine Learning](#general-purpose-machine-learning-28)
  - [Tools](#tools-1)
      - [Neural Networks](#neural-networks-2)
      - [Misc](#misc-2)
  - [Credits](#credits)



<a name="c"></a>
## C

<a name="c-general-purpose"></a>
#### General-Purpose Machine Learning
* [Darknet](https://github.com/pjreddie/darknet) - Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
* [Hybrid Recommender System](https://github.com/SeniorSA/hybrid-rs-trainner) - A hybrid recommender system based upon scikit-learn algorithms. **[Deprecated]**
* [neonrvm](https://github.com/siavashserver/neonrvm) - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [Recommender](https://github.com/GHamrouni/Recommender) - A C library for product recommendations/suggestions using collaborative filtering (CF).

<a name="c-cv"></a>
#### Computer Vision

* [CCV](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library.
* [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has Matlab toolbox.

<a name="cpp"></a>
## C++

<a name="cpp-cv"></a>
#### Computer Vision

* [DLib](http://dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
* [EBLearn](http://eblearn.sourceforge.net/) - Eblearn is an object-oriented C++ library that implements various machine learning models **[Deprecated]**
* [libfacedetection](https://github.com/ShiqiYu/libfacedetection) - An open source library for face detection in images. The face detection speed can reach 1500FPS. 
* [Multitarget-tracker](https://github.com/Smorodov/Multitarget-tracker) - Multiple Object Tracker, Based on Hungarian algorithm + Kalman filter.
* [nvvl](https://github.com/NVIDIA/nvvl) - A library that uses hardware acceleration to load sequences of video frames to facilitate machine learning training 
* [OpenCV](https://opencv.org) - OpenCV has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.
* [VIGRA](https://github.com/ukoethe/vigra) - VIGRA is a generic cross-platform C++ computer vision and machine learning library for volumes of arbitrary dimensionality with Python bindings.

<a name="cpp-general-purpose"></a>
#### General-Purpose Machine Learning

* [BanditLib](https://github.com/jkomiyama/banditlib) - A simple Multi-armed Bandit library. **[Deprecated]**
* [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind. [DEEP LEARNING]
* [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, contains fast inference implementation and supports CPU and GPU (even multi-GPU) computation.
* [CNTK](https://github.com/Microsoft/CNTK) - The Computational Network Toolkit (CNTK) by Microsoft Research, is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph.
* [CUDA](https://code.google.com/p/cuda-convnet/) - This is a fast C++/CUDA implementation of convolutional [DEEP LEARNING]
* [DeepDetect](https://github.com/jolibrain/deepdetect) - A machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications.
* [Distributed Machine learning Tool Kit (DMTK)](http://www.dmtk.io/) - A distributed machine learning (parameter server) framework by Microsoft. Enables training models on large data sets across multiple machines. Current tools bundled with it include: LightLDA and Distributed (Multisense) Word Embedding.
* [DLib](http://dlib.net/ml.html) - A suite of ML tools designed to be easy to imbed in other applications.
* [DSSTNE](https://github.com/amznlabs/amazon-dsstne) - A software library created by Amazon for training and deploying deep neural networks using GPUs which emphasizes speed and scale over experimental flexibility.
* [DyNet](https://github.com/clab/dynet) - A dynamic neural network library working well with networks that have dynamic structures that change for every training instance. Written in C++ with bindings in Python.
* [Feast](https://github.com/gojek/feast) - A feature store for the management, discovery, and access of machine learning features. Feast provides a consistent view of feature data for both model training and model serving.
* [Featuretools](https://github.com/featuretools/featuretools) - A library for automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning using reusable feature engineering "primitives". 
* [Fido](https://github.com/FidoProject/Fido) - A highly-modular C++ machine learning library for embedded electronics and robotics.
* [igraph](http://igraph.org/) - General purpose graph library.
* [Intel(R) DAAL](https://github.com/intel/daal) - A high performance software library developed by Intel and optimized for Intel's architectures. Library provides algorithmic building blocks for all stages of data analytics and allows to process data in batch, online and distributed modes.
* [libffm](https://github.com/ycjuan/libffm) - A Library for Field-aware Factorization Machines 
* [libfm](https://github.com/srendle/libfm) - A generic approach that allows to mimic most factorization models by feature engineering.
* [LightGBM](https://github.com/Microsoft/LightGBM) - Microsoft's fast, distributed, high performance gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.
* [LKYDeepNN](https://github.com/mosdeo/LKYDeepNN) - A header-only C++11 Neural Network library. Low dependency, native traditional chinese document.
* [mkl-dnn](https://github.com/intel/mkl-dnn) - Deep Neural Network Library (DNNL) 
* [MLDB](https://mldb.ai) - The Machine Learning Database is a database designed for machine learning. Send it commands over a RESTful API to store data, explore it using SQL, then train machine learning models and expose them as APIs.
* [mlpack](https://www.mlpack.org/) - A scalable C++ machine learning library.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, Javascript and more.
* [nmslib](https://github.com/nmslib/nmslib) - Non-Metric Space Library (NMSLIB): An efficient similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces.
* [Polyaxon](https://github.com/polyaxon/polyaxon) - A platform for reproducible and scalable machine learning and deep learning.
* [proNet-core](https://github.com/cnclabs/proNet-core) - A general-purpose network embedding framework: pair-wise representations optimization Network Edit.
* [PyCUDA](https://mathema.tician.de/software/pycuda/) - Python interface to CUDA
* [ROOT](https://root.cern.ch) - A modular scientific software framework. It provides all the functionalities needed to deal with big data processing, statistical analysis, visualization and storage.
* [shark](http://image.diku.dk/shark/sphinx_pages/build/html/index.html) - A fast, modular, feature-rich open-source C++ machine learning library.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox.
* [skynet](https://github.com/Tyill/skynet) - A library for learning neural network, has C-interface, net set in JSON. Written in C++ with bindings in Python, C++ and C#. 
* [sofia-ml](https://code.google.com/archive/p/sofia-ml) - Suite of fast incremental algorithms.
* [Stan](http://mc-stan.org/) - A probabilistic programming language implementing full Bayesian statistical inference with Hamiltonian Monte Carlo sampling.
* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) - A fast library for GBDTs and Random Forests on GPUs.
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) - A fast SVM library on GPUs and CPUs.
* [Timbl](https://languagemachines.github.io/timbl/) - A software package/C++ library implementing several memory-based learning algorithms, among which IB1-IG, an implementation of k-nearest neighbor classification, and IGTree, a decision-tree approximation of IB1-IG. Commonly used for NLP.
* [Vowpal Wabbit (VW)](https://github.com/VowpalWabbit/vowpal_wabbit) - A fast out-of-core learning system.
* [Warp-CTC](https://github.com/baidu-research/warp-ctc) - A fast parallel implementation of Connectionist Temporal Classification (CTC), on both CPU and GPU.
* [XGBoost](https://github.com/dmlc/xgboost) - A parallelized optimized general purpose gradient boosting library.
* [xLearn](https://github.com/aksnzhy/xlearn) - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertisement and recommender systems.

<a name="cpp-nlp"></a>
#### Natural Language Processing

* [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser).
* [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks. **[Deprecated]**
* [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data. **[Deprecated]**
* [fastText](https://github.com/facebookresearch/fastText) - Library for fast text representation and classification. 
* [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
* [libfolia](https://github.com/LanguageMachines/libfolia) - C++ library for the [FoLiA format](https://proycon.github.io/folia/)
* [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
* [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
* [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
* [udpipe](https://github.com/ufal/udpipe) - UDPipe: Trainable pipeline for tokenizing, tagging, lemmatizing and parsing Universal Treebanks and other CoNLL-U files
* [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) - Unsupervised text tokenizer focused on computational efficiency 

<a name="cpp-speech-recognition"></a>
#### Speech Recognition
* [Kaldi](https://github.com/kaldi-asr/kaldi) - Kaldi is a toolkit for speech recognition written in C++ and licensed under the Apache License v2.0. Kaldi is intended for use by speech recognition researchers.

<a name="cpp-sequence"></a>
#### Sequence Analysis
* [ToPS](https://github.com/ayoshiaki/tops) - This is an objected-oriented framework that facilitates the integration of probabilistic models for sequences over a user defined alphabet. **[Deprecated]**

<a name="cpp-gestures"></a>
#### Gesture Detection
* [grt](https://github.com/nickgillian/grt) - The Gesture Recognition Toolkit (GRT) is a cross-platform, open-source, C++ machine learning library designed for real-time gesture recognition.

<a name="python"></a>
## Python

<a name="python-cv"></a>
#### Computer Vision

* [albumentations](https://github.com/albu/albumentations) - А fast and framework agnostic image augmentation library that implements a diverse set of augmentation techniques. Supports classification, segmentation, detection out of the box. Was used to win a number of Deep Learning competitions at Kaggle, Topcoder and those that were a part of the CVPR workshops.
* [Detectron](https://github.com/facebookresearch/Detectron) - FAIR's software system that implements state-of-the-art object detection algorithms, including Mask R-CNN. It is written in Python and powered by the Caffe2 deep learning framework.
* [dockerface](https://github.com/natanielruiz/dockerface) - Easy to install and use deep learning Faster R-CNN face detection for images and video in a docker container.
* [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library that recognize and manipulate faces from Python or from the command line.
* [imgaug](https://github.com/aleju/imgaug) - Image augmentation for machine learning experiments.
* [imutils](https://github.com/jrosebr1/imutils) - A library containg Convenience functions to make basic image processing operations such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV and Python.
* [mmdetection](https://github.com/open-mmlab/mmdetection) - object detection toolbox based on PyTorch, part of the open-mmlab project.
* [mtcnn](https://github.com/ipazc/mtcnn) - MTCNN face detection implementation for TensorFlow, as a PIP package.  
* [OpenFace](https://cmusatyalab.github.io/openface/) - Free and open source face recognition with deep neural networks.
* [PCV](https://github.com/jesolem/PCV) - Open source Python module for computer vision. **[Deprecated]**
* [pytessarct](https://github.com/madmaze/pytesseract) - Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and "read" the text embedded in images.Python-tesseract is a wrapper for [Google's Tesseract-OCR Engine](https://github.com/tesseract-ocr/tesseract)>.
* [PyTorchCV](https://github.com/donnyyou/PyTorchCV) - A PyTorch-Based Framework for Deep Learning in Computer Vision.
* [Scikit-Image](https://github.com/scikit-image/scikit-image) - A collection of algorithms for image processing in Python.
* [SimpleCV](http://simplecv.org/) - An open source computer vision framework that gives access to several high-powered computer vision libraries, such as OpenCV. Written on Python and runs on Mac, Windows, and Ubuntu Linux.
* [Vigranumpy](https://github.com/ukoethe/vigra) - Python bindings for the VIGRA C++ computer vision library.

<a name="python-nlp"></a>
#### Natural Language Processing

* [allennlp](https://github.com/allenai/allennlp) - An open-source NLP research library, built on PyTorch. 
* [BigARTM](https://github.com/bigartm/bigartm) - topic modelling platform.
* [BLLIP Parser](https://pypi.org/project/bllipparser/) - Python bindings for the BLLIP Natural Language Parser (also known as the Charniak-Johnson parser). **[Deprecated]**
* [CLTK](https://github.com/cltk/cltk) - The Classical Language Toolkit.
* [colibri-core](https://github.com/proycon/colibri-core) - Python binding to C++ library for extracting and working with with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [Dedupe](https://github.com/dedupeio/dedupe) - A python library for accurate and scalable fuzzy matching, record deduplication and entity-resolution.
* [DeepPavlov](https://github.com/deepmipt/DeepPavlov/) - conversational AI library with many pretrained Russian NLP models.
* [Distance](https://github.com/doukremt/distance) - Levenshtein and Hamming distance computation. **[Deprecated]**
* [dostoevsky](https://github.com/bureaucratic-labs/dostoevsky) - Sentiment analysis library for russian language
* [DrQA](https://github.com/facebookresearch/DrQA) - Reading Wikipedia to answer open-domain questions.
* [editdistance](https://pypi.org/project/editdistance/) - fast implementation of edit distance.
* [FARM](https://github.com/deepset-ai/FARM) - Fast & easy transfer learning for NLP. Harvesting language models for the industry. 
* [Fuzzy Wuzzy](https://github.com/seatgeek/fuzzywuzzy) - Fuzzy String Matching in Python.
* [genius](https://github.com/duanhongyi/genius) - A Chinese segment base on Conditional Random Field.
* [jellyfish](https://github.com/jamesturk/jellyfish) - a python library for doing approximate and phonetic matching of strings.
* [jieba](https://github.com/fxsjy/jieba#jieba-1) - Chinese Words Segmentation Utilities.
* [KoNLPy](http://konlpy.org) - A Python package for Korean natural language processing.
* [langdetect](https://github.com/Mimino666/langdetect) - Port of Google's language-detection library to Python. **[DEPRECATED]**
* [langid.py](https://github.com/saffsd/langid.py) - Stand-alone language identification system **[DEPRECATED]**
* [LASER](https://github.com/facebookresearch/LASER) - Language-Agnostic SEntence Representations 
* [loso](https://github.com/fangpenlin/loso) - Another Chinese segmentation library. **[Deprecated]**
* [natasha](https://github.com/natasha/natasha) - Rule-based named entity recognition library for russian language 
* [NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER) - Named-entity recognition using neural networks providing state-of-the-art-results
* [NLTK](https://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
* [num2words](https://github.com/savoirfairelinux/num2words) - Modules to convert numbers to words. 42 --> forty-two 
* [nut](https://github.com/pprett/nut) - Natural language Understanding Toolkit. **[Deprecated]**
* [Pattern](http://www.clips.ua.ac.be/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
* [pkuseg-python](https://github.com/lancopku/pkuseg-python) - A better version of Jieba, developed by Peking University. 
* [Polyglot](https://github.com/aboSamoor/polyglot) - Multilingual text (NLP) processing toolkit.
* [pymorphy2](https://github.com/kmike/pymorphy2) - Morphological analyzer / inflection engine for Russian and Ukrainian languages.
* [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](https://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
* [PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies) - Python interface for converting Penn Treebank trees to Stanford Dependencies.
* [python-frog](https://github.com/proycon/python-frog) - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
* [python-ucto](https://github.com/proycon/python-ucto) - Python binding to ucto (a unicode-aware rule-based tokenizer for various languages).
* [python-zpar](https://github.com/EducationalTestingService/python-zpar) - Python bindings for [ZPar](https://github.com/frcchang/zpar), a statistical part-of-speech-tagger, constiuency parser, and dependency parser for English.
* [Quepy](https://github.com/machinalis/quepy) - A python framework to transform natural language questions to queries in a database query language.
* [rasa_nlu](https://github.com/RasaHQ/rasa_nlu) - turn natural language into structured data.
* [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
* [Snips NLU](https://github.com/snipsco/snips-nlu) - Natural Language Understanding library for intent classification and entity extraction
* [SnowNLP](https://github.com/isnowfy/snownlp) - A library for processing Chinese text.
* [spacy-ru](https://github.com/buriy/spacy-ru) - Russian language models for spaCy 
* [spaCy](https://github.com/explosion/spaCy) - Industrial strength NLP with Python and Cython.
* [spammy](https://github.com/tasdikrahman/spammy) - A library for email Spam filtering built on top of nltk
* [stanford-corenlp-python](https://github.com/dasmith/stanford-corenlp-python) - Python wrapper for [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP) **[Deprecated]**
* [textacy](https://github.com/chartbeat-labs/textacy) - higher-level NLP built on Spacy.
* [TextBlob](http://textblob.readthedocs.io/en/dev/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of NLTK and Pattern, and plays nicely with both.
* [transformers](https://github.com/huggingface/transformers) - Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.  
* [wmd-relax](https://github.com/src-d/wmd-relax) - Calculates Word Mover's Distance Insanely Fast 
* [YAlign](https://github.com/machinalis/yalign) - A sentence aligner, a friendly tool for extracting parallel sentences from comparable corpora. **[Deprecated]**
* [yase](https://github.com/PPACI/yase) - Transcode sentence (or other sequence) to list of word vector .

<a name="python-recsys"></a>
#### Recommender systems
* [Cornac](https://github.com/PreferredAI/cornac) - A comparative framework for multimodal recommender systems with a focus on models leveraging auxiliary data.
* [Crab](https://github.com/muricoca/crab) - A flexible, fast recommender engine. **[Deprecated]**
* [implicit](https://github.com/benfred/implicit) - Fast Python Collaborative Filtering for Implicit Feedback Datasets 
* [Microsoft Recommenders](https://github.com/Microsoft/Recommenders): Examples and best practices for building recommendation systems, provided as Jupyter notebooks. The repo contains some of the latest state of the art algorithms from Microsoft Research as well as from other companies and institutions.
* [python-recsys](https://github.com/ocelma/python-recsys) - A Python library for implementing a Recommender System.
* [recommender_test_bench](https://github.com/rampeer/recommender_test_bench) - Recommender system test bench 
* [Surprise](https://surpriselib.com) - A scikit for building and analyzing recommender systems.

<a name="python-general-purpose"></a>
#### General-Purpose Machine Learning
* [Annoy](https://github.com/spotify/annoy) - Approximate nearest neighbours implementation.
* [Apache SINGA](https://singa.apache.org) - An Apache Incubating project for developing an open source machine learning library.
* [astroML](https://www.astroml.org/) - Machine Learning and Data Mining for Astronomy.
* [auto_ml](https://github.com/ClimbsRocks/auto_ml) - Automated machine learning for production and analytics. Lets you focus on the fun parts of ML, while outputting production-ready code, and detailed analytics of your dataset and results. Includes support for NLP, XGBoost, CatBoost, LightGBM, and soon, deep learning. 
* [batchflow](https://github.com/analysiscenter/batchflow) - BatchFlow helps you conveniently work with random or sequential batches of your data and define data processing and machine learning workflows even for datasets that do not fit into memory. 
* [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - Book/iPython notebooks on Probabilistic Programming in Python.
* [BentoML](https://github.com/bentoml/bentoml): Toolkit for package and deploy machine learning models for serving in production
* [BigML](https://bigml.com) - A library that contacts external servers.
* [Bolt](https://github.com/pprett/bolt) - Bolt Online Learning Toolbox. **[Deprecated]**
* [Brainstorm](https://github.com/IDSIA/brainstorm) - Fast, flexible and fun neural networks. This is the successor of PyBrain.
* [breze](https://github.com/breze-no-salt/breze) - Theano based library for deep and recurrent neural networks. **[Deprecated]**
* [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, well documented and supports CPU and GPU (even multi-GPU) computation.
* [Chainer](https://github.com/chainer/chainer) - Flexible neural network framework.
* [CNTK](https://github.com/Microsoft/CNTK) - Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit. Documentation can be found [here](https://docs.microsoft.com/cognitive-toolkit/).
* [Cogitare](https://github.com/cogitare-ai/cogitare): A Modern, Fast, and Modular Deep Learning and Machine Learning framework for Python. 
* [CoverTree](https://github.com/patvarilly/CoverTree) - Python implementation of cover trees, near-drop-in replacement for scipy.spatial.kdtree **[Deprecated]**
* [creme](https://github.com/creme-ml/creme): A framework for online machine learning.
* [deap](https://github.com/deap/deap) - Evolutionary algorithm framework.
* [DIGITS](https://github.com/NVIDIA/DIGITS) - The Deep Learning GPU Training System (DIGITS) is a web application for training deep learning models.
* [Edward](http://edwardlib.org/) - A library for probabilistic modeling, inference, and criticism. Built on top of TensorFlow.
* [eli5](https://github.com/TeamHG-Memex/eli5) - A library for debugging/inspecting machine learning classifiers and explaining their predictions
* [feature-selector](https://github.com/WillKoehrsen/feature-selector) - Feature selector is a tool for dimensionality reduction of machine learning datasets 
* [Featureforge](https://github.com/machinalis/featureforge) A set of tools for creating and testing machine learning features, with a scikit-learn compatible API.
* [fuku-ml](https://github.com/fukuball/fuku-ml) - Simple machine learning library, including Perceptron, Regression, Support Vector Machine, Decision Tree and more, it's easy to use and easy to learn for beginners.
* [gensim](https://github.com/RaRe-Technologies/gensim) - Topic Modelling for Humans.
* [graphlab-create](https://turi.com/products/create/docs/) - A library with various machine learning models (regression, clustering, recommender systems, graph analytics, etc.) implemented on top of a disk-backed DataFrame.
* [hebel](https://github.com/hannes-brt/hebel) - GPU-Accelerated Deep Learning Library in Python. **[Deprecated]**
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) - a service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [hyperlearn](https://github.com/danielhanchen/hyperlearn) - 50% faster, 50% less RAM Machine Learning. Numba rewritten Sklearn. SVD, NNMF, PCA, LinearReg, RidgeReg, Randomized, Truncated SVD/PCA, CSR Matrices all 50+% faster 
* [Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/williamFalcon/pix2pix-keras) - Implementation of image to image (pix2pix) translation from the paper by [isola et al](https://arxiv.org/pdf/1611.07004.pdf).[DEEP LEARNING]
* [imbalanced-learn](https://imbalanced-learn.org/en/stable/index.html) - Python module to perform under sampling and over sampling with various techniques.
* [keras](https://github.com/keras-team/keras) - High-level neural networks frontend for [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/CNTK) and [Theano](https://github.com/Theano/Theano).
* [Lasagne](https://github.com/Lasagne/Lasagne) - Lightweight library to build and train neural networks in Theano.
* [luminol](https://github.com/linkedin/luminol) - Anomaly Detection and Correlation library 
* [machine learning](https://github.com/jeff1evesque/machine-learning) - automated build consisting of a [web-interface](https://github.com/jeff1evesque/machine-learning#web-interface), and set of [programmatic-interface](https://github.com/jeff1evesque/machine-learning#programmatic-interface) API, for support vector machines. Corresponding dataset(s) are stored into a SQL database, then generated model(s) used for prediction(s), are stored into a NoSQL datastore.
* [metric-learn](https://github.com/metric-learn/metric-learn) - A Python module for metric learning.
* [milk](https://github.com/luispedro/milk) - Machine learning toolkit focused on supervised classification. **[Deprecated]**
* [MindsDB](https://github.com/mindsdb/mindsdb) - Open Source framework to streamline use of neural networks.
* [MiraiML](https://github.com/arthurpaulino/miraiml): An asynchronous engine for continuous & autonomous machine learning, built for real-time usage.
* [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) - Implementations of Machine Learning models from scratch in Python with a focus on transparency. Aims to showcase the nuts and bolts of ML in an accessible way.
* [MLBox](https://github.com/AxeldeRomblay/MLBox) - MLBox is a powerful Automated Machine Learning python library. 
* [mlens](https://github.com/flennerhag/mlens) - A high performance, memory efficient, maximally parallelized ensemble learning, integrated with scikit-learn.
* [MLlib in Apache Spark](http://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [mlxtend](https://github.com/rasbt/mlxtend) - A library consisting of useful tools for data science and machine learning tasks.
* [modAL](https://github.com/modAL-python/modAL) - A modular active learning framework for Python, built on top of scikit-learn.
* [mrjob](https://pythonhosted.org/mrjob/) - A library to let Python program run on Hadoop.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, Javascript and more.
* [neon](https://github.com/NervanaSystems/neon) - Nervana's [high-performance](https://github.com/soumith/convnet-benchmarks) Python-based Deep Learning framework [DEEP LEARNING].
* [neonrvm](https://github.com/siavashserver/neonrvm) - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [Netron](https://github.com/lutzroeder/netron) - Visualizer for machine learning models.
* [Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning) - Code samples for my book "Neural Networks and Deep Learning" [DEEP LEARNING].
* [Neuraxle](https://github.com/Neuraxio/Neuraxle): A framework providing the right abstractions to ease research, development, and deployment of your ML pipelines.
* [neurolab](https://github.com/zueve/neurolab)
* [neuropredict](https://github.com/raamana/neuropredict) - Aimed at novice machine learners and non-expert programmers, this package offers easy (no coding needed) and comprehensive machine learning (evaluation and full report of predictive performance WITHOUT requiring you to code) in Python for NeuroImaging and any other type of features. This is aimed at absorbing the much of the ML workflow, unlike other packages like nilearn and pymvpa, which require you to learn their API and code to produce anything useful.
* [nilearn](https://github.com/nilearn/nilearn) - Machine learning for NeuroImaging in Python.
* [numpy-ML](https://github.com/ddbourgin/numpy-ml): Reference implementations of ML models written in numpy
* [NuPIC](https://github.com/numenta/nupic) - Numenta Platform for Intelligent Computing.
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search.
* [Orange](https://orange.biolab.si/) - Open source data visualization and data analysis for novices and experts.
* [Parris](https://github.com/jgreenemi/Parris) - Parris, the automated infrastructure setup tool for machine learning algorithms.
* [pattern](https://github.com/clips/pattern) - Web mining module for Python.
* [Pebl](https://github.com/abhik/pebl/) - Python Environment for Bayesian Learning. **[Deprecated]**
* [pgmpy](https://github.com/pgmpy/pgmpy) A python library for working with Probabilistic Graphical Models.
* [pomegranate](https://github.com/jmschrei/pomegranate) - Hidden Markov Models for Python, implemented in Cython for speed and efficiency.
* [prophet](https://facebook.github.io/prophet/) - Fast and automated time series forecasting framework by Facebook.
* [PyBrain](https://github.com/pybrain/pybrain) - Another Python Machine Learning Library.
* [pydeep](https://github.com/andersbll/deeppy) - Deep Learning In Python. **[Deprecated]**
* [Pyevolve](https://github.com/perone/Pyevolve) - Genetic algorithm framework. **[Deprecated]**
* [pyhsmm](https://github.com/mattjj/pyhsmm) - library for approximate unsupervised inference in Bayesian Hidden Markov Models (HMMs) and explicit-duration Hidden semi-Markov Models (HSMMs), focusing on the Bayesian Nonparametric extensions, the HDP-HMM and HDP-HSMM, mostly with weak-limit approximations.
* [Pylearn2](https://github.com/lisa-lab/pylearn2) - A Machine Learning library based on [Theano](https://github.com/Theano/Theano).
* [PyOD](https://github.com/yzhao062/pyod) -> Python Outlier Detection, comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. Featured for Advanced models, including Neural Networks/Deep Learning and Outlier Ensembles.
* [python-timbl](https://github.com/proycon/python-timbl) - A Python extension module wrapping the full TiMBL C++ programming interface. Timbl is an elaborate k-Nearest Neighbours machine learning toolkit.
* [PyTorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python with strong GPU acceleration
* [REP](https://github.com/yandex/rep) - an IPython-based environment for conducting data-driven research in a consistent and reproducible way. REP is not trying to substitute scikit-learn, but extends it and provides better user experience. **[Deprecated]**
* [Restricted Boltzmann Machines](https://github.com/echen/restricted-boltzmann-machines) -Restricted Boltzmann Machines in Python. [DEEP LEARNING]
* [rgf_python](https://github.com/RGF-team/rgf) - Python bindings for Regularized Greedy Forest (Tree) Library.
* [scikit-learn](https://scikit-learn.org/) - A Python module for machine learning built on top of SciPy.
* [shap](https://github.com/slundberg/shap) - A unified approach to explain the output of any machine learning model.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox.
* [SimpleAI](https://github.com/simpleai-team/simpleai) Python implementation of many of the artificial intelligence algorithms described on the book "Artificial Intelligence, a Modern Approach". It focuses on providing an easy to use, well documented and tested library.
* [skbayes](https://github.com/AmazaspShumik/sklearn-bayes) - Python package for Bayesian Machine Learning with scikit-learn API.
* [SKLL](https://github.com/EducationalTestingService/skll) - A wrapper around scikit-learn that makes it simpler to conduct experiments.
* [Spearmint](https://github.com/HIPS/Spearmint) - Spearmint is a package to perform Bayesian optimization according to the algorithms outlined in the paper: Practical Bayesian Optimization of Machine Learning Algorithms. Jasper Snoek, Hugo Larochelle and Ryan P. Adams. Advances in Neural Information Processing Systems, 2012. **[Deprecated]**
* [stacked_generalization](https://github.com/fukatani/stacked_generalization) - Implementation of machine learning stacking technic as handy library in Python.
* [StellarGraph](https://github.com/stellargraph/stellargraph): Machine Learning on Graphs, a Python library for machine learning on graph-structured (network-structured) data.
* [steppy-toolkit](https://github.com/neptune-ml/steppy-toolkit) -> Curated collection of the neural networks, transformers and models that make your machine learning work faster and more effective.
* [steppy](https://github.com/neptune-ml/steppy) -> Lightweight, Python library for fast and reproducible machine learning experimentation. Introduces very simple interface that enables clean machine learning pipeline design.
* [TensorFlow](https://github.com/tensorflow/tensorflow/) - Open source software library for numerical computation using data flow graphs.
* [TFLearn](https://github.com/tflearn/tflearn) - Deep learning library featuring a higher-level API for TensorFlow.
* [Thampi](https://github.com/scoremedia/thampi) - Machine Learning Prediction System on AWS Lambda
* [Theano](https://github.com/Theano/Theano/) - Optimizing GPU-meta-programming code generating array oriented optimizing math compiler in Python.
* [thinking bayes](https://github.com/AllenDowney/ThinkBayes) - Book on Bayesian Analysis.
* [topik](https://github.com/ContinuumIO/topik) - Topic modelling toolkit. **[Deprecated]**
* [TPOT](https://github.com/EpistasisLab/tpot) - Tool that automatically creates and optimizes machine learning pipelines using genetic programming. Consider it your personal data science assistant, automating a tedious part of machine learning.
* [Turi Create](https://github.com/apple/turicreate) - Machine learning from Apple. Turi Create simplifies the development of custom machine learning models. You don't have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.
* [vecstack](https://github.com/vecxoz/vecstack) - Python package for stacking (machine learning technique) 
* [Xcessiv](https://github.com/reiinakano/xcessiv) - A web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling.
* [xgbfir](https://github.com/limexp/xgbfir) - XGBoost Feature Interactions Reshaped **[DEPRECATED]**
* [XGBoost](https://github.com/dmlc/xgboost) - Python bindings for eXtreme Gradient Boosting (Tree) Library.
* [xLearn](https://github.com/aksnzhy/xlearn) - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertisement and recommender systems.
* [xRBM](https://github.com/omimo/xRBM) - A library for Restricted Boltzmann Machine (RBM) and its conditional variants in Tensorflow.

<a name="python-data-analysis"></a>
#### Data Analysis / Data Visualization

* [altair](https://github.com/altair-viz/altair) - A Python to Vega translator.
* [astropy](https://www.astropy.org/) - A community Python library for Astronomy.
* [Blaze](https://github.com/blaze/blaze) - NumPy and Pandas interface to Big Data.
* [bokeh](https://github.com/bokeh/bokeh) - Interactive Web Plotting for Python.
* [Bowtie](https://github.com/jwkvam/bowtie) - A dashboard library for interactive visualizations using flask socketio and react.
* [bqplot](https://github.com/bloomberg/bqplot) - An API for plotting in Jupyter (IPython).
* [cerebro2](https://github.com/numenta/nupic.cerebro2) A web-based visualization and debugging platform for NuPIC. **[Deprecated]**
* [d3py](https://github.com/mikedewar/d3py) - A plotting library for Python, based on [D3.js](https://d3js.org/).
* [Dash](https://github.com/plotly/dash) - A framework for creating analytical web applications built on top of Plotly.js, React, and Flask
* [Dora](https://github.com/nathanepstein/dora) - Tools for exploratory data analysis in Python.
* [emcee](https://github.com/dfm/emcee) - The Python ensemble sampling toolkit for affine-invariant MCMC.
* [folium](https://github.com/python-visualization/folium) - folium builds on the data wrangling strengths of the Python ecosystem and the mapping strengths of the Leaflet.js library. Manipulate your data in Python, then visualize it in a Leaflet map via folium.
* [ggfortify](https://github.com/sinhrks/ggfortify) - Unified interface to ggplot2 popular R packages.
* [ggplot](https://github.com/yhat/ggpy) - Same API as ggplot2 for R. **[Deprecated]**
* [HDBScan](https://github.com/lmcinnes/hdbscan) - implementation of the hdbscan algorithm in Python - used for clustering
* [igraph](https://igraph.org/python/) - binding to igraph library - General purpose graph library.
* [Kartograph.py](https://github.com/kartograph/kartograph.py) - Rendering beautiful SVG maps in Python.
* [Lambdo](https://github.com/asavinov/lambdo) - A workflow engine for solving machine learning problems by combining in one analysis pipeline (i) feature engineering and machine learning (ii) model training and prediction (iii) table population and column evaluation via user-defined (Python) functions.
* [lime](https://github.com/marcotcr/lime) - Lime is about explaining what machine learning classifiers (or models) are doing. It is able to explain any black box classifier, with two or more classes.
* [Mars](https://github.com/mars-project/mars) - A tensor-based framework for large-scale data computation which often regarded as a parallel and distributed version of NumPy. 
* [matplotlib](https://matplotlib.org/) - A Python 2D plotting library.
* [NetworkX](https://networkx.github.io/) - A high-productivity software for complex networks.
* [Numba](https://numba.pydata.org/) - Python JIT (just in time) compiler to LLVM aimed at scientific Python by the developers of Cython and NumPy.
* [NumPy](https://www.numpy.org/) - A fundamental package for scientific computing with Python.
* [NuPIC Studio](https://github.com/htm-community/nupic.studio) An all-in-one NuPIC Hierarchical Temporal Memory visualization and debugging super-tool! **[Deprecated]**
* [Open Mining](https://github.com/mining/mining) - Business Intelligence (BI) in Python (Pandas web interface) **[Deprecated]**
* [Pandas](https://pandas.pydata.org/) - A library providing high-performance, easy-to-use data structures and data analysis tools.
* [pastalog](https://github.com/rewonc/pastalog) - Simple, realtime visualization of neural network training performance.
* [Petrel](https://github.com/AirSage/Petrel) - Tools for writing, submitting, debugging, and monitoring Storm topologies in pure Python.
* [plotly](https://plot.ly/python/) - Collaborative web plotting for Python and matplotlib.
* [plotly_express](https://github.com/plotly/plotly_express) - Plotly Express - simple syntax for complex charts 
* [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) - Latex code for making neural networks diagrams 
* [pycascading](https://github.com/twitter/pycascading) **[Deprecated]**
* [PyCM](https://github.com/sepandhaghighi/pycm) - PyCM is a multi-class confusion matrix library written in Python that supports both input data vectors and direct matrix, and a proper tool for post-classification model evaluation that supports most classes and overall statistics parameters
* [PyDexter](https://github.com/D3xterjs/pydexter) - Simple plotting for Python. Wrapper for D3xterjs; easily render charts in-browser.
* [PyDy](https://www.pydy.org/) - Short for Python Dynamics, used to assist with workflow in the modeling of dynamic motion based around NumPy, SciPy, IPython, and matplotlib.
* [pygal](http://pygal.org/en/stable/) - A Python SVG Charts Creator.
* [PyMC](https://github.com/pymc-devs/pymc) - Markov Chain Monte Carlo sampling toolkit.
* [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) - A pure-python graphics and GUI library built on PyQt4 / PySide and NumPy.
* [Ruffus](http://www.ruffus.org.uk) - Computation Pipeline library for python.
* [scikit-plot](https://github.com/reiinakano/scikit-plot) - A visualization library for quick and easy generation of common plots in data analysis and machine learning.
* [SciPy](https://www.scipy.org/) - A Python-based ecosystem of open-source software for mathematics, science, and engineering.
* [Seaborn](https://seaborn.pydata.org/) - A python visualization library based on matplotlib.
* [somoclu](https://github.com/peterwittek/somoclu) Massively parallel self-organizing maps: accelerate training on multicore CPUs, GPUs, and clusters, has python API.
* [SOMPY](https://github.com/sevamoo/SOMPY) - Self Organizing Map written in Python (Uses neural networks for data analysis).
* [SparklingPandas](https://github.com/sparklingpandas/sparklingpandas) Pandas on PySpark (POPS).
* [statsmodels](https://github.com/statsmodels/statsmodels) - Statistical modeling and econometrics in Python.
* [Superset](https://github.com/apache/incubator-superset) - A data exploration platform designed to be visual, intuitive, and interactive.
* [SymPy](https://github.com/sympy/sympy) - A Python library for symbolic mathematics.
* [TensorWatch](https://github.com/microsoft/tensorwatch) - Debugging and visualization tool for machine learning and data science. It extensively leverages Jupyter Notebook to show real-time visualizations of data in running processes such as machine learning training.
* [vispy](https://github.com/vispy/vispy) - GPU-based high-performance interactive OpenGL 2D/3D data visualization library.
* [visualize_ML](https://github.com/ayush1997/visualize_ML) - A python package for data exploration and data analysis. **[Deprecated]**
* [windML](https://github.com/cigroup-ol/windml) - A Python Framework for Wind Energy Analysis and Prediction.
* [zipline](https://github.com/quantopian/zipline) - A Pythonic algorithmic trading library.

<a name="python-misc"></a>
#### Misc Scripts / iPython Notebooks / Codebases
* [2012-paper-diginorm](https://github.com/dib-lab/2012-paper-diginorm)
* [A gallery of interesting IPython notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks)
* [Allen Downey’s Data Science Course](https://github.com/AllenDowney/DataScience) - Code for Data Science at Olin College, Spring 2014.
* [Allen Downey’s Think Bayes Code](https://github.com/AllenDowney/ThinkBayes) - Code repository for Think Bayes.
* [Allen Downey’s Think Complexity Code](https://github.com/AllenDowney/ThinkComplexity) - Code for Allen Downey's book Think Complexity.
* [Allen Downey’s Think OS Code](https://github.com/AllenDowney/ThinkOS) - Text and supporting code for Think OS: A Brief Introduction to Operating Systems.
* [BayesPy](https://github.com/maxsklar/BayesPy) - Bayesian Inference Tools in Python.
* [BioPy](https://github.com/jaredthecoder/BioPy) - Biologically-Inspired and Machine Learning Algorithms in Python. **[Deprecated]**
* [climin](https://github.com/BRML/climin) - Optimization library focused on machine learning, pythonic implementations of gradient descent, LBFGS, rmsprop, adadelta and others.
* [Crab](https://github.com/marcelcaraciolo/crab) - A recommendation engine library for Python.
* [data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks) - Continually updated Data Science Python Notebooks: Spark, Hadoop MapReduce, HDFS, AWS, Kaggle, scikit-learn, matplotlib, pandas, NumPy, SciPy, and various command lines.
* [decision-weights](https://github.com/CamDavidsonPilon/decision-weights)
* [Diffusion Segmentation](https://github.com/Wavelets/diffusion-segmentation) - A collection of image segmentation algorithms based on diffusion methods.
* [Dive into Machine Learning  with Python Jupyter notebook and scikit-learn](https://github.com/hangtwenty/dive-into-machine-learning) - "I learned Python by hacking first, and getting serious *later.* I wanted to do this with Machine Learning. If this is your style, join me in getting a bit ahead of yourself."
* [GreatCircle](https://github.com/mwgg/GreatCircle) - Library for calculating great circle distance.
* [group-lasso](https://github.com/fabianp/group_lasso) - Some experiments with the coordinate descent algorithm used in the (Sparse) Group Lasso model.
* [Homemade Machine Learning](https://github.com/trekhleb/homemade-machine-learning) - Python examples of popular machine learning algorithms with interactive Jupyter demos and math being explained
* [hyperopt](https://github.com/hyperopt/hyperopt-sklearn)
* [Introduction to Machine Learning with Python](https://github.com/amueller/introduction_to_ml_with_python) - Notebooks and code for the book "Introduction to Machine Learning with Python"
* [Introduction to machine learning with scikit-learn](https://github.com/justmarkham/scikit-learn-videos) - IPython notebooks from Data School's video tutorials on scikit-learn.
* [ipython-notebooks](https://github.com/ogrisel/notebooks)
* [jProcessing](https://github.com/kevincobain2000/jProcessing) - Kanji / Hiragana / Katakana to Romaji Converter. Edict Dictionary & parallel sentences Search. Sentence Similarity between two JP Sentences. Sentiment Analysis of Japanese Text. Run Cabocha(ISO--8859-1 configured) in Python.
* [keras_telegram_callback](https://github.com/qubvel/keras_telegram_callback) - Telegram-bot callback for your Keras model 
* [Map/Reduce implementations of common ML algorithms](https://github.com/Yannael/BigDataAnalytics_INFOH515): Jupyter notebooks that cover how to implement from scratch different ML algorithms (ordinary least squares, gradient descent, k-means, alternating least squares), using Python NumPy, and how to then make these implementations scalable using Map/Reduce and Spark. 
* [mne-python-notebooks](https://github.com/mne-tools/mne-python-notebooks) - IPython notebooks for EEG/MEG data processing using mne-python.
* [Neon Course](https://github.com/NervanaSystems/neon_course) - IPython notebooks for a complete course around understanding Nervana's Neon.
* [numpic](https://github.com/numenta/nupic)
* [Optunity examples](http://optunity.readthedocs.io/en/latest/notebooks/index.html) - Examples demonstrating how to use Optunity in synergy with machine learning libraries.
* [pandas cookbook](https://github.com/jvns/pandas-cookbook) - Recipes for using Python's pandas library.
* [pattern_classification](https://github.com/rasbt/pattern_classification)
* [Practical XGBoost in Python](https://parrotprediction.teachable.com/p/practical-xgboost-in-python) - comprehensive online course about using XGBoost in Python.
* [Prodmodel](https://github.com/prodmodel/prodmodel) - Build tool for data science pipelines.
* [Pydata book](https://github.com/wesm/pydata-book) - Materials and IPython notebooks for "Python for Data Analysis" by Wes McKinney, published by O'Reilly Media
* [Python Programming for the Humanities](https://www.karsdorp.io/python-course/) - Course for Python programming for the Humanities, assuming no prior knowledge. Heavy focus on text processing / NLP.
* [pytorch2keras](https://github.com/nerox8664/pytorch2keras) - PyTorch to Keras model convertor 
* [Sarah Palin LDA](https://github.com/Wavelets/sarah-palin-lda) - Topic Modeling the Sarah Palin emails.
* [scikit-learn tutorials](https://github.com/GaelVaroquaux/scikit-learn-tutorial) - Series of notebooks for learning scikit-learn.
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) - Sequential model-based optimization with a `scipy.optimize` interface 
* [Scipy Tutorials](https://github.com/Wavelets/scipy-tutorials) - SciPy tutorials. This is outdated, check out scipy-lecture-notes.
* [sentiment-analyzer](https://github.com/madhusudancs/sentiment-analyzer) - Tweets Sentiment Analyzer
* [sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier) - Sentiment classifier using word sense disambiguation.
* [Suiron](https://github.com/kendricktan/suiron/) - Machine Learning for RC Cars.
* [SVM Explorer](https://github.com/plotly/dash-svm) - Interactive SVM Explorer, using Dash and scikit-learn
* [TDB](https://github.com/ericjang/tdb) - TensorDebugger (TDB) is a visual debugger for deep learning. It features interactive, node-by-node debugging and visualization for TensorFlow.
* [the-elements-of-statistical-learning](https://github.com/maitbayev/the-elements-of-statistical-learning) - This repository contains Jupyter notebooks implementing the algorithms found in the book and summary of the textbook.
* [thinking stats 2](https://github.com/Wavelets/ThinkStats2)

<a name="python-neural-networks"></a>
#### Neural Networks

* [NeuralTalk](https://github.com/karpathy/neuraltalk) - NeuralTalk is a Python+numpy project for learning Multimodal Recurrent Neural Networks that describe images with sentences.
* [Neuron](https://github.com/molcik/python-neuron) - Neuron is simple class for time series predictions. It's utilize LNU (Linear Neural Unit), QNU (Quadratic Neural Unit), RBF (Radial Basis Function), MLP (Multi Layer Perceptron), MLP-ELM (Multi Layer Perceptron - Extreme Learning Machine) neural networks learned with Gradient descent or LeLevenberg–Marquardt algorithm.
* [nn_builder](https://github.com/p-christ/nn_builder) - nn_builder is a python package that lets you build neural networks in 1 line
=======
* [catalyst](https://github.com/catalyst-team/catalyst) - High-level utils for PyTorch DL & RL research with a focus on reproducibility, fast experimentation and code/ideas reusing. 
* [Data Driven Code](https://github.com/atmb4u/data-driven-code) - Very simple implementation of neural networks for dummies in python without using any libraries, with detailed comments.
* [graph_nets](https://github.com/deepmind/graph_nets) - Build Graph Nets in Tensorflow 
* [Machine Learning, Data Science and Deep Learning with Python](https://www.manning.com/livevideo/machine-learning-data-science-and-deep-learning-with-python) - LiveVideo course that covers machine learning, Tensorflow, artificial intelligence, and neural networks.
* [MMdnn](https://github.com/microsoft/MMdnn) - MMdnn is a set of tools to help users inter-operate among different deep learning frameworks. E.g. model conversion and visualization. Convert models between Caffe, Keras, MXNet, Tensorflow, CNTK, PyTorch Onnx and CoreML. 
* [NeuralTalk](https://github.com/karpathy/neuraltalk2) - NeuralTalk is a Python+numpy project for learning Multimodal Recurrent Neural Networks that describe images with sentences. **[Deprecated]**
* [Neuron](https://github.com/molcik/python-neuron) - Neuron is simple class for time series predictions. It's utilize LNU (Linear Neural Unit), QNU (Quadratic Neural Unit), RBF (Radial Basis Function), MLP (Multi Layer Perceptron), MLP-ELM (Multi Layer Perceptron - Extreme Learning Machine) neural networks learned with Gradient descent or LeLevenberg–Marquardt algorithm. **[Deprecated]**
* [pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune) - Fine-tune pretrained Convolutional Neural Networks with PyTorch 

<a name="python-kaggle"></a>
#### Kaggle Competition Source Code
* [heamy](https://github.com/rushter/heamy) - A set of useful tools for competitive data science.
* [kaggle amazon](https://github.com/zygmuntz/kaggle-amazon) - Amazon access control challenge.
* [Kaggle Dogs vs. Cats](https://github.com/kastnerkyle/kaggle-dogs-vs-cats) - Code for Kaggle Dogs vs. Cats competition.
* [Kaggle Galaxy Challenge](https://github.com/benanne/kaggle-galaxies) - Winning solution for the Galaxy Challenge on Kaggle.
* [Kaggle Gender](https://github.com/zygmuntz/kaggle-gender) - A Kaggle competition: discriminate gender based on handwriting.
* [kaggle insults](https://github.com/amueller/kaggle_insults) - Kaggle Submission for "Detecting Insults in Social Commentary".
* [Kaggle Merck](https://github.com/zygmuntz/kaggle-merck) - Merck challenge at Kaggle.
* [Kaggle Stackoverflow](https://github.com/zygmuntz/kaggle-stackoverflow) - Predicting closed questions on Stack Overflow.
* [kaggle-accelerometer](https://github.com/zygmuntz/kaggle-accelerometer) - Code for Accelerometer Biometric Competition at Kaggle.
* [kaggle-advertised-salaries](https://github.com/zygmuntz/kaggle-advertised-salaries) - Predicting job salaries from ads - a Kaggle competition.
* [kaggle-bestbuy_big](https://github.com/zygmuntz/kaggle-bestbuy_big) - Code for the Best Buy competition at Kaggle.
* [kaggle-bestbuy_small](https://github.com/zygmuntz/kaggle-bestbuy_small)
* [kaggle-blackbox](https://github.com/zygmuntz/kaggle-blackbox) - Deep learning made easy.
* [kaggle-cifar](https://github.com/zygmuntz/kaggle-cifar) - Code for the CIFAR-10 competition at Kaggle, uses cuda-convnet.
* [kaggle_acquire-valued-shoppers-challenge](https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge) - Code for the Kaggle acquire valued shoppers challenge.
* [kaggle_acquire-valued-shoppers-challenge](https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge) - Code for the Kaggle acquire valued shoppers challenge.
* [open-solution-data-science-bowl-2018](https://github.com/neptune-ml/open-solution-data-science-bowl-2018) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Data-Science-Bowl-2018) for [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018).
* [open-solution-googleai-object-detection](https://github.com/neptune-ml/open-solution-googleai-object-detection) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Google-AI-Object-Detection-Challenge) for [Google AI Open Images - Object Detection Track](https://www.kaggle.com/c/google-ai-open-images-object-detection-track).
* [open-solution-home-credit](https://github.com/neptune-ml/open-solution-home-credit) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Home-Credit-Default-Risk) for [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).
* [open-solution-salt-identification](https://github.com/neptune-ml/open-solution-salt-identification) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Salt-Detection) for [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge).
* [open-solution-ship-detection](https://github.com/neptune-ml/open-solution-ship-detection) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Ships) for [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection).
* [open-solution-toxic-comments](https://github.com/neptune-ml/open-solution-toxic-comments) -> source code for [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
* [open-solution-value-prediction](https://github.com/neptune-ml/open-solution-value-prediction) -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Santander-Value-Prediction-Challenge) for [Santander Value Prediction Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge).
* [wiki challenge](https://github.com/hammer/wikichallenge) - An implementation of Dell Zhang's solution to Wikipedia's Participation Challenge on Kaggle.
* [wine-quality](https://github.com/zygmuntz/wine-quality) - Predicting wine quality.

<a name="python-reinforcement-learning"></a>
#### Reinforcement Learning
* [Coach](https://github.com/NervanaSystems/coach) - Reinforcement Learning Coach by Intel® AI Lab enables easy experimentation with state of the art Reinforcement Learning algorithms
* [DeepMind Lab](https://github.com/deepmind/lab) - DeepMind Lab is a 3D learning environment based on id Software's Quake III Arena via ioquake3 and other open source software. Its primary purpose is to act as a testbed for research in artificial intelligence, especially deep reinforcement learning.
* [Gym](https://github.com/openai/gym) - OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms.
* [Retro](https://github.com/openai/retro) - Retro Games in Gym
* [Roboschool](https://github.com/openai/roboschool) - Open-source software for robot simulation, integrated with OpenAI Gym.
* [Serpent.AI](https://github.com/SerpentAI/SerpentAI) - Serpent.AI is a game agent framework that allows you to turn any video game you own into a sandbox to develop AI and machine learning experiments. For both researchers and hobbyists. 
* [SLM Lab](https://github.com/kengz/SLM-Lab) - Modular Deep Reinforcement Learning framework in PyTorch.
* [ViZDoom](https://github.com/mwydmuch/ViZDoom) - ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.


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
