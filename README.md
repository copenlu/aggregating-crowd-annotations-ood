# Code release for the paper "Aggregating Soft Labels from Crowd Annotations Improves Uncertainty Estimation Under Distribution Shift"

The code to run experiments on each dataset are given in their respective script:

```
cifar10h.py
jigsaw.py
pos.py
rte.py
```

Running the code is dependent on python >= 3.8 and the following packages:

```
crowd-kit==1.2.1
datasets==2.12.0
matplotlib==3.6.2
nltk==3.7
numpy
pandas==1.4.4
pyarrow==10.0.0
scikit-learn
seaborn==0.12.1
tokenizers==0.13.2
torch==1.13.0
```

We provide the crowd annotations for RTE and POS in this repository.

RTE data comes from the following papers:

> Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL Recognis-
ing Textual Entailment Challenge. In Joaquin Quiñonero Candela, Ido Da-
gan, Bernardo Magnini, and Florence d’Alché-Buc, editors, Machine Learn-
ing Challenges, Evaluating Predictive Uncertainty, Visual Object Classifica-
tion and Recognizing Textual Entailment, First PASCAL Machine Learning
Challenges Workshop, MLCW 2005, Southampton, UK, April 11-13, 2005, Re-
vised Selected Papers, volume 3944 of Lecture Notes in Computer Science,
pages 177–190. Springer, 2005. doi: 10.1007/11736790\_9

> Rion Snow, Brendan O’Connor, Daniel Jurafsky, and Andrew Y. Ng. Cheap and
Fast - But is it Good? Evaluating Non-Expert Annotations for Natural Language
Tasks. In 2008 Conference on Empirical Methods in Natural Language Processing,
EMNLP 2008, Proceedings of the Conference, 25-27 October 2008, Honolulu,
Hawaii, USA, A meeting of SIGDAT, a Special Interest Group of the ACL, pages
254–263. ACL, 2008.

POS data come from the following papers:

> Kevin Gimpel, Nathan Schneider, Brendan O’Connor, Dipanjan Das, Daniel
Mills, Jacob Eisenstein, Michael Heilman, Dani Yogatama, Jeffrey Flanigan,
and Noah A. Smith. Part-of-Speech Tagging for Twitter: Annotation, Fea-
tures, and Experiments. In The 49th Annual Meeting of the Association for
Computational Linguistics: Human Language Technologies, Proceedings of
the Conference, 19-24 June, 2011, Portland, Oregon, USA - Short Papers,
pages 42–47. The Association for Computer Linguistics, 2011

> Dirk Hovy, Barbara Plank, and Anders Søgaard. Experiments with crowdsourced
re-annotation of a POS tagging data set. In Proceedings of the 52nd Annual
Meeting of the Association for Computational Linguistics, ACL 2014, June 22-
27, 2014, Baltimore, MD, USA, Volume 2: Short Papers, pages 377–382. The
Association for Computer Linguistics, 2014

> Barbara Plank, Dirk Hovy, and Anders Søgaard. Linguistically debatable or just
plain wrong? In Proceedings of the 52nd Annual Meeting of the Association
for Computational Linguistics, ACL 2014, June 22-27, 2014, Baltimore, MD,
USA, Volume 2: Short Papers, pages 507–511. The Association for Computer
Linguistics, 2014. doi: 10.3115/v1/p14-2083

The crowd annotated dats for Jigsaw and CIFAR10H can be found at the following:

- Jigsaw: [Gold data](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification); 
[Crowd annotations](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) 

- [CIFAR10H](https://github.com/jcpeterson/cifar-10h) 

For example, to run the ensembling/averaging on POS tagging, execute the following:

```
python pos.py \
  --data_loc data/POS \
  --output_dir output \
  --run_name pos_ensemble \
  --n_epochs 5 \
  --seed 1000 \
  --metrics_dir metrics \
  --loss_function KLLoss \
  --tags kld pos-ood ensemble\
  --aggregation ensemble_basic
```

