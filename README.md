# arxivsearch

Repo that automatically pulls links to new eprints published on arXiv
for machine learning research related to Bayesian methods for machine learning.
Papers are found for key search terms listed against specific catagories.
Current catagories are, bnn, causal, interpretable and variational.
Papers are displayed in a markdown format, inspired by
[arXausality by Logan Graham](https://github.com/logangraham/arxausality),
who displays new papers relating to causality and machine learning.
This project works in a similar (though less comprehensive in implelementation),
but allows for additional search catagories to be included. The papers for
each catagory are listed in each corresponding directory.

Currently Supported Search Catagories:

[BNNs (Bayesian Neural Netyworks)](https://github.com/ethangoan/arxivsearch/tree/master/bnn)

[Causal Machine Learning](https://github.com/ethangoan/arxivsearch/tree/master/causal)

[Interpretable Machine Learning](https://github.com/ethangoan/arxivsearch/tree/master/interpretable)

[Variational (and other approximation) Methods](https://github.com/ethangoan/arxivsearch/tree/master/variational)



Repo is automatically updated once a week.


## Install
```bash
#install arxivpy
pip install git+https://github.com/titipata/arxivpy
#clone this repo
git clone https://github.com/ethangoan/arxivsearch.git
#add this module to your PYTHONPATH
echo PYTHONPATH=$PYTHONPATH/<where you cloned this repo>/arxivsearch
```

## Running
```bash
#update for the previous 30 days
./bin/search variational --days 30
```
I have made a bash script that will change to the directory of this repo,
and then run the update script to push everything to this repo. This script
is then added to cron so that it can be set to automatically run once a week.


## Contact
If you have any questions, or want me to add a catagory, please feel free to email me

Ethan Goan
ethanjgoan@gmail.com

