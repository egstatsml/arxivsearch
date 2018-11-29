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

[Causal Inference (Link to arXausality by Logan Graham)](https://github.com/logangraham/arxausality)

[Fairness in Statistics and Machine Learning](https://github.com/ethangoan/arxivsearch/tree/master/fairness)

[Interpretable Machine Learning](https://github.com/ethangoan/arxivsearch/tree/master/interpretable)

[Variational (and other approximation) Methods](https://github.com/ethangoan/arxivsearch/tree/master/variational)



Repo is automatically updated once a week on Fridays 12:30pm AEST (+10 GMT)

## Install
```bash
#install arxivpy
pip install git+https://github.com/titipata/arxivpy
#clone this repo
git clone https://github.com/ethangoan/arxivsearch.git
#add this module to your PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/<where you cloned this repo>/
```
To add this package to your Python path permanently (on Linux), you can run the commands
```bash
echo PYTHONPATH=$PYTHONPATH:/<where you cloned this repo>/ >> ~/.bashrc
source ~/.bashrc
```

## Running
```bash
#update for the previous 30 days
./bin/search variational --days 30
```
I have made a bash script that will change to the directory of this repo,
and then run the update script to push everything to this repo. This script
is then added to cron so that it can be set to automatically run once a week.

## Adding Your own Terms
To add your own search catagory,
1. add a your own category class in `category.py` that inherits from the `category` class
2. set your general and specific terms as done in the other classes (look for some notes below on how you might want to specify these)
3. Add your class selection to the `get_category()` class
4. Update the `InvalidCategoryError` exception string at the top to an informative error message of your choice
5. You can stop here if you just want to run the `search` script, but if you want to run the concatenate and update all script keep playing along
6. Update the `bin/update_all` script to include the new search classes you made
7. If you want to use the Git markdown functionality, you will have to store your credentials using `git config credential.helper store`. You will also have to change the remote path to your own repo where you can push to.
8. On a Linux machine, you can add the running the `update_all` script to `cron` to schedule it to run automatically (Will be able to do something similar on Windows or Mac)


## Contact
If you have any questions, or want me to add a catagory, please feel free to email me

Ethan Goan
ethanjgoan@gmail.com



### TODO

[] Change the way I combine searches from previous week (save a dataframe instead of a markdown file)
