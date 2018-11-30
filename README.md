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



This Repo is automatically updated once a week on Fridays 12:30pm AEST (+10 GMT)


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
echo export PYTHONPATH=\$PYTHONPATH:/<where you cloned this repo>/ >> ~/.bashrc
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

## How the search will work
The search function has 3 fields, the `subject`, `generic_terms` and `specific_terms`.
The `subject` refers to the arXiv subject category where the preprint was submitted. These are listed at the bottom of this page. You can specify which subject/category you want to search for in the `self.subject` attribute in the `category` parent class. The search function will only search within these categories.

The next terms are the `general_terms` and the `specific_terms`. The `general_terms` should be broad terms that are related to your field, for example, for my `fairness` category, I include general terms such as `equality` and `bias` which are common terms used when discussing fairness in systems. The `specific_terms` should be specific to your field of interest. Again, for the `fairness` category, I use specific terms such as `algorithms` and `statistics` to limit my search results to fairness preprints within the context of statistics and machine learning.

The search will work in the format of,

`within_category & general_terms & specific_terms`

where it will iterate over all of the different combinations of your terms.

### Formatting terms
The [arXiv API](https://arxiv.org/help/api/user-manual) describes a funky way for formatting your search queries. If you have multiple words in a term you want to search for, you need to format it in a way that arXiv will handle properly. For example, say you want to search for the term `Monte Carlo`, you will need to set your search term to `%22Monte+Carlo%22`. The `%22` is basically a code that translates into double quotes, and the `+` sign is there as the URL that is sent to arXiv for the query can't handle spaces. I might automate this formatting at some point so you don't need to do this.

## Contact
If you have any questions, or want me to add a catagory, please feel free to email me

Ethan Goan
ethanjgoan@gmail.com


## arXiv Subject Categories
```
astro-ph	Astrophysics
astro-ph.CO	Cosmology and Nongalactic Astrophysics
astro-ph.EP	Earth and Planetary Astrophysics
astro-ph.GA	Astrophysics of Galaxies
astro-ph.HE	High Energy Astrophysical Phenomena
astro-ph.IM	Instrumentation and Methods for Astrophysics
astro-ph.SR	Solar and Stellar Astrophysics
cond-mat.dis-nn	Disordered Systems and Neural Networks
cond-mat.mes-hall	Mesoscale and Nanoscale Physics
cond-mat.mtrl-sci	Materials Science
cond-mat.other	Other Condensed Matter
cond-mat.quant-gas	Quantum Gases
cond-mat.soft	Soft Condensed Matter
cond-mat.stat-mech	Statistical Mechanics
cond-mat.str-el	Strongly Correlated Electrons
cond-mat.supr-con	Superconductivity
cs.AI	Artificial Intelligence
cs.AR	Hardware Architecture
cs.CC	Computational Complexity
cs.CE	Computational Engineering, Finance, and Science
cs.CG	Computational Geometry
cs.CL	Computation and Language
cs.CR	Cryptography and Security
cs.CV	Computer Vision and Pattern Recognition
cs.CY	Computers and Society
cs.DB	Databases
cs.DC	Distributed, Parallel, and Cluster Computing
cs.DL	Digital Libraries
cs.DM	Discrete Mathematics
cs.DS	Data Structures and Algorithms
cs.ET	Emerging Technologies
cs.FL	Formal Languages and Automata Theory
cs.GL	General Literature
cs.GR	Graphics
cs.GT	Computer Science and Game Theory
cs.HC	Human-Computer Interaction
cs.IR	Information Retrieval
cs.IT	Information Theory
cs.LG	Machine Learning
cs.LO	Logic in Computer Science
cs.MA	Multiagent Systems
cs.MM	Multimedia
cs.MS	Mathematical Software
cs.NA	Numerical Analysis
cs.NE	Neural and Evolutionary Computing
cs.NI	Networking and Internet Architecture
cs.OH	Other Computer Science
cs.OS	Operating Systems
cs.PF	Performance
cs.PL	Programming Languages
cs.RO	Robotics
cs.SC	Symbolic Computation
cs.SD	Sound
cs.SE	Software Engineering
cs.SI	Social and Information Networks
cs.SY	Systems and Control
econ.EM	Econometrics
eess.AS	Audio and Speech Processing
eess.IV	Image and Video Processing
eess.SP	Signal Processing
gr-qc	General Relativity and Quantum Cosmology
hep-ex	High Energy Physics - Experiment
hep-lat	High Energy Physics - Lattice
hep-ph	High Energy Physics - Phenomenology
hep-th	High Energy Physics - Theory
math.AC	Commutative Algebra
math.AG	Algebraic Geometry
math.AP	Analysis of PDEs
math.AT	Algebraic Topology
math.CA	Classical Analysis and ODEs
math.CO	Combinatorics
math.CT	Category Theory
math.CV	Complex Variables
math.DG	Differential Geometry
math.DS	Dynamical Systems
math.FA	Functional Analysis
math.GM	General Mathematics
math.GN	General Topology
math.GR	Group Theory
math.GT	Geometric Topology
math.HO	History and Overview
math.IT	Information Theory
math.KT	K-Theory and Homology
math.LO	Logic
math.MG	Metric Geometry
math.MP	Mathematical Physics
math.NA	Numerical Analysis
math.NT	Number Theory
math.OA	Operator Algebras
math.OC	Optimization and Control
math.PR	Probability
math.QA	Quantum Algebra
math.RA	Rings and Algebras
math.RT	Representation Theory
math.SG	Symplectic Geometry
math.SP	Spectral Theory
math.ST	Statistics Theory
math-ph	Mathematical Physics
nlin.AO	Adaptation and Self-Organizing Systems
nlin.CD	Chaotic Dynamics
nlin.CG	Cellular Automata and Lattice Gases
nlin.PS	Pattern Formation and Solitons
nlin.SI	Exactly Solvable and Integrable Systems
nucl-ex	Nuclear Experiment
nucl-th	Nuclear Theory
physics.acc-ph	Accelerator Physics
physics.ao-ph	Atmospheric and Oceanic Physics
physics.app-ph	Applied Physics
physics.atm-clus	Atomic and Molecular Clusters
physics.atom-ph	Atomic Physics
physics.bio-ph	Biological Physics
physics.chem-ph	Chemical Physics
physics.class-ph	Classical Physics
physics.comp-ph	Computational Physics
physics.data-an	Data Analysis, Statistics and Probability
physics.ed-ph	Physics Education
physics.flu-dyn	Fluid Dynamics
physics.gen-ph	General Physics
physics.geo-ph	Geophysics
physics.hist-ph	History and Philosophy of Physics
physics.ins-det	Instrumentation and Detectors
physics.med-ph	Medical Physics
physics.optics	Optics
physics.plasm-ph	Plasma Physics
physics.pop-ph	Popular Physics
physics.soc-ph	Physics and Society
physics.space-ph	Space Physics
q-bio.BM	Biomolecules
q-bio.CB	Cell Behavior
q-bio.GN	Genomics
q-bio.MN	Molecular Networks
q-bio.NC	Neurons and Cognition
q-bio.OT	Other Quantitative Biology
q-bio.PE	Populations and Evolution
q-bio.QM	Quantitative Methods
q-bio.SC	Subcellular Processes
q-bio.TO	Tissues and Organs
q-fin.CP	Computational Finance
q-fin.EC	Economics
q-fin.GN	General Finance
q-fin.MF	Mathematical Finance
q-fin.PM	Portfolio Management
q-fin.PR	Pricing of Securities
q-fin.RM	Risk Management
q-fin.ST	Statistical Finance
q-fin.TR	Trading and Market Microstructure
quant-ph	Quantum Physics
stat.AP	Applications
stat.CO	Computation
stat.ME	Methodology
stat.ML	Machine Learning
stat.OT	Other Statistics
stat.TH	Statistics Theory
```

### TODO

- [ ] Change the way I combine searches from previous week (save a dataframe instead of a markdown file)

- [ ] Add automated formatting
