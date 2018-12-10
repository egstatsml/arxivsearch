# 05/12/2018
Changed naming convention to be more consistent with arXiv API (commit dd0f18b59459922bad853ec8fe05c678ec3f2ac0)
The arXiv API defines what a category is, such as `stat.ML`, `cs.CV` etc. In my code I called a category something different, so I updated my code to be consistent with that. 

- changed `category` to `topic`
- changed `subject` to `category`
