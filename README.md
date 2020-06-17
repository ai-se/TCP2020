# How Different is Test Case Prioritization for Open and Closed Source Projects

## RQ Results of Prioritization Schemes

- Clone the repository and install the required packages list in **requirements.txt**.
- To reproduce our results for RQ1 and RQ2, run **rqs.py** in command line by typing
  ```
  $ python3 rqs.py [data set name]
  ```
  For example:
  ```
  $ python3 rqs.py TCT_diaspora@diaspora.csv
  ```
  Simulation results will store in **result.txt** file. This file is necessary for Scott-Knott analysis.
- Run Scott-Knott analysis in command line by typing
  ```
  $ type result.txt| python2 sk_stats.py --text 30 --latex False
  ```
  Scott-Knott analysis will show in your terminal.
