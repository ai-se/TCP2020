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
  or with Latex text
  ```
  $ type result.txt| python2 sk_stats.py --text 30 --latex True
  ```
  Scott-Knott analysis will show in your terminal.
- Examples of Scott-Knott analysis and run time are showed in **result** folder and **runtime** folder.
- To reproduce our results for table 16 in the discussion section, run **Discussion1.py** in command line by typing
  ```
  $ python3 Discussion1.py
  ```
  The code contains all the projects in an input list. If you want to add your own project, please put the .csv file into the /data folder and put the file name in the "project" list.
