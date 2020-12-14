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
  The code contains all the projects in an input list called "projects" in the code. If you want to add your own projects, please put the .csv files into the **/data** folder and put the file name in the "projects" list. (We cannot share our proprietary data, so this project is excluded in this repo)
- To reproduce our results for figure 3 in the discussion section, run **Discussion2.py** in command line by typing
  ```
  $ python3 Discussion2.py
  ```
  The code contains all the projects and their names in two input lists called "projects" and "project_name" in the code. If you want to add your own projects, please put the .csv files into the **/data** folder and put the file name in the "projects" and "project_name" list. (We cannot share our proprietary data, so this project is excluded in this repo)
