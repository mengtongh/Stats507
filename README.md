# Stats507
This repository stores jupyter notebooks for STATS507 homeworks for Fall 2021.

### PS6 
[NHANES_data_clean.ipynb](/HW6/NHANES_data_clean.ipynb) is created for reading, cleaning, and appending several data files from the National Health and Nutrition Examination Survey [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm) for four cohorts spanning the years 2011-2018. 
\
The demographic columns include id (SEQN), age (RIDAGEYR), race and ethnicity (RIDRETH3), education (DMDEDUC2), and marital status (DMDMARTL), along with the following variables related to the survey weighting: (RIDSTATR, SDMVPSU, SDMVSTRA, WTMEC2YR, WTINT2YR). Save the resulting data frame pickle object.
\
Repeat the procedure for the oral health and dentition data (OHXDEN_*.XPT) retaining the following variables: SEQN, OHDDESTS, tooth counts (OHXxxTC), and coronal cavities (OHXxxCTC).
