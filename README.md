# TTC Subway Delay Exploratory Data Analysis (EDA)

## Project Overview

This repository contains the Exploratory Data Analysis (EDA) for Milestone 3
of our Data Science course. The project analyzes TTC subway delay records to
prepare for predictive modeling.

Our business objective is to support emergency shuttle-bus deployment
decisions. The primary classification target is whether a reported delay is
greater than 15 minutes (`Prolonged_Disruption`). The secondary regression
target is the reported `Min Delay` in minutes.

## Notebook Structure

`TTC_Delay_EDA.ipynb` runs top to bottom and covers the Milestone 3
requirements in order:

1. **Dataset overview** - size, column names, data types.
2. **Feature categorization** - temporal, categorical, numerical, identifier,
   and target variables.
3. **Train/test split** - stratified 80/20 split (`random_state=313`) taken
   before the detailed analysis, so the test set stays unexamined.
4. **Missing data and quality assessment** - missingness audit, investigation
   of the 36.5% missing `Bound` values, duplicate and range checks, and the
   row-level cleaning applied to both subsets.
5. **Descriptive statistics** - summary statistics for numerical features and
   frequency counts for categorical ones.
6. **Outliers** - IQR analysis, boxplot and distribution charts, and the
   decision to retain genuine extreme delays.
7. **Target variable exploration** - class balance, threshold sensitivity,
   regression target shape, correlation analysis, and prolonged-disruption
   rates by incident code, line, and hour.
8. **Preprocessing summary** - per-feature keep/exclude decisions and the
   steps to be fitted on the training set during modeling.

### Key findings

- About 64% of logged incidents have a delay of exactly 0 minutes.
- The classification target is heavily imbalanced: roughly 3% of records are
  prolonged disruptions at the 15-minute threshold.
- `Min Gap` correlates with `Min Delay` at 0.96 and, by its definition, is
  only known after the incident resolves, so it is **excluded as leakage**.
- `Code` is the strongest predictor available at incident onset on every
  measure (mutual information and Cramér's V), separating incident types from
  a 0% to a 19% prolonged rate.
- `Bound` is the second most informative feature because its *missingness*
  is tied to incident type, so missing values are kept as `Unknown`.
- Calendar features (`Day`, `Is_Weekend`, `Month`, `Season`) show no
  association with the target above chance.
- Restricting the model to incidents with a non-zero delay would raise the
  positive share from about 3% to about 9%. This scope choice is left open
  for the group.

## Dataset Scope

This project uses **only the 2023 and 2024 TTC subway delay datasets**. The
2020 and 2021 data are excluded because pandemic-era service patterns and
ridership were abnormal compared with the period selected for this project.

The notebook also uses the TTC subway delay-code dictionary to translate
incident codes into readable descriptions.

The required data files are:

```text
ttc-subway-delay-data-2023.xlsx
ttc-subway-delay-data-2024.xlsx
ttc-subway-delay-codes.xlsx
```

Place all three files in the same directory as `TTC_Delay_EDA.ipynb`.

## Data Source

The files are sourced from the official City of Toronto Open Data portal:

- [TTC Subway Delay Data](https://open.toronto.ca/dataset/ttc-subway-delay-data/)
- [Toronto Open Data portal](https://open.toronto.ca/)

If the filenames provided by the portal are different, rename the downloaded
files to the exact filenames listed above so the notebook can load them.

## Setup Instructions

1. Clone this repository.
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the three required `.xlsx` files listed above.
4. Place the files in the project folder beside `TTC_Delay_EDA.ipynb`.
5. Open and run the notebook cells from top to bottom.

## Data Files and GitHub

The `.xlsx` files are intentionally **not committed to GitHub**. Dataset files
can be large and should be downloaded from the official source rather than
duplicated in the repository. The repository `.gitignore` excludes `.xlsx`,
`.xls`, and `.csv` files, so each group member must download the required files
locally before running the notebook.
