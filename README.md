# TTC Subway Delay Exploratory Data Analysis (EDA)

## Project Overview

This repository contains the Exploratory Data Analysis (EDA) for Milestone 3
of our Data Science course. The project analyzes TTC subway delay records to
prepare for predictive modeling.

Our business objective is to support emergency shuttle-bus deployment
decisions. The primary classification target is whether a reported delay is
greater than 15 minutes (`Prolonged_Disruption`). The secondary regression
target is the reported `Min Delay` in minutes.

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
