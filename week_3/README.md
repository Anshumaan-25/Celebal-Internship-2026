# Customer Intelligence System

End-to-end notebook for clustering (K-Means, DBSCAN) and ensemble classification (Random Forest, XGBoost) on country development indicators.

## Dataset
The dataset files for this assignment are included in the `data/` directory in this folder. Files present:

- `data/Country-data.csv` (primary table)
- `data/Country-data.xls` (original spreadsheet)
- `data/data-dictionary.*` (metadata)

If you prefer to re-download from Kaggle instead of using the included files, you can use:

```
kaggle kernels output jatin2bagga/unsupervised-learning-on-country-data -p data
kaggle datasets download rohan0301/unsupervised-learning-on-country-data -p data --unzip
```

If using the Kaggle download, make sure your Kaggle API token is configured at `~/.kaggle/kaggle.json`.

## Setup
Install dependencies:

```
pip install -r requirements.txt
```

## Run
Open the notebook and run all cells:

- customer_intelligence_country_segmentation.ipynb

## Outputs
- K-Means model selection (elbow + silhouette)
- DBSCAN clustering with k-distance guidance
- PCA cluster visualization
- Cluster profiles and draft insights
- Development index labels and ensemble model metrics
- Feature importance charts

## Notes
- Update the "Draft insights" section with 3 to 5 concrete observations from your results.

## Submission checklist
- Include these files/folders in your upload: `customer_intelligence_country_segmentation.ipynb`, `unsupervised-learning-on-country-data.ipynb`, `requirements.txt`, `README.md`, and the `data/` directory.
- Ensure the notebook runs end-to-end (I executed it successfully on my side).
- No virtual environments are included; remove any local `.venv` before uploading.

If you want me to produce a lighter `customer_intelligence_country_segmentation.ipynb` without outputs (smaller file), I can create a cleared copy for upload instead.
