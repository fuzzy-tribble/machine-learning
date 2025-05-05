# Machine Learning Experiments

![Website](https://img.shields.io/badge/launch-website-yellow)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fuzzy-tribble/machine-learning-experiments/main)

# Usage
- For browsing and demos [reference the workbook content online](https://fuzzy-tribble.github.io/machine-learning-experiments)
- Or run workbooks interactively online here: https://mybinder.org/v2/gh/fuzzy-tribble/machine-learning-experiments/main
- Or run workbooks interactively on local machine like this:

```bash
# open VS code, clone or fork this repo then open terminal and run the following commands to create a virtual environment and install dependencies then you can run the notebooks locally

# create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Select `.venv` kernel in Jupyter Notebook

# Contributing

- Contribute to lecture notes in `notes/`
- Contribute to website configs in `website/` (note that this content is auto-generated so refrain from editing content manually)
- Contribute to ml experiments/notebooks in `notebooks/`
    + notebooks are generally labelled with `exp_*` for experiments, `pg_*` for playgrounds, and `eda_*` for exploratory data analysis

When you are happy with your changes rebuild the static site from the notebooks folder

```bash
cd tools
python convert_ipynbs.py
```