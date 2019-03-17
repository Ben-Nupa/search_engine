# Search engine project - MSc in AI CentraleSupelec

Search engine project for the Information Retrieval course from CentraleSupelec.

## Requirements
Python 3.5+ is needed.
For the libraries, see `requirements.txt`:
```bash
pip install -r requirements.txt
```
You also need to put the data for both collections in a folder `data` (more precisely: `data/CACM` and `data/pa1-data`).

## Report
Please run `Rapport.ipynb` or [check the notebook](https://github.com/BenoitLaures/search_engine/blob/master/Rapport.ipynb).

## Technical details
We built our model using SciPy [sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html).
