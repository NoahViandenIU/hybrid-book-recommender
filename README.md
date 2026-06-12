# Hybrid Book Recommendation System

This repository contains the implementation for the IU portfolio project
**Design and Implementation of a Hybrid Recommendation System for a Small
Online Bookstore**.

## Overview

The system combines content-based recommendation with user-based collaborative
filtering. It stores only pseudonymous user identifiers and synthetic ratings.
The implementation is intentionally compact so that it can be inspected,
tested, and reproduced easily.

The hybrid weight is selected on a validation holdout and then evaluated on a
separate test holdout. The evaluation also reports content-only,
collaborative-only, and popularity-only baselines.

## Repository Structure

```text
hybrid-book-recommender/
|-- app.py                  # Flask web API and static frontend server
|-- data/                   # Synthetic book catalogue and ratings
|-- frontend/               # Browser-based demo UI
|-- reports/                # Generated evaluation outputs
|-- scripts/evaluate.py     # Reproducible evaluation script
|-- src/recommender.py      # Hybrid recommendation engine
|-- tests/                  # Unit tests
`-- requirements.txt        # Python dependencies
```

## Installation

Linux and macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

If `python` is the configured Windows launcher, it can be used instead of `py`.

## Run the Web Demo

```bash
python app.py
```

Windows:

```powershell
py app.py
```

Open `http://127.0.0.1:5000` and enter a user ID such as `U003`.

## Run Tests

```bash
python -m unittest discover -s tests
```

Windows:

```powershell
py -m unittest discover -s tests
```

The suite covers the recommendation engine, deterministic ranking, validation,
API responses, invalid input handling, and the evaluation protocol.

## Reproduce Evaluation

```bash
python scripts/evaluate.py
```

Windows:

```powershell
py scripts\evaluate.py
```

The script writes `reports/evaluation_summary.json` and
`reports/evaluation_rows.csv`.

The current reproducible evaluation selects `alpha = 0.7` on the validation
split. On the separate test split, both the hybrid and content-only models
recover 14 of 15 held-out positive items in the top five. Content-only has the
higher MRR@5, so the results support the hybrid implementation but do not claim
that it outperforms every simpler baseline.

## API

- `GET /api/books`
- `GET /api/recommendations/<user_id>?k=5`

The API accepts between 1 and 20 recommendations and returns JSON errors with
HTTP status 400 for invalid parameters.

## Privacy Note

The dataset contains no real personal data. All users are represented by
pseudonymous IDs such as `U001`. The catalogue and ratings are synthetic and
designed for reproducible academic evaluation.
