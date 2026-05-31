# Bayesian Optimizer

**Gaussian Process**-based Bayesian optimization with Expected Improvement acquisition.

## Overview

- GP surrogate model with RBF kernel
- Expected Improvement acquisition function
- Convergence tracking with best-so-far curve
- **Streamlit dashboard** with interactive optimization

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main optimize --dataset wine --n-iter 30
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## License

MIT
# Bayesian-Optimizer
