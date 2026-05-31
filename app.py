import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data
from src.model import BayesianOptimizer

st.set_page_config(page_title="Bayesian Optimizer", page_icon="🧠", layout="wide")
st.title("🧠 Bayesian Optimizer")
st.markdown("Gaussian Process-based Bayesian optimization with Expected Improvement acquisition.")

dataset_name = st.selectbox("Dataset", ["wine", "iris", "breast_cancer"])
X, y, fn = load_data(dataset_name)

c1, c2 = st.columns(2)
with c1:
    n_iter = st.slider("Iterations", 10, 100, 30, 5)
with c2:
    n_initial = st.slider("Initial Random Points", 2, 20, 5)

if st.button("Run Bayesian Optimization", type="primary"):
    with st.spinner(f"Optimizing for {n_iter} iterations..."):
        param_space = {
            "n_estimators": [10, 300],
            "max_depth": [1, 30],
            "min_samples_split": [2, 20],
            "min_samples_leaf": [1, 10],
        }
        bo = BayesianOptimizer(n_iter=n_iter, n_initial=n_initial)
        results = bo.optimize(X, y, param_space)
    st.success(f"Best Score: **{results['best_score']:.3f}**")
    st.json(results["best_params"])
    history = results["history"]
    df_hist = pd.DataFrame({"Iteration": range(len(history["scores"])), "Score": history["scores"]})
    df_hist["Best So Far"] = df_hist["Score"].cummax()
    st.line_chart(df_hist.set_index("Iteration")[["Score", "Best So Far"]])
