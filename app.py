from __future__ import annotations

import os


for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import streamlit as st

from alpha_cluster_dashboard.ablation_app import render_ablation_atlas


def main() -> None:
    st.set_page_config(page_title="Model Zoo Spectral Atlas", layout="wide")
    render_ablation_atlas()


if __name__ == "__main__":
    main()
