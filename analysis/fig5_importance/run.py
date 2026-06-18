"""Figure 5 (feature importance) — CUT from the paper.

Feature importance was removed: it is dataset-specific, orthogonal to the
targeted-learning contribution, and lacks valid uncertainty quantification.
This script is no longer wired into the Snakemake workflow. The original
implementation is retained, commented out, below for reference only.
"""

# """Figure 5: per-metric feature importance for Adult (top) and Law (bottom).
#
# A 2x5 grid of horizontal bar plots, one column per fairness metric (paper
# Section 5, Figure 5).
# """
#
# import argparse
# import pickle
# import sys
# from pathlib import Path
#
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# from tlfair.plotting import configure_matplotlib, FULL_WIDTH
#
# # Map importance-dict keys to display titles. Robust to extra metrics (e.g. the
# # TMLE variants) being present: unknown keys fall back to the key itself.
# METRIC_TITLES = {
#     'parity': 'Parity',
#     'prob_parity': 'Prob. Parity',
#     'opportunity': 'Opportunity',
#     'prob_opp': 'Prob. Opportunity',
#     'cmi': 'CMI',
#     'prob_parity_tmle': 'Prob. Parity (TMLE)',
#     'prob_opp_tmle': 'Prob. Opportunity (TMLE)',
#     'cmi_tmle': 'CMI (TMLE)',
# }
#
#
# def _load(path):
#     with open(path, 'rb') as f:
#         return pickle.load(f)
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--adult-input', default='data/generated/adult_results.pkl')
#     parser.add_argument('--law-input', default='data/generated/law_results.pkl')
#     parser.add_argument('--output', default='results/figures/fig5_importance.pdf')
#     args = parser.parse_args()
#
#     configure_matplotlib()
#     adult = _load(args.adult_input)
#     law = _load(args.law_input)
#     if not law.get('importance') or not adult.get('importance'):
#         sys.exit(
#             "No feature-importance results in the analyze_* pickles. The "
#             "importance analysis is off by default; regenerate with a positive "
#             "permutation count, e.g.:\n"
#             "  uv run snakemake --forcerun analyze_adult analyze_law "
#             "fig5_importance --cores 16 --config njobs=16 importance_samples=1000"
#         )
#     metrics = list(law['importance'].keys())
#
#     fig, axs = plt.subplots(2, len(metrics), figsize=(FULL_WIDTH * 2.3, 8))
#     for i, m in enumerate(metrics):
#         adult_res = adult['importance'][m][0]
#         law_res = law['importance'][m][0]
#         adult_df = pd.DataFrame({'Variable': list(adult_res.keys()), 'Importance': list(adult_res.values())})
#         law_df = pd.DataFrame({'Variable': list(law_res.keys()), 'Importance': list(law_res.values())})
#
#         sns.barplot(adult_df, y='Variable', x='Importance', ax=axs[0, i])
#         sns.barplot(law_df, y='Variable', x='Importance', ax=axs[1, i])
#         axs[0, i].set_title(METRIC_TITLES.get(m, m), fontsize=11)
#         axs[0, i].set_xlabel('')
#         if i >= 1:
#             axs[0, i].set_yticklabels([])
#             axs[1, i].set_yticklabels([])
#             axs[0, i].set_ylabel('')
#             axs[1, i].set_ylabel('')
#
#     fig.tight_layout()
#     fig.savefig(args.output)
#     print(f'Wrote {args.output}', flush=True)
#
#
# if __name__ == '__main__':
#     main()
