# Section 2 — revision plan and progress log

Companion to [`section2-critique.md`](section2-critique.md), which states the problems this
plan fixes. Bracketed tags below (`[A1]`, `[B6]`, …) point at the corresponding critique
item.

Goal: a Section 2 that (a) is self-contained in setup and notation, (b) treats its five
estimands in parallel, (c) states the inferential claim it is named after, and (d) prepares
the reader for every result Sections 3–5 report.

## Target structure

```
§2  Inference for Data Fairness
    §2.1  Setup                                              [NEW]
          - observed data O = (X,G,Y) ~ P ∈ M, nonparametric model            [A7]
          - joint conditional law p(g,y|x) and its margins D, π, ρ_g:
            one nuisance object, four names                                   [B6]
          - decision-rule view: D, D_c, choice of c, and an honest paragraph
            on what "a Bayes-optimal G-blind rule" does and does not describe [A8, D6]
          - formal statement of what data fairness means
          - the ATE contrast, compressed, keeping the mirror-image argument   [C1]
          - contrast with model fairness, moved up, with m(x) defined here     [B3, C2]
          - inference recipe: sample splitting, EIF, one-step correction,
            Wald interval + asymptotic-linearity proposition                   [A7, C2]
    §2.2  Fairness Metrics
          §2.2.1  Thresholded metrics: DP and EO in parallel; margin
                  condition; no nuisance-correction term, and why             [A2, A3, B1]
          §2.2.2  Probabilistic metrics: DP and EO in parallel; π and ρ_g
                  as margins of the same joint                                [B1, B6]
    §2.3  Double Robustness                                   [promoted]
          - product-bias statement; connect to App. F's o_P(n^{-1/4});
            promise interval validity; forward-ref Setting 3                  [A6, B3, B8]
    §2.4  Conditional Mutual Information
          - bridge in both directions: DP=0 ⇏ CMI=0 and CMI=0 ⇏ DP=0;
            drop "strong form of data fairness"; explain Table 1 in advance   [A1]
          - estimand / EIF / estimator as numbered displays                   [B4]
          - boundary degeneracy; plug-in can be negative; forward refs        [B8, D7, D8]
    §2.5  Summary of Estimands                                [NEW table]
          rows: DP, prob. DP, EO, prob. EO, CMI
          cols: estimand | nuisances | correction term? | DR? | regular? | derivation
                                                                              [A4, B7]
```

## Commit breakdown

Each row is one logical commit in the `paper` submodule unless noted. Parent-repo pointer
bumps are interleaved rather than left to the end.

All landed on branch `section2-revision` in both repos.

| # | commit | fixes | sha |
| --- | --- | --- | --- |
| 1 | *(parent)* add `docs/` with critique and this plan | — | `3e62e8a` |
| 2 | reindent CMI derivation displays; discard word-break regression | C6 | `38e7b50` |
| 3 | fix equal-opportunity derivations and display typography | D2, D3, D4, D5 | `0344183` |
| 4 | stop shipping mediation appendix; drop dead placeholders | C4, C5 | `1a3b30a` |
| — | *(parent)* pointer bump | — | `51f91db` |
| 5 | add §2.1 Setup | A7, A8, B3, B6, C1, C2, D6 | `7b3bdb9` |
| 6 | parity and EO in parallel; margin condition | A2, A3, B1, B2 | `dcc7147` |
| 7 | recast DR as product-bias; promote to §2.3 | A6, B3, B8 | `96b1894` |
| 8 | correct CMI/parity relationship and boundary caveats | A1, B4, B5, D7, D8 | `847f0b7` |
| — | *(parent)* pointer bump | — | `cc76524` |
| 9 | add §2.5 summary-of-estimands table | A4, B7 | `8a43430` |
| 10 | rename CMI coupling weight to `\kappa` | C3 | `f956630` |
| 11 | label-bias caveat, margin condition, data-fairness definition | A5, C6 | `8ccff6e` |
| 12 | derive the parity and EO influence functions | D1, B6 | `8afe6a4` |

## Resulting structure

```
2    Inference for Data Fairness            p. 2
2.1    Setup                                p. 3
       Proposition 1 (asymptotic linearity) p. 4
2.2    Fairness Metrics                     p. 5
2.2.1    Thresholded Metrics                p. 5
2.2.2    Probabilistic Metrics              p. 6
2.3    Double Robustness                    p. 7
2.4    Conditional Mutual Information       p. 7
2.5    Summary of Estimands (Table 1)       p. 9
```

Section 2 grew from ~3.5 pages to ~7. Most of the growth is the Setup subsection and
the two additions that close logic gaps (the margin condition and the CMI non-nesting
argument); the ATE material was cut, not expanded.

## Discovered during the revision, beyond the original critique

- **The `associati / on` corruption was never committed.** It existed only in the working
  tree, introduced alongside a latexindent run. The committed text was correct, so the
  fix was to discard the change rather than repair the line.
- **Appendix D was credited with derivations it did not contain.** It listed four
  equations with no argument, and the probabilistic *demographic parity* EIF was derived
  nowhere in the paper. All four are now derived from a single ratio rule.
- **The two commented-out causal paragraphs in `discussion.tex` cited `\cite{causal}` and
  `\cite{shroud}`, neither of which exists in `refs.bib`** (21 entries). They could not
  have compiled if uncommented — further evidence they were dead rather than pending.
- **`tlfair/metrics.py` docstrings cited paper equation numbers** ("paper Eq. 5",
  "Eqs. 8, 10"), which the renumbering invalidated. They now cite section numbers, which
  are stable under equation insertion.

## Decision: the `c` collision is fixed in the paper only

`c` is not only a symbol in the manuscript — it is a **column key in generated results**
(`results/data/table2_cmi_truth.csv`, written by `analysis/sim_cmi/run.py:73` and
`tlfair/cmi_sim.py:237`). Renaming it in code would invalidate the cached CMI outputs and
force a re-run of the simulations, which is out of proportion to a notation fix.

Resolution:

- **paper**: `c` keeps its conventional meaning as the decision threshold (it also matches
  the `D_c` subscript and the `metrics.py` docstrings). The CMI simulation's coupling
  weight is renamed `\kappa`.
- **code**: untouched. `c` remains the coupling-weight parameter and results column.

Symbol mapping, for anyone reading the paper next to the pipeline:

| paper | code | meaning |
| --- | --- | --- |
| `c` | `threshold` / the `_c` in `D_c` | decision threshold, default `1/2` |
| `\kappa` | `c` (`cmi_sim.py`, `run.py`, results column `"c"`) | CMI simulation coupling weight, swept 0 → 4 |

If the CMI simulations are ever re-run from scratch, rename the code parameter to `kappa`
at that point and delete this table.

## Verification

- `cd paper && latexmk -pdf main.tex` — clean build, no undefined references
  (`app:equal`, `app:proof`, `app:assume`, `app:cmi_eif`, `sec:tl`, and the new
  `tab:estimands`).
- `grep -n "associati" sections/*.tex` returns nothing.
- Symbol audit: every symbol used in `inference.tex` (`m(x)`, `\bar\Psi`, `\Prob_n`, `D`,
  `D_c`, `π`, `ρ_g`, `p(g,y|x)`, `c`) is defined at or before first use in the main text.
- No remaining use of `c` as the CMI coupling weight in `simulations.tex` or
  `appendix_cmi.tex`.
- §2.5's table matches `tlfair/metrics.py`: `parity`/`opportunity` plug-in (no correction
  term), `prob_parity`/`prob_opportunity` carry the residual term and DR, `cmi` plug-in.
- Read §2 → §4 in sequence and confirm every number in Table 1 (including `CMI = -0.00`
  next to `Parity = 0.17`) is interpretable from §2 alone.
