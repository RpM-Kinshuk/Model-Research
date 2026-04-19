# Alpha Near-2 Hypothesis Check

## Overall verdict
- No size-adjusted tested pair showed a concave quadratic peak near alpha=2.
- 31 size-adjusted pairs had a concave peak somewhere inside their observed alpha range.
- Distance-to-2 models were usually no better than plain linear-alpha models once the same covariates were included.

## Best size-adjusted quadratic gains
- self_attn.q_proj__start vs MATH Lvl 5: linear CV R2=0.280, quadratic CV R2=0.368, delta=0.088, peak=3.652, x-range=[1.372, 4.597]
- alpha_global_mean vs MATH Lvl 5: linear CV R2=0.380, quadratic CV R2=0.450, delta=0.070, peak=5.192, x-range=[4.148, 7.089]
- alpha_mean_across_modules vs MATH Lvl 5: linear CV R2=0.415, quadratic CV R2=0.481, delta=0.066, peak=5.071, x-range=[4.114, 6.878]
- self_attn.k_proj__start vs IFEval: linear CV R2=0.034, quadratic CV R2=0.098, delta=0.065, peak=5.008, x-range=[1.345, 9.902]
- alpha_start_across_modules vs MATH Lvl 5: linear CV R2=0.373, quadratic CV R2=0.433, delta=0.061, peak=5.861, x-range=[2.428, 7.035]
- self_attn.q_proj__start vs Average ⬆️: linear CV R2=0.346, quadratic CV R2=0.405, delta=0.059, peak=3.400, x-range=[1.372, 4.597]
- self_attn.q_proj__start vs MMLU-PRO: linear CV R2=0.473, quadratic CV R2=0.526, delta=0.052, peak=3.345, x-range=[1.372, 4.597]
- self_attn.v_proj__median vs MATH Lvl 5: linear CV R2=0.447, quadratic CV R2=0.496, delta=0.049, peak=5.549, x-range=[2.509, 13.732]
- self_attn.k_proj__start vs MMLU-PRO: linear CV R2=0.495, quadratic CV R2=0.543, delta=0.048, peak=5.837, x-range=[1.345, 9.902]
- self_attn.k_proj__start vs Average ⬆️: linear CV R2=0.396, quadratic CV R2=0.436, delta=0.039, peak=6.369, x-range=[1.345, 9.902]

## Pairs with observed support for a concave peak
- self_attn.q_proj__start vs MATH Lvl 5: peak=3.652, x-range=[1.372, 4.597], delta=0.088
- self_attn.k_proj__start vs IFEval: peak=5.008, x-range=[1.345, 9.902], delta=0.065
- alpha_start_across_modules vs MATH Lvl 5: peak=5.861, x-range=[2.428, 7.035], delta=0.061
- self_attn.q_proj__start vs Average ⬆️: peak=3.400, x-range=[1.372, 4.597], delta=0.059
- self_attn.q_proj__start vs MMLU-PRO: peak=3.345, x-range=[1.372, 4.597], delta=0.052
- self_attn.k_proj__start vs MMLU-PRO: peak=5.837, x-range=[1.345, 9.902], delta=0.048
- self_attn.k_proj__start vs Average ⬆️: peak=6.369, x-range=[1.345, 9.902], delta=0.039
- alpha_start_across_modules vs Average ⬆️: peak=5.155, x-range=[2.428, 7.035], delta=0.034
- alpha_start_across_modules vs IFEval: peak=4.455, x-range=[2.428, 7.035], delta=0.028
- self_attn.q_proj__start vs BBH: peak=3.155, x-range=[1.372, 4.597], delta=0.025
- self_attn.q_proj__start vs IFEval: peak=3.378, x-range=[1.372, 4.597], delta=0.024
- self_attn.k_proj__start vs BBH: peak=5.366, x-range=[1.345, 9.902], delta=0.022
- alpha_start_across_modules vs MMLU-PRO: peak=5.222, x-range=[2.428, 7.035], delta=0.016
- alpha_global_median vs Average ⬆️: peak=5.505, x-range=[3.761, 6.468], delta=0.015
- alpha_global_median vs MMLU-PRO: peak=5.433, x-range=[3.761, 6.468], delta=0.014
- self_attn.q_proj__start vs GPQA: peak=3.094, x-range=[1.372, 4.597], delta=0.011
- alpha_global_median vs IFEval: peak=5.096, x-range=[3.761, 6.468], delta=0.009
- alpha_global_median vs BBH: peak=5.275, x-range=[3.761, 6.468], delta=0.008
- alpha_start_across_modules vs BBH: peak=4.641, x-range=[2.428, 7.035], delta=0.006
- self_attn.v_proj__median vs BBH: peak=12.723, x-range=[2.509, 13.732], delta=0.003
- alpha_global_mean vs IFEval: peak=6.326, x-range=[4.148, 7.089], delta=0.003
- alpha_global_median vs MUSR: peak=5.470, x-range=[3.761, 6.468], delta=0.002
- alpha_median_across_modules vs IFEval: peak=5.937, x-range=[3.755, 6.741], delta=0.001
- alpha_mean_across_modules vs IFEval: peak=6.571, x-range=[4.114, 6.878], delta=0.001
- alpha_median_across_modules vs BBH: peak=6.162, x-range=[3.755, 6.741], delta=0.000
- alpha_median_across_modules vs MUSR: peak=5.954, x-range=[3.755, 6.741], delta=-0.001
- self_attn.k_proj__start vs GPQA: peak=7.264, x-range=[1.345, 9.902], delta=-0.001
- alpha_global_mean vs MUSR: peak=6.983, x-range=[4.148, 7.089], delta=-0.002
- alpha_start_across_modules vs GPQA: peak=5.657, x-range=[2.428, 7.035], delta=-0.004
- alpha_global_median vs GPQA: peak=5.631, x-range=[3.761, 6.468], delta=-0.014
- self_attn.v_proj__median vs IFEval: peak=12.203, x-range=[2.509, 13.732], delta=-0.064

## Raw mode check
- Raw mode near-2 peaks found: 0 out of 56 tested pairs.
- alpha_start_across_modules vs MATH Lvl 5: delta=0.080, peak=5.532, x-range=[2.428, 7.035]
- self_attn.q_proj__start vs MATH Lvl 5: delta=0.075, peak=3.621, x-range=[1.372, 4.597]
- self_attn.k_proj__start vs IFEval: delta=0.060, peak=4.843, x-range=[1.345, 9.902]
- alpha_start_across_modules vs Average ⬆️: delta=0.054, peak=4.748, x-range=[2.428, 7.035]
- self_attn.q_proj__start vs Average ⬆️: delta=0.045, peak=3.321, x-range=[1.372, 4.597]

## Interpretation
- If a feature never gets close to 2 in the observed sample, the data cannot really test whether performance peaks at 2 for that feature.
- If the quadratic optimum lands well above 2, the data is more consistent with a monotone-improving or late-peaking trend over the observed region than with a near-2 optimum.
- If the distance-to-2 model does not beat the linear model, that is evidence against the idea that closeness to 2 is the main explanatory pattern in this sample.
