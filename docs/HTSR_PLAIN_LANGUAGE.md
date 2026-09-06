# HTSR study in plain language

Heavy-tailed self-regularization (HTSR) summarizes the eigenvalues of model
weight matrices. We use those summaries to compare models without prompts,
inference data, architecture labels, or family labels.

## The rule

Lineage, family, architecture, model type, parameter count, source, and ancestry
links are hidden while the method groups models. They are opened only after the
answers are saved.

Performance prediction learns from benchmark scores in the training data. The
test model's score stays hidden until its prediction is saved. Descriptive model
metadata is never used as a prediction input.

## Lineage

```text
complete seven-role summaries -> fit 17 stable groups
missing-role pattern ---------> hide the same roles from complete models
                                -> include only patterns that keep neighbors stable
                                -> attach accepted models to the frozen groups
                                -> save results -> open labels and score
```

Seven common transformer roles define the strict core, but they are no longer a
general requirement:

- 10,431 complete models define the 17 groups;
- 515 models with reliable partial summaries are added;
- 10,946 models receive a lineage group.

The 515 additions pass a weight-only check: when the same roles are hidden from
complete models, the original nearest model remains in the new top ten at least
90% of the time.

After the groups are saved, agreement with 11 broad family labels is:

- adjusted Rand index 0.913 on 174 human-reviewed models;
- adjusted Rand index 0.924 on 2,981 models where metadata rules agree.

An adjusted Rand index of 1 means the two groupings match exactly after
correcting for chance. Agreement with 529 individual root IDs is lower, 0.600,
because 17 broad groups cannot represent 529 separate answers.

The displayed cluster center is a real core model closest to the group's mean.
It is a representative model, not a claim about the historical parent.

## Supplied base models

For comparison with the paper, the method still requires all seven roles. It
compares every query with a fixed list of candidate bases.

We supplied 38 base IDs; 29 had complete summaries. On 3,836 queries whose known
root was one of those 29:

- top-1 accuracy was 82.27%;
- top-3 accuracy was 98.41%;
- top-1 was 84.07% when every root received equal weight.

This answers which supplied base is most similar. It cannot return an omitted
base or determine which similar model came first.

## Performance

```text
at least four observed roles + training scores
    -> five fixed train/test parts
    -> require at least four shared roles
    -> three nearest training models
    -> weighted score estimate
    -> open test score and measure error
```

The four-role cutoff is selected without benchmark scores: masking those roles
on complete models retains the original nearest model in the top ten 90.5% of
the time.

| Scores | Models | Mean absolute error | Spearman rank correlation |
| --- | ---: | ---: | ---: |
| Modern | 2,537 | 3.043 | 0.875 |
| Legacy | 3,515 | 3.188 | 0.897 |

This covers 394 more modern and 519 more legacy scored models than the strict
seven-role method. Error rises slightly, so the improvement is coverage rather
than accuracy.

The harder test removes a model's complete weight group from training. Modern
error is 7.721 versus 9.102 for the mean baseline. Legacy error is 14.169 versus
13.182 for the mean baseline, and its rank correlation is only 0.032.

## Conclusion

Use the strict seven-role method for paper comparison. For general use, retain
the stable complete-model groups, add only missing-role patterns that pass the
masking check, and use wider performance prediction only when comparable
training models exist. The current evidence does not support exact parent
recovery, training direction, or reliable score prediction in an unseen weight
group.
