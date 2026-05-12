# Social Norms Definitions Report

## Canonical Definitions From The Literature

### 1. Descriptive norms

In the focus theory of normative conduct, descriptive norms guide behavior via perceptions of what most others do. Cialdini, Kallgren, and Reno define descriptive norms as norms that guide behavior through the perception of how most others behave.
Source: Cialdini, Kallgren, & Reno (1991), *A Focus Theory of Normative Conduct*
Link: https://doi.org/10.1016/S0065-2601(08)60330-5

Rimal and Real make the same distinction in communication theory: descriptive norms are peopleâ€™s perceptions about the prevalence of a behavior.
Source: Rimal & Real (2005), *How Behaviors are Influenced by Perceived Norms*
Link: https://doi.org/10.1177/0093650205275385

Important implication:

- a descriptive norm is **not** just any mention of behavior
- it is specifically about what is perceived as common, typical, or prevalent in a relevant group

### 2. Injunctive norms

Cialdini et al. define injunctive norms as perceptions of what behaviors others approve or disapprove of.
Source: Cialdini, Kallgren, & Reno (1991)
Link: https://doi.org/10.1016/S0065-2601(08)60330-5

Rimal and Real define injunctive norms as perceptions that important referents expect compliance with a behavior. They also tie injunctive norms to social approval and potential sanction.
Source: Rimal & Real (2005)
Link: https://doi.org/10.1177/0093650205275385

Important implication:

- injunctive norms are not just any moral statement
- the social piece matters: approval, expectation, obligation, sanction, or pressure from others

### 3. Social norms more broadly

Bicchieriâ€™s framework is stricter than the common descriptive/injunctive split. A social norm exists when peopleâ€™s preference to conform depends on social expectations:

- empirical expectations: belief about what others do
- normative expectations: belief about what others think one should do
- conditionality: behavior depends on those expectations

Source: Bicchieri, *Norms in the Wild* and Bicchieri, *Measuring Social Norms*
Links:
- https://doi.org/10.1093/acprof:oso/9780190622046.001.0001
- https://www.irh.org/wp-content/uploads/2016/09/Bicchieri_MeasuringSocialNorms.pdf

Important implication:

- a lone behavior report is not automatically a social norm
- a norm claim should be tied to a relevant reference network and social expectations

### 4. Reference group / reference network

Bicchieri defines the relevant people as the **reference network**: the people whose behaviors and beliefs matter for my behavior.
Source: Bicchieri, *Measuring Social Norms*
Link: https://www.irh.org/wp-content/uploads/2016/09/Bicchieri_MeasuringSocialNorms.pdf

Rimal and Real similarly emphasize important referents, group identity, similarity, aspiration, and social networks.
Source: Rimal & Real (2005)
Link: https://doi.org/10.1177/0093650205275385

Tankard and Paluck also stress that people build norm perceptions from behavior, summary information, and institutional signals, and that the effect depends on the personâ€™s relationship to the source.
Source: Tankard & Paluck (2016), *Norm Perception as a Vehicle for Social Change*
Link: https://doi.org/10.1111/sipr.12022

Important implication:

- a reference group does **not** have to be only `my family / my coworkers / my neighbors`
- it can also be a salient in-group or relevant social category:
  - `people here`
  - `this subreddit`
  - `vegans`
  - `EV owners`
  - `people in my town`
  - `people like me`

So the current rule requiring a personal relationship is too narrow.

---

## Climate-Mitigation Perspective

The climate-mitigation literature treats social norms as a substantive mechanism of behaviour change rather than a side detail. In this setting, what matters is not an exhaustive ontology of every group mention, but a disciplined way to capture the social channels through which mitigation-relevant narratives, approval, disapproval, imitation, and identity signals travel. That supports a selective reference-group taxonomy centered on broad publics, identity groups, online publics, close ties, and local community ties. It also supports keeping dynamic commonness claims such as â€œbecoming normalâ€ or â€œmore common nowâ€ inside the descriptive-norm family, because climate transitions are often discussed as moving social patterns rather than only static majorities.

## Comment-Level Versus Community-Level Norms

Reddit requires one extra interpretive step: a single comment may be an explicit norm claim, a public evaluative performance, or neither. The prompts therefore aim to distinguish direct descriptive and injunctive signals from mere behavior evidence, while still treating visible public praise, blame, ridicule, or condemnation as norm-relevant when those utterances clearly help mark a behaviour as acceptable, unacceptable, expected, or mocked in the forum.

---

## Our approach

### A. Norm-relevant signal

The gate question is used to isolate comments that contain social expectation content rather than generic lifestyle discussion. The relevant signal is evidence about what relevant others typically do, what they approve or expect, or public evaluative / prescriptive discourse that contributes to a perceived norm climate in the forum. Isolated self-reports, product facts, technical descriptions, and neutral population statistics are not enough by themselves.

### B. Descriptive norm

Descriptive norms are handled as information about what a relevant group commonly does, typically does, or is understood to be doing. This is why the current approach does **not** treat lone self-report as a descriptive norm. The logic is:

- `I am vegan` is **behavior evidence**
- `many people in my friend group are vegan` is **descriptive norm information**
- `most people here are vegan` is a **clear descriptive norm statement**

Statistics are also not sufficient by themselves. Reddit contains many numeric statements that describe populations, technologies, or markets without functioning as norm signals. The prompt therefore treats statistics as descriptive norms only when they support an inference about what a relevant group typically does in a way that could guide behaviour.

### C. Injunctive norm

Injunctive norms are handled as claims about approval, disapproval, expectation, pressure, sanction, or social-regulatory force. For Reddit this includes explicit â€œshould / should notâ€ statements, but also public praise, blame, ridicule, and condemnation when the comment clearly marks the behaviour as appropriate or inappropriate in a socially meaningful way. The key distinction is not morality in general, but the presence of social approval or disapproval.

### D. Reference group

Reference groups are treated more broadly than intimate personal ties. The current approach includes close ties, local community, online publics, identity groups, and broad publics because climate discourse on Reddit often travels through diffuse publics and salient identities rather than only through family or friends. The point is not to extract every possible group mention, but to capture the kinds of social referents that plausibly mediate comparison, approval, imitation, and narrative spread.

### E. Schema logic

The resulting four-label dashboard slice is:

- `gate`: norm-relevant discourse
- `descriptive`: commonness / prevalence / normality / trend
- `injunctive`: approval / disapproval / expectation / sanction
- `reference group`: the relevant social referent for the norm claim

This keeps the current public workflow compact while remaining aligned with the literature-backed distinction between behaviour evidence, empirical expectations, normative expectations, and reference networks.

---

## Prompts used

### `1.1_gate`

```text
You are identifying norm-relevant content in online comments.

A comment contains a social-norm signal only if it refers to a relevant social group's expectations:
1. what people in that group typically do, or
2. what people in that group approve, disapprove, expect, pressure, or sanction.

Code YES when the comment contains:
- a claim about what most / many / people like X usually do
- a claim about what a group thinks people should do
- social approval or disapproval
- social pressure, criticism, shame, or sanction
- a public evaluative or prescriptive stance that helps mark the behavior as acceptable, unacceptable, admirable, embarrassing, mocked, or expected in the forum

Code NO when the comment is only:
- an individual's self-report ("I am vegan", "I bought solar")
- a private preference or logistics statement without social-regulatory force
- technical facts, product facts, or market information
- company behavior
- neutral statistics that do not indicate what a relevant social group typically does or expects

Important:
- A lone self-report is not enough.
- A single anecdote about one person is not enough.
- The social expectation component can be explicit or implicit through public approval/disapproval.

Answer with exactly one word: yes or no.
```

### `1.2.1_descriptive`

```text
You are identifying DESCRIPTIVE NORMS.

Descriptive norms are claims about what members of a relevant social group typically do, commonly do, or are perceived to do.

Code PRESENT when the comment says or clearly implies that:
- most / many / people like X do Y
- a behavior is common, typical, routine, or mainstream in a group
- a relevant group habitually engages in the behavior
- or a behavior is described as becoming more common or more normal in a group

Code ABSENT when the comment is only:
- a lone self-report ("I am vegan", "I drive an EV")
- one anecdote about one person
- technical or demographic statistics without norm relevance
- prescriptive or moral language about what people should do
- product, policy, or company information

Code UNCLEAR only when the comment gestures toward group typicality but the relevant group or prevalence claim is too ambiguous.

Important:
- Do not treat a single person's behavior as a descriptive norm by itself.
- Do not treat any statistic as a norm. It must indicate what a relevant group typically does.
- Repeated opinions may reveal a forum norm after aggregation, but a single opinion comment is not automatically descriptive.

Answer with exactly one of: present, absent, unclear.
```

### `1.2.2_injunctive`

```text
You are identifying INJUNCTIVE NORMS.

Injunctive norms are claims about what a relevant social group approves, disapproves, expects, pressures, or sanctions.

Code PRESENT when the comment says or clearly implies that:
- others think people should or should not do something
- a behavior is socially approved or disapproved
- people face pressure, criticism, shame, praise, or sanction for compliance or noncompliance
- a rule of appropriateness is being invoked
- or the speaker publicly praises, blames, ridicules, or condemns the behavior in a way that contributes to visible approval/disapproval in the forum

Code ABSENT when the comment is only:
- a pure personal taste statement with no real social-regulatory force
- private advice with no group expectation
- technical analysis or factual description
- a descriptive claim about what people do

Code UNCLEAR only when the comment has prescriptive force but the social source of approval or expectation is ambiguous.

Important:
- The key test is social-regulatory force, not just morality.
- "People should..." is stronger than "I think..."
- But public condemnation or praise can still count when it clearly marks the behavior as socially acceptable or unacceptable.

Answer with exactly one of: present, absent, unclear.
```

### `1.3.1_reference_group`

```text
Identify the REFERENCE GROUP for the norm claim.

A reference group is the social group whose behavior, beliefs, approval, or expectations matter for the norm claim.

This can include:
- direct personal ties: family, partner/spouse, friends, coworkers, neighbors
- local publics: local community
- online publics: online community, other reddit users
- salient identity groups: vegans, EV owners, political groups, people like us
- broad publics: most people, society, everyone

Choose OTHER only when no meaningful social referent is present.

Answer with exactly one of:
family, partner/spouse, friends, coworkers, neighbors, local community,
online community, other reddit users, identity group, general public, other.
```

---

## Sources

1. Cialdini, R. B., Kallgren, C. A., & Reno, R. R. (1991). *A Focus Theory of Normative Conduct: A Theoretical Refinement and Reevaluation of the Role of Norms in Human Behavior*.
Link: https://doi.org/10.1016/S0065-2601(08)60330-5

2. Rimal, R. N., & Real, K. (2005). *How Behaviors are Influenced by Perceived Norms: A Test of the Theory of Normative Social Behavior*.
Link: https://doi.org/10.1177/0093650205275385
Accessible PDF used: https://terpconnect.umd.edu/~nan/398readings/398%20background%20readings/6%20Rimal%202005%20How%20behaviors%20are%20influenced%20by%20norms.pdf

3. Bicchieri, C. (2017). *Norms in the Wild: How to Diagnose, Measure, and Change Social Norms*.
Link: https://doi.org/10.1093/acprof:oso/9780190622046.001.0001

4. Bicchieri, C. *Measuring Social Norms*. Penn Social Norms Group slides / guide.
Link: https://www.irh.org/wp-content/uploads/2016/09/Bicchieri_MeasuringSocialNorms.pdf

5. Tankard, M. E., & Paluck, E. L. (2016). *Norm Perception as a Vehicle for Social Change*.
Link: https://doi.org/10.1111/sipr.12022

6. Schultz, P. W., Nolan, J. M., Cialdini, R. B., Goldstein, N. J., & Griskevicius, V. (2007). *The Constructive, Destructive, and Reconstructive Power of Social Norms*.
Link: https://doi.org/10.1111/j.1467-9280.2007.01917.x
PubMed: https://pubmed.ncbi.nlm.nih.gov/17576283/

7. IPCC (2022). *AR6 WGIII Chapter 5: Demand, services and social aspects of mitigation*.
Links:
- https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-5/
- https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_Chapter_05.pdf

8. De Dominicis, S., Sokoloski, R., Jaeger, C. M., & Schultz, P. W. (2019). *Making the smart meter social promotes long-term energy conservation*.
Link: https://doi.org/10.1080/00224545.2019.1611512

9. Mortensen, C. R., Neel, R., Cialdini, R. B., Jaeger, C. M., Jacobson, R. P., & Ringel, M. M. (2019). *Trending Norms: A Lever for Encouraging Behaviors Performed by the Minority*.
Link: https://www.neellab.ca/uploads/1/2/1/1/121173522/mortenson_et_al_2019_spps.pdf

10. Jachimowicz, J. M., Hauser, O. P., O'Brien, J. D., Sherman, E., & Galinsky, A. D. (2018). *The critical role of second-order normative beliefs in predicting energy conservation*.
Link: https://doi.org/10.1038/s41562-018-0434-0

11. Nyborg, K., Anderies, J. M., Dannenberg, A., et al. (2016). *Social norms as solutions*.
PubMed: https://pubmed.ncbi.nlm.nih.gov/27846488/

## Inference Note

The prompt recommendations above are an inference from the cited literature. They are not copied from a single paper; they synthesize the common definitional core across the sources.
