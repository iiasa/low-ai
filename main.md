# IPCC Social Norm Knowledge Gap

## Overview

Brief placeholder for IPCC-related social norm and knowledge-gap framing (climate, housing, transport, food).

## Social norms & knowledge gaps

- **IPCC context:** …
\section{Motivation: measuring socio-cultural drivers of mitigation in the wild}

Demand-side mitigation is not only a matter of technical feasibility but of social dynamics: norms, narratives, trust, cultural meanings, and collective action shape whether low-carbon practices and technologies become socially acceptable and thus scalable. The IPCC emphasizes that policies and technologies that run counter to social norms or cultural meanings are less likely to be effective, and highlights mechanisms such as socially comparative feedback, injunctive norms, and second-order normative beliefs as levers for behavioral change. It also frames behavioral contagion and social tipping points as key routes by which adoption can accelerate, and stresses that trust, procedural legitimacy, and participatory mechanisms affect acceptance of mitigation policies and infrastructures.

A central knowledge gap identified by the IPCC concerns the dynamic interaction between individual, social, and structural drivers of change, and how these interactions vary over time. Social media platforms constitute a natural observatory for these dynamics: they are not merely channels of communication, but environments where norms are negotiated, trust is built or eroded, narratives compete, and collective action is coordinated. Yet the empirical mapping from these IPCC driver constructs to large-scale, longitudinal measurements remains limited.

Here we operationalize IPCC socio-cultural drivers as a structured annotation scheme and measure their prevalence and co-occurrence in Reddit discussions about three mitigation-relevant sectors: electric vehicles, solar power, and dietary change (non-meat / plant-based). To do so at scale, we use large language models as probabilistic classifiers, producing multi-label signals (e.g., norms, trust targets, contagion cues, justice frames, movement dynamics) for each post/comment. This enables (i) time-resolved tracking of attention to each driver, (ii) interaction analyses among drivers (co-occurrence and conditional dependencies), and (iii) sector comparisons that reveal where societal readiness is being constructed versus contested.

- **Social norms:** …
- **Knowledge gap:** …


- **Notes to self:** …
So all this analsyis is for a paper I am working on. Focus: societal readiness for different climate technologies on social media, particularly about electronic vehicles, veganism, and solar power. So that's my paper. And I'm using Reddit data set where I have three sectors and posts related to each sector. That's my data set. Now, I'm trying to find surveys which correspond to questions about social nature of these climate drivers of climate action or drivers of change. So that's why I attached the IPCC chapter on this. And I paste below what my supervisor wants me to focus on. So I'm looking for surveys which have questions related to these three, namely solar, EVs, and non-meat or diet. So these three, diet, solar power, electronic vehicles, and surveys which ask questions about these, but of the nature of social drivers of change. So like social norms, trust, behavior contagion, collective action, and social movements, and cultural and religion. Now, at first, I want to find the surveys which have these questions, hopefully recent, or at least after 2020 and before 2025, or 2025 is always fine. And the next step would be to map those questions to actually how we will track this in Reddit data. So I have GPUs which I will basically use large language models to classify into a structured. I will pass the Reddit post and I would like to say yes or no to some categories, basically, based on the survey questions. my supervisor" Following upon our conversation yesterday, I thought about the main source of this terminology I am so familiar with, because it is apparently a tacit knowledge obtained in the community, and I found out that it is the IPCC report. I am sending you my copy of chapter 5. You can look at Section 5.4, and Table 5.4 for a summary. For the quickest overview, the figure below from the supplementary material of this chapter. I also highlighted the knowledge gap section I mentioned yesterday. That is, a figure on how the attention to those drivers on Reddit changed over time, in relation to each other, can help to address this knowledge gap. I hope this helps you. Let me know if you find additional questions on the surveys relevant to those.
Anchor: what IPCC Ch5 is asking you to measure
1.1 The chapter's "drivers of change" framing is explicitly social: norms, meanings/identity, narratives, trust, participation, social movements, etc., and it treats these as interacting over time with infrastructure/technology and institutions. 

IPCC_AR6_WGIII_Chapter05

 

IPCC_AR6_WGIII_Chapter05


1.2 The "knowledge gap" your supervisor pointed to is basically: measure the dynamic interaction of (i) individual, (ii) social, (iii) structural drivers, including how social media shapes narratives and how social movements affect norms and political change. 

IPCC_AR6_WGIII_Chapter05

Surveys (2020–2025) that actually touch your 3 sectors + social drivers
2.1 Yale/Meta/Rare: "Household Climate Actions: Adoption and Barriers" (multi-country, 2023)

Covers all three: EVs, solar panels, eating less meat.

Has direct social-driver hooks in the barrier batteries (peer effects / diffusion): e.g., "no one I know is doing this / I don't see others doing it", plus perceived feasibility and social friction.

This is your highest-yield instrument because you can map the barrier options almost one-to-one into Reddit labels.

2.2 European Investment Bank (EIB) Climate Survey 2023–2024

Explicitly includes policy acceptability questions that touch EVs, renewables/solar, and diet/meat (via food / behavioral policy items), and it's designed around collective responsibility + legitimacy rather than tech-only preferences.

2.3 Pew Research Center (US) – local solar/wind development + energy transition (2024)

Solar is treated as a community conflict / governance object (local support/opposition, perceived local impacts), which is exactly where trust, norms, and collective action show up in text.

(Even when questions are not phrased as "trust", the structure is: "who should decide / what is acceptable / what tradeoffs are legitimate".)

2.4 Eurobarometer: "Climate Change" (EU, 2023)

Includes self-reported actions and responsibility attribution ("who should tackle climate change"), which is a clean measurement of collective action norms and institutional role expectations.

It also references enabling actions like installing solar / buying electric cars in the "what should authorities encourage" framing (institutional driver channel).

2.5 ISSP 2020 Environment IV (fielded 2020–2023; released 2023)

Strong on trust in institutions + diet behavior frequency (meat consumption days/week), plus broad environmental attitudes. This is good for the "trust / norm climate" layer that will condition tech narratives.

2.6 (Optional, if you want a "social norms battery" to borrow verbatim) Yale "Climate Change in the American Mind" tracking

Not always tech-specific, but it's the best source for clean perceived norms items (descriptive + injunctive) you can directly port as Reddit classification questions.

A practical mapping: from survey question → Reddit classifier label
3.1 Define a small orthogonal basis of social-driver labels (binary or ternary). Suggested minimal set aligned to IPCC 5.4 + your supervisor:

Descriptive norms ("people around me do X / adoption is common") 

IPCC_AR6_WGIII_Chapter05

Injunctive norms / moral pressure ("people should do X; shame/praise; 'good/bad person'") 

IPCC_AR6_WGIII_Chapter05

Trust / distrust in actors (gov, utilities, companies, science, media, "they're lying") 

IPCC_AR6_WGIII_Chapter05

Collective action & movements (organizing, protests, strikes, petitions, "movement", "culture war") 

IPCC_AR6_WGIII_Chapter05

Identity / culture / religion (status, masculinity, tradition, purity, tribe markers) 

IPCC_AR6_WGIII_Chapter05

Contagion / diffusion / social learning ("I switched because friends did"; "everyone's installing panels now") 

IPCC_AR6_WGIII_Chapter05

Legitimacy of policy / mandates (coercion vs consent; fairness; "nanny state") 

IPCC_AR6_WGIII_Chapter05

3.2 For each sector (EV / solar / diet), you then ask the LLM a survey-anchored yes/no battery per post:

"Does this text reference what others do/approve?" → norms

"Does it cite trustworthiness of institutions/companies/experts?" → trust

"Does it invoke identity groups / culture / religion / status?" → identity

"Does it call for / describe collective action?" → movements

"Does it mention peer influence / diffusion explicitly?" → contagion

The figure your supervisor wants (directly hits IPCC 'knowledge gap 4')
4.1 Build, per month (or quarter), for each sector:

share of posts containing each driver label in 3.1

plus a cross-driver coupling metric (e.g., PMI between labels, or a small driver–driver co-occurrence network)
4.2 Then show relative attention trajectories across drivers (norms vs trust vs movements vs identity…) and across sectors. That is exactly "dynamic interaction" on social media. 

IPCC_AR6_WGIII_Chapter05


## Data Sources

### Subreddits Used in Analysis

We analyzed Reddit discussions from 16 subreddits across three climate-related behavioral sectors: transport (\textit{r/electricvehicles}, \textit{r/ElectricScooters}, \textit{r/Electricmotorcycles}), housing (\textit{r/solar}), and food (\textit{r/vegan}, \textit{r/veganarchism}, \textit{r/vegancirclejerk}), supplemented by nine general climate-related subreddits (\textit{r/climate}, \textit{r/ClimateActionPlan}, \textit{r/climatechange}, \textit{r/ClimateChaos}, \textit{r/climatedisalarm}, \textit{r/ClimateMemes}, \textit{r/ClimateOffensive}, \textit{r/ClimateShitposting}, \textit{r/climateskeptics}). Posts and comments were filtered using regex patterns to identify sector-relevant content and processed through a hierarchical norms labelling pipeline to extract social norm signals.

**Subreddit List (16 unique subreddits):**

**Transport (3):** electricvehicles, ElectricScooters, Electricmotorcycles | **Housing (1):** solar | **Food (3):** vegan, veganarchism, vegancirclejerk | **Climate (9):** climate, ClimateActionPlan, climatechange, ClimateChaos, climatedisalarm, ClimateMemes, ClimateOffensive, ClimateShitposting, climateskeptics

## Links to this project

- Sector cache: `paper4data/sector_to_comments_cache.json` (transport, housing, food).
- Disagreement labelling: vLLM pipeline in `00_vLLM_hierarchical.py`.
- Labels: `paper4data/disagreement_labels.json`.

## References

- IPCC AR6 (and relevant chapters).
- (Add key papers on social norms / knowledge gaps.)

@article{Pearce2019SocialMediaLife,
  author  = {Pearce, Warren and Niederer, Sabine and {\"O}zkula, Suay Melisa and S{\'a}nchez Querub{\'i}n, Natalia},
  title   = {The social media life of climate change: Platforms, publics, and future imaginaries},
  journal = {Wiley Interdisciplinary Reviews: Climate Change},
  year    = {2019},
  volume  = {10},
  number  = {2},
  pages   = {e569},
  doi     = {10.1002/wcc.569}
}

@article{Jachimowicz2018SecondOrderNorms,
  author  = {Jachimowicz, Jon M. and Hauser, Oliver P. and O'Brien, Julia D. and Sherman, Erin and Galinsky, Adam D.},
  title   = {The critical role of second-order normative beliefs in predicting energy conservation},
  journal = {Nature Human Behaviour},
  year    = {2018},
  volume  = {2},
  number  = {10},
  pages   = {757--764},
  doi     = {10.1038/s41562-018-0434-0}
}

@article{Otto2020SocialTipping,
  author  = {Otto, Ilona M. and Donges, Jonathan F. and Cremades, Roger and Bhowmik, Amar and Hewitt, Richard J. and Lucht, Wolfgang and Rockstr{\"o}m, Johan and Allerberger, Franz and McCaffrey, Mark and Doe, Samuel S. P. and Lenferna, Alexander and Mor{\'a}n, Nils and van Vuuren, Detlef P. and Schellnhuber, Hans Joachim},
  title   = {Social tipping dynamics for stabilizing Earth's climate by 2050},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2020},
  volume  = {117},
  number  = {5},
  pages   = {2354--2365},
  doi     = {10.1073/pnas.1900577117}
}

@article{IvesKidwell2019ReligionValues,
  author  = {Ives, Chris and Kidwell, Jeremy},
  title   = {Religion and social values for sustainability},
  journal = {Sustainability Science},
  year    = {2019},
  volume  = {14},
  number  = {5},
  pages   = {1355--1362},
  doi     = {10.1007/s11625-019-00657-0}
}

@article{Wustenhagen2007SocialAcceptance,
  author  = {W{\"u}stenhagen, Rolf and Wolsink, Maarten and B{\"u}rer, Mary Jean},
  title   = {Social acceptance of renewable energy innovation: An introduction to the concept},
  journal = {Energy Policy},
  year    = {2007},
  volume  = {35},
  number  = {5},
  pages   = {2683--2691},
  doi     = {10.1016/j.enpol.2006.12.001}
}
