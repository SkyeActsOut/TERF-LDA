# Grievance articulation among TERF & Gender Critical communities on Twitter

The main repository for the project: _"Just Questions": Grievance articulation among Gender Critical communities on Twitter_ by Skye Kychenthal.

### Abstract

Trans Exclusionary Radical Feminism (TERF) or Gender Critcial (GC) feminism is an anti-trans hate movement using the guise of radical feminism. It focuses on support of Lesbian, Gay, Bisexual and Heterosexual individuals (particularly women) meanwhile actively excluding trans individuals and specifically trans-feminine individuals. Trans individuals have been defined as a sort of boogie-man within TERF and GC ideology, as is the case with other prominent anti-trans groups, and is the target of wide-spread delegitimization efforts. This article hopes to add to existing literature on the topic, by exploring the main grievances articulated by Gender Critical communities. Through gathering the Twitter posts of three prominent TERF and Gender Critical accounts from Twitter’s API, and subsequent analysis using LDA (Latent Dirichlet allocation), I will test the hypothesis that TERFs and Gender Critical individuals online are radicalized and further polarized through a few key thoughts and ideas (focus on reactions of disgust, fear, and anger; a false belief of GCs being silenced, a perception of a pervasiveness of trans women in societies institutions), and that these subsequent ideas are used to delegitimize trans individuals. 

### Citations

> Kychenthal Skye. _[Proposal to article on grievance articulation among TERF & Gender Critical communities on Twitter](https://skymocha.github.io/Papers/Proopsal_Twitter_TERF_Grievance_Articulation.pdf)_. [skymocha.github.io](https://skymocha.github.io). November 8, 2022.

### Packages Used

NumPy, Pandas, Re, NLTK, Spacy, Gensim, Sci-kit learn, Tweepy

### Structure

```
.
│ 
├──  exports/*         # Exported LDA models in .CSV
├──  ALL_SETTINGS.json # A json file containing all inputs and outputs from the LDA model alongside scores
├──  TERF-LDA.py       # The main LDA model
├──  consolidate.py    # Consolidates twitter .CSV into a .TXT file with all tweets (no QRTs or RTs) 
├──  CleanText.py      # The lemmatization and cleaning process on the text. Separate from model to save time.
├──  CleanCSV.py       # CleavCSV for creating a data set from the cleaned and lemmatized CSV file
├──  couny.py          # Counting!
├──  pull.py           # Pulls tweets from individual Twitter accounts and exports them to ./tweets/ as a .CSV
│   
├──  proposal_LDA
│    ├──  JK-LDA-gen.py      # Proposal for dissecting JK Rowling's blog post for proposal methodology
│    └──  sussana-LDA-gen.py # same as JK-LDA-gen.py with minor edits fro susanna Rustin's articles 
│
├── /tweets/           # Tweets form the groups. Private for ethical reasons.
│
└── README.md  
```

### Methodology

Refer to the [proposal](https://skymocha.github.io/Papers/Proopsal_Twitter_TERF_Grievance_Articulation.pdf) and [final paper]() for methodology

Notes taken to document methodology and fine-tuning of LDA can be found in [/notes/](/notes/)