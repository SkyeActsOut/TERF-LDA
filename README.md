# Grievance articulation among TERF & Gender Critical communities on Twitter

The main repository for the project: _"Just Right." Grievance articulation among TERF & Gender Critical communities on Twitter_ by Skye Kychenthal.

### Abstract

Trans Exclusionary Radical Feminism (TERF) or Gender Critcial (GC) feminism is a form of radical feminism whose core values are antithetical to feminist ideology. It focuses on support of Lesbian, Gay, Bisexual and Heterosexual individuals (particularly women) meanwhile actively excluding trans individuals and specifically trans-feminine individuals. Trans individuals have been defined as a sort of boogie-man within this sect of radical feminism, as is the case with other prominent anti-trans groups. TERF ideology holds a loud megaphone, with prominent TERF activists including J.K. Rowling, The BBC, The Guardian, Dave Chapelle, and the LGBAlliance. The hashtag #SexNotGender regularly trends on Twitter advocating lesbians, gay men, and bisexual individuals to be attracted to their preferred sex (the bimodal in which someone is born) not gender (how one would identify). This proposal and eventual article hopes to add to existing literature on the topic, by exploring the main grievances articulated by TERF and Gender Critical communities. Through scraping Twitter posts from Twitter's API of three prominent TERF and Gender Critical accounts, and subsequent analysis using LDA (Latent Dirichlet allocation), I will test the hypothesis that TERFs and Gender Critical individuals online are radicalized and further polarized through a few key thoughts and ideas:

1. A focus on reactions of disgust, fear, and anger. 
2. A false belief in the 'silencing' of TERF voices as marginalized and an 'invasion' of women's spaces.
3. A perception there is a pervasiveness of trans women in societies' institutions.

### Citations

**Currently only the proposal is done. The paper is a WIP.**

> Kychenthal Skye. _[Proposal to article on grievance articulation among TERF & Gender Critical communities on Twitter](https://skymocha.github.io/Twitter_TERF_Grievance_Articulation.pdf)_. [skymocha.github.io](https://skymocha.github.io). November 11, 2022. Retrieved (https://skymocha.github.io/Twitter_TERF_Grievance_Articulation.pdf).

### Packages Used

NumPy, Pandas, re, NLTK, spacy, gensim, sci-kit learn, tweepy

### Structure

```
.
|- JK-LDA-gen.py --> The main model used to test against JK-TEXT.txt 
|- exports/* --> exported LDA models in .CSV
|- consolidate --> consolidates twitter .CSV into a .TXT file with all tweets (no QRTs or RTs) 
|- CleanText.py --> The lemmatization and cleaning process on the text. Separate from model to save time.
|- pull.py --> pulls tweets from individual Twitter accounts and exports them to ./tweets/ as a .CSV
|- proposal_LDA
|    - JK-LDA-gen.py --> Proposal for dissecting JK Rowling's blog post for proposal methodology
|    - sussana-LDA-gen.py --> same as JK-LDA-gen.py with minor edits fro susanna Rustin's articles 
```

### Methods used

Refer to proposal or final paper for methodology