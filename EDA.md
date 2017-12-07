---
title: EDA
---

## Contents
{:.no_toc}
*  
{: toc}

Fixed "year" effects are probably an important part of a prediction model.
The median and mean number of murders varies by about 25% peak to trough over the 10 years.
For out-of-sample predictions in new years, however, these cross-city effects would make predictive models weaker unless 
(1) national trends can be predicted, or 
(2) national trends are a function of or endogenous to local variables like economic conditions at the city level.

*Figure 1: Murders scale faster than linearly in population*
![murders scaling](EDA_files/histogram of murder rate to 2014.png "Murder rates skew large, so the best model will predict wide variation" )

