---
title: EDA
---

## Contents
{:.no_toc}
*  
{: toc}

## Murder rates
The analysis that follows is restricted to pooled 2006-2014 data in order to avoid peeking at our test set.
A histogram of murder rates shows importance of right-tail of distribution.
**The model will need to predict cities with much higher murder rates than the median.**

*Figure 1: Murder rates skew large, so the best model will predict wide variation.*
![murders scaling](EDA_files/histogram of murder rate to 2014.png "Murders distribution" )

Fixed "year" effects are probably an important part of a prediction model.
The median and mean number of murders varies by about 25% peak to trough over the 10 years.
For out-of-sample predictions in new years, however, these cross-city effects would make predictive models weaker unless 
(1) national trends can be predicted, or 
(2) national trends are a function of or endogenous to local variables like economic conditions at the city level.

*Figure 1: Total murders vary substantially across years with a clear trend*
![](EDA_files/total_murders_y_fixed.png "Murders trend" )

## Murder rates and metro characteristics

Metro areas with higher populations have more crime (in line with theory), but nevertheless does not explain significant variance and is not a sufficient predictor by itself.

*Figure 3: Murders scale faster than linearly in population*
![murders scaling](EDA_files/scatter reg_murder by pop.png "Murder rates skew large, so the best model will predict wide variation")

Median HH income appears to be a strong predictor of murder rates, but the relationship may not be linear.
The variable's predictive power may be strongest in conjunction with other variables.

*Figure 4:  Income and poverty are also clear predictors of murder rates*



