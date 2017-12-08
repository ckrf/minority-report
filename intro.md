---
title: Introduction and Description of Data 
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}

###Literature Review

What factors lead to higher murder rates? Or even more simply, what factors are associated with violent crime that might help us predict and forecast murder rates? 

At a micro within-city level, researchers like Green et al. (2017) have been able to model gun violence using social contagion models, and they found that interpersonal relationships could explain 63% of gunshot incidences in Chicago from 2006 to 2014. Nevertheless, their social model was a complement to demographics, which we are able to use more easily in our models of murder across cities over time.

Glaeser and Sacerdote (1999) discuss possible causal mechanisms behind the strong association of higher populations (e.g. in bigger cities versus smaller cities or rural areas) with higher crime rates. They showed in 1999 that population could predict murder per capita with an $R^2$ of0.178 (or 0.123 for crime in general). 

The same authors showed that the other variables are associated positively (poverty rate, unemployment) or negatively (non-white population, attainment of tertiary education, house ownership) with crime arrests. (Not all associations were statistically significant predictors.) In particular, they found that the percentage of female-headed households is strongly and positively associated with higher murder crime rates.

Fox and Piquero (2003) highlight the importance of basic demographic make-up as a predictor of murder. In addition to demographic breakdowns on age, gender, or race being predictive of murder rates, they also found that *changes* in make-up over time could also be predictive.

While many demographic features of a city can be relatively more stable over time, variables like unemployment rate (and changes in unemployment rate) may be more volatile *across* time (in addition to geography), so it could be an interesting variable to use in a prediction model for different years. Using data from before 2001, Raphael and Winter-Ebmer (2001) found that lower unemployment was significantly and strongly negatively associated with lower property crimes, but its relationship with violent crime was not statistically significant. Based on this, we decided to see how useful economic indicators like unemployment would be in our own prediction model.

<u>Citations</u>

*Fox, J.A. and Piquero, A.R.,2003. Deadly demographics: Population characteristics and forecasting homicide trends. NCCD news, 49(3), pp.339-359.*

*Glaeser, E.L. and Sacerdote, B., 1999. Why is there more crime in cities?. Journal of political economy, 107(S6), pp.S225-S258.*

*Green, B., Horel, T. and Papachristos, A.V., 2017. Modeling contagion through social networks to explain and predict gunshot violence in Chicago, 2006 to 2014. JAMA internal medicine, 177(3), pp.326-333.*

*Raphael, S. and Winter-Ebmer, R., 2001. Identifying the effect of unemployment on crime. The Journal of Law and Economics, 44(1), pp.259-283.*

## Data description
_Outcome variable: number of murders per MSA-year_ from the FBI. 
- Located tables on separate FBI pages by year
- Programmatically scraped data using BeautifulSoup 
- Data present for all MSA-years except 1 (Flint, 2015)
- For the purpose of EDA, we focus on relating the murder rate to predictors of interest, based on the assumption that murders scale with population even if not exactly linearly.

_Predictors: MSA averages and totals of household traits_ from the American Community Survey
- Used American FactFinder Download Center from USCB to download numerous tables per year
- Downloaded the first table in each set of summary tables (major indicators) 
- Many missing values due to:
  * Some predictors are not available and we have not yet removed them
  * Some predictors are only available for MSAs above a certain population
  * Several MSA-years with no data available
- 5953 predictors, of which at most 4134 in the most complete observation

_Supplementary (predictors): unemployment data_ from Bureau of Labor Statisticso
- Downloaded as a panel (MSA x month) 
- Calculated annual statistics:
  * maximum level
  * minimum level
  * change over year
- merged with remaining data