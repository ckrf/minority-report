---
title: Introduction and Description of Data 
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}

## Data description
- _Outcome variable: number of murders per MSA-year_ from the FBI. 
- Located tables on separate FBI pages by year
- Programmatically scraped data using BeautifulSoup 
- Data present for all MSA-years except 1 (Flint, 2015)
- For the purpose of EDA, we focus on relating the murder rate to predictors of interest, based on the assumption that murders scale with population even if not exactly linearly.

_Predictors: MSA averages and totals of household traits_ from the American Community Survey
- Used American FactFinder Download Center from USCB to download numerous tables per year
- Downloaded the first table in each set of summary tables (major indicators) 
- Many missing values due to:
- * Some predictors are not available and we have not yet removed them
- * Some predictors are only available for MSAs above a certain population
- * Several MSA-years with no data available
- 5953 predictors, of which at most 4134 in the most complete observation
