---
title: EDA
notebook: EDA.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}








```python

df_crime = pd.read_csv("../Processed Data/crimeAndCensus.csv")
```




```python
missing = df_crime.agg(func = lambda x: np.mean(pd.isnull(x)))
```




```python
# incorporate year in missing analysis

missing.hist(bins=50)

```




```python
np.sum(missing < .20)
```




```python
missing[ missing < .20]
```




```python

df_crime['murder_pc'] = df_crime['murder_fbi'] / df_crime['population_fbi'] * 100000
df_crime['ln_pop'] = np.log(df_crime['population_fbi'])

df_crime['murder_fbi_smallcity'] = df_crime['murder_fbi'].where(df_crime['population_fbi'] < 100000, df_crime['murder_fbi']) 


df_crime.head(10)



```




```python
s_years = df_crime["year"]
s_years = s_years.value_counts()
s_years = s_years.sort_index(ascending=False)
count = s_years.plot(kind="barh")
count.set(xlabel="count of metro areas")
plt.savefig('../EDA/obsr_per_year')
```




```python
df_crime = df_crime[ df_crime.year<2015]
year_data = df_crime[['year','murder_fbi']]

year_summary = year_data.groupby('year')

#year_summary.head()

year_totals = year_summary.sum()
year_totals['year'] = year_totals.index

print(year_totals.shape)
year_totals


```




```python
plot_total = year_totals.plot(x="year", y ="murder_fbi", legend=False, ylim=(0,15000) )
plot_total.set(xlabel="Year",ylabel="Total murders")
plt.savefig('../EDA/total_murders_y_fixed')
```




```python
plot_pop = df_crime.plot(x="ln_pop", y="murder_pc", kind="scatter",alpha=0.5)
plot_pop.set(xlabel="Log of population",ylabel="Murder per 1000")
#sns.lmplot("ln_pop", "murder_pc", data=df_crime, fit_reg=True)
plt.savefig('../EDA/population_murder_pc')
```




```python
d_name_change = {
    "S01_HC01_EST_VC27":"pop_16over",
    "S01_HC01_EST_VC01":"ln_pop_total_census",
    "S06_HC01_EST_VC63":"income_median",
    "S05_HC01_EST_VC143":"poverty_rate",
    "S23_HC04_EST_VC44":"unemployment_16over",
   "S01_HC01_EST_VC36":"demog_sex_ratio",
      "S01_HC01_EST_VC39":"demog_child_dep_ratio",
     "S05_HC01_EST_VC21":"demog_black",
    "S05_HC01_EST_VC22":"demog_amerindian",
    "S05_HC01_EST_VC54":"education_over25_lessthanhs",
    "S05_HC01_EST_VC53":"pop_over25"
    
}

x_of_interest = list(d_name_change.values())

df_crime = df_crime.rename(columns=d_name_change)


```




```python
df_pair_plot = df_crime[x_of_interest + ["murder_pc"] ]
df_pair_plot["pop_16over"] = pd.to_numeric(df_pair_plot["pop_16over"],errors='coerce') 
df_pair_plot["unemployment_16over"] = pd.to_numeric(df_pair_plot["unemployment_16over"],errors='coerce') 

df_pair_plot.apply(lambda x: sum(np.isnan(x)))
```




```python
for s_x in x_of_interest:
    df_local = df_pair_plot[ [s_x,"murder_pc"]]
    df_local= df_local.dropna()
    sns.lmplot(s_x, "murder_pc", data=df_local, fit_reg=True ) 
    plt.savefig('../EDA/bivariage_'+s_x)
    plt.show()
```


** Total murders by year **



```python


murder_pc = df_crime.murder_pc

f, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.despine(left=True)

sns.distplot(murder_pc, kde=False, bins=18, color="m", ax=ax)
ax.set_title("Distribution of Murder Rate")
ax.set_xlabel("Murder Rate (per 100K)")
f.savefig('../EDA/histogram of murder rate to 2014')
```


** Population vs. Murders **

Scatterplot (2016)

### Census Data Analysis



```python
sns.set(style="white", color_codes=True)

g = sns.lmplot("ln_pop_total", "murder_pc", data=df_eda, fit_reg=True)
#g.set_titles("test")
g.savefig('../EDA/scatter reg_murder by pop_census')
```


** Histogram **



```python
print(df_eda.shape)
df_eda_nomissing = df_eda.dropna(subset=['murder_pc'])


f, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.despine(left=True)

sns.distplot(df_eda_nomissing['murder_pc'], kde=False, bins=18, color="m", ax=ax)
ax.set_title("Distribution of Murder Rates (2016)")
ax.set_xlabel("Murder Rate (per 100K)")
f.savefig('../EDA/histogram of murder rate')
```




```python
g = sns.distplot(df_eda_nomissing['murder_pc'], kde=False, bins=18, color="m", ax=ax)
#ax.set_title("Distribution of Murder Rates (2016)")
#ax.set_xlabel("Murder Rate (per 100K)")
#g.savefig('../EDA/histogram of murder rate')
```


** Income **



```python
fig, ax = plt.subplots(1,1, figsize=(12,6))

ax = plt.scatter(df_eda['income_hh_median'], df_eda['murder_pc'], alpha=0.75)
plt.title('2016 murder rate by median HH income')
plt.xlabel('Median HH Income')
plt.ylabel('Murders per 100K')
plt.savefig('../EDA/scatter_murder by HH income')
```




```python
sns.set(style="white", color_codes=True)

g = sns.lmplot("income_hh_median", "murder_pc", data=df_eda, fit_reg=True)
#g.set_titles("test")
g.savefig('../EDA/scatter reg_murder by HH income')
```




```python
fig, ax = plt.subplots(1,1, figsize=(12,6))

ax = plt.scatter(df_eda['poverty_rate_18over'], df_eda['murder_pc'], alpha=0.75)
plt.title('2016 murder rate by 18+ poverty rate')
plt.xlabel('Poverty Rate (18+)')
plt.ylabel('Murders per 100K')
plt.savefig('../EDA/scatter_murder by poverty rate')
```




```python
sns.set(style="white", color_codes=True)

g = sns.lmplot("poverty_rate_18over", "murder_pc", data=df_eda, fit_reg=True)
#g.set_titles("test")
g.savefig('../EDA/scatter reg_murder by poverty rate')
```




```python


```


** Other Demographics **



```python
sns.set(style="white", color_codes=True)

g = sns.lmplot("demog_black", "murder_pc", data=df_eda, fit_reg=True)
#g.set_titles("test")
g.savefig('../EDA/scatter reg_murder by race=black')
```




```python
fig, ax = plt.subplots(1,1, figsize=(12,6))

ax = plt.scatter(df_eda['demog_black'], df_eda['murder_pc'], alpha=0.75)
plt.title('2016 murder rate by % black population')
plt.xlabel('African American Population (%)')
plt.ylabel('Murders per 100K')
plt.savefig('../EDA/scatter_murder by race=black')
```




```python
# American Indian

sns.set(style="white", color_codes=True)

g = sns.lmplot("demog_amerindian", "murder_pc", data=df_eda, fit_reg=True)
#g.set_titles("test")
g.savefig('../EDA/scatter reg_murder by race=amerindian')
```




```python
# Less than HS Education (limited data availability)

sns.set(style="white", color_codes=True)

g = sns.lmplot("education_over25_lessthanhs", "murder_pc", data=df_eda, fit_reg=True)
#g.set_titles("test")
g.savefig('../EDA/scatter reg_murder by % over 25 lacking HS diploma')
```




```python
fig, ax = plt.subplots(1,1, figsize=(12,6))

ax = plt.scatter(df_eda['demog_black'], df_eda['murder_pc'], alpha=0.75)
plt.title('2016 murder rate by % lacking HS diploma')
plt.xlabel('% >25yrs without HS diploma')
plt.ylabel('Murders per 100K')
plt.savefig('scatter_murder by % over 25 lacking HS diploma')
```




```python
# only variables with non-missing data (2016)

y = df_eda['murder_pc']
X = df_eda[['ln_pop_total', 'poverty_rate_18over']]

X = sm.add_constant(X)

# Create OLS class instance
est = sm.OLS(y, X, missing='drop')

# Use the fit method in the instance for fitting a linear regression model
results_temp = est.fit()

print(results_temp.summary())
```




```python
# Including variables with significant missing data (2016)

y = df_eda['murder_pc']
X = df_eda[['ln_pop_total', 'income_hh_median', 'poverty_rate_18over', 'demog_black', 'demog_amerindian', 'education_over25_lessthanhs']]

X = sm.add_constant(X)

# Create OLS class instance
est = sm.OLS(y, X, missing='drop')

# Use the fit method in the instance for fitting a linear regression model
results_temp = est.fit()

print(results_temp.summary())
```




```python
# Data available for 2006

y = df_eda['murder_pc']
X = df_eda[['ln_pop_total', 'pop_16over', 'demog_sex_ratio', 'demog_child_dep_ratio']]

X = sm.add_constant(X)

# Create OLS class instance
est = sm.OLS(y, X, missing='drop')

# Use the fit method in the instance for fitting a linear regression model
results_temp = est.fit()

print(results_temp.summary())
```




```python

# census_pop
Total; Estimate; Total households
Total; Estimate; Total population

Total; Estimate; Total population - SELECTED AGE CATEGORIES - 16 years and over
Total; Estimate; Total population - SUMMARY INDICATORS - Sex ratio (males per 100 females)

Total; Estimate; Total population - SUMMARY INDICATORS - Child dependency ratio


# income
Total; Estimate; EARNINGS IN THE PAST 12 MONTHS (IN 2006 INFLATION-ADJUSTED DOLLARS) FOR FULL-TIME, YEAR-ROUND WORKERS - Population 16 years and over with earnings - Median earnings (dollars) for full-time, year-round workers: - Female
Total; Estimate; EARNINGS IN THE PAST 12 MONTHS (IN 2006 INFLATION-ADJUSTED DOLLARS) FOR FULL-TIME, YEAR-ROUND WORKERS - Population 16 years and over with earnings - Median earnings (dollars) for full-time, year-round workers: - Male
Total; Estimate; Median Household income (dollars)
Total; Estimate; Median earnings (dollars) for full-time, year-round workers: - Female
Total; Estimate; Median earnings (dollars) for full-time, year-round workers: - Male

Percent; Estimate; POVERTY STATUS IN THE PAST 12 MONTHS - Civilian population 18 years and over for whom poverty status is determined - Income in the past 12 months below poverty level

# unemployment
Unemployment rate; Estimate; PERCENT IMPUTED - Employment status for population 16 years and over
Total; Estimate; EMPLOYMENT STATUS - Population 16 years and over - In labor force - Civilian labor force - Unemployed - Percent of civilian labor force



# demographics 
Total; Estimate; Median number of rooms

Total; Estimate; EDUCATIONAL ATTAINMENT - Population 25 years and over - Less than high school graduate
Total; Estimate; EDUCATIONAL ATTAINMENT - Population 25 years and over

Total; Estimate; RACE AND HISPANIC OR LATINO ORIGIN - One race - American Indian and Alaska Native
Total; Estimate; RACE AND HISPANIC OR LATINO ORIGIN - One race - Black or African American




```

