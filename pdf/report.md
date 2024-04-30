---
 layout: wide_default
 ---    
## Summary

The goal of this project was to answer the question of whether 10-K filings contain value-relevant information in the sentiment of the text. Furthermore, it was also about if the sentiment in a 10k statement impacted stocks returns - either in a positive or negative way. This was a cross-sectional event study. After looking at the results, I realized that many of the results were inconclusive due to many contradictions. Some variables were positively affected by positive sentiment, but some were negatively affected. This made analysis and conclusions unpredictable, especially given the time frame of the reports.

The report is focused on 2022 stock returns and 500 10K files based on the S&P500. The sentiment analysis was done using an LM and ML sentiment. The LM was a sentinment dictionary, while the ML was a list generated via a machine learning approach. In this project, 10 sentiment variables were added to the data for analysis. 4 used the sentiment dictionary, but 6 used word lists that I created for 3 different topics. I picked 3 topics to analyze sentimentality in the 10K. These topics focused words surrounding environment, diversity, and patent. The three topics were very different in order to get a larger net in analysis. 

## Data

The data sample contained 503 firms that were part of the S&P 500 index during the 2022 fiscal year.

The data also consisted of two different return variables. One is the cumulative returns of the stock from the day of filing to two days later, and the other is 3 days after the filing date to 10 days after. To create the return variables, two for loops had to be created, one to get the filing dates, and another to pull out the correct returns for each return variable. Starting with the first, for loop, I had to first find the filing date of each 10k that was downloaded from SEC-Edgar. Each firm's specific webpage could be accessed only by using a specific CIK and accession number. This was the code used:

` cik = row['CIK']
  accession_number = fpath.split('/')[-2] `

Once the filing dates were added another loop needed to be done in order to pull out the correct returns for each return variable. The returns were located in a different dataframe called sp_returns which had the returns for all stocks in the sp500. When creating the second for loop to create the variables on the dataset of sp500. It started with this specific line of code:

` for firm, row in tqdm(sp500.iterrows()): `

Within the for loop, it had to look through the sp_returns database and locate the row where the firm and date in the loop were the same as the sp_returns database. The returns were then cumulated using these codes: 

`returns = returns.cumprod()
        returns = returns.to_list()
        cumret = returns[2]
        cumret = cumret - 1 `
        
Finally, the variable was then created and added to a column in the data frame called 'T-T2' using this code:

` sp500.loc[index, 'T-T2'] = cumret `

This was also done a second time on another for loop except slightly adjusted in order to get day 3 to 10 rather than 0 to 2. When creating the variable, this was the code:

` sp500.loc[index, 'T3-T10'] = cumret `



The data also contained 10 sentiment variables that were split up into two categories, like I mentioned before. There were 4 that were taken straight from the provided dictionaries, and 6 that were created using custom word lists based upon the 3 selected topics. 

To create the first 4 variables, the first step was to split the dictionaries into positive and negative categories. This was done using columns in their dataframes called positive and negative. These values were used to load a positive and negative dictionary of each database. In order to do this the near_regex function was provided. This function essentially took an input and looked at how many times it was found in a specific document. For this project, the near_regex function would look at the 10Ks. After the dictionaries were loaded into their categories, I had to put them in the proper format using this code:

` BHR_negative = ['('+'|'.join(BHR_negative)+')'] ` 

This was done 4 times for loading ML negative words, ML positive words, LM negative words, and LM positive words. Within the ML_negative dictionary there were 94 words and within the ML positive dictionary there were 75 words. After loading all 4 dictionaries, I created a for loop that looped through each firm, opened up the 10k that was downloaded, and pulled out just the text of that document. This was done using this code: 

` with zipfolder.open(fpath) as report_file:
           
            html = report_file.read().decode(encoding="utf-8")
           
           
        # do more stuff here...
            soup = BeautifulSoup(html,features='lxml-xml')

        for div in soup.find_all("div", {'style':'display:none'}):
            div.decompose() `

I also had to clean the htmls, which was done with this code:

`lower = soup.get_text().lower()    
        no_punc = re.sub(r'\W',' ',lower)    
        document = re.sub(r'\s+',' ',no_punc) `
        
Finally, I had to get the ratio and create the column in the dataframe, so I used this code:

` BHR_pos = (len(re.findall(NEAR_regex(BHR_positive),document)) / len(document.split()))
  BHR_neg = (len(re.findall(NEAR_regex(BHR_negative),document)) / len(document.split()))
  
  sp500.loc[index, 'BHR_positive'] = BHR_pos
  sp500.loc[index, 'BHR_negative'] = BHR_neg `
  
Next, to create the next 6 variables, I generated custom lists based on the three topics: environment, diversity, and patent. I chose these three topics as they were all pretty broad, talked about in probably every 10K, and all very different from one another. I skimmed through a few 10Ks and found 10-20 words the correlated with the topics. These variables were meant to analyze specific topics in a 10k in order to get a good idea of a company's position in the three topics. Once the word lists were completed they had to be cleaned in the same way that the sentiment dictionaries were previously. Once that step was completed the near_regex function was needed once again in order to complete the task.

## Data Summary Stats


```python
# import pandas as pd
sp500 =pd.read_csv('output/analysis_sample.csv')
sp500
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Security</th>
      <th>SEC filings</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date first added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>BHR_positive</th>
      <th>...</th>
      <th>ppe_a</th>
      <th>cash_a</th>
      <th>xrd_a</th>
      <th>dltt_a</th>
      <th>invopps_FG09</th>
      <th>sales_g</th>
      <th>dv_a</th>
      <th>short_debt</th>
      <th>T-T2</th>
      <th>T3-T10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1976-08-09</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>...</td>
      <td>0.218538</td>
      <td>0.101228</td>
      <td>0.042361</td>
      <td>0.355625</td>
      <td>2.564301</td>
      <td>0.098527</td>
      <td>0.072655</td>
      <td>0.086095</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>reports</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0.024460</td>
      <td>...</td>
      <td>0.183974</td>
      <td>0.181729</td>
      <td>0.027113</td>
      <td>0.061075</td>
      <td>NaN</td>
      <td>0.222291</td>
      <td>0.048958</td>
      <td>0.080191</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1964-03-31</td>
      <td>1800</td>
      <td>1888</td>
      <td>0.021590</td>
      <td>...</td>
      <td>0.134475</td>
      <td>0.136297</td>
      <td>0.036465</td>
      <td>0.242726</td>
      <td>3.559664</td>
      <td>0.244654</td>
      <td>0.042582</td>
      <td>0.051893</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0.019753</td>
      <td>...</td>
      <td>0.040074</td>
      <td>0.067086</td>
      <td>0.054911</td>
      <td>0.442929</td>
      <td>2.144449</td>
      <td>0.227438</td>
      <td>0.063203</td>
      <td>0.163364</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0.027968</td>
      <td>...</td>
      <td>0.111674</td>
      <td>0.189283</td>
      <td>0.025902</td>
      <td>0.063702</td>
      <td>5.023477</td>
      <td>0.140013</td>
      <td>0.051790</td>
      <td>0.215661</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>reports</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>0.026854</td>
      <td>...</td>
      <td>0.337915</td>
      <td>0.123366</td>
      <td>0.000000</td>
      <td>1.019505</td>
      <td>8.944086</td>
      <td>0.164897</td>
      <td>0.099229</td>
      <td>0.012864</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>reports</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>0.028396</td>
      <td>...</td>
      <td>0.064843</td>
      <td>0.055350</td>
      <td>0.091231</td>
      <td>0.167820</td>
      <td>5.301699</td>
      <td>0.265063</td>
      <td>0.000000</td>
      <td>0.089083</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>0.021506</td>
      <td>...</td>
      <td>0.097530</td>
      <td>0.020400</td>
      <td>0.021892</td>
      <td>0.242318</td>
      <td>1.415104</td>
      <td>0.115553</td>
      <td>0.008531</td>
      <td>0.227553</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>reports</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
      <td>0.019965</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>reports</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>0.021790</td>
      <td>...</td>
      <td>0.187266</td>
      <td>0.250719</td>
      <td>0.036547</td>
      <td>0.485108</td>
      <td>8.792744</td>
      <td>0.164349</td>
      <td>0.034101</td>
      <td>0.006044</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 89 columns</p>
</div>




```python
sp500.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CIK</th>
      <th>BHR_positive</th>
      <th>BHR_negative</th>
      <th>LM_positive</th>
      <th>LM_negative</th>
      <th>environment_topic_negative</th>
      <th>environment_topic_positive</th>
      <th>diversity_topic_negative</th>
      <th>diversity_topic_positive</th>
      <th>patent_topic_negative</th>
      <th>patent_topic_positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.030000e+02</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
      <td>501.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.919979e+05</td>
      <td>0.023948</td>
      <td>0.025892</td>
      <td>0.004986</td>
      <td>0.015906</td>
      <td>0.000226</td>
      <td>0.000226</td>
      <td>0.000408</td>
      <td>0.000408</td>
      <td>0.001029</td>
      <td>0.001030</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.522469e+05</td>
      <td>0.003489</td>
      <td>0.003392</td>
      <td>0.001315</td>
      <td>0.003691</td>
      <td>0.000133</td>
      <td>0.000133</td>
      <td>0.000178</td>
      <td>0.000179</td>
      <td>0.000803</td>
      <td>0.000806</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.800000e+03</td>
      <td>0.007966</td>
      <td>0.008953</td>
      <td>0.001226</td>
      <td>0.006609</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000079</td>
      <td>0.000079</td>
      <td>0.000088</td>
      <td>0.000088</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.761050e+04</td>
      <td>0.021971</td>
      <td>0.023964</td>
      <td>0.004095</td>
      <td>0.013296</td>
      <td>0.000140</td>
      <td>0.000140</td>
      <td>0.000290</td>
      <td>0.000290</td>
      <td>0.000595</td>
      <td>0.000599</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.848870e+05</td>
      <td>0.024117</td>
      <td>0.025899</td>
      <td>0.004895</td>
      <td>0.015646</td>
      <td>0.000202</td>
      <td>0.000202</td>
      <td>0.000384</td>
      <td>0.000384</td>
      <td>0.000891</td>
      <td>0.000895</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.137782e+06</td>
      <td>0.026129</td>
      <td>0.027808</td>
      <td>0.005656</td>
      <td>0.017859</td>
      <td>0.000285</td>
      <td>0.000285</td>
      <td>0.000510</td>
      <td>0.000512</td>
      <td>0.001270</td>
      <td>0.001270</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.868275e+06</td>
      <td>0.037982</td>
      <td>0.038030</td>
      <td>0.010899</td>
      <td>0.030185</td>
      <td>0.001205</td>
      <td>0.001205</td>
      <td>0.001636</td>
      <td>0.001636</td>
      <td>0.012171</td>
      <td>0.012171</td>
    </tr>
  </tbody>
</table>
</div>



I created a .describe() above and based on this there are a few things to note:
- There are 501 observations in total
- The sentiment and return variables have unique statistics which means that there is variation in the analysis variables

## Results

To make results easier to analyze for the new variables, I am creating a new dataframe called sp500_variables. 


```python
sp500_variables = sp500[['Symbol', 'BHR_positive', 'BHR_negative', 'LM_positive', 'LM_negative', 'environment_topic_negative', 'environment_topic_positive', 'diversity_topic_negative', 'diversity_topic_positive', 'patent_topic_negative', 'patent_topic_positive', 'filing_date', 'T-T2', 'T3-T10' ]]
sp500_variables = sp500_variables.dropna()
```


```python
corr_matrix = sp500_variables.corr()
# import seaborn as sns
sns.heatmap(corr_matrix.loc[['BHR_positive', 'BHR_negative', 'LM_positive', 'LM_negative', 'environment_topic_negative', 'environment_topic_positive', 'diversity_topic_negative', 'diversity_topic_positive', 'patent_topic_negative', 'patent_topic_positive'], ['T-T2','T3-T10']], annot=True, cmap='coolwarm')
```

    /Users/matt/opt/anaconda3/lib/python3.9/site-packages/seaborn/matrix.py:198: RuntimeWarning: All-NaN slice encountered
      vmin = np.nanmin(calc_data)
    /Users/matt/opt/anaconda3/lib/python3.9/site-packages/seaborn/matrix.py:203: RuntimeWarning: All-NaN slice encountered
      vmax = np.nanmax(calc_data)





    <AxesSubplot:>




    
![png](output_13_2.png)
    


For some reason, I have nothing within my return variables of T-T2 and T3-T10, even though my code worked and generated returns in the build_sample. It is difficult to find a concrete conclusion from the data I had. Using the data discussed with my peers though, it was apparent though that many of the positive sentiment variables led to negative returns and vice versa. I think this had something to do with the LM and BHR dictionaries having contradicting results based on what you would expect. It is possible that investors may see a 10k be negative and decide that it is a good time to buy low on the stock, or they might think the opposite. In general, there is not enough data in this report to back up that idea.


```python
variables = sp500_variables[['BHR_positive', 'BHR_negative', 'LM_positive', 'LM_negative', 'environment_topic_negative', 'environment_topic_positive', 'diversity_topic_negative', 'diversity_topic_positive', 'patent_topic_negative', 'patent_topic_positive']]
y_variables = sp500_variables[['T-T2','T3-T10']]
sns.pairplot(sp500_variables, x_vars=variables, y_vars=y_variables)
```




    <seaborn.axisgrid.PairGrid at 0x7f8c2ab1d790>




    
![png](output_15_1.png)
    


## Discussion Topics

(1) When comparing the LM-Returns correlation to the ML-returns correlation, I couldn't unfortunately do it based off my data due to having a problem getting the return variables. However, based on my friends' results, I can still make conclusions and that is what I will be basing my discussion and analysis on. The main takeaway I drew would be that the ML(BHR) sentiment variables both have a positive correlation with the first return variable, whereas the LM variables both have a negative correlation with the return variable.

(2) When comparing results with ML_JFE, the report states that the LM correlations are statistically insignificant, but the t-stat from the ML variables of my friend indicates some correlation between returns and sentiment. It is possible that due to the small size of our 500 firms with a much larger sample probably taken for the report, makes it hard to make a correct assertion. Their study also implemented a lot more control variables and more data which probably led to more precise results.

(3) Due to my missing returns, I can't comment as to the relationships of my sentiment variables with those return variables, based off my data. However, I will say that it is valuable to investigate further. At the end of the day, based off my friends' results and what the report discussed, sentiment analysis in regards to returns does seem to have merit. Even though my report can't support this answer, it's hard to deny the merit of looking at sentiment in regards to returns. This all goes back to the idea of the importance of humans' subjective thinking. People's ability to interpret sentiment is very subjective, so any study that attempts to study and find patterns of subjective thinking on objective data can definitely be valuable. This could give investors a lot more to work with when analyzing the market. 

(4) It is hard to make a concrete answer when it comes to size and magnitude in comparing return variables due to some of the inaccuracies in the results with missing values. However, I can probably make an educated guess that there would be a difference in size and magnitude for the various sentiment measures. Each one is different in size and magnitude as a topic on their own, so their level of positive and negative sentiment values would probably vary.
