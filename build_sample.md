```python
import fnmatch
import glob
import os
import re
from time import sleep
from zipfile import ZipFile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from near_regex import NEAR_regex  # copy this file into the folder this script is in
from tqdm import tqdm  # progress bar on loops

os.makedirs("output", exist_ok=True)
```

## Load Sentiment Dictionaries


```python
# Q1 - Loading ML negative words
with open ('inputs/ML_negative_unigram.txt', 'r') as file:
    words = [line.strip() for line in file]
    print(len(words))
    BHR_negative = list(words)
    
BHR_negative = ['('+'|'.join(BHR_negative)+')']
    
    
# Q2 - Loading ML positive words
with open ('inputs/ML_positive_unigram.txt') as file:
    words = [line.strip() for line in file]
    print(len(words))
    BHR_positive = list(words)
    
BHR_positive = ['('+'|'.join(BHR_positive)+')']
print(BHR_positive)
print(len(BHR_positive))

# Q3 - Loading LM negative words
LM = pd.read_csv('inputs/LM_MasterDictionary_1993-2021.csv')
LM_negative = LM.query('Negative > 0')['Word'].to_list()

LM_negative = ['('+'|'.join(LM_negative)+')']
LM_negative = [word.lower() for word in LM_negative]


# Q4 - Loading LM positive words
LM = pd.read_csv('inputs/LM_MasterDictionary_1993-2021.csv')
LM_positive = LM.query('Positive > 0')['Word'].to_list()

LM_positive = ['('+'|'.join(LM_positive)+')']
```

    94
    75
    ['(strong|strength|great|improvement|nice|improved|momentum|congratulations|pleased|helped|impressive|exceeded|record|congrats|good|leverage|raising|sustainable|really|job|benefited|continue|outperformance|increased|excellent|growth|increase|driving|helping|drove|grew|performance|pretty|above|margin|better|curious|across|continued|results|up|increasing|share|outstanding|improvements|operating|success|expansion|income|over|benefiting|lot|terrific|growing|favorable|generated|proud|repurchase|exceeding|solid|benefit|nicely|basis|flow|gains|well|achieved|upside|improving|cash|years|continues|delivered|think|fantastic)']
    1


## Stock Returns


```python
sp500 = pd.read_csv('inputs/sp500_2022.csv')   
sp500= sp500

url = 'https://github.com/LeDataSciFi/data/raw/main/Stock%20Returns%20(CRSP)/crsp_2022_only.zip'

from urllib.request import urlopen
from io import BytesIO

with urlopen(url) as request:
    data = BytesIO(request.read())

with ZipFile(data) as archive:
    with archive.open(archive.namelist()[0]) as stata:
        sp_returns = pd.read_stata(stata)
```


```python
sp_returns = sp_returns.rename(columns = {'ticker': 'Symbol'})
```

## Opening a 10-K File


```python
# open the zip file (do this before the for loop
# so you only open it one time... faster)
with ZipFile('10k_files/10k_files.zip','r') as zipfolder:
   
    # before the loop, get list of files in zipped folder
    file_list = zipfolder.namelist()
       
    # replace this with how you'd loop over the dataframe
    # which you already know...
    for index, row in tqdm(sp500.iterrows()):

        # MY ADDITION: grab the CIK to create the file path
        firm = row['CIK']

        # get a list of possible files for this firm
        firm_folder    = f"sec-edgar-filings/{str(firm).zfill(10)}/10-K/*/*.html"
        possible_files = fnmatch.filter(file_list, firm_folder)
        if len(possible_files) == 0:
            continue
           
        fpath = possible_files[0] # the first match is the path to the file
       

        # open the file (this is a little different!)
        with zipfolder.open(fpath) as report_file:
           
            html = report_file.read().decode(encoding="utf-8")
           
           
        # do more stuff here...
            soup = BeautifulSoup(html,features='lxml-xml')

        for div in soup.find_all("div", {'style':'display:none'}):
            div.decompose()
   
        lower = soup.get_text().lower()    
        no_punc = re.sub(r'\W',' ',lower)    
        document = re.sub(r'\s+',' ',no_punc)
       
           
        BHR_pos = (len(re.findall(NEAR_regex(BHR_positive),document)) / len(document.split()))
        BHR_neg = (len(re.findall(NEAR_regex(BHR_negative),document)) / len(document.split()))
           
        sp500.loc[index, 'BHR_positive'] = BHR_pos
        sp500.loc[index, 'BHR_negative'] = BHR_neg
```

    503it [16:44,  2.00s/it]



```python
# open the zip file (do this before the for loop
# so you only open it one time... faster)
with ZipFile('10k_files/10k_files.zip','r') as zipfolder:
    
    # before the loop, get list of files in zipped folder
    file_list = zipfolder.namelist()
        
    # replace this with how you'd loop over the dataframe
    # which you already know...
    for index, row in tqdm(sp500.iterrows()): 
        
        # MY ADDITION: grab the CIK to create the file path
        firm = row['CIK']
        
        # get a list of possible files for this firm
        firm_folder    = f"sec-edgar-filings/{str(firm).zfill(10)}/10-K/*/*.html"
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

        # open the file (this is a little different!)
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
            
        # do more stuff here...
            soup = BeautifulSoup(html,features='lxml-xml')

        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()
    
        lower = soup.get_text().lower()    
        no_punc = re.sub(r'\W',' ',lower)    
        document = re.sub(r'\s+',' ',no_punc)
            
        LM_pos = (len(re.findall(NEAR_regex(LM_positive),document)) / len(document.split()))
        LM_neg = (len(re.findall(NEAR_regex(LM_negative),document)) / len(document.split()))
            
        sp500.loc[index, 'LM_positive'] = LM_pos
        sp500.loc[index, 'LM_negative'] = LM_neg
```

    503it [38:43,  4.62s/it]


## Getting 10K Dates and Returns


```python
# pip install requests_html
```


```python
with ZipFile('10k_files/10k_files.zip','r') as zipfolder:
    
    
    file_list = zipfolder.namelist()
    
    for index, row in tqdm(sp500.iterrows()):
       
        firm = row['CIK']
        
        firm_folder    = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] 

       
        with zipfolder.open(fpath) as report_file:
            from requests_html import HTMLSession
            session = HTMLSession()
            cik = row['CIK']
            accession_number = fpath.split('/')[-2]
            url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}-index.html'
            r = session.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            filing_date = soup.find('div',text='Filing Date').find_next_sibling('div').text.strip()
        sp500.loc[index, 'filing_date'] = filing_date
```

    503it [00:00, 664.60it/s]



```python
sp500['T-T2'] = None

for index, row in tqdm(sp500.iterrows()):
    try:
        filing_index = sp_returns.loc[(sp_returns['Symbol'] == row['Symbol']) & (sp_returns['date'] == row['filing_date'])].index[0]
        returns = sp_returns.loc[filing_index:filing_index+2, 'ret']
        returns = returns +1 
        returns = returns.cumprod()
        returns = returns.to_list()
        cumret = returns[2]
        cumret = cumret -1
        sp500.loc[index, 'T-T2'] = cumret
    except IndexError:
        pass
```

    503it [01:44,  4.81it/s]



```python
sp500['T3-T10'] = None

for index, row in tqdm(sp500.iterrows()):
    try:
        filing_index = sp_returns.loc[(sp_returns['Symbol'] == row['Symbol']) & (sp_returns['date'] == row['filing_date'])].index[0]
        returns = sp_returns.loc[filing_index+3:filing_index+10, 'ret']
        returns = returns +1 
        returns = returns.cumprod()
        returns = returns.to_list()
        cumret = returns[7]
        cumret = cumret -1
        sp500.loc[index, 'T3-T10'] = cumret
    except IndexError:
        pass
   
```

    503it [01:38,  5.12it/s]



```python
ccm_cleaned = pd.read_stata('https://github.com/LeDataSciFi/data/raw/main/Firm%20Year%20Datasets%20(Compustat)/2021_ccm_cleaned.dta')
```


```python
ccm_cleaned= ccm_cleaned.rename(columns = {'tic': 'Symbol'})
sp500 = pd.merge(sp500, ccm_cleaned, on = 'Symbol', how = 'left', validate = '1:1')
```

## Choosing 3 Topics and Sentiment Variables


```python
environment_topic = ('environment', 'environmental', 'regulations', 'emissions', 'pollution', 'pollutant', 'pollutants', 'natural resource', 'remediation', 'remediations', 'protection', 'cleanup', 'contamination', 'wastewater', 'contaminant', 'contaminants', 'waste', 'wastes')
environment_topic ='('+'|'.join(environment_topic)+')'
```


```python
diversity_topic = ('diversity', 'diverse', 'belonging', 'inclusion', 'inclusive', 'workforce', 'share', 'perspective', 'perspectives', 'talent', 'culture', 'respect', 'unique', 'underrepresented', 'membership', 'engage', 'engaging', 'engaged', 'engagement', 'mission')
diversity_topic ='('+'|'.join(diversity_topic)+')'
```


```python
patent_topic = ('patent', 'patents', 'pending', 'utility', 'design', 'intellectual property', 'contract', 'contracts', 'rights', 'protect', 'protects', 'protecting', 'copying', 'copyright', 'copyrights', 'trademark', 'trademarks', 'filed', 'file', 'expiration', 'license', 'licenses', 'licensing', 'IP')
patent_topic ='('+'|'.join(patent_topic)+')'
```


```python
environment_topic_negative = [BHR_negative, environment_topic]
environment_topic_positive = [BHR_positive, environment_topic]

environment_topic_negative = [str(item) for item in environment_topic_negative]
environment_topic_positive = [str(item) for item in environment_topic_positive]

environment_topic_negative = NEAR_regex(environment_topic_negative, max_words_between=4)
environment_topic_positive = NEAR_regex(environment_topic_positive, max_words_between=4)
```


```python
diversity_topic_negative = [BHR_negative, diversity_topic]
diversity_topic_positive = [BHR_positive, diversity_topic]

diversity_topic_negative = [str(item) for item in diversity_topic_negative]
diversity_topic_positive = [str(item) for item in diversity_topic_positive]

diversity_topic_negative = NEAR_regex(diversity_topic_negative, max_words_between=4)
diversity_topic_positive = NEAR_regex(diversity_topic_positive, max_words_between=4)
```


```python
patent_topic_negative = [BHR_negative, patent_topic]
patent_topic_positive = [BHR_positive, patent_topic]

patent_topic_negative = [str(item) for item in patent_topic_negative]
patent_topic_positive = [str(item) for item in patent_topic_positive]

patent_topic_negative = NEAR_regex(patent_topic_negative, max_words_between=4)
patent_topic_positive = NEAR_regex(patent_topic_positive, max_words_between=4)
```


```python
with ZipFile('10k_files/10k_files.zip','r') as zipfolder:
    
    
    file_list = zipfolder.namelist()
        
    for index, row in tqdm(sp500.iterrows()): 
        
        # MY ADDITION: grab the CIK to create the file path
        firm = row['CIK']
        
        firm_folder    = f"sec-edgar-filings/{str(firm).zfill(10)}/10-K/*/*.html"
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0]

        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
            
            soup = BeautifulSoup(html,features='lxml-xml')

            for div in soup.find_all("div", {'style':'display:none'}): 
                    div.decompose()
    
            lower = soup.get_text().lower()    
            no_punc = re.sub(r'\W',' ',lower)    
            document = re.sub(r'\s+',' ',no_punc)
            
            hits1 = (len(re.findall(environment_topic_negative,document)) / len(document.split()))
            hits2 = (len(re.findall(environment_topic_positive,document)) / len(document.split()))
            
            sp500.loc[index, 'environment_topic_negative'] = hits1
            sp500.loc[index, 'environment_topic_positive'] = hits2
            
            hits3 = (len(re.findall(diversity_topic_negative,document)) / len(document.split()))
            hits4 = (len(re.findall(diversity_topic_positive,document)) / len(document.split()))
            
            sp500.loc[index, 'diversity_topic_negative'] = hits3
            sp500.loc[index, 'diversity_topic_positive'] = hits4
            
            hits5 = (len(re.findall(patent_topic_negative,document)) / len(document.split()))
            hits6 = (len(re.findall(patent_topic_positive,document)) / len(document.split()))
            
            sp500.loc[index, 'patent_topic_negative'] = hits5
            sp500.loc[index, 'patent_topic_positive'] = hits6
```

    503it [20:19,  2.42s/it]


## Putting it all Together


```python
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
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
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 89 columns</p>
</div>




```python
saved_final = 'output/analysis_sample.csv'
if not os.path.exists(saved_final):
    sp500.to_csv('output/analysis_sample.csv', index = False)
```


```python

```
