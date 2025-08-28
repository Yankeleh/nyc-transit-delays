# NYC Transit Delay Prediction

## 🎯 Project Overview
Predicting transit delays in NYC using real-time transit data, weather conditions, and historical patterns to help commuters make informed travel decisions.


## Getting Started

To install the dependencies either use Conda

```bash
conda env create -f environment.yml
conda activate transit-env
```

or pip

```bash
pip install -r requirements.txt
```


## Project Tracker:

#### August 8, 2025

**Completed:**

- Initiated Github repo
- Created Conda environment
- Set up and tested relevant APIs (MTA Live Bus Data, Open Weather, Google Maps, NYC Open Data)


**Next steps:**

[ ] Build MTA real-time bus and train position collector

[ ] Create historical delay data scrapper

[ ] Set up weather data collection pipeline

[ ] Implement traffic data ingestion (NYC DOT, Google Maps)

[ ] Create holiday/event data collection

[ ] Build data validation checks

[ ] Initial analysis of delay patterns by time/route/weather

[ ] Create summary statistics and distributions

[ ] Build initial visualizations (delay heatmaps, time series)

[ ] Identify data gaps and anomalies


**Later:**

[ ] Clean data

[ ] Naive baseline and finer-tuned models

[ ] Model validation

[ ] Visualization

**Maybe?**

[ ] Live web application?
