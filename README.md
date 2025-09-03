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

#### September 2, 2025

- Decided to begin with historical data, monthly aggregates from 2022 until 2024.
- MTA data is accessible [via NY's data site](https://data.ny.gov/) and weather data from [NOAA](https://www.ncdc.noaa.gov/cdo-web/search).
- Initial analysis is in the 20250902 .ipynb file.
- There is an issue with the weather data - max temperature is only recorded at 48 degrees Farenheit.

**Next steps**

- Review code, especially historical_weather_collector.py
- Rerun analysis.
- Clean up code as suggested in the issues page.


#### August 8, 2025

**Completed:**

- Initiated Github repo
- Created Conda environment
- Set up and tested relevant APIs (MTA Live Bus Data, Open Weather, Google Maps, NYC Open Data)
