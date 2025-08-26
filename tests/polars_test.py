import polars as pl
from datetime import datetime

buses = [{'timestamp':datetime.now(), 'route_id':'B6', 'delay':100},{'timestamp':datetime.now(), 'route_id':'B11', 'delay':800}]

df = pl.DataFrame(buses)

print(df)