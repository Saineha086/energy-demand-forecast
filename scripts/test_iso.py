from gridstatus import NYISO
from datetime import datetime, timedelta

iso = NYISO()
end = datetime.utcnow()
start = end - timedelta(days=1)

df = iso.get_load(start=start, end=end)
print(df.head())
print(df.tail())
