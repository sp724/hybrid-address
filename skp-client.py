import polars as pl

import data_structuring
from data_structuring.components.readers.dataframe_reader import DataFrameReader
from data_structuring.pipeline import AddressStructuringPipeline
from data_structuring.components.runners import ResultPostProcessing

addresses = [
"""1234567890
JOHN DOE
42 MAIN ST A 5TH AVE
NEW YORK, 10001 US"""
]

df = pl.DataFrame({"addresses":addresses})
ds = AddressStructuringPipeline()
results = ds.run(DataFrameReader(df,"addresses"), batch_size=1024)


for result in results:
	print(result.i_th_best_match_country(0))
	print(result.i_th_best_match_town(0))

ResultPostProcessing.save_list_as_json(results)