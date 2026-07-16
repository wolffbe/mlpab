-- Create a temporary table from the CSV file
CREATE OR REPLACE TEMPORARY VIEW training_data_temp
USING CSV
OPTIONS (header = "true", inferSchema = "true", path = "/Volumes/workspace/mlpabb1ccad/training_data_volume/training_data.csv");