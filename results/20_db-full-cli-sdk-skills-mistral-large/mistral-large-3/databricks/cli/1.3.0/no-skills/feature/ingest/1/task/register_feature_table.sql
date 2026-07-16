-- Register the table as a feature table
CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpabd62957.transactions4adadd
AS SELECT * FROM workspace.mlpabd62957.transactions4adadd;

-- Enable online access for low-latency lookup
CREATE OR REFRESH ONLINE TABLE workspace.mlpabd62957.transactions4adadd
FROM FEATURES workspace.mlpabd62957.transactions4adadd;