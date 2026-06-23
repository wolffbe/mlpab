import hopsworks
proj = hopsworks.login()
fs = proj.get_feature_store()
t = proj.get_trino_api()
conn = t.connect(verify=False)
cur = conn.cursor()
cur.execute("SHOW CATALOGS")
print('CATALOGS:', [r[0] for r in cur.fetchall()])
