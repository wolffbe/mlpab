# Notebook to enable online/low-latency access to predictions table
import json, os
import urllib.request, urllib.error

catalog    = dbutils.widgets.get("catalog")
db_schema  = dbutils.widgets.get("db_schema")
prefix     = dbutils.widgets.get("prefix")
store_name = dbutils.widgets.get("store_name")
host_param = dbutils.widgets.get("host")
token_param = dbutils.widgets.get("token")

cs = f"{catalog}.{db_schema}"
pred_table = f"{cs}.airqpredfdfb59"
online_table_name = f"{prefix}_airqpredfdfb59"

host  = host_param
token = token_param
print(f"Host: {host[:40]}...")
print(f"Source table: {pred_table}")
print(f"Online store: {store_name}")

status = "not_started"

# Approach 1: REST API with URL-encoded table name (encode dots as %2E)
try:
    table_encoded = pred_table.replace('.', '%2E')
    url = f"https://{host}/api/2.0/feature-store/tables/{table_encoded}/publish"
    print(f"Trying REST: {url[:80]}")

    payload = json.dumps({
        'publish_spec': {
            'online_store': store_name,
            'online_table_name': online_table_name,
            'publish_mode': 'SNAPSHOT'
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode()
        print(f"REST success: {body[:200]}")
        status = "published_via_rest"
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"REST error {e.code}: {body[:300]}")
except Exception as e:
    print(f"REST exception: {type(e).__name__}: {e}")

# Approach 2: databricks.sdk from within Databricks environment
if "published" not in status:
    try:
        from databricks.sdk import WorkspaceClient
        w2 = WorkspaceClient()
        # Try feature_store publish_table with workspace catalog workaround
        from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
        result_pub = w2.feature_store.publish_table(
            source_table_name=pred_table,
            publish_spec=PublishSpec(
                online_store=store_name,
                online_table_name=online_table_name,
                publish_mode=PublishSpecPublishMode.SNAPSHOT
            )
        )
        print(f"SDK publish success: {result_pub}")
        status = "published_via_sdk"
    except Exception as e:
        print(f"SDK publish error: {type(e).__name__}: {str(e)[:300]}")

print(f"Final status: {status}")
result = {"status": status, "pred_table": pred_table, "online_store": store_name}
print(json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
