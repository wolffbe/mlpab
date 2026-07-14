import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()

for exec_id in ["3271", "3270"]:
    base = f"Logs/Python/airq_verify_754fa9/{exec_id}"
    try:
        items = ds.list(base)
    except Exception as e:  # noqa: BLE001
        print(base, "list error:", e)
        continue
    print(base, "->", items if not isinstance(items, list) else [str(i) for i in items])

# fallback: known layout Logs/Python/<job>/<exec_id>/stdout.log
for exec_id in ["3271"]:
    for fname in ["stdout.log", "stderr.log"]:
        p = f"Logs/Python/airq_verify_754fa9/{exec_id}/{fname}"
        try:
            content = ds.read_content(p)
            print(f"===== {p} =====")
            print(content.decode() if isinstance(content, bytes) else content)
        except Exception as e:  # noqa: BLE001
            print(p, "read error:", e)
