import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
pid = 'a5784a7e-d6f5-4711-aa1b-a39e7595f10f'

for _ in range(60):
    p = w.pipelines.get(pid)
    print('pipeline state:', p.state)
    latest = p.latest_updates[0] if p.latest_updates else None
    if latest:
        print('  latest update:', latest.state)
    st = str(p.state)
    if 'IDLE' in st or 'FAILED' in st:
        break
    time.sleep(15)

# show pipeline name and any update detail
print('pipeline name:', p.name)
if p.latest_updates:
    print('updates:', [(u.update_id, str(u.state)) for u in p.latest_updates])
