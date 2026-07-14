import hw_env  # noqa: F401
import requests

for url in (
    "https://***REDACTED***/hopsworks-api/api/variables/versions",
    "https://10.99.99.99/",
):
    try:
        r = requests.get(url, verify=False, timeout=20)
        print(url, "->", r.status_code, r.headers.get("server"), r.text[:80].replace("\n", " "))
    except Exception as e:
        print(url, "-> EXC", type(e).__name__, str(e)[:200])
