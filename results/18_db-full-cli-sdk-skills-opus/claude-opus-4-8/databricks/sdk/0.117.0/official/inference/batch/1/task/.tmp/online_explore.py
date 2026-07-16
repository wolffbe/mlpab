import databricks.sdk.service.catalog as c
import inspect
print("OnlineTable:", [x for x in dir(c) if "Online" in x])
print(inspect.signature(c.OnlineTableSpec.__init__))
print("---triggered---")
print([x for x in dir(c) if "Triggered" in x or "Continuous" in x or "Scheduling" in x])
