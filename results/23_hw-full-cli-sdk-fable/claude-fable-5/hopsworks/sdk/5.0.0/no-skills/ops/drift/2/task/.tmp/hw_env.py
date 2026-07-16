import os

# Route Hopsworks traffic through the local proxy: the default NO_PROXY
# includes 10.0.0.0/8 which forces a direct (sandbox-blocked) connection.
for var in ("NO_PROXY", "no_proxy"):
    os.environ[var] = "localhost,127.0.0.1,::1"

import urllib3

urllib3.disable_warnings()
