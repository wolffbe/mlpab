import os,json,math
from http.server import BaseHTTPRequestHandler,HTTPServer
A=1.160441
B=0.853525
C=1.347386
D=0.351469
def score(t):
    ll=0.0
    for i in range(len(t)-2):
        s=t[i:i+3]
        o0=ord(s[0]);o1=ord(s[1]);o2=ord(s[2])
        ll+=math.sin(A*o0+B*o1+C*o2+D)
    return {"score":round(ll,6)}
HP=os.environ.get("AIP_HEALTH_ROUTE","/health")
PP=os.environ.get("AIP_PREDICT_ROUTE","/predict")
PORT=int(os.environ.get("AIP_HTTP_PORT","8080"))
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path==HP:
            self.send_response(200);self.end_headers();self.wfile.write(b"{}")
        else:
            self.send_response(404);self.end_headers()
    def do_POST(self):
        if self.path==PP:
            n=int(self.headers.get("Content-Length","0"))
            body=self.rfile.read(n)
            data=json.loads(body.decode("utf-8"))
            insts=data.get("instances",[])
            preds=[]
            for x in insts:
                if isinstance(x,str):
                    preds.append(score(x))
                elif isinstance(x,dict):
                    preds.append(score(x.get("text","")))
                else:
                    preds.append(score(str(x)))
            out=json.dumps({"predictions":preds}).encode("utf-8")
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(out)
        else:
            self.send_response(404);self.end_headers()
    def log_message(self,*a):
        return
HTTPServer(("0.0.0.0",PORT),H).serve_forever()
