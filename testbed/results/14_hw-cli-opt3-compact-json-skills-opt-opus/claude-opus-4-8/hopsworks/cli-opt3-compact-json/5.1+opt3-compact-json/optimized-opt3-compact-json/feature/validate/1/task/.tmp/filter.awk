BEGIN { FS="," }
NR==1 { print > ".tmp/valid.csv"; next }
{
  amt=$4; cat=$5;
  ok=1;
  if (amt=="") { ok=0 }
  else {
    a=amt+0;
    if (a<0 || a>10000) { ok=0 }
  }
  if (cat!="grocery" && cat!="travel" && cat!="salary" && cat!="rent" && cat!="other") { ok=0 }
  if (ok==1) { print >> ".tmp/valid.csv" }
  else { print $1 >> ".tmp/rejected.txt" }
}
