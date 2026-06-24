BEGIN { FS="," }
NR==1 { next }
{
  amt=$4; cat=$5;
  ok=1;
  if (amt=="") ok=0;
  else if (amt+0<0 || amt+0>10000) ok=0;
  if (cat!="grocery" && cat!="travel" && cat!="salary" && cat!="rent" && cat!="other") ok=0;
  if (ok==1) print $0 >> ".tmp/valid.csv";
  else print $1 >> ".tmp/rejected.txt";
}
