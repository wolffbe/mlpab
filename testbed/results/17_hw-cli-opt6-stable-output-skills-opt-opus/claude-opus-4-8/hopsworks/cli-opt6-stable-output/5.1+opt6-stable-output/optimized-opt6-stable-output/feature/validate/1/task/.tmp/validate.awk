BEGIN { FS=","; OFS="," }
NR==1 { print > validfile; next }
{
  nf = NF
  rid = $1
  amount = $4
  cat = $5
  valid = 1
  # rule: well-formed (5 fields)
  if (nf != 5) valid = 0
  # rule 1: amount present
  if (amount == "") valid = 0
  # rule 2: amount in [0,10000]
  else {
    a = amount + 0
    if (a < 0 || a > 10000) valid = 0
  }
  # rule 3: category whitelist
  if (cat != "grocery" && cat != "travel" && cat != "salary" && cat != "rent" && cat != "other") valid = 0

  if (valid == 1) print $0 > validfile
  else print rid > rejectfile
}
