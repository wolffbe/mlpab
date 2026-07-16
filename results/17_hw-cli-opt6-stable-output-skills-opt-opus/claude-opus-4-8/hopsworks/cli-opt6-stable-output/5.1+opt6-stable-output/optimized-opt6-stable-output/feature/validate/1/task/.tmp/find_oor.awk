BEGIN { FS="," }
NR>1 && $4 != "" {
  a = $4 + 0
  if (a > 10000 || a < 0) print $1" amount="$4
}
