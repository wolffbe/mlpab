BEGIN { FS="," }
NR>1 {
    cat = $5
    gsub(/\r/, "", cat)
    amt = $4
    is_bad = 0
    if (amt == "") is_bad = 1
    else if (amt+0 < 0 || amt+0 > 10000) is_bad = 1
    else if (cat != "grocery" && cat != "travel" && cat != "salary" && cat != "rent" && cat != "other") is_bad = 1
    if (is_bad) print $1
}
