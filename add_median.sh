#!/bin/bash
set -euo pipefail

# 사용법:
#   ./add_median.sh topk_alignrec_item*_k500.txt
# 또는:
#   ./add_median.sh ./RQ3/baby/topk_*.txt

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <topk_file1> [topk_file2 ...]"
  exit 1
fi

for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "[SKIP] not a file: $f"
    continue
  fi

  out="${f%.txt}_with_median.txt"

  # 2hop: col4, vision_cos: col5, text_cos: col6 (헤더/avg 제외)
  med_line=$(
    awk -F'\t' '
      BEGIN { n=0 }
      # 데이터 행만 수집: 첫 컬럼이 숫자(rank)인 라인만
      $1 ~ /^[0-9]+$/ {
        twohop[n]=$4+0
        vision[n]=$5+0
        text[n]=$6+0
        n++
      }
      function sort_array(a, n,   i,j,tmp){
        for(i=0;i<n;i++){
          for(j=i+1;j<n;j++){
            if(a[i] > a[j]){ tmp=a[i]; a[i]=a[j]; a[j]=tmp }
          }
        }
      }
      function median(a, n,   mid){
        if(n==0) return "NA"
        sort_array(a,n)
        mid=int(n/2)
        if(n%2==1) return a[mid]
        else return (a[mid-1]+a[mid])/2.0
      }
      END {
        m2=median(twohop,n)
        mv=median(vision,n)
        mt=median(text,n)
        # 형식: median  -  -  <2hop>  <vision>  <text>
        if(m2=="NA"){
          print "median\t-\t-\t-\t-\t-"
        } else {
          # twohop은 정수처럼 보이게, 나머지는 소수 4자리
          printf "median\t-\t-\t%.2f\t%.4f\t%.4f\n", m2, mv, mt
        }
      }
    ' "$f"
  )

  # avg 라인은 유지하되, median 라인을 avg 아래에 추가 (새 파일로 저장)
  # (만약 원본 맨 끝에 이미 median이 있으면 중복 방지)
  if tail -n 5 "$f" | grep -q "^median[[:space:]]"; then
    echo "[SKIP] already has median: $f"
    cp -f "$f" "$out"
  else
    cp -f "$f" "$out"
    echo "$med_line" >> "$out"
    echo "[OK] $f -> $out"
    echo "     $med_line"
  fi
done
