#!/usr/bin/env bash 

ft=caporaso-gut
#ft=american-gut-mf
map=../data/$ft.txt
data=../data/$ft.biom
tdir=../files
 { time supervised_learning.py -i $data -m $map -c SEX -o $tdir/qiime-dir/ -f ; } 2> $tdir/qiime.time &
 { time fizzy -i $data -m $map -l SEX -o $tdir/lasso.txt -n 100 -f CMIM ; } 2> $tdir/lasso.time & 
 wait 
 
 for selects in `seq 100 100 900`; do 
   echo "Running $selects"
   { time fizzy -i $data -m $map -l SEX -o $tdir/cmim-$selects.txt -n $selects -f CMIM ; } 2> $tdir/cmim-$selects.time &
   { time fizzy -i $data -m $map -l SEX -o $tdir/mim-$selects.txt -n $selects -f MIM ; } 2> $tdir/mim-$selects.time &
   { time fizzy -i $data -m $map -l SEX -o $tdir/mrmr-$selects.txt -n $selects -f mRMR ; } 2> $tdir/mrmr-$selects.time &
   { time fizzy -i $data -m $map -l SEX -o $tdir/jmi-$selects.txt -n $selects -f JMI ; } 2> $tdir/jmi-$selects.time &
   wait 
 done

{ time npfs -l SEX -f MIM -i $data -m $map -o $tdir/npfs.txt -n 25 -b 100 ; } 2> $tdir/npfs.time
