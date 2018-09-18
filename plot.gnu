#!/usr/bin/gnuplot -persist
# gnuplot -e "filename='oldlogs/mHopper.out';outname='1.png'" a.gnu
set terminal wxt persist 
set title "Plot"
set xlabel "TimeSteps"
set ylabel "Avg Reward"
set grid
set term png
set output outname
plot filename using 4:10 with lines
