#!/usr/bin/gnuplot -persist
# gnuplot -e "filename='oldlogs/mHopper.out';outname='oldlogs/1.png'" a.gnu
set terminal wxt persist 
set title "Plot"
set xlabel "TimeSteps"
set ylabel "Avg Reward"
set grid
set term png
set output outname
# Plotting Avg rewards
plot filename using 4:10 with lines
# plot filename using (60*$4):24 with lines
# plot filename using (60*$4):26 with lines
