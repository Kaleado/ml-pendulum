#!/usr/bin/perl

$test_name = $ARGV[0];
$num_runs = $ARGV[1];

chdir($test_name);

for(my $i = 0; $i < $num_runs; $i += 1) {
    print("Graphing $i of $num_runs.\n");
    $scr = "jpeg(\"$i.jpg\")\ndf <- read.csv(\"$i.csv\")\nx <- df\$X0\ny <- df\$X0.1\nplot(x, y)\nlines(x, y)";
    `echo '$scr' | R --no-save`;
}
