#!/usr/bin/perl

$test_name = $ARGV[0];
$num_runs  = $ARGV[1];

`mkdir $test_name`;

for(my $i = 0; $i < $num_runs; $i += 1) {
    print("Run $i of $num_runs.\n");
    `python ./Neural_QTrain.py > $test_name/$i.csv`;
}
