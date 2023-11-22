import pstats

p = pstats.Stats('output.pstats')
p.strip_dirs().sort_stats('cumulative').print_stats('bimaru\.py')

