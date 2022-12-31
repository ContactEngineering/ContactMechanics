import sys

import muFFT

mpi_required = sys.argv[1] in ["yes", "1"]

print('`muFFT` FFT engines:')
for engine, (a, b, c) in muFFT.fft_engines.items():
    print(f'* {engine}')

if mpi_required:
    # Make sure that we have the parallel version running
    #assert muFFT.__has_parallel4_support__
    assert muFFT.has_mpi
