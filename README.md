# fast_lcs

An CUDA implementation of [FAST_LCS](https://scholar.google.com/scholar?q=A+fast+parallel+algorithm+for+finding+the+longest+common+sequence+of+multiple+biosequences&hl=zh-CN&as_sdt=0&as_vis=1&oi=scholart)

* dp: A plain 2D dynamic programming algorithm.
* cpu_fastlcs: A Serial version FAST_LCS.
* no_fastlcs: A CUDA parallel version FAST_LCS.
* cu_fastlcs: A CUDA parallel version FAST_LCS with zero-copy.
* gen: Random biosequence generator.
* sw: An implementation of Smith-Waterman algorithm.
