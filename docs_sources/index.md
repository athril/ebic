# EBIC

*EBIC* is a next-generation biclustering algorithm based on artificial intelligence (AI).
*EBIC* is probably the first algorithm capable of discovering the most challenging patterns (i.e. row-constant, column-constant, shift, scale, shift-scale and trend-preserving)
It is also one of the very few parallel biclustering algorithms that use at least one graphics processing unit (GPU) and is ready for big-data challenges.

*EBIC* is mainly implemented in C++11. CUDA with OpenMP used for parallelization.


If you find our work useful, please cite it using the following BibTex entry:

```
@article{doi:10.1093/bioinformatics/bty401,
  author = {Orzechowski, Patryk and Sipper, Moshe and Huang, Xiuzhen and Moore, Jason H},
  title = {EBIC: an evolutionary-based parallel biclustering algorithm for pattern discovery},
  journal = {Bioinformatics},
  volume = {},
  number = {},
  pages = {bty401},
  year = {2018},
  doi = {10.1093/bioinformatics/bty401},
  URL = {http://dx.doi.org/10.1093/bioinformatics/bty401},
  eprint = {/oup/backfile/content_public/journal/bioinformatics/pap/10.1093_bioinformatics_bty401/3/bty401.pdf}
}
```

