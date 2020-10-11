documents in this file is trying to find the best priors for LRG in SGC

for 1-7, I tried different prior ranges, wishing that larger priors will ensure that the posterior is complete
unfirtunately, sigma and Vceil are highly degenerate that this idea doesn't work.

for 4-1~4-4, I tried diffferent number of tracers for the same prior ranges because I suspect that increasing Ntracer can reduce the degeneracy. 
Luckily, it is the case, but again, no posterior is completely within the prior. 

So I'd like to try the pre-cut SHAM instead of the after-cut SHAM
