several problems for iminuit package until the 2nd run:

1. errordef should be 0.5 instead of 1 for the negative log-likelihood function

2. sigma.values[#] didn't provide us with the correct result: use sigma.values[<names>]

3. the txt files provide two calculation result, making it difficult to distinguish 
the best results.

4. the results are not the optimal

****************************
the 3rd run:
****************************
1. the chi2 didn't converge, so the results are use less: 
neither because it didn't have a local min nor because it exceed the call limit. ==> 4th run to debug


****************************
the 4th run:
****************************
1. iminuit finished LRG NGC fitting, costing ~40 min=> ddin't converge;(value min is False.)
the unbiased C^{-1} should be corrected

2. scipy erroneously converged or even failed to produce a result in 4 hours

*****************************
the 5th run:
*****************************
1. without minos:
unbiased C^{-1} corrected, but hesse error is larger and edm is also larger 0.048 (previous 0.044, should be lower than 5e-6)

2. with minos:
it's not converged, so it cannot be produced.

3. chi2 robust test:
there are multipule local minimums. =>chi2 is non-parabola, so we should use another minimizer

4. until the 5th run, we didn't use the correct UNIT simulation redshift slice.

*****************************
the 6th run: (but files are in 7th run)
*****************************
1. chi2 in the same range for nseed=3 and nseed=30 produce different chi2-sigma relation.
plot nseed=3,10,20,30 with range 5-25Mpc/h to see the difference->chi2 is smaller(fewer data points), but still non-parabolic
(but use the z=0.895 slice)

it means that the statistical fluctuation will affect the chi2 relation, nseed=30 is enough, but still non-parabolic

2. use Vpeak instead of Vmax in quadrupole: 
It successfully converged! But the galaxy distribution is not as good as Vmax.
nseed = 30 is converged, but nseed=3,10,20 all have potential problems for being non-parabolic.

*********************************
the 7th run: the correct UNIT slice + quadrupole fitting 5-25 Mpc/h +nseed=30(except for stability test)
*********************************
1. preference for quadrupole??????
covariance matrices are correct=> different due to float precision
quadrupoles is not suseptible to sigma, and LRG quadrupole is completely out. => maybe the wrong z?????? no

2. Vpeak vs Vmax quadrupole fitting
Vmax best fitting is only a local minimum instead of a global minimum.


3. SGC Vmax converged, but quadrupole still away from good. chi2 zoom-in is not parabolic.


************************************
the 8th run: the same as 7th run + use Vpeak + the same model for LRG & ELG +nseed 60
************************************
LRG_2cuts: poor fitting and large error -> LRG_1cut still poor fitting and smaller error (error meaningless due to un-convergence)
-> tune the number density for LRG_2cuts to solve the poor fitting

ELG_nseeds60 : out of memory ->nseed40: large error

use ELG: neff = 2.93e-4 & LRG: neff = 6.26e-5 for the best fit and MCMC

LRG precut =170km/s: speed up the code and didn't strongly affect the result. 
for 1 chi2 calculation:
t_previous =1min 37s ; t_cut = 18.6 s
chi2_previous = 73.34(69.41); chi2_cut=63.17971(74.74)=>chi2 changes if we use different uniform random arrays=> save one



**************************************
the 9th_run:
**************************************
Vpeak, range 5-25Mpc/h, nseed=15*2=30, use n_eff to determine gal_num 

1.ELG LRG quadrupole errorbar inconsistent with the values before!!!!!!!(plot error)
2. LRG 1scat model is poorly fitted(discarded); HAM-LRG didn't behave well for SGC


**************************************
the 10th_run: (redirect to comparison 3rd_run)
**************************************
the same as 9th run, but change the LRG observation to PIP weighted and change its prior to [500,1200]






s=np.array(*post_equal*.dat) # remove the chi2

s = MCSamples(samples = s, labels=parameters) # !!!!!??????
#s = MCSamples(samples = s)
g = plots.getSubplotPlotter()
g.triangle_plot(s, filled=True)




