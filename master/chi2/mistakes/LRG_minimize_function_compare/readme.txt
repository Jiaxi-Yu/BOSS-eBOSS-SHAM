the comparison of scipy.optimize.minimize and iminuit.Minute


the former one can produce result in 150s but not accurate;

the latter one cannot produce result in the beginning because we didn't 
specify the precision in the sigma.migrad() minimising process.