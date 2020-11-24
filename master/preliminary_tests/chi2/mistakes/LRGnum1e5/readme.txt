the LRG number for the HAM method should be 5468750 instead of len(obs).

This is because we should keep the same number density in the box as the 
observation, instead of the same galaxy number. 

The difference between LRGnum and len(obs) is because of the footprint is
not as complete as the simulation box.