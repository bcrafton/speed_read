
import numpy as np

############################

array = np.array([4, 20, 40, 72, 144, 288])
narray = 2048
nlayer = 6

nmac = np.array([1769472, 37748736, 18874368, 37748736, 18874368, 37748736])
mac = array / np.array([0.085, 0.19, 0.145, 0.13, 0.12, 0.06])

dup = [1] * nlayer
ndup = narray // array

############################

def cost():
    return dup
    
############################

done = False
while not done:
    # print (cost())
    for layer in range(nlayer):
        if dup[layer] == ndup[layer]:
            dup[layer] = 1
            print (cost())
            if (layer == (nlayer - 1)):
                done = True
        else:
            dup[layer] += 1
            break

############################

'''
imagine this is a recursive solution
where we get an object

and we then we branch 100 times off that object.

> start with some initial solution.
> then keep branching.

=====

> initial solution should be an empty list.
> then add all possible combinations for layer 1 in it.
  > then branch.

kill the branches that should not continue.

=====

we need to come up with a decent API then.
> bound(choice_list)

our 'object' will just be a list I think.
> we cud make classes but i dont really feel like it.
> although i think it wud be useful now that i think about it.

this will be SOO easy, because u can only pick 1 of each.

=====

what does our object API look like ?
> bound()
> value() ... minimize cost, maximize value
> branch() ? 

loop over all objects in list
all new objects if they make the bound.

u can just go right in order really.
use a list internal to the class and u will be good to go.

> start making your class for this.

'''






















