"""
attemtping at this phucking mixing coefficient...

"""



set_1 = set(['a','b','c','d','e','f','g','h'])
set_2 = set(['a','c','d','f','h'])
set_3 = set(['q','w','e','r','t','y','g','h'])

# set intersection: what two sets have in common:
# set_a.intersection(set_b) = set_b in this case...


set_diff21 = set_2.difference(set_1)
set_diff12 = set_1.difference(set_2)

intersect_12 = set_1.intersection(set_2)
intersect_21 = set_2.intersection(set_1)
intersect_13 = set_1.intersection(set_3)

#set_1.update(set_3)

union_13 = set_1.union(set_3)




In [47]: set_diff13 = set_1.difference(set_3)                                                                                        

In [48]: set_diff13                                                                                                                  
Out[48]: {'a', 'b', 'c', 'd', 'f'}

In [49]: set_diff31 = set_3.difference(set_1)                                                                                        

In [50]: set_diff31                                                                                                                  
Out[50]: {'q', 'r', 't', 'w', 'y'}

In [51]:                      
