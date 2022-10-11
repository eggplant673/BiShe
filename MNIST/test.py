nums = [4,1,5,2,9,6,8,7]
 
m_sorted = sorted(enumerate(nums), key=lambda x:x[1])
 
sorted_inds = [m[0] for m in m_sorted]
 
sorted_nums = [m[1] for m in m_sorted]

print(sorted_inds)