def addToIndex(toindex, pixel):
    if pixel>60 and pixel<720:
        toindex.append(pixel)
        toindex.append(pixel+1)
        toindex.append(pixel-1)
        toindex.append(pixel-28)
        toindex.append(pixel-29)
        toindex.append(pixel-27)
        toindex.append(pixel+28)
        toindex.append(pixel+29)
        toindex.append(pixel+27) 
index = [22,33,44,88]
for ele in index:
    if(ele >60 and ele<720):
        addToIndex(index,ele)
        print('----')
indexMatter = list(set(index))
print(indexMatter)