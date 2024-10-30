import numpy  as np

mylist = [10,
         11]
#vector - has one column
#matrice/matrix - has multiple rows and columns
myarray = np.array(mylist)
print(myarray)
print(myarray.shape)

mylist = [  [20, 20, 50],
            [40, 40, 50]]
myarray = np.array(mylist)
print(myarray.shape)

mylist = mylist * 2
print(mylist)
myarray = myarray * 2
print(myarray)

myfancyMatrice = np.ones((3, 4))
print(myfancyMatrice)
