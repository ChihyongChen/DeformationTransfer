# DeformationTransfer
- The function "DTSumAndPop" takes the reference source mesh, the reference target mesh and deformed source mesh.  4
- Inside the function "DTSumAndPop", the instruction "p=Pool()" distributes the faces to all cores of the CPU. That means this code works faaster if the number of Core is higher.  
