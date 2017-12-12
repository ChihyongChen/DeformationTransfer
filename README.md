# DeformationTransfer
- The function "DTSumAndPop" takes the reference source mesh, the reference target mesh and deformed source mesh.  4
- Inside the function "DTSumAndPop", the instruction "p=Pool()" distributes the faces to all cores of the CPU. That means this code works faaster if the number of Core is higher.  


sir I found above error while running code:


Traceback (most recent call last):
  File "/home/orienit/.config/blender/2.72/scripts/addons/DTSumnerPopovic.py", line 198, in execute
    DTSumAndPop(sourceInpt[:,0:2],TrgtInpt,F)
  File "/home/orienit/.config/blender/2.72/scripts/addons/DTSumnerPopovic.py", line 126, in DTSumAndPop
    tmp=A[:,:(NV-1)].transpose()
  File "/home/orienit/.local/lib/python3.4/site-packages/scipy/sparse/csc.py", line 167, in __getitem__
    return self.T[col, row].T
  File "/home/orienit/.local/lib/python3.4/site-packages/scipy/sparse/csr.py", line 304, in __getitem__
    return self._get_submatrix(row, col)
  File "/home/orienit/.local/lib/python3.4/site-packages/scipy/sparse/csr.py", line 455, in _get_submatrix
    i0, i1 = process_slice(row_slice, M)
  File "/home/orienit/.local/lib/python3.4/site-packages/scipy/sparse/csr.py", line 438, in process_slice
    i0, i1, stride = sl.indices(num)
TypeError: slice indices must be integers or None or have an __index__ method
