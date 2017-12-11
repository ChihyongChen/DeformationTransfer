########################################################################################################
#                           How to use it
########################################################################################################
""" 
1. Rename sequence in assending oreder of pose
2. Import all meshes in blender one by one in order m to n (m<n)
3. Change the "path" in this file to new path where you save this file
3. Load this add-ons in blender as per following instructions:

    i:   go to "file"
    ii:  go to "user preferences"
    iii: go to "install from files"
    iv:  go to directory where you save this file
    v:   activate add ons by clicking rectanle box in right (every time when you start blender you need to activate this add ons)
    vi:  go to "object tools" in "3D view" and in in botton you find 5 buttons 
4. Select source sequence (atleast two poses) and then by clicking button "source sequence" you can store source sequnce 
5. Similarly select target poses (atleast one) and then by clicking button "target sequence" you can store target sequnce     
6. click the buttons whatever the methods you want to use 
"""


bl_info = {
    "name": "DTSumnerPopovic",
    "author": "Prashant Domadiya",
    "version": (1, 0),
    "blender": (2, 67, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Transfer Deformation From Sorce Temporal Sequence to Target Temporal Sequence",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

import bpy
import numpy as np
import os
from scipy import sparse as sp
from scipy.sparse import linalg as sl
from multiprocessing import Pool
from functools import partial


#########################################################################################
#                   Display Mesh
#########################################################################################

def CreateMesh(V,F,NPs):
    E=np.zeros(np.shape(V))
    F = [ [int(i) for i in thing] for thing in F]
    for i in range(NPs):
        E[:,3*i]=V[:,3*i]
        E[:,3*i+1]=-V[:,3*i+2]
        E[:,3*i+2]=V[:,3*i+1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('Myobj', me)
        scn = bpy.context.scene
        scn.objects.link(ob)
        #scn.objects.active = ob
        #ob.select = True
        
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()

    
#############################################################################################
#                       Sumner and Popovic
#############################################################################################

def GetL(InFace):
    L=np.zeros([3,3])
    InFace=np.reshape(InFace,(3,3))-InFace[0:3]
    #print np.shape(InFace[1:,:].T)
    Q,R=np.linalg.qr(InFace[1:,:].T)
    L[:,1:]=np.dot(np.linalg.inv(R),Q.T).T
    L[:,0]=-L[:,1]-L[:,2]
    return L

def GetY(InFace):
    S1=np.reshape(InFace[0:9],[3,3])-InFace[0:3]
    S2=np.reshape(InFace[9:],[3,3])-InFace[9:12]
    Q,R=np.linalg.qr(S1[1:,:].T)
    tmp=np.dot(np.linalg.inv(R),Q.T)
    return (np.dot(S2[1:,:].T,tmp)).T

def DTSumAndPop(sourceInpt,TrgtInpt,F):

    NV=np.size(TrgtInpt)/3
    NF,NVF=np.shape(F)

    S=np.zeros([NV,6])
    S[:,0:3]=np.reshape(sourceInpt[:,0],[NV,3])                
    S[:,3:6]=np.reshape(sourceInpt[:,1],[NV,3])

    P=np.zeros([NV,6])
    P[:,0:3]=np.reshape(TrgtInpt,[NV,3])


    A=sp.lil_matrix((3*NF,NV))
    p=Pool()
    fcs=np.reshape(F,NVF*NF)
    PrllIn=np.reshape(P[fcs,0:3],(NF,NVF**2))
    PrllOut=p.map(GetL,PrllIn)

    for t in range(len(F)):
        A[3*t:3*t+3,F[t]]=PrllOut[t]
    p.close()
    A=A.tocsc()
             
    c=sp.lil_matrix(P[NV-1,0:3])
    print(c)

    Y=sp.lil_matrix((3*NF,3))

    p=Pool()
    fcs=np.reshape(F,NVF*NF)
    PrllIn=np.concatenate((np.reshape(S[fcs,0:3],(NF,NVF**2)),
                          np.reshape(S[fcs,3:6],(NF,NVF**2))),axis=1)
    PrllOut=p.map(GetY,PrllIn)

    for t in range(len(F)):
        Y[3*t:3*t+3,:]=PrllOut[t]
    p.close()

    Y=Y-A[:,NV-1].dot(c)
    A=A.tocsc()
    tmp=A[:,:(NV-1)].transpose()
    A=tmp.dot(tmp.transpose())
    b=tmp.dot(Y)
    P[:NV-1,3]=sl.spsolve(A,b[:,0])
    P[:NV-1,4]=sl.spsolve(A,b[:,1])
    P[:NV-1,5]=sl.spsolve(A,b[:,2])

    P[NV-1,3:6]=c.toarray()
    CreateMesh(P[:,3:6],F,1)
    return


#####################################################################################################
#           Deformation Transfer
#####################################################################################################

class DTToolsPanel(bpy.types.Panel):
    bl_label = "DT Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
 
    def draw(self, context):
        self.layout.operator("get.seq",text='Source Seq').seqType="source"
        self.layout.operator("get.seq",text='Target Seq').seqType="target" 
        self.layout.operator("dt.tools",text='DTSumnerPopovic').seqType="DTSumnerPopovic"
         
        

# Operator
class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        path=bpy.utils.resource_path('USER')
        print(path)
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        F=np.zeros([len(obj.data.polygons),len(obj.data.polygons[0].vertices)],dtype=int)
        V=np.zeros([3*len(obj.data.vertices),len(Selected_Meshes)])
        t=0
        for f in obj.data.polygons:
                F[t,:]=f.vertices[:]
                t+=1
        
        for i in range(len(Selected_Meshes)):
            bpy.context.scene.objects.active = Selected_Meshes[-i-1]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world*v.co
                V[3*t:3*t+3,i]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
        np.savetxt(path+self.seqType+'_vertz.txt',V,delimiter=',')
        np.savetxt(path+'facez.txt',F,delimiter=',')                      
        return{'FINISHED'}    
 

class DeformationTransferTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType = bpy.props.StringProperty()
    def execute(self,context):
        path=bpy.utils.resource_path('USER') 
        
        sourceInpt=np.loadtxt(path+'source_vertz.txt',delimiter=',')
        TrgtInpt=np.loadtxt(path+'target_vertz.txt',delimiter=',')
        F=np.loadtxt(path+'facez.txt',delimiter=',').astype(int)
        
        if np.size(TrgtInpt)/len(TrgtInpt)==1: 
            DTSumAndPop(sourceInpt[:,0:2],TrgtInpt,F)
        else:
            DTSumAndPop(sourceInpt[:,0:2],TrgtInpt[:,0],F)
        return {'FINISHED'}

def register():
    bpy.utils.register_class(DTToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.utils.register_class(DeformationTransferTools)
    
   

def unregister():
    bpy.utils.unregister_class(DTToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    bpy.utils.unregister_class(DeformationTransferTools)
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 

