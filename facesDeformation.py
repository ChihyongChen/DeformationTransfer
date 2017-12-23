
bl_info = {
    "name": "deformation",
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
import subprocess as subp
from scipy import sparse as sp
from scipy.sparse import linalg as sl
from multiprocessing import Pool
from functools import partial


#########################################################################################
#                   Display Mesh
#########################################################################################

def CreateMesh(V, F, NPs):
    E = np.zeros(np.shape(V))
    F = [[int(i) for i in thing] for thing in F]
    for i in range(NPs):
        E[:, 3 * i] = V[:, 3 * i]
        E[:, 3 * i + 1] = -V[:, 3 * i + 2]
        E[:, 3 * i + 2] = V[:, 3 * i + 1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('Myobj', me)
        scn = bpy.context.scene
        scn.objects.link(ob)
        # scn.objects.active = ob
        # ob.select = True

        me.from_pydata(E[:, 3 * i:3 * i + 3], [], F)
        me.update()



#####################################################################################################
#           Deformation Transfer
#####################################################################################################
def GetL(InFace):

    L=np.zeros([3,3])

    InFace=np.reshape(InFace,(3,3))-InFace[0:3]

    Q,R=np.linalg.qr(InFace[1:,:].T)

    L[:,1:]=np.dot(np.linalg.inv(R),Q.T).T
    L[:,0]=-L[:,1]-L[:,2]
    return L


numOfMeshes = 0

class DTToolsPanel(bpy.types.Panel):
    bl_label = "DT Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"

    def draw(self, context):
        self.layout.operator("get.seq", text='Source Seq').seqType = "source"
        self.layout.operator("get.seq", text='Target Seq').seqType = "target"
        self.layout.operator("dt.tools", text='deformation').seqType = "deformation"



# Operator
class GetSequence(bpy.types.Operator):

    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType = bpy.props.StringProperty()

    def execute(self, context):

        path = bpy.utils.resource_path('USER')
        print(path)
        Selected_Meshes = bpy.context.selected_objects
        obj = bpy.context.active_object
        F = np.zeros([len(obj.data.polygons), len(obj.data.polygons[0].vertices)], dtype=int)
        t = 0
        for f in obj.data.polygons:
            F[t, :] = f.vertices[:]
            t += 1

        if self.seqType == "source":
            global  numOfMeshes
            numOfMeshes = len(Selected_Meshes)
            sourceInpt = np.zeros([3 * len(obj.data.vertices), len(Selected_Meshes)])
            NV = int(np.size(sourceInpt[:, 0:1]) / 3)
            NF, NVF = np.shape(F)

            for i in range(len(Selected_Meshes)):
                bpy.context.scene.objects.active = Selected_Meshes[-i - 1]
                obj = bpy.context.active_object

                t = 0
                for v in obj.data.vertices:
                    co_final = obj.matrix_world * v.co
                    sourceInpt[3 * t:3 * t + 3, i] = np.array([co_final.x, co_final.z, -co_final.y])
                    # arranging vertices of mesh in V
                    t += 1
            S = np.zeros([NV, 6])
            for i in range(len(Selected_Meshes)-1):
                S[:, 0:3] = np.reshape(sourceInpt[:, 0], [NV, 3])
                S[:, 3:6] = np.reshape(sourceInpt[:, i+1], [NV, 3])
                fcs = np.reshape(F, NVF * NF)
                PrllIn = np.concatenate((np.reshape(S[fcs, 0:3], (NF, NVF ** 2)),
                                         np.reshape(S[fcs, 3:6], (NF, NVF ** 2))), axis=1)
                for t in range(10):
                    np.savetxt(self.seqType+str(t+1)+'_vertx.txt', PrllIn[t*len(PrllIn)/10:(t+1)*len(PrllIn)/10,:], delimiter=',')

        else:
            TrgtInpt = np.zeros([3 * len(obj.data.vertices), len(Selected_Meshes)])
            # TrgtInpt = TrgtInpt[:,0:1]
            NV = int(np.size(TrgtInpt) / 3)
            NF, NVF = np.shape(F)

            for i in range(len(Selected_Meshes)):
                bpy.context.scene.objects.active = Selected_Meshes[-i - 1]
                obj = bpy.context.active_object

                t = 0
                for v in obj.data.vertices:
                    co_final = obj.matrix_world * v.co
                    TrgtInpt[3 * t:3 * t + 3, i] = np.array([co_final.x, co_final.z, -co_final.y])
                    # arranging vertices of mesh in V
                    t += 1

                P = np.zeros([NV, 6])
                P[:, 0:3] = np.reshape(TrgtInpt, [NV, 3])
                np.savetxt(self.seqType + '_init_vertz.txt', P, delimiter=',')
                # similarly for target input

                fcs = np.reshape(F, NVF * NF)
                # Transposing the Face matrix F i.e. faces are in vertical rep.=> 9x1
                PrllIn = np.reshape(P[fcs, 0:3], (NF, NVF ** 2))
            np.savetxt(self.seqType + '_vertz.txt', PrllIn, delimiter=',')
        # reshaping the matrix sourceInpt into S by changing vertical orientation of faces into horizontal


        np.savetxt('facez.txt', F, delimiter=',')

        return {'FINISHED'}


class DeformationTransferTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType = bpy.props.StringProperty()


    def execute(self, context):
        matrixA = []
        PrllInA = np.loadtxt('target_vertz.txt', delimiter=',')
        #PrllInY = np.loadtxt('source_vertz.txt', delimiter=',')
        P = np.loadtxt('target_init_vertz.txt', delimiter=',')
        F = np.loadtxt('facez.txt', delimiter=',').astype(int)  # path+

        NV = int(np.size(P) / 6)

        NF, NVF = np.shape(F)

        A = sp.lil_matrix((3 * NF, NV))
        p = Pool()
        PrllOut = p.map(GetL, PrllInA)

        for t in range(NF):
            matrixA.append(PrllOut[t][0])
            matrixA.append(PrllOut[t][1])
            matrixA.append(PrllOut[t][2])
        p.close()
        MA = np.array(matrixA)
        np.savetxt('/home/orienit/outputA.txt', MA, delimiter=',')
        strArg = ""
        for u in range(10):
            strng = "/uday/source"+str(u+1)+"_vertx.txt"
            strngIn = "/home/orienit/source"+str(u+1)+"_vertx.txt"
            args = "hadoop fs -put "+strngIn+" /uday"
            subp.call(args, shell=True)
            strArg = strng+" "+strArg

        args = "hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.6.0.jar -file /home/orienit/mapper.py -mapper /home/orienit/mapper.py  -input "+strArg+"-output /demo/streaming-py -cacheFile '/uday/facez.txt#facez.txt' '/uday/target_init_vertz.txt#target_init_vertz.txt' '/uday/outputA.txt#outputA.txt'"
        subp.call(args,shell=True)
        x = []

        for u in range(10):
            print('output'+str(u+1)+'.txt')
            P = np.loadtxt('output'+str(u+1)+'.txt', delimiter=',')
            for t in range(len(P)):
                x.append(P[t])
        P = np.array(x)
        CreateMesh(P[:, 3:6], F, 1)
        subp.call("hadoop fs -rm -r /demo/streaming-py", shell=True)
        for u in range(10):
            strngIn = "/uday/source"+str(u+1)+"_vertx.txt"
            args = "hadoop fs -rm "+strngIn
            subp.call(args, shell=True)

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


