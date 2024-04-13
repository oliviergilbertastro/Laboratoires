from raytracing import *


l1 = Lens(f=20, diameter=50, label="Objectif")
l2 = Lens(f=-5, diameter=50, label="Oculaire")

#def transfer_matrix(f1,f2):
#    return np.array([[-f1/f2, 0], [f1+f2, -f2/f1]])


path = ImagingPath()
path.label="Simple example"
path.append(Space(d=l1.f))
path.append(l1) 
path.append(Space(d=(l1.f+l2.f)))
path.append(l2) 
path.append(Space(d=l2.f))


print(path.transferMatrix())
print('IsImaging (i.e. B=0):', path.isImaging)
path.display()