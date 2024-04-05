from raytracing import *


l1 = Lens(f=1200, diameter=50, label="Objectif")
l2 = Lens(f=-200, diameter=38, label="Oculaire")



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