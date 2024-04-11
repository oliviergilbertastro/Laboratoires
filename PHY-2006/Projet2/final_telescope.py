from raytracing import *

#all lengths are in mm
l1 = Lens(f=1000, diameter=50, label="Objectif")

px_size = 3.75E-3 #mm
sensor_width = px_size*1304
sensor_height = px_size*976

#We make the path such as the object is the sensor to see what the FOV is
path = ImagingPath()
path.objectHeight = sensor_height
path.label="FOV"
path.append(Space(d=l1.f))
path.append(l1) 
path.append(Space(d=(l1.f*5)))

print('FOV:', np.tan(sensor_width/l1.f)*2/(2*np.pi)*360)
#transferMatrix method returns the transfer matrix between the focals, not between the lenses
print(path.transferMatrix())
print('IsImaging (i.e. B=0):', path.isImaging)
print(path.fieldStop())
print(path.fieldOfView())
path.display()

