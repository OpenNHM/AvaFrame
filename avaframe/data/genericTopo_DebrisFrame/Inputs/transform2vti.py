import numpy as np
import vtk
from vtk.util import numpy_support

# ASC einlesen
with open("DEM_GT_Topo.asc") as f:
    header = {}
    for _ in range(6):
        k, v = f.readline().split()
        header[k.lower()] = float(v)

    data = np.loadtxt(f)

# Header
nx = int(header["ncols"])
ny = int(header["nrows"])
cell = header["cellsize"]
nodata = header.get("nodata_value", -9999)

# NoData behandeln
data[data == nodata] = np.nan

# VTK ImageData
vtk_img = vtk.vtkImageData()
vtk_img.SetDimensions(nx, ny, 1)
vtk_img.SetSpacing(cell, cell, 1.0)
vtk_img.SetOrigin(
    header.get("xllcorner", 0.0),
    header.get("yllcorner", 0.0),
    0.0
)

vtk_array = numpy_support.numpy_to_vtk(
    data[::-1].ravel(order="C"),
    deep=True,
    array_type=vtk.VTK_FLOAT
)
vtk_img.GetPointData().SetScalars(vtk_array)

# Schreiben
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("terrain.vti")
writer.SetInputData(vtk_img)
writer.Write()

## Open .vti in Paraview
# Filter -> Warp by Scalar
# Apply
# change to 3D view mode
