import avaframe.out3Plot.outAIMEC as outAIMEC
import avaframe.ana5Utils.regionalThalwegTools as thalwegTools
from avaframe.in3Utils import cfgUtils
from avaframe.ana3AIMEC import ana3AIMEC



"""
    rasterTransfo: dict
        domain transformation information:
            gridx: 2D numpy array
                x coord of the new raster points in old coord system
            gridy: 2D numpy array
                y coord of the new raster points in old coord system
            s: 1D numpy array
                new coord system in the polyline direction
            l: 1D numpy array
                new coord system in the cross direction
            x: 1D numpy array
                coord of the resampled polyline in old coord system
            y: 1D numpy array
                coord of the resampled polyline in old coord system
            rasterArea: 2D numpy array
                real area of the cells of the new raster
            indStartOfRunout: int
                index for start of the runout area (in s)
                if defineRunoutArea is False - indStartOfRunout=0 (start of thalweg)
"""

startRow = 112
startCol = 101

pathDict = {'projectName': "/home/paula/repos/github/AvaFrame/avaframe/data/avaBowl_FP/Outputs/com4FlowPy/peakFiles/res_b29b7b1962/ThalwegPlots",
            "pathResult": "/home/paula/repos/github/AvaFrame/avaframe/data/avaBowl_FP/Outputs/com4FlowPy/peakFiles/res_b29b7b1962"}
datathalweg = thalwegTools.readThalwegData(f"{pathDict['pathResult']}/thalwegData", startRow, startCol, "CoF")
#print(datathalweg)


rasterTransfo = {"indStartOfRunout": 0,"z": datathalweg["z"],
"s": datathalweg["s"],
"startOfRunoutAreaAngle": False}

pfvCrossMax = thalwegTools.zDelta2velocity(datathalweg["zDelta"])

pftCrossMax = datathalweg["flux"] * 100
cfg = cfgUtils.getModuleConfig(ana3AIMEC)

cfgPlots = cfg['PLOTS']
simName = "test"


outAIMEC.plotVelThAlongThalweg(pathDict, rasterTransfo, pftCrossMax, pfvCrossMax, cfgPlots, simName)