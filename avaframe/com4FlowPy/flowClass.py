#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Functions for calculations at the 'cell level' (vgl. D'Amboise et al., 2022)
"""

import numpy as np
import math


class Cell:

    def __init__(
        self,
        rowindex,
        colindex,
        dem_ng,
        cellsize,
        flux,
        z_delta,
        parent,
        alpha,
        exp,
        flux_threshold,
        max_z_delta,
        startcell,
        fluxDistOldVersionBool=False,
        FSI=None,
        forestParams=None,
        startcellVol=None,
    ):
        """constructor for the Cell class that describes a raster cell that is hit by the GMF.
        the constructor function is called every time a new instance of type 'Cell' is
        initialized.
        NOTE/TODO: parent can be of different data types still, maybe split into two separate variables
                   * bool --> isStart
                   * Cell --> startCell
        """
        self.rowindex = rowindex  # index of the Cell in row-direction (i.e. local y-index in the calculation domain)
        self.colindex = colindex  # index of the Cell in column-direction (i.e. local x-index in the calculation domain)
        self.dem_ng = dem_ng  # elevation values in the 3x3 neigbourhood around the Cell
        self.altitude = dem_ng[1, 1]  # elevation value of the cell (central cell of 3x3 neighbourhood)
        self.cellsize = cellsize  # cellsize in meters

        self.tan_beta = np.zeros_like(self.dem_ng)
        self.dist = np.zeros_like(self.dem_ng)
        self.persistence = np.zeros_like(self.dem_ng)
        self.r_t = np.zeros_like(self.dem_ng)
        self.no_flow = np.ones_like(self.dem_ng)

        self.flux = flux
        self.fluxDep = 0
        self.z_delta = z_delta

        self.alpha = float(alpha)
        self.exp = int(exp)
        self.max_z_delta = float(max_z_delta)
        self.flux_threshold = float(flux_threshold)

        self.fluxDistOldVersionBool = fluxDistOldVersionBool

        self.tanAlpha = np.tan(np.deg2rad(self.alpha))  # moved to constructor, so this doesn't have to be calculated on
        # every iteration of calc_z_delta(self)

        self.min_distance = 0  # minimal distance to start-cell (i.e. along shortest path) min_distance >=
        self.minDistXYZ = 0  # minimal distance to start-cell (Actual 3D lenght, not projected!!)
        self.max_distance = 0  # NOTE: self.max_distance is never used - maybe remove!?
        self.min_gamma = 0  # NOTE: self.min_gamma (assumingly minimal travel angle to cell) never used - maybe remove!?
        self.max_gamma = 0
        self.sl_gamma = 0

        self._SQRT2 = np.sqrt(2.0)
        self._RAD90 = np.deg2rad(90.0)

        self.startcellVolMin = startcellVol
        self.startcellVolMax = startcellVol

        # NOTE: Forest Interaction included here
        # if FSI != None AND forestParams != None - then self.ForestBool = True and forestParams and
        # FSI are accordingly initialized
        if (FSI is not None) and (forestParams is not None):

            self.forestBool = True
            self.forestModule = forestParams["forestModule"]
            self.skipForestDist = forestParams["skipForestDist"]

            # forestInteraction:
            self.forestInteraction = forestParams["forestInteraction"]
            if self.forestInteraction:
                if FSI > 0:
                    self.isForest = 1
                else:
                    self.isForest = 0
                self.forestIntCount = self.isForest
            else:
                self.forestInteraction = False
            if (self.forestModule == "forestFriction") or (self.forestModule == "forestDetrainment"):
                self.FSI = FSI
                self.maxAddedFrictionForest = forestParams["maxAddedFriction"]
                self.minAddedFrictionForest = forestParams["minAddedFriction"]
                self.noFrictionEffectV = forestParams["velThForFriction"]
                self.maxAddedDetrainmentForest = forestParams["maxDetrainment"]
                self.minAddedDetrainmentForest = forestParams["minDetrainment"]
                self.noDetrainmentEffectV = forestParams["velThForDetrain"]

                _vThFr = self.noFrictionEffectV
                _vThDe = self.noDetrainmentEffectV
                _sqrt2xG = self._SQRT2 * 9.81
                self.noFrictionEffectZDelta = (_vThFr * _vThFr) / _sqrt2xG
                self.noDetrainmentEffectZdelta = (_vThDe * _vThDe) / _sqrt2xG

            elif self.forestModule == "forestFrictionLayer":

                if forestParams["fFrLayerType"] == "absolute":
                    self.AlphaFor = FSI
                elif forestParams["fFrLayerType"] == "relative":
                    self.AlphaFor = self.alpha + FSI

                self.AlphaFor = max(self.AlphaFor, self.alpha)  # Friction in Forest can't be lower than without forest
                self.tanAlphaFor = np.tan(np.deg2rad(self.AlphaFor))

            # NOTE: This is a quick hack to check if all values for Detrainment are set to 0 (as provided in the
            #      .ini file)
            #      if this is the case, then the self.forest_detrainment function does not have to be called inside
            #      self.calc_distribution
            # TO-DO: clean this up and handle it better
            if (self.forestModule == "forestFriction") or (self.forestModule == "forestFrictionLayer"):
                self.forestDetrainmentBool = False
            elif (
                (self.maxAddedDetrainmentForest == 0)
                and (self.minAddedDetrainmentForest == 0)
                and (self.noDetrainmentEffectV == 0)
            ):
                self.forestDetrainmentBool = False
            else:
                self.forestDetrainmentBool = True

        else:
            self.forestBool = False
            self.forestInteraction = False

            self.FSI = 0.0
            self.maxAddedFrictionForest = 0
            self.minAddedFrictionForest = 0
            self.noFrictionEffectV = 0
            self.maxAddedDetrainmentForest = 0
            self.minAddedDetrainmentForest = 0
            self.noDetrainmentEffectV = 0
            self.noFrictionEffectZDelta = 0
            self.noDetrainmentEffectZdelta = 0

        if type(startcell) == bool:  # if a boolean variable (i.e.'True') is passed to the constructor
            self.is_start = True  # set is_start to True
        else:
            self.startcell = startcell  # give startcell to cell
            self.is_start = False  # set is_start to False

        self.lOfParents = []

        if type(parent) == Cell:
            self.lOfParents.append(parent)
            if self.forestInteraction:
                self.forestIntCount += parent.forestIntCount

    def add_os(self, flux):
        """
        Adds flux to 'existing' flux

        Parameters
        -----------
        flux: float
            added flux
        """
        self.flux += flux

    def add_parent(self, parent):
        """
        Adds parent to parents list
        and optionally the forest interaction value of the parent

        Parameters
        -----------
        parent: class Cell
            Cell object to add to the list of parent cells 'lOfParents'
        """
        self.lOfParents.append(parent)
        if self.forestInteraction:
            # check if new/ younger parent has a lower forest interaction number
            # than the older one -> take minimum!
            if parent.forestIntCount < (self.forestIntCount - self.isForest):
                self.forestIntCount = parent.forestIntCount + self.isForest

    def calc_startCellVol(self, startcellVolNew):
        self.startcellVolMin = min(self.startcellVolMin, startcellVolNew)
        self.startcellVolMax = max(self.startcellVolMax, startcellVolNew)

    def calcDistMin(self, calc3D=False):
        """
        function calculates the projected horizontal (self.min_distance) and 3D (self.minDistXYZ) length
        of the shortest flow path from the start-cell to the current cell.

        Parameters
        -----------
        calc3D: bool
            if True, the 3D distance is computed additionally (default: False)
        """
        if calc3D:
            _ldistMin = []
            _lDistMinXYZ = []
            for parent in self.lOfParents:
                _dx = abs(parent.colindex - self.colindex) * self.cellsize
                _dy = abs(parent.rowindex - self.rowindex) * self.cellsize
                _dz = abs(parent.altitude - self.altitude)
                _ldistMin.append(math.sqrt(_dx * _dx + _dy * _dy) + parent.min_distance)
                _lDistMinXYZ.append(math.sqrt(_dy*_dy + _dy*_dy + _dz*_dz) + parent.minDistXYZ)
            self.min_distance = np.amin(_ldistMin)
            self.minDistXYZ = np.amin(_lDistMinXYZ)
        else:
            _ldistMin = []
            for parent in self.lOfParents:
                _dx = abs(parent.colindex - self.colindex) * self.cellsize
                _dy = abs(parent.rowindex - self.rowindex) * self.cellsize
                _ldistMin.append(math.sqrt(_dx * _dx + _dy * _dy) + parent.min_distance)
            self.min_distance = np.amin(_ldistMin)

    def calc_fp_travelangle(self):
        """function calculates the travel-angle along the shortest flow-path from the start-cell
        to the current cell. The travel-angle along the shortest flow-path is equivalent to the
        maximum travel angle along all paths from the startcell to this cell.
        """
        _dh = self.startcell.altitude - self.altitude  # elevation difference from cell to start-cell
        self.max_gamma = np.rad2deg(np.arctan(_dh / self.min_distance))

    def calc_sl_travelangle(self):
        """function calculates the travel-angle between the start cell and the current cell
        using the shortest connection or straight line (sl) between the two cells.
        The travel angle calculated with this shortest horizontal distance 'sl_gamma' is always
        larger or equal to the travel angle  'max_gamma', which is calculated along the
        (potentially curved) flow path (fp). (vgl. Meißl, 1998)
        """
        _dx = abs(self.startcell.colindex - self.colindex)
        _dy = abs(self.startcell.rowindex - self.rowindex)
        _dh = self.startcell.altitude - self.altitude

        _ds = math.sqrt(_dx * _dx + _dy * _dy) * self.cellsize
        self.sl_gamma = np.rad2deg(np.arctan(_dh / _ds))

    def calc_z_delta(self):
        """
        function calculates zDelta (velocity or energy line height) to the eligible neighbours
        NOTE: forestFriction related mechanics are implemented here!
        """
        self.z_delta_neighbour = np.zeros((3, 3))
        self.z_gamma = self.altitude - self.dem_ng
        ds = np.array([[self._SQRT2, 1, self._SQRT2], [1, 0, 1], [self._SQRT2, 1, self._SQRT2]])

        if (not self.is_start):
            if (not self.forestBool):
                self.calcDistMin()
            else:
                self.calcDistMin(calc3D=True)

        if self.forestBool:

            if self.forestModule == "forestFrictionLayer":
                if (not self.is_start) and (self.skipForestDist < self.minDistXYZ):
                    _tanAlpha = self.tanAlphaFor
                else:
                    _tanAlpha = self.tanAlpha

            if self.forestModule in ["forestFriction", "forestDetrainment"]:
                if (not self.is_start) and (self.FSI > 0.) and (self.skipForestDist < self.minDistXYZ):
                    # if forestBool, we assume that forestFriciton is activated
                    # and if FSI > 0 then we also calculate _tanAlpha with forestEffect
                    # NOTE: We also don't assume a forest Effect on potential Start Zells, since this should
                    #      ideally be handled by a separate release-area algorithm in the pre-processing
                    # NOTE-TODO: The rest of this implementation is also just copy+pasted from 'foreste_detraiment'
                    #      branch and not yet fully tested!!
                    if self.z_delta < self.noFrictionEffectZDelta:
                        # friction at rest v=0 would be applied to start cells
                        _rest = self.maxAddedFrictionForest * self.FSI
                        # rise over run
                        _slope = (_rest - self.minAddedFrictionForest) / (0 - self.noFrictionEffectZDelta)
                        # y = mx + b, shere z_delta is the x
                        friction = max(self.minAddedFrictionForest, _slope * self.z_delta + _rest)
                        _alpha_calc = self.alpha + max(0, friction)  # NOTE: not sure what this does, seems redundant!
                    else:
                        _alpha_calc = self.alpha + self.minAddedFrictionForest

                    _tanAlpha = np.tan(np.deg2rad(_alpha_calc))
                else:
                    _tanAlpha = self.tanAlpha

        else:
            # else simply use tanAlpha
            _tanAlpha = self.tanAlpha

        self.z_alpha = ds * self.cellsize * _tanAlpha
        self.z_delta_neighbour = self.z_delta + self.z_gamma - self.z_alpha
        self.z_delta_neighbour[self.z_delta_neighbour < 0] = 0
        self.z_delta_neighbour[self.z_delta_neighbour > self.max_z_delta] = self.max_z_delta

    def calc_tanbeta(self):
        """calculates the normalized terrain based routing
        """
        _ds = np.array([[self._SQRT2, 1, self._SQRT2], [1, 1, 1], [self._SQRT2, 1, self._SQRT2]])
        _distance = _ds * self.cellsize

        _beta = np.arctan((self.altitude - self.dem_ng) / _distance) + self._RAD90
        self.tan_beta = np.tan(_beta / 2)

        self.tan_beta[self.z_delta_neighbour <= 0] = 0
        self.tan_beta[self.persistence <= 0] = 0
        self.tan_beta[1, 1] = 0
        if abs(np.sum(self.tan_beta)) > 0:
            self.r_t = self.tan_beta**self.exp / np.sum(self.tan_beta**self.exp)

    def calc_persistence(self):
        """
        calculates persistence-based routing
        """
        self.persistence = np.zeros_like(self.dem_ng)
        if self.is_start:
            self.persistence += 1
        elif self.lOfParents[0].is_start:
            self.persistence += 1
        else:
            for parent in self.lOfParents:
                dx = parent.colindex - self.colindex
                dy = parent.rowindex - self.rowindex

                self.no_flow[dy + 1, dx + 1] = 0  # 3x3 Matrix of ones, every parent gets a 0, no flow to a parent field

                maxweight = parent.z_delta
                # Old Calculation
                if dx == -1:
                    if dy == -1:
                        self.persistence[2, 2] += maxweight
                        self.persistence[2, 1] += 0.707 * maxweight
                        self.persistence[1, 2] += 0.707 * maxweight
                    if dy == 0:
                        self.persistence[1, 2] += maxweight
                        self.persistence[2, 2] += 0.707 * maxweight
                        self.persistence[0, 2] += 0.707 * maxweight
                    if dy == 1:
                        self.persistence[0, 2] += maxweight
                        self.persistence[0, 1] += 0.707 * maxweight
                        self.persistence[1, 2] += 0.707 * maxweight

                if dx == 0:
                    if dy == -1:
                        self.persistence[2, 1] += maxweight
                        self.persistence[2, 0] += 0.707 * maxweight
                        self.persistence[2, 2] += 0.707 * maxweight
                    if dy == 1:
                        self.persistence[0, 1] += maxweight
                        self.persistence[0, 0] += 0.707 * maxweight
                        self.persistence[0, 2] += 0.707 * maxweight

                if dx == 1:
                    if dy == -1:
                        self.persistence[2, 0] += maxweight
                        self.persistence[1, 0] += 0.707 * maxweight
                        self.persistence[2, 1] += 0.707 * maxweight
                    if dy == 0:
                        self.persistence[1, 0] += maxweight
                        self.persistence[0, 0] += 0.707 * maxweight
                        self.persistence[2, 0] += 0.707 * maxweight
                    if dy == 1:
                        self.persistence[0, 0] += maxweight
                        self.persistence[0, 1] += 0.707 * maxweight
                        self.persistence[1, 0] += 0.707 * maxweight

    def calc_distribution(self):
        """
        calculates flux and zdelta that is distributed to the adjacent cells

        Returns
        -----------
        tuple
            list of row, col, flux, zdelta of adjacent cells that receive flux/zdelta
        """
        self.calc_z_delta()
        self.calc_persistence()
        self.persistence *= self.no_flow
        self.calc_tanbeta()
        # print(self.persistence)

        if not self.is_start:
            # FOREST-Detrainment --> we only assume detrainment if the cell is not a start-cell
            if self.forestBool and self.forestDetrainmentBool:
                self.forest_detrainment()

            self.calc_fp_travelangle()
            self.calc_sl_travelangle()

            # FOREST-Detrainment
            # here we subtract the detrainment from the flux before moving flux to new cells.
            if self.forestBool and self.forestDetrainmentBool:
                # NOTE-TODO: check/test what the hard-coded 0.0003 does here or if this should be
                # substituted by self.flux_threshold????
                self.flux = max(0.0003, self.flux - self.detrainment)

        threshold = self.flux_threshold
        if np.sum(self.r_t) > 0:
            self.dist = (self.persistence * self.r_t) / np.sum(self.persistence * self.r_t) * self.flux

        # This handles (local) flux re-distribution if n cells are below threshold, but lager 0 and m cells are
        # still above threshold
        # NOTE: this only works if "0 < n < 8" AND "0 < m < 8", the case where
        # "0<n<8" AND "m=0" is not handled!!! (in this case flux is "lost")

        if self.fluxDistOldVersionBool:
            """
            legacy version of code (pre 01-2025) WITH BUG (is used if self.fluxDistOldVersionBool)
            here 'count' returns the number of neighbor/child cells that receive flux > 0, but
            below the flux_threshold.
            """
            count = ((0 < self.dist) & (self.dist < threshold)).sum()
        else:
            """
            default/correct version with FIXED BUG (used if self.fluxDistOldVersionBool == False)
            her 'count' returns the number of neighbor/child cells which receive
            flux >= the flux_threshold.
            """
            count = (self.dist >= threshold).sum()
        # massToDistribute ~ sum of flux of child cells below the flux_threshold
        mass_to_distribute = np.sum(self.dist[self.dist < threshold])

        if mass_to_distribute > 0 and count > 0:
            # local flux redistribution to eligible child cells
            self.dist[self.dist >= threshold] += mass_to_distribute / count
            self.dist[self.dist < threshold] = 0
        if np.sum(self.dist) != self.flux and count > 0:
            # correction/flux conservation for potential rounding losses or gains
            # (self.flux - np.sum(self.dist)) will either be negative or positive
            # depending on the direction of the rounding error
            self.dist[self.dist >= threshold] += (self.flux - np.sum(self.dist)) / count
        if count == 0:
            # if all child cells are below flux_threshold, the flux is deposited
            # TODO: check alternatives (e.g. 'global' redistribution or within generations)
            self.fluxDep = self.flux
        row_local, col_local = np.where(self.dist >= threshold)

        return (
            self.rowindex - 1 + row_local,
            self.colindex - 1 + col_local,
            self.dist[row_local, col_local],
            self.z_delta_neighbour[row_local, col_local],
        )

    def forest_detrainment(self):
        """
        linear decrease of forest effect with regard to alpha increase and kinetic energy height
        This is the detrainment routine for forest. It should reduce the routing flux of the avalanche.
        NOTE: This is more or less copied+pasted from 'foreste_detrainment' branch in avaframe/FlowPy repo
        TODO: Definitely re-check/test this function!!
        """
        _noDetrainmentEffectZdelta = self.noDetrainmentEffectZdelta

        # detrainment effect scaled to forest, 0 for non-forest
        _rest = (self.maxAddedDetrainmentForest * self.FSI)
        # rise over run (should be negative slope)
        slope = (_rest - self.minAddedDetrainmentForest) / (0 - _noDetrainmentEffectZdelta)
        # y=mx+b, where zDelta is x
        self.detrainment = max(self.minAddedDetrainmentForest, slope * self.z_delta + _rest)
