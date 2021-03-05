#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:
from concurrencytest import ConcurrentTestSuite, fork_for_tests
import sys
import unittest
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from sympy import (
    Symbol,
    Matrix,
    diag,
    symbols,
    sin,
    Piecewise,
    DiracDelta,
    Function,
    simplify,
    zeros,
    var
)
import CompartmentalSystems.helpers_reservoir as hr 
from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel


class TestHelpers_reservoir(unittest.TestCase):

    def test_compartmental_model_split_and_merge(self):
        sym_list = [
            'C_foliage',
            'C_roots',
            'C_wood',
            'C_metlit',
            'C_stlit',
            'C_fastsom',
            'C_slowsom',
            'C_passsom',
            'In_foliage',
            'In_roots',
            'In_wood',
            'cr_foliage',
            'cr_wood',
            'cr_fineroot',
            'cr_metlit',
            'cr_stlit',
            'cr_fastsom',
            'cr_slowsom',
            'cr_passsom',
            'f_foliage2metlit',
            'f_foliage2stlit',
            'f_wood2stlit',
            'f_fineroots2metlit',
            'f_fineroots2stlit',
            'f_metlit2fastsom',
            'f_stlit2fastsom',
            'f_stlit2slowsom',
            'f_fastsom2slowsom',
            'f_fastsom2passsom',
            'f_slowsom2fastsom',
            'f_slowsom2passsom',
            'f_passsom2fastsom',
        ]

        for name in sym_list:
            var(name)

        sv_set_comb = set([
            C_foliage,
			C_wood,
			C_roots,
			C_metlit,
			C_stlit,
			C_fastsom,
			C_slowsom,
		    C_passsom
        ])

        sv_set_veg = set([
            C_foliage,
			C_wood,
			C_roots
        ])

        sv_set_soil = set([
		    C_metlit,
		    C_stlit,
		    C_fastsom,
		    C_slowsom,
		    C_passsom
        ])

        in_fluxes_comb = {
            C_foliage: In_foliage,
            C_wood: In_wood,
            C_roots:In_roots
        }

        in_fluxes_veg= {
            C_foliage: In_foliage,
            C_wood: In_wood,
            C_roots:In_roots
        }

        in_fluxes_soil= {
            C_metlit:
            (
                C_foliage*f_foliage2metlit +
			    C_roots*f_fineroots2metlit
            ),
			C_stlit:
            (
                C_foliage*f_foliage2stlit +
			    C_wood*f_wood2stlit +
			    C_roots*f_fineroots2stlit
            ),
        }

        out_fluxes_comb = {
            C_foliage: C_foliage*(cr_foliage - f_foliage2metlit - f_foliage2stlit),
	        C_wood: C_wood*(cr_wood - f_wood2stlit),
	        C_roots: C_roots*(cr_fineroot - f_fineroots2metlit - f_fineroots2stlit),
	        C_metlit: C_metlit*(cr_metlit - f_metlit2fastsom),
	        C_stlit: C_stlit*(cr_stlit - f_stlit2fastsom - f_stlit2slowsom),
	        C_fastsom: C_fastsom*(cr_fastsom - f_fastsom2passsom - f_fastsom2slowsom),
	        C_slowsom: C_slowsom*(cr_slowsom - f_slowsom2fastsom - f_slowsom2passsom),
	        C_passsom: C_passsom*(cr_passsom - f_passsom2fastsom),
        }

        out_fluxes_veg = {
            C_foliage:
            (
                C_foliage*(cr_foliage - f_foliage2metlit - f_foliage2stlit) +
                C_foliage*f_foliage2metlit +
			    C_foliage*f_foliage2stlit
            ),
	        C_wood:
            (
                C_wood*(cr_wood - f_wood2stlit) +
			    C_wood*f_wood2stlit
            ),
	        C_roots:
            (
                C_roots*(cr_fineroot - f_fineroots2metlit - f_fineroots2stlit) +
			    C_roots*f_fineroots2metlit +
			    C_roots*f_fineroots2stlit
            ),
        }

        out_fluxes_soil = {
	        C_metlit: C_metlit*(cr_metlit - f_metlit2fastsom),
	        C_stlit: C_stlit*(cr_stlit - f_stlit2fastsom - f_stlit2slowsom),
	        C_fastsom: C_fastsom*(cr_fastsom - f_fastsom2passsom - f_fastsom2slowsom),
	        C_slowsom: C_slowsom*(cr_slowsom - f_slowsom2fastsom - f_slowsom2passsom),
	        C_passsom: C_passsom*(cr_passsom - f_passsom2fastsom),
        }

        internal_fluxes_comb = {
            (C_foliage, C_metlit): C_foliage*f_foliage2metlit,
			(C_foliage, C_stlit): C_foliage*f_foliage2stlit,
			(C_wood, C_stlit): C_wood*f_wood2stlit,
			(C_roots, C_metlit): C_roots*f_fineroots2metlit,
			(C_roots, C_stlit): C_roots*f_fineroots2stlit,
			(C_metlit, C_fastsom): C_metlit*f_metlit2fastsom,
			(C_stlit, C_fastsom): C_stlit*f_stlit2fastsom,
			(C_stlit, C_slowsom): C_stlit*f_stlit2slowsom,
			(C_fastsom, C_slowsom): C_fastsom*f_fastsom2slowsom,
			(C_fastsom, C_passsom): C_fastsom*f_fastsom2passsom,
			(C_slowsom, C_fastsom): C_slowsom*f_slowsom2fastsom,
			(C_slowsom, C_passsom): C_slowsom*f_slowsom2passsom,
			(C_passsom, C_fastsom): C_passsom*f_passsom2fastsom,
        }

        internal_fluxes_veg = {}

        internal_fluxes_soil = {
			(C_metlit, C_fastsom): C_metlit*f_metlit2fastsom,
			(C_stlit, C_fastsom): C_stlit*f_stlit2fastsom,
			(C_stlit, C_slowsom): C_stlit*f_stlit2slowsom,
			(C_fastsom, C_slowsom): C_fastsom*f_fastsom2slowsom,
			(C_fastsom, C_passsom): C_fastsom*f_fastsom2passsom,
			(C_slowsom, C_fastsom): C_slowsom*f_slowsom2fastsom,
			(C_slowsom, C_passsom): C_slowsom*f_slowsom2passsom,
			(C_passsom, C_fastsom): C_passsom*f_passsom2fastsom,
        }

        combined = (
            sv_set_comb,
            in_fluxes_comb,
            out_fluxes_comb,
            internal_fluxes_comb
        )

        veg = (
            sv_set_veg,
            in_fluxes_veg,
            out_fluxes_veg,
            internal_fluxes_veg
        )

        soil = (
            sv_set_soil,
            in_fluxes_soil,
            out_fluxes_soil,
            internal_fluxes_soil
        )

        self.assertEqual(
            hr.extract(combined, sv_set_veg),
            veg
        )

        self.assertEqual(
            hr.extract(combined, sv_set_soil),
            soil
        )

        veg_to_soil = {
           (pool_from, pool_to): flux
           for (pool_from, pool_to), flux in internal_fluxes_comb.items()
           if (pool_from in sv_set_veg) and (pool_to in sv_set_soil)
        }

        soil_to_veg = {
           (pool_from, pool_to): flux
           for (pool_from, pool_to), flux in internal_fluxes_comb.items()
           if (pool_from in sv_set_soil) and (pool_to in sv_set_veg)
        }

        self.assertEqual(
            hr.combine(veg, soil, veg_to_soil, soil_to_veg),
            combined
        )

        self.assertEqual(
            hr.combine(soil, veg, soil_to_veg, veg_to_soil),
            combined
        )

        # the single models overlap now when combined to a big model

        t = symbols("t")
        E = symbols("E")
        B_L, C_L = symbols("B_L C_L")
        C_R, B_R = symbols("C_R B_R")
        C_S = symbols("C_S")
        B_OS, B_OH = symbols("B_OS B_OH")
        B_TS, B_TH = symbols("B_TS B_TH")
        
        combined_sv_set = set([
            E,
            B_L, C_L,
            C_R, B_R,
#            C_S,
#            B_OS, B_OH,
#            B_TS, B_TH
        ])
        
        leaves_sv_set = set([
            E,
            B_L, C_L,
        ])

        roots_sv_set = set([
            E,
            B_R, C_R,
        ])

        GPP = symbols("GPP")
        input_fluxes = {
            E: GPP,
        }

        combined_in_fluxes = {E: GPP}
        leaves_in_fluxes = {E: GPP}
        roots_in_fluxes = {E: GPP}
        
        zeta, zeta_gluc, zeta_dw = symbols("zeta, zeta_gluc zeta_dw")
        C_gL, delta_L, R_mL, S_L = symbols("C_gL delta_L R_mL S_L")
        C_gR, delta_R, R_mR, S_R = symbols("C_gR delta_R R_mR S_R")
#        C_gW, delta_W, delta_S, R_mS, S_O = symbols("C_gW delta_W delta_S R_mS S_O")
        
        # dynamic carbon allocation coefficients to leaves, coarse roots,
        # fine roots + branches, and trunk
        f_L = symbols("f_L")
        f_R = symbols("f_R")
#        f_O = symbols("f_O")
#        f_T = symbols("f_T")
        
        # CUEs in production of leaves, roots, wood
        eta_L = zeta_dw / (zeta_gluc * C_gL)
        eta_R = zeta_dw / (zeta_gluc * C_gR)
#        eta_W = zeta_dw / (zeta_gluc * C_gW)
        
        # maintenance respiration fluxes for leaves, roots, sapwood
        ML = zeta_gluc/zeta_dw * R_mL * B_L
        MR = zeta_gluc/zeta_dw * R_mR * B_R
#        B_S_star = symbols("B_S_star")
#        MS = R_mS * B_S_star
        
        # growth respiration fluxes for leaves, roots, sapwood
        GL = f_L * C_gL/(C_gL+delta_L) * (1-eta_L) * E
        GR = f_R * C_gR/(C_gR+delta_R) * (1-eta_R) * E
#        GS = (f_O+f_T) * C_gW/(C_gW+delta_W) * (1-eta_W) * E
        
        combined_out_fluxes = {
            E: ML + GL + MR + GR,# + MS + GS,
            B_L: S_L * B_L,
            B_R: S_R * B_R,
#            B_OS: S_O * B_OS,
#            B_OH: S_O * B_OH
        }
        
        leaves_out_fluxes = {
            E: ML + GL,
            B_L: S_L * B_L,
        }

        roots_out_fluxes = {
            E: MR + GR,
            B_R: S_R * B_R,
        }

#        # heartwood production rates from sapwood
#        v_O, v_T = symbols("v_O v_T") 
        
        combined_internal_fluxes = {
            # leaves
            (E, B_L): f_L * C_gL/(C_gL+delta_L) * eta_L * E,
            (E, C_L): f_L * delta_L/(C_gL+delta_L) * E,
            (C_L, E): S_L * C_L,
        
            # fine roots
            (E, B_R): f_R * C_gR/(C_gR+delta_R) * eta_R *E,
            (E, C_R): f_R * delta_R/(C_gR+delta_R) * E,
            (C_R, E): S_R * C_R,
        
#            # sapwood production and senesence
#            (E, C_S): (f_O+f_T) * delta_W/(C_gW+delta_W) * E,
#            (C_S, E): S_O * B_OS/(B_OS+B_TS) * C_S,
#            (E, B_OS): f_O * C_gW/(C_gW+delta_W) * eta_W * E,
#            (E, B_TS): f_T * C_gW/(C_gW+delta_W) * eta_W * E,
#        
#            # coarse roots and branches heartwood production
#            (B_OS, B_OH): v_O * B_OS,
#            (C_S, B_OH): v_O * 1/zeta * B_OS/(B_OS+B_TS) * zeta_dw/zeta_gluc * C_S,
#        
#            # trunk heartwood production
#            (B_TS, B_TH): v_T * B_TS,
#            (C_S, B_TH): v_T * 1/zeta * B_TS/(B_OS+B_TS) * zeta_dw/zeta_gluc * C_S
        }
        
        leaves_internal_fluxes = {
            (E, B_L): f_L * C_gL/(C_gL+delta_L) * eta_L * E,
            (E, C_L): f_L * delta_L/(C_gL+delta_L) * E,
            (C_L, E): S_L * C_L,
        }

        roots_internal_fluxes = {
            # fine roots
            (E, B_R): f_R * C_gR/(C_gR+delta_R) * eta_R *E,
            (E, C_R): f_R * delta_R/(C_gR+delta_R) * E,
            (C_R, E): S_R * C_R,
        }

        combined = (
            combined_sv_set,
            combined_in_fluxes,
            combined_out_fluxes,
            combined_internal_fluxes
        )
        
        leaves = (
            leaves_sv_set,
            leaves_in_fluxes,
            leaves_out_fluxes,
            leaves_internal_fluxes
        )

        roots = (
            roots_sv_set,
            roots_in_fluxes,
            roots_out_fluxes,
            roots_internal_fluxes
        )
   
        with self.assertRaises(ValueError):
            hr.combine(leaves, roots, {}, {})

        intersect = (
            {E: GPP},
            {E: leaves[2][E] + roots[2][E]}
        )

        self.assertEqual(
            combined,
            hr.combine(leaves, roots, {}, {}, intersect)
        )

        supersede = (
            {},
            {E: leaves[2][E]},
            {}
        )
        self.assertEqual(
            hr.extract(
                combined,
                leaves_sv_set,
                ignore_other_pools=True,
                supersede=supersede
            ),
            leaves
        )
        
        supersede = (
            {},
            {E: roots[2][E]},
            {}
        )
        self.assertEqual(
            hr.extract(
                combined,
                roots_sv_set,
                ignore_other_pools=True,
                supersede=supersede
            ),
            roots
        )

        # more than one pool overlapping

        combined_sv_set = set([
            E,
            C_S,
            B_OS, B_OH,
            B_TS, B_TH
        ])
        
        # "other" means branches + coarse roots
        other_sv_set = set([
            E,
            C_S,
            B_OS, B_OH,
        ])

        trunk_sv_set = set([
            E,
            C_S,
            B_TS, B_TH
        ])

        GPP = symbols("GPP")
        input_fluxes = {
            E: GPP,
        }

        combined_in_fluxes = {E: GPP}
        other_in_fluxes = {E: GPP}
        trunk_in_fluxes = {E: GPP}
        
        zeta, zeta_gluc, zeta_dw = symbols("zeta, zeta_gluc zeta_dw")
        C_gW, delta_W, delta_S, R_mS, S_O = symbols("C_gW delta_W delta_S R_mS S_O")
        
        # dynamic carbon allocation coefficients to leaves, coarse roots,
        # fine roots + branches, and trunk
        f_O = symbols("f_O")
        f_T = symbols("f_T")
        
        # CUEs in production of leaves, roots, wood
        eta_W = zeta_dw / (zeta_gluc * C_gW)
        
        # maintenance respiration fluxes for leaves, roots, sapwood
        B_S_star = symbols("B_S_star")
        MS = R_mS * B_S_star
        
        # growth respiration fluxes for leaves, roots, sapwood
        GS_O = f_O * C_gW/(C_gW+delta_W) * (1-eta_W) * E
        GS_T = f_T * C_gW/(C_gW+delta_W) * (1-eta_W) * E
        
        combined_out_fluxes = {
            E: MS + GS_O + GS_T,
            B_OS: S_O * B_OS,
            B_OH: S_O * B_OH
        }
        
        other_out_fluxes = {
            E: MS + GS_O,
            B_OS: S_O * B_OS,
            B_OH: S_O * B_OH
        }

        trunk_out_fluxes = {
            E: MS + GS_T
        }

        # heartwood production rates from sapwood
        v_O, v_T = symbols("v_O v_T") 
        
        combined_internal_fluxes = {
            # sapwood production and senesence
            (E, C_S): (f_O+f_T) * delta_W/(C_gW+delta_W) * E,
            (C_S, E): S_O * B_OS/(B_OS+B_TS) * C_S,
            (E, B_OS): f_O * C_gW/(C_gW+delta_W) * eta_W * E,
            (E, B_TS): f_T * C_gW/(C_gW+delta_W) * eta_W * E,
        
            # coarse roots and branches heartwood production
            (B_OS, B_OH): v_O * B_OS,
            (C_S, B_OH): v_O * 1/zeta * B_OS/(B_OS+B_TS) * zeta_dw/zeta_gluc * C_S,
        
            # trunk heartwood production
            (B_TS, B_TH): v_T * B_TS,
            (C_S, B_TH): v_T * 1/zeta * B_TS/(B_OS+B_TS) * zeta_dw/zeta_gluc * C_S
        }
        
        other_internal_fluxes = {
            # sapwood production and senesence
            (E, C_S): f_O * delta_W/(C_gW+delta_W) * E,
            (C_S, E): S_O * B_OS/(B_OS+B_TS) * C_S,
            (E, B_OS): f_O * C_gW/(C_gW+delta_W) * eta_W * E,
        
            # coarse roots and branches heartwood production
            (B_OS, B_OH): v_O * B_OS,
            (C_S, B_OH): v_O * 1/zeta * B_OS/(B_OS+B_TS) * zeta_dw/zeta_gluc * C_S,
        }

        trunk_internal_fluxes = {
            # sapwood production and senesence
            (E, C_S): f_T * delta_W/(C_gW+delta_W) * E,
            (E, B_TS): f_T * C_gW/(C_gW+delta_W) * eta_W * E,
            
            # trunk heartwood production
            (B_TS, B_TH): v_T * B_TS,
            (C_S, B_TH): v_T * 1/zeta * B_TS/(B_OS+B_TS) * zeta_dw/zeta_gluc * C_S
        }

        combined = (
            combined_sv_set,
            combined_in_fluxes,
            combined_out_fluxes,
            combined_internal_fluxes
        )
        
        other = (
            other_sv_set,
            other_in_fluxes,
            other_out_fluxes,
            other_internal_fluxes
        )

        trunk = (
            trunk_sv_set,
            trunk_in_fluxes,
            trunk_out_fluxes,
            trunk_internal_fluxes
        )

        # test combine
   
        with self.assertRaises(ValueError):
            hr.combine(other, trunk, {}, {})

        intersect = (
            {E: GPP, C_S: 0},
            {E: MS + GS_O + GS_T, C_S: 0}
        )

        res = hr.combine(other, trunk, {}, {}, intersect)
        self.assertEqual(combined[0], res[0])
        self.assertEqual(combined[1], res[1])
        self.assertEqual(combined[2], res[2])
        for pool_from, pool_to in res[3].keys():
            c = combined[3][pool_from, pool_to]
            r = res[3][pool_from, pool_to]
            self.assertEqual(simplify(c-r), 0)

        for pool_from, pool_to in combined[3].keys():
            c = combined[3][pool_from, pool_to]
            r = res[3][pool_from, pool_to]
            self.assertEqual(simplify(c-r), 0)

        # test extract

        supersede = (
            {},
            {E: MS + GS_O},
            {(E, C_S): delta_W*f_O/(C_gW+delta_W) * E}
        )

        res = hr.extract(
           combined,
           other_sv_set,
           ignore_other_pools=True,
           supersede=supersede
       )
        self.assertEqual(other[0], res[0])
        self.assertEqual(other[1], res[1])
        self.assertEqual(other[2], res[2])
        for pool_from, pool_to in res[3].keys():
            o = other[3][pool_from, pool_to]
            r = res[3][pool_from, pool_to]
            self.assertEqual(simplify(o-r), 0)

        for pool_from, pool_to in other[3].keys():
            o = other[3][pool_from, pool_to]
            r = res[3][pool_from, pool_to]
            self.assertEqual(simplify(o-r), 0)

        supersede = (
            {},
            {E: MS + GS_T},
            {
                (E, C_S): delta_W*f_T/(C_gW+delta_W) * E,
                (C_S, E): 0
            }
        )

        res = hr.extract(
           combined,
           trunk_sv_set,
           ignore_other_pools=True,
           supersede=supersede
       )
        self.assertEqual(trunk[0], res[0])
        self.assertEqual(trunk[1], res[1])
        self.assertEqual(trunk[2], res[2])
        for pool_from, pool_to in res[3].keys():
            t = trunk[3][pool_from, pool_to]
            r = res[3][pool_from, pool_to]
            self.assertEqual(simplify(t-r), 0)

        for pool_from, pool_to in trunk[3].keys():
            t = trunk[3][pool_from, pool_to]
            r = res[3][pool_from, pool_to]
            self.assertEqual(simplify(t-r), 0)
        
        # test entire four-part model

        combined_sv_set = set([
            E,
            B_L, C_L,
            C_R, B_R,
            C_S,
            B_OS, B_OH,
            B_TS, B_TH
        ])

        combined_out_fluxes = {
            E: ML + GL + MR + GR,# + MS + GS,
            B_L: S_L * B_L,
            B_R: S_R * B_R,
            B_OS: S_O * B_OS,
            B_OH: S_O * B_OH
        }

        combined_in_fluxes = {E: GPP}

        combined_out_fluxes = {
            E: ML + GL + MR + GR + MS + GS_O + GS_T,
            B_L: S_L * B_L,
            B_R: S_R * B_R,
            B_OS: S_O * B_OS,
            B_OH: S_O * B_OH
        }
        
        combined_internal_fluxes = {
            # leaves
            (E, B_L): f_L * C_gL/(C_gL+delta_L) * eta_L * E,
            (E, C_L): f_L * delta_L/(C_gL+delta_L) * E,
            (C_L, E): S_L * C_L,
        
            # fine roots
            (E, B_R): f_R * C_gR/(C_gR+delta_R) * eta_R *E,
            (E, C_R): f_R * delta_R/(C_gR+delta_R) * E,
            (C_R, E): S_R * C_R,
        
            # sapwood production and senesence
            (E, C_S): (f_O+f_T) * delta_W/(C_gW+delta_W) * E,
            (C_S, E): S_O * B_OS/(B_OS+B_TS) * C_S,
            (E, B_OS): f_O * C_gW/(C_gW+delta_W) * eta_W * E,
            (E, B_TS): f_T * C_gW/(C_gW+delta_W) * eta_W * E,
        
            # coarse roots and branches heartwood production
            (B_OS, B_OH): v_O * B_OS,
            (C_S, B_OH): v_O * 1/zeta * B_OS/(B_OS+B_TS) * zeta_dw/zeta_gluc * C_S,
        
            # trunk heartwood production
            (B_TS, B_TH): v_T * B_TS,
            (C_S, B_TH): v_T * 1/zeta * B_TS/(B_OS+B_TS) * zeta_dw/zeta_gluc * C_S
        }

        combined = (
            combined_sv_set,
            combined_in_fluxes,
            combined_out_fluxes,
            combined_internal_fluxes
        )

        intersect = (
            {E: GPP},
            {E: ML + GL + MR + GR}
        )
        LR = hr.combine(leaves, roots, {}, {}, intersect)
        intersect = (
            {E: GPP},
            {E: ML + GL + MR + GR + MS + GS_O}
        )
        LRO = hr.combine(LR, other, {}, {}, intersect)
        intersect = (
            {E: GPP, C_S: 0},
            {E: ML + GL + MR + GR + MS + GS_O + GS_T, C_S: 0}
        )
        LROT = hr.combine(LRO, trunk, {}, {}, intersect)
        self.assertEqual(combined[0], LROT[0])
        self.assertEqual(combined[1], LROT[1])
        self.assertEqual(combined[2], LROT[2])

        for pool_from, pool_to in LROT[3].keys():
            c = combined[3][pool_from, pool_to]
            r = LROT[3][pool_from, pool_to]
            self.assertEqual(simplify(c-r), 0)

        for pool_from, pool_to in combined[3].keys():
            c = combined[3][pool_from, pool_to]
            r = LROT[3][pool_from, pool_to]
            self.assertEqual(simplify(c-r), 0)




#        # in case sv_set_veg and sv_set_soil are not disjoint
#        # for this purpose we invent another pool that will be
#        # part of both systems
#        for s in ('C_labile','f_labile2metlit','cr_labile', 'f_labile2stlit'):
#            var(s)
#
#        sv_set_comb2 = sv_set_comb.union([C_foliage])
#        sv_set_veg2 = sv_set_veg.union([C_foliage])
#
#        sv_set_soil2 = sv_set_soil2.union([ C_labile ])
#       
#        in_fluxes_comb = {
#            C_foliage: In_foliage,
#            C_wood: In_wood,
#            C_roots:In_roots
#        }
#
#        in_fluxes_veg= {
#            C_foliage: In_foliage,
#            C_wood: In_wood,
#            C_roots:In_roots
#        }
#
#        in_fluxes_soil= {
#            C_metlit:
#            (
#                C_foliage*f_foliage2metlit +
#			    C_roots*f_fineroots2metlit
#            ),
#			C_stlit:
#            (
#                C_foliage*f_foliage2stlit +
#			    C_wood*f_wood2stlit +
#			    C_roots*f_fineroots2stlit
#            ),
#        }
#        out_fluxes_comb = {
#            C_foliage: C_foliage*(cr_foliage - f_foliage2metlit - f_foliage2stlit),
#	        C_wood: C_wood*(cr_wood - f_wood2stlit),
#	        C_roots: C_roots*(cr_fineroot - f_fineroots2metlit - f_fineroots2stlit),
#	        C_labile: C_labile*(cr_labile- f_labile2metlit - f_labile2stlit), #new
#	        C_metlit: C_metlit*(cr_metlit - f_metlit2fastsom),
#	        C_stlit: C_stlit*(cr_stlit - f_stlit2fastsom - f_stlit2slowsom),
#	        C_fastsom: C_fastsom*(cr_fastsom - f_fastsom2passsom - f_fastsom2slowsom),
#	        C_slowsom: C_slowsom*(cr_slowsom - f_slowsom2fastsom - f_slowsom2passsom),
#	        C_passsom: C_passsom*(cr_passsom - f_passsom2fastsom),
#        }
#
#        out_fluxes_veg = {
#            C_foliage:
#            (
#                C_foliage*(cr_foliage - f_foliage2metlit - f_foliage2stlit - f_foliage2labile) +
#                C_foliage*f_foliage2metlit +
#			    C_foliage*f_foliage2stlit +
#			    C_foliage*f_foliage2labile #new
#            ),
#	        C_wood:
#            (
#                C_wood*(cr_wood - f_wood2stlit) +
#			    C_wood*f_wood2stlit
#            ),
#	        C_roots:
#            (
#                C_roots*(cr_fineroot - f_fineroots2metlit - f_fineroots2stlit) +
#			    C_roots*f_fineroots2metlit +
#			    C_roots*f_fineroots2stlit
#            ),
#	        C_labile: #new
#            (
#                C_labile*(cr_labile- f_labile2metlit - f_labile2stlit) +
#			    C_labile*f_labile2metlit +
#			    C_labile*f_labile2stlit
#            ),
#        }
#
#        out_fluxes_soil = {
#	        C_labile: #new
#            (
#                C_labile*(cr_labile- f_labile2metlit - f_labile2stlit) +
#			    C_labile*f_labile2metlit +
#			    C_labile*f_labile2stlit
#            ),
#	        C_metlit: C_metlit*(cr_metlit - f_metlit2fastsom),
#	        C_stlit: C_stlit*(cr_stlit - f_stlit2fastsom - f_stlit2slowsom),
#	        C_fastsom: C_fastsom*(cr_fastsom - f_fastsom2passsom - f_fastsom2slowsom),
#	        C_slowsom: C_slowsom*(cr_slowsom - f_slowsom2fastsom - f_slowsom2passsom),
#	        C_passsom: C_passsom*(cr_passsom - f_passsom2fastsom),
#        }
#        internal_fluxes_comb = {
#            (C_foliage, C_metlit): C_foliage*f_foliage2metlit,
#			(C_foliage, C_stlit): C_foliage*f_foliage2stlit,
#			(C_foliage, C_labile): C_foliage*f_foliage2labile, #new
#            (C_labile, C_metlit): C_labile*f_labile2metlit, #new 
#            (C_labile, C_stlit): C_labile*f_labile2stlit, #new 
#			(C_wood, C_stlit): C_wood*f_wood2stlit,
#			(C_roots, C_metlit): C_roots*f_fineroots2metlit,
#			(C_roots, C_stlit): C_roots*f_fineroots2stlit,
#			(C_metlit, C_fastsom): C_metlit*f_metlit2fastsom,
#			(C_stlit, C_fastsom): C_stlit*f_stlit2fastsom,
#			(C_stlit, C_slowsom): C_stlit*f_stlit2slowsom,
#			(C_fastsom, C_slowsom): C_fastsom*f_fastsom2slowsom,
#			(C_fastsom, C_passsom): C_fastsom*f_fastsom2passsom,
#			(C_slowsom, C_fastsom): C_slowsom*f_slowsom2fastsom,
#			(C_slowsom, C_passsom): C_slowsom*f_slowsom2passsom,
#			(C_passsom, C_fastsom): C_passsom*f_passsom2fastsom,
#        }
#
#        internal_fluxes_veg = {}
#
#        internal_fluxes_soil = {
#			(C_labile, C_metlit): C_labile*f_labile2metlit, #new
#			(C_labile, C_stlit): C_labile*f_labile2stlit, #new
#			(C_metlit, C_fastsom): C_metlit*f_metlit2fastsom,
#			(C_stlit, C_fastsom): C_stlit*f_stlit2fastsom,
#			(C_stlit, C_slowsom): C_stlit*f_stlit2slowsom,
#			(C_fastsom, C_slowsom): C_fastsom*f_fastsom2slowsom,
#			(C_fastsom, C_passsom): C_fastsom*f_fastsom2passsom,
#			(C_slowsom, C_fastsom): C_slowsom*f_slowsom2fastsom,
#			(C_slowsom, C_passsom): C_slowsom*f_slowsom2passsom,
#			(C_passsom, C_fastsom): C_passsom*f_passsom2fastsom,
#        }
#
#        combined = (
#            sv_set_comb,
#            in_fluxes_comb,
#            out_fluxes_comb,
#            internal_fluxes_comb
#        )


        
    def test_matrices_to_fluxes_and_back(self):
        # f = xi*T*N*C + u
        t, C_1, C_2, C_3, gamma, k_1, k_2, k_3, t_12, t_13, t_21, t_23, t_31, t_32, u_1, u_2, u_3, xi \
            = symbols('t C_1 C_2 C_3 gamma k_1 k_2 k_3 t_12 t_13 t_21 t_23 t_31 t_32 u_1 u_2 u_3 xi')
        C = Matrix(3,1, [C_1, C_2, C_3])
        u = Matrix(3,1, [u_1, u_2, u_3])
        xi = gamma
        T = Matrix([[  -1, t_12, t_13],
                    [t_21,   -1, t_23],
                    [t_31, t_32,   -1]])
        N = diag(k_1, k_2, k_3)
        R = gamma*N
        B = T*R
        
        # outfluxes
        ofbi = hr.out_fluxes_by_index(C, B)
        for key, val in ofbi.items():
            with self.subTest():
                ref_val = {
                    0: gamma*k_1*(1-t_21-t_31)*C_1,
                    1: gamma*k_2*(1-t_12-t_32)*C_2,
                    2: gamma*k_3*(1-t_13-t_23)*C_3
                }[key]
                self.assertEqual(simplify(val-ref_val), 0)

        ofbs = hr.out_fluxes_by_symbol(C, B)
        for key, val in ofbs.items():
            with self.subTest():
                ref_val = {
                    C_1: gamma*k_1*(1-t_21-t_31)*C_1,
                    C_2: gamma*k_2*(1-t_12-t_32)*C_2,
                    C_3: gamma*k_3*(1-t_13-t_23)*C_3
                }[key]
                self.assertEqual(simplify(val-ref_val), 0)

        # internalfluxes
        ifbi = hr.internal_fluxes_by_index(C, B)
        for key, val in ifbi.items():
            with self.subTest():
                ref_val = {
                    (0, 1): gamma*t_21*k_1*C_1, (0, 2): gamma*t_31*k_1*C_1,
                    (1, 0): gamma*t_12*k_2*C_2, (1, 2): gamma*t_32*k_2*C_2,
                    (2, 0): gamma*t_13*k_3*C_3, (2, 1): gamma*t_23*k_3*C_3
                }[key]
                self.assertEqual(simplify(val-ref_val), 0)

        ifbs = hr.internal_fluxes_by_symbol(C, B)
        for key, val in ifbs.items():
            with self.subTest():
                ref_val = {
                    (C_1, C_2): gamma*t_21*k_1*C_1, (C_1, C_3): gamma*t_31*k_1*C_1,
                    (C_2, C_1): gamma*t_12*k_2*C_2, (C_2, C_3): gamma*t_32*k_2*C_2,
                    (C_3, C_1): gamma*t_13*k_3*C_3, (C_3, C_2): gamma*t_23*k_3*C_3
                }[key]
                self.assertEqual(simplify(val-ref_val), 0)

        # ## test backward conversion to compartmental matrix 
        R2 = hr.release_operator_1(
            out_fluxes_by_index=ofbi,
            internal_fluxes_by_index=ifbi,
            state_vector=C
        )

        xi2 = hr.factor_out_from_matrix(R2)

        T2 = hr.tranfer_operator_1(
            out_fluxes_by_index=ofbi,
            internal_fluxes_by_index=ifbi,
            state_vector=C
        )
        B2= hr.compartmental_matrix_1(
            out_fluxes_by_index=ofbi,
            internal_fluxes_by_index=ifbi,
            state_vector=C
        )


        R3 = hr.release_operator_2(
            out_fluxes_by_symbol=ofbs,
            internal_fluxes_by_symbol=ifbs,
            state_vector=C
        )
        xi3 = hr.factor_out_from_matrix(R3)

        T3 = hr.transfer_operator_2(
            out_fluxes_by_symbol=ofbs,
            internal_fluxes_by_symbol=ifbs,
            state_vector=C
        )
        B3= hr.compartmental_matrix_2(
            out_fluxes_by_symbol=ofbs,
            internal_fluxes_by_symbol=ifbs,
            state_vector=C
        )
        self.assertEqual(simplify(R-R2), zeros(*R.shape))
        self.assertEqual(simplify(R-R3), zeros(*R.shape))
        self.assertEqual(simplify(T-T2), zeros(*R.shape))
        self.assertEqual(simplify(T-T3), zeros(*R.shape))
        self.assertEqual(simplify(B-B2), zeros(*R.shape))
        self.assertEqual(simplify(B-B3), zeros(*R.shape))

        self.assertEqual(simplify(B-T2*R2), zeros(*R.shape))
        self.assertEqual(simplify(N-R2/xi2), zeros(*R.shape))

        self.assertEqual(simplify(T3*R3-B3), zeros(*R.shape))
        self.assertEqual(simplify(N-R3/xi3), zeros(*R.shape))


    def test_in_fluxes_by_index_and_symbol(self):
        C_1, C_2, C_3, u_1, u_2, u_3 = symbols('C_1 C_2 C_3 u_1 u_2 u_3')
        C = Matrix(3, 1, [C_1, C_2, C_3])
        u = Matrix(3, 1, [u_1, u_2, u_3])
        self.assertEqual(
            hr.in_fluxes_by_index(C, u),
            {0: u_1, 1: u_2, 2: u_3}
        )
        u_by_sym = hr.in_fluxes_by_symbol(C, u)
        self.assertEqual(
            u_by_sym,
            {C_1: u_1, C_2: u_2, C_3: u_3}
        )
        # test backward conversion to matrix
        u2 = hr.in_or_out_flux_tuple(state_vector=C, in_or_out_fluxes_by_symbol=u_by_sym)
        self.assertEqual(simplify(u-u2), zeros(*u.shape))

    def test_out_fluxes_by_index_and_symbol(self):
        t,C_1, C_2, C_3, k_1, k_2, k_3, a_12, a_13, a_21, a_23, a_31, a_32, gamma\
        = symbols('t,C_1 C_2 C_3 k_1 k_2 k_3 a_12 a_13 a_21 a_23 a_31 a_32 gamma')
        C = Matrix(3, 1, [C_1, C_2, C_3])
        B = gamma*Matrix([
                [-k_1, a_12, a_13],
                [a_21, -k_2, a_23],
                [a_31, a_32, -k_3]
            ])

        ofbi = hr.out_fluxes_by_index(C, B)
        for key, val in ofbi.items():
            with self.subTest():
                ref_val = {
                    0: C_1*(-a_21*gamma-a_31*gamma+k_1*gamma),
                    1: C_2*(-a_12*gamma-a_32*gamma+k_2*gamma),
                    2: C_3*(-a_13*gamma-a_23*gamma+k_3*gamma)
                }[key]
                self.assertEqual(simplify(val-ref_val), 0)

        ofbi = hr.out_fluxes_by_symbol(C, B)
        for key, val in ofbi.items():
            with self.subTest():
                ref_val = {
                    C_1: C_1*(-a_21*gamma-a_31*gamma+k_1*gamma),
                    C_2: C_2*(-a_12*gamma-a_32*gamma+k_2*gamma),
                    C_3: C_3*(-a_13*gamma-a_23*gamma+k_3*gamma)
                }[key]
                self.assertEqual(simplify(val-ref_val), 0)

    def test_internal_fluxes_by_index_and_symbol(self):
        t,C_1, C_2, C_3, k_1, k_2, k_3, a_12, a_13, a_21, a_23, a_31, a_32, gamma\
        = symbols('t,C_1 C_2 C_3 k_1 k_2 k_3 a_12 a_13 a_21 a_23 a_31 a_32 gamma')
        C = Matrix(3,1, [C_1, C_2, C_3])
        B = gamma*Matrix([
            [-k_1, a_12, a_13],
            [a_21, -k_2, a_23],
            [a_31, a_32, -k_3]
        ])

        self.assertEqual(
            hr.internal_fluxes_by_index(C, B),
            {
                (0, 1): gamma*a_21*C_1, (0, 2): gamma*a_31*C_1,
                (1, 0): gamma*a_12*C_2, (1, 2): gamma*a_32*C_2,
                (2, 0): gamma*a_13*C_3, (2, 1): gamma*a_23*C_3
            }
        )
        
        self.assertEqual(
            hr.internal_fluxes_by_symbol(C, B),
            {
                (C_1, C_2): gamma*a_21*C_1, (C_1, C_3): gamma*a_31*C_1,
                (C_2, C_1): gamma*a_12*C_2, (C_2, C_3): gamma*a_32*C_2,
                (C_3, C_1): gamma*a_13*C_3, (C_3, C_2): gamma*a_23*C_3
            }
        )
             
        # ## test backward conversion to compartmental matrix 
        # B2 = rm.compartmental_matrix
        # u2 = rm.external_inputs
        # self.assertEqual(simplify(u-u2), zeros(*u.shape))
        # self.assertEqual(simplify(B-B2), zeros(*B.shape))


    def test_numerical_function_from_expression(self):
        C_0, C_1, C_2 = symbols('C_0 C_1 C_2')
        t = Symbol('t')
        u_0_sym = Function('u_0')
        u_2_sym = Function('u_2')
        
        u_0_expr = u_0_sym(C_0, C_1, t)
        u_2_expr = u_2_sym(t)
        
        X = Matrix([C_0, C_1, C_2])
        t_min, t_max = 0, 10
        symbolic_input_fluxes = {0: u_0_expr, 2: u_2_expr}
        
        def u0_func(C_0_val, C_1_val, t_val):
            return C_0_val+ C_1_val + t_val
        
        def u2_func(t_val):
            return t_val
        parameter_dict={}
        func_set = {u_0_expr: u0_func, u_2_expr: u2_func}
        
        tup = (C_0,C_1) + (t,)
        u_0_func = hr.numerical_function_from_expression(u_0_expr,tup,parameter_dict,func_set)
        self.assertEqual(u_0_func(1,2,3),1+2+3)
        
        tup = (t,)
        u_2_func=hr.numerical_function_from_expression(u_2_expr,tup,parameter_dict,func_set)
        self.assertEqual(u_2_func(2),2)
        
        # wrong tup: C_1 is not necessary but it does not hurt
        # this behavior is convinient to make everything a variable of 
        # ALL statevariables and time 
        tup = (C_1,t,)
        u_2_func=hr.numerical_function_from_expression(u_2_expr,tup,parameter_dict,func_set)
        # the superflous first argument just does not have any influence
        self.assertEqual(u_2_func(1002103413131,2),2)
        
        # wrong tup: C_0 is missing but necessary
        # this is a real error
        tup = (C_1,t)
        with self.assertRaises(Exception) as e:
            u_0_func=hr.numerical_function_from_expression(u_0_expr,tup,parameter_dict,func_set)

    def test_func_subs(self):
        # t is in the third position
        C_0, C_1  = symbols('C_0 C_1')
        t= Symbol('t')  
        u_0_expr = Function('u_0')(C_0, C_1, t)
        def u_0_func(C_0,C_1,t):
            return (C_0*C_1)**t

        ref=u_0_func(1,2,3)
        u_0_part=hr.func_subs(t,u_0_expr,u_0_func,3)
        res=u_0_part(1,2)
        self.assertEqual(ref,res)
       
        # this time t is in the first position
        u_0_expr = Function('u_0')(t,C_0, C_1)
        def u_0_func(C_0,C_1,t):
            return (C_0*C_1)**t

        ref=u_0_func(1,2,3)
        u_0_part=hr.func_subs(t,u_0_expr,u_0_func,1)
        res=u_0_part(2,3)
        self.assertEqual(ref,res)
    

    def test_parse_input_function(self):
        t = symbols('t')
        u = (1+sin(t) + DiracDelta(2-t) 
            +5*DiracDelta(3-t) + Piecewise((1,t<=1), (2,True)))
        impulses, jump_times = hr.parse_input_function(u, t)
        self.assertEqual(impulses, [{'time': 2, 'intensity': 1}, 
                                    {'time': 3,'intensity': 5}])
        self.assertEqual(jump_times, [1,2,3])

        u = 1
        impulses, jump_times = hr.parse_input_function(u, t)
        self.assertEqual(impulses, [])
        self.assertEqual(jump_times, [])


    def test_factor_out_from_matrix(self):
        gamma, k_1 = symbols('gamma k_1')
        M = Matrix([[12*gamma*k_1, 0], [3*gamma**2, 15*gamma]])
        cf = hr.factor_out_from_matrix(M)

        self.assertEqual(cf, 3*gamma)


    def test_melt(self):
        ndarr = np.arange(24).reshape(3,4,2)
        
        a_ref = [[0,  0,  0,  0], 
                 [0,  0,  1,  1],
                 [0,  1,  0,  2],
                 [0,  1,  1,  3],
                 [0,  2,  0,  4],
                 [0,  2,  1,  5],
                 [0,  3,  0,  6],
                 [0,  3,  1,  7],
                 [1,  0,  0,  8],
                 [1,  0,  1,  9],
                 [1,  1,  0, 10],
                 [1,  1,  1, 11],
                 [1,  2,  0, 12],
                 [1,  2,  1, 13],
                 [1,  3,  0, 14],
                 [1,  3,  1, 15],
                 [2,  0,  0, 16],
                 [2,  0,  1, 17],
                 [2,  1,  0, 18],
                 [2,  1,  1, 19],
                 [2,  2,  0, 20],
                 [2,  2,  1, 21],
                 [2,  3,  0, 22],
                 [2,  3,  1, 23]]
        ref = np.array(a_ref).reshape((24,4))
        melted = hr.melt(ndarr)
        self.assertTrue(np.all(melted==ref))

        ages = np.linspace(0,4,3)
        times = np.linspace(0,0.75,4)
        pools = [0,1]

        a_ref = [[0.,     0.  ,   0.,     0.  ], 
                 [0.,     0.  ,   1.,     1.  ],
                 [0.,     0.25,   0.,     2.  ],
                 [0.,     0.25,   1.,     3.  ],
                 [0.,     0.5 ,   0.,     4.  ],
                 [0.,     0.5 ,   1.,     5.  ],
                 [0.,     0.75,   0.,     6.  ],
                 [0.,     0.75,   1.,     7.  ],
                 [2.,     0.  ,   0.,     8.  ],
                 [2.,     0.  ,   1.,     9.  ],
                 [2.,     0.25,   0.,    10.  ],
                 [2.,     0.25,   1.,    11.  ],
                 [2.,     0.5 ,   0.,    12.  ],
                 [2.,     0.5 ,   1.,    13.  ],
                 [2.,     0.75,   0.,    14.  ],
                 [2.,     0.75,   1.,    15.  ],
                 [4.,     0.  ,   0.,    16.  ],
                 [4.,     0.  ,   1.,    17.  ],
                 [4.,     0.25,   0.,    18.  ],
                 [4.,     0.25,   1.,    19.  ],
                 [4.,     0.5 ,   0.,    20.  ],
                 [4.,     0.5 ,   1.,    21.  ],
                 [4.,     0.75,   0.,    22.  ],
                 [4.,     0.75,   1.,    23.  ]]
        melted = hr.melt(ndarr, [ages, times, pools])
        ref = np.array(a_ref).reshape((24,4))
        self.assertTrue(np.all(melted==ref))

    def test_MH_sampling(self):
        # fixme:
        # fails sometimes because of bad random values
        # perhaps a fixed random seed would help


        # uniform distribution on [0,1]
        PDF = lambda x: 1 if x>=0 and x<=1 else 0
        rvs = hr.MH_sampling(200000, PDF)
        self.assertTrue(abs(np.mean(rvs)-0.5) < 0.05)
        self.assertTrue(abs(np.std(rvs, ddof=1)-np.sqrt(1/12)) < 0.05)

        # exponential distribution
        l = 2
        PDF = lambda x: l*np.exp(-l*x)
        rvs = hr.MH_sampling(100000, PDF, start = 1/l)
        #print(rvs)
        #print(np.mean(rvs))
        #count, bins, ignored = plt.hist(rvs, 100, normed=True)
        #ts = np.linspace(0, 5, 101)
        #plt.plot(ts, [PDF(t) for t in ts], color='red')
        #plt.show()
        self.assertTrue(abs(np.mean(rvs)-1/l) < 0.05)
        self.assertTrue(abs(np.std(rvs, ddof=1)-1/l) < 0.05)


    def test_stride(self):
        data = np.array([i*10+np.linspace(0,9,10) for i in range(20)])
        strided_data = hr.stride(data, (2,4))
        ref = np.array([[   0.,    4.,    8.,    9.], 
                        [  20.,   24.,   28.,   29.],
                        [  40.,   44.,   48.,   49.],
                        [  60.,   64.,   68.,   69.],
                        [  80.,   84.,   88.,   89.],
                        [ 100.,  104.,  108.,  109.],
                        [ 120.,  124.,  128.,  129.],
                        [ 140.,  144.,  148.,  149.],
                        [ 160.,  164.,  168.,  169.],
                        [ 180.,  184.,  188.,  189.],
                        [ 190.,  194.,  198.,  199.]])
        self.assertTrue(np.all(strided_data==ref))

        times = np.linspace(0,100,101)
        strided_times = hr.stride(times, 25)
        self.assertTrue(np.all(strided_times==np.array([0, 25, 50, 75, 100])))
        
        strided_times = hr.stride(times, 1)
        self.assertTrue(np.all(strided_times==times))

    def test_is_compartmental(self):
        k=Symbol('k',positive=True, real=True)
        l=Symbol('l',positive=True, real=True)
        M=Matrix([[-k,0],[0,-l]])
        self.assertTrue(hr.is_compartmental(M))



################################################################################


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.discover(".",pattern=__file__)
    # Run same tests across 16 processes
    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(16))
    runner = unittest.TextTestRunner()
    res=runner.run(concurrent_suite)
    # to let the buildbot fail we set the exit value !=0 if either a failure or 
    # error occurs
    if (len(res.errors)+len(res.failures))>0:
        sys.exit(1)

