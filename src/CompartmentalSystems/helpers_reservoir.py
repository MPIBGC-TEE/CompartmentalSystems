# vim:set ff=unix expandtab ts=4 sw=4:
from typing import Callable, Tuple, Sequence, Set, Dict, Iterator
from functools import reduce, lru_cache, _CacheInfo, _lru_cache_wrapper
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import inspect
from collections import namedtuple
from numbers import Number
from scipy.integrate import odeint, quad
from scipy.interpolate import lagrange
from scipy.optimize import brentq
from scipy.stats import norm
from string import Template
from sympy import (
    gcd,
    diag,
    sympify,
    lambdify,
    DiracDelta,
    solve,
    Matrix,
    Symbol,
    Expr,
    diff,
    simplify,
    eye,
    ImmutableMatrix,
    latex
)

from sympy.polys.polyerrors import PolynomialError
from sympy.core.function import UndefinedFunction, Function, sympify
from sympy import Symbol
from collections.abc import Iterable
import networkx as nx
import igraph as ig
import itertools as it
from frozendict import frozendict
from .BlockOde import BlockOde
from .myOdeResult import solve_ivp_pwc
from copy import deepcopy


ALPHA_14C = 1.18e-12
DECAY_RATE_14C_YEARLY = np.log(2) / 5730
DECAY_RATE_14C_DAILY = DECAY_RATE_14C_YEARLY / 365.25


def combine(m1, m2, m1_to_m2, m2_to_m1, intersect=False):
    m1_sv_set, m1_in_fluxes, m1_out_fluxes, m1_internal_fluxes = m1
    m2_sv_set, m2_in_fluxes, m2_out_fluxes, m2_internal_fluxes = m2

    intersect_sv_set = m1_sv_set & m2_sv_set
    if intersect_sv_set and not intersect:
        raise(ValueError("How to handle pools %s?" % str(intersect_sv_set)))

    sv_set = m1_sv_set | m2_sv_set

    # create external in_fluxes
    in_fluxes = dict()

    # add all external in_fluxes of m1
    for k, v in m1_in_fluxes.items():
        if k in in_fluxes.keys():
            in_fluxes[k] += v
        else:
            in_fluxes[k] = v

    # remove flux from in_flux if it becomes internal
    for pool_to in in_fluxes.keys():
        for (pool_from, a), flux in m2_to_m1.items():
            if a == pool_to:
                in_fluxes[pool_to] -= flux

    # add all external in_fluxes of m2
    for k, v in m2_in_fluxes.items():
        if k in in_fluxes.keys():
            in_fluxes[k] += v
        else:
            in_fluxes[k] = v

    # remove flux from in_flux if it becomes internal
    for pool_to in in_fluxes.keys():
        for (pool_from, a), flux in m1_to_m2.items():
            if a == pool_to:
                in_fluxes[pool_to] -= flux

    # create external out_fluxes
    out_fluxes = dict()

    # add all external out_fluxes from m1
    for k, v in m1_out_fluxes.items():
        if k in out_fluxes.keys():
            out_fluxes[k] += v
        else:
            out_fluxes[k] = v

    # remove flux from out_flux if it becomes internal
    for pool_from in out_fluxes.keys():
        for (a, pool_to), flux in m1_to_m2.items():
            if a == pool_from:
                out_fluxes[pool_from] -= flux

    # add all external out_fluxes from m2
    for k, v in m2_out_fluxes.items():
        if k in out_fluxes.keys():
            out_fluxes[k] += v
        else:
            out_fluxes[k] = v

    # remove flux from out_flux if it becomes internal
    for pool_from in out_fluxes.keys():
        for (a, pool_to), flux in m2_to_m1.items():
            if a == pool_from:
                out_fluxes[pool_from] -= flux

    # create internal fluxes
    internal_fluxes = dict()

    dicts = [m1_internal_fluxes, m2_internal_fluxes, m1_to_m2, m2_to_m1]
    for d in dicts:
        for k, v in d.items():
            if k in internal_fluxes.keys():
                internal_fluxes[k] += v
            else:
                internal_fluxes[k] = v
    
    # overwrite in_fluxes and out_fluxes for intersection pools
    for sv in intersect_sv_set:
        in_fluxes[sv] = intersect[0][sv]
        out_fluxes[sv] = intersect[1][sv]

    clean_in_fluxes = {k: v for k, v in in_fluxes.items() if v != 0}
    clean_out_fluxes = {k: v for k, v in out_fluxes.items() if v != 0}
    clean_internal_fluxes = {k: v for k, v in internal_fluxes.items() if v != 0}

    return sv_set, clean_in_fluxes, clean_out_fluxes, clean_internal_fluxes


def extract(m, sv_set, ignore_other_pools=False, supersede=False):
    m_sv_set, m_in_fluxes, m_out_fluxes, m_internal_fluxes = m
    assert(sv_set.issubset(m_sv_set))

    in_fluxes = {pool: flux for pool, flux in m_in_fluxes.items() if pool in sv_set}
    out_fluxes = {pool: flux for pool, flux in m_out_fluxes.items() if pool in sv_set}
    internal_fluxes = {
        (pool_from, pool_to): flux 
        for (pool_from, pool_to), flux in m_internal_fluxes.items() 
        if (pool_from in sv_set) and (pool_to in sv_set)
    }

    for (pool_from, pool_to), flux in m_internal_fluxes.items():
        # internal flux becomes influx if not ignored
        if not ignore_other_pools:
            if (pool_from not in sv_set) and (pool_to in sv_set):
                if pool_to in in_fluxes.keys():
                    in_fluxes[pool_to] += flux
                else:
                    in_fluxes[pool_to] = flux
        
        # internal flux becomes outflux if not ignored
        if not ignore_other_pools:
            if (pool_from in sv_set) and (pool_to not in sv_set):
                if pool_from in out_fluxes.keys():
                    out_fluxes[pool_from] += flux
                else:
                    out_fluxes[pool_from] = flux

    # overwrite in_fluxes and out_fluxes if desired
    if supersede:
        for sv, flux in supersede[0].items():
            in_fluxes[sv] = flux
        for sv, flux in supersede[1].items():
            out_fluxes[sv] = flux
        for (pool_from, pool_to), flux in supersede[2].items():
            internal_fluxes[pool_from, pool_to] = flux

    clean_in_fluxes = {k: v for k, v in in_fluxes.items() if v != 0}
    clean_out_fluxes = {k: v for k, v in out_fluxes.items() if v != 0}
    clean_internal_fluxes = {k: v for k, v in internal_fluxes.items() if v != 0}

    return sv_set, clean_in_fluxes, clean_out_fluxes, clean_internal_fluxes


def nxgraphs(
    state_vector: Tuple[Symbol],
    in_fluxes: Dict[Symbol, Expr],
    internal_fluxes: Dict[Tuple[Symbol, Symbol], Expr],
    out_fluxes: Dict[Symbol, Expr],
) -> nx.DiGraph:
    G = nx.DiGraph()
    node_names = [str(sv) for sv in state_vector]
    G.add_nodes_from(node_names)
    in_flux_targets, out_flux_sources = [
        [str(k) for k in d.keys()]
        for d in (in_fluxes, out_fluxes)
    ]

    virtual_in_flux_sources = ["virtual_in_" + str(t) for t in in_flux_targets]
    for n in virtual_in_flux_sources:
        G.add_node(n, virtual=True)
    for i in range(len(in_flux_targets)):
        G.add_edge(
            virtual_in_flux_sources[i],
            in_flux_targets[i],
            expr=in_fluxes[Symbol(in_flux_targets[i])]
        )

    virtual_out_flux_targets = [
        "virtual_out_" + str(t)
        for t in out_flux_sources
    ]
    for n in virtual_out_flux_targets:
        G.add_node(n, virtual=True)
    for i in range(len(out_flux_sources)):
        G.add_edge(
            out_flux_sources[i], 
            virtual_out_flux_targets[i],
            expr=out_fluxes[Symbol(out_flux_sources[i])]
        )

    #for c in internal_connections:
    for c in internal_fluxes.keys():
        G.add_edge(str(c[0]), str(c[1]),expr=internal_fluxes[c])


    return G


def igraph_func_plot(
    Gnx: nx.DiGraph, # note that Gnx has to have a 'virtual' attribute on some verteces 
    node_color_func: Callable[[nx.DiGraph,str],str],
    edge_color_func: Callable[[nx.DiGraph,str,str],str],
    path=None
) -> ig.drawing.Plot:

    G = ig.Graph.from_networkx(Gnx)

    vertex_size = [1 if v['virtual'] else 50 for v in G.vs]
    vertex_color= [node_color_func(Gnx,v) for v in Gnx.nodes]
    vertex_label = [v['_nx_name'] if not v['virtual'] else '' for v in G.vs]

    edge_color = [edge_color_func(Gnx,s,t) for s, t in Gnx.edges]
    edge_label= [Gnx.get_edge_data(s,t)['expr'] for s, t in Gnx.edges]


    layout = G.layout('sugiyama')
    if path is None:
        pl = ig.plot(
            G,
            layout=layout,
            vertex_size=vertex_size,
            vertex_label=vertex_label,
            vertex_color=vertex_color,
            vertex_label_size=9,
            edge_color=edge_color,
            edge_label=edge_label,
            edge_label_size=4,
        )
    else:
        pl = ig.plot(
            G,
            layout=layout,
            vertex_size=vertex_size,
            vertex_label=vertex_label,
            vertex_color=vertex_color,
            vertex_label_size=9,
            edge_color=edge_color,
            #edge_label=edge_label,
            edge_label_size=4,
            target=str(path)
        )
    return pl

def igraph_plot(
    state_vector: Matrix,
    in_fluxes: frozendict,
    internal_fluxes: frozendict,
    out_fluxes: frozendict,
) -> ig.drawing.Plot:
    Gnx = nxgraphs(state_vector, in_fluxes, internal_fluxes, out_fluxes)
   
    def n_color(
            G: nx.DiGraph,
            node_name: str
    ) -> str:
        return 'grey' 


    def e_color(
            G: nx.DiGraph,
            s: str,
            t: str
    ) -> str:
        return "blue" if G.in_degree(s) ==0  else (
                'red' if G.out_degree(t) == 0 else 'black'
        )

    return igraph_func_plot(
        Gnx,
        node_color_func=n_color,
        edge_color_func=e_color,
    )

def igraph_part_plot(
    state_vector: Tuple[Symbol],
    in_fluxes: Dict[Symbol, Expr],
    internal_fluxes: Dict[Tuple[Symbol, Symbol], Expr],
    out_fluxes: Dict[Symbol, Expr],
    part_dict: Dict[Set[str], str],
    path=None
) -> ig.drawing.Plot:
    Gnx = nxgraphs(state_vector, in_fluxes, internal_fluxes, out_fluxes)
   
    
    def n_color(G,node_name):
        cs=set({})
        for var_set, color in part_dict.items():
            var_set_str = frozenset({str(v) for v in var_set})
            # we could have multicolored nodes if the variable set overlap 
            # but igraph does not support it
            cs = cs.union(set({color})) if node_name in var_set_str else cs
        return 'grey' if len(cs) == 0 else list(cs)[0] 

    
    def e_color(
            G: nx.DiGraph,
            s: str,
            t: str
    ) -> str:
        return "blue" if G.in_degree(s) ==0  else (
                'red' if G.out_degree(t) == 0 else 'black'
        )

    return igraph_func_plot(
        Gnx,
        node_color_func=n_color,
        edge_color_func=e_color,
        path=path
    )

###############################################################################
def matplotlib_part_plot(
    state_vector: Tuple[Symbol],
    IFBS: Dict[Symbol, Expr],
    IntFBS: Dict[Tuple[Symbol, Symbol], Expr],
    OFBS: Dict[Symbol, Expr],
    part_dict: Dict[Set[str], str],
    ax: matplotlib.axes._axes.Axes,
    rotate: bool=True
):
    # VirtualInfluxesBySymbol
    VIFBVP = {
        (Symbol(f"In_{p}"), p): v
        for p, v in IFBS.items()
    }

    # VirtualOutfluxesByVirtualPools
    VOFBVP = {
        (p, Symbol(f"Out_{p}")): v
        for p, v in OFBS.items()
    }

    def flipped(tup):
        s, t = tup
        return(t, s)

    def f_maker(target):
        def count_and_remove(acc, el):
            c, filtered = acc
            if flipped(el) == target:
                return c+1, filtered
            else:
                return c, filtered+[el]
        return count_and_remove

    def tl(tup):
        twes, cands, owes = tup
        if len(cands) == 0:
            return (twes, [], owes)
        else:
            fst = cands[0]
            c, rest = reduce(f_maker(fst), cands[1:], (0, []))
            return tl((twes+[fst], rest, owes)) if c > 0\
            else tl((twes, rest, owes + [fst]))

    two_way_edges, _, one_way_edges = tl(
        ([], list(IntFBS.keys()), [])
    )

    visible_nodes = state_vector
    def name_or_empty(sym):
        return latex(sym) if sym in visible_nodes else " "
    
    def flux_name(tup,exp):
        src, target=tup
        return f"$F_{{ {name_or_empty(src)} \\rightarrow {name_or_empty(target)} }} = {latex(exp)}$"

    e_d = {**VIFBVP, **VOFBVP, **IntFBS}
    e_l_d = {
        k: flux_name(k, v)
        for k, v in e_d.items()
    }

    G = nx.DiGraph()
    G.add_edges_from(e_d.keys())

    g = ig.Graph.from_networkx(G)

    n_l_d = {n: f"${latex(n)}$" for n in state_vector}

    elfs_test = 10 #edge label font size in typographic points

    def labelwidth_in_typo_pts(label):
        # measure the size of the textpath in dots for the current 
        t = matplotlib.textpath.TextPath(
            (0, 0),
            label,
            size=elfs_test
        )
        bb = t.get_extents()
        return bb.width


    nw = max(
        map(
            labelwidth_in_typo_pts,
            n_l_d.values()
        )
    )
    ew = max(
        map(
            labelwidth_in_typo_pts,
            e_l_d.values()
        )
    )

    # create a first layout of the graph without any constraints 
    # from the text for the labels
    l = g.layout_sugiyama(
        vgap=1,
        hgap=1,
        #maxiter=10000
    )
    l.center(0,0)
    l.rotate(180)

    # compute the minimum distance between two nodes in this
    # layout
    def dist(p1,p2):
        a1,a2=map(np.array,(p1,p2))
        v=a2-a1
        return np.sqrt(np.dot(v,v))
    
    def sd(d,cands):
        if len(cands)<2:
            return d,cands
        else:
            fst,rest=cands[0],cands[1:]
            d_fst=min([dist(fst,el) for el in rest])
            return sd(min(d,d_fst),rest)
    
    def min_dist(points):
        if len(points)<2:
            return 0
        else:
            start_d = dist(points[0],points[1])
            d_min,_= sd(start_d,points)
            return d_min
    
    ## minimum distance between two nodes
    d = min_dist(list(l.coords))
    # compute the first scale factor sf1 that stretches the 
    # arbitraty layout to length of a nodelabel + edgelabel
    sf1=(nw+ew)/d

    l1 = deepcopy(l)
    l1.scale(sf1,sf1)

    # we now add a boundary layer around all the nodecoordinates
    # For the nodelabel to be fully visible  the bounddary must 
    # be at least a node lable width big

    bb1=l1.bounding_box(border=nw)
    w1=bb1.width
    h1=bb1.height

    # To fill the whole subplot we scale the layout to the width
    # of the subplot computed from the figure and ax extends


    f = ax.get_figure()
    fh = f.get_figheight()
    fw = f.get_figwidth() # inches
    bb = ax.get_position() # relative bounding box

    plot_width_in_typo_pts = fw * 72 *bb.width
    plot_height_in_typo_pts = fh* 72 *bb.height
    ax.set_xlim((-plot_width_in_typo_pts/2, plot_width_in_typo_pts/2))
    ax.set_ylim((-plot_height_in_typo_pts/2,plot_height_in_typo_pts/2))

    sf2x= plot_width_in_typo_pts/ w1 
    sf2y= plot_height_in_typo_pts/ h1 
    sf2=min(sf2x,sf2y)


    l2=deepcopy(l1)
    #l2.scale(sf2x, sf2y)
    l2.scale(sf2, sf2)

    df = g.get_vertex_dataframe()
    pos = {
        df.iat[i, 0]: np.array(l2.coords[i])
        for i in df.index
    }
    

    ##pos = nx.layout.kamada_kawai_layout(G)
    ##pos = nx.layout.shell_layout(G)


    rad = 0.5

    def x_angle(pos,silent=True):
        vec_x = np.array([1,0])
        pos_n = pos/np.sqrt(np.dot(pos,pos))
        # compute the angle between x_axis and pos
        x_a = np.arccos(np.dot(vec_x,pos_n))*180/np.pi
        x, y = pos
        if (x> 0) & (y > 0) :
            if not silent:
                print("I")
            t_a =  x_a
        elif (x <= 0) & (y > 0):
            if not silent:
                print("II")
            t_a = 90- x_a
        elif (x <= 0) & (y<= 0):
            if not silent:
                print("III")
            t_a = 180- x_a
        else:
            if not silent:
                print("IV")
            t_a=x_a-90
        return(t_a)
        
    def rotate_90(vec):
        x,y=vec
        phi=np.pi/2
        return np.array([
            
            x*np.cos(phi)-y*np.sin(phi),
            x*np.sin(phi)+y*np.cos(phi)
        ])
    
    def label_pos(src, target):
        return (src+target) / 2 + rotate_90((src - target) * rad/2)
    
    def draw_edge_labels(
            ax,
            pos,
            edge_dict,
            fontsize=10,
            rotate=True,
            bbox=dict(
                boxstyle="round",
                color="white",
                #alpha=0.1
            ),
        ):
        def draw_edge_label(sym_tup):
            src, target = sym_tup
            src_pos = pos[src]
            target_pos = pos[target]
            angle = x_angle(target_pos-src_pos)
            ax.text(
                *label_pos(src_pos,target_pos),
                flux_name(sym_tup,edge_dict[sym_tup]),
                bbox=bbox,
                verticalalignment="center",
                #verticalalignment="center_baseline",
                horizontalalignment="center",
                rotation=angle if rotate else None,
                fontsize=fontsize
            )
        for k in edge_dict.keys():
            draw_edge_label(k)

    def draw_node_labels(
        ax,
	pos,
	labels,
	fontsize=10,
    ):
        def draw_node_label(sym):
            target_pos = pos[sym]
            ax.text(
                *target_pos,
                labels[sym],
                #bbox=bbox,
                verticalalignment="center",
                #verticalalignment="center_baseline",
                horizontalalignment="center",
                fontsize=fontsize
            )
        for k in labels.keys():
            draw_node_label(k)


    d_n_s=(nw*sf2)**2 #in squared typo_points...
    d_n_c="gray" # default node color
    uncolored_nodes = frozenset(visible_nodes).difference(
            reduce(
            lambda acc,el: acc.union(el),
            part_dict.keys()
        )
    )
    
    def dict_maker(tup):
        ns, col = tup
        return {n: col for n in ns}

    node_color_dict=reduce(
        lambda acc, tup: {**acc, **dict_maker(tup)},
        [(ns,col) for ns,col in part_dict.items()],
        {   
            n: d_n_c 
            for n in uncolored_nodes
        }
    )    
    #import pdb; pdb.set_trace()
    #from IPython import embed; embed()

    edge_label_bbox=dict(
        # boxstyle="round",
        color="white"
    )
    nx.draw_networkx_nodes(
        G=G,
        pos=pos,
        ax=ax,
        nodelist=visible_nodes,
        node_size=d_n_s,
        alpha=0.5,
        #node_color="red"
        node_color=[node_color_dict[n] for n in visible_nodes]
    )
    draw_node_labels(
        pos=pos,
        ax=ax,
        labels=n_l_d,
        fontsize=elfs_test*sf2,
        #arrowsize=ars,
    )    
    # draw in fluxes straight
    nx.draw_networkx_edges(
        G=G,
        pos=pos,
        ax=ax,
        node_size=d_n_s,
        edgelist=[k for k in VIFBVP.keys()], 
    )
    # draw out fluxes straight
    nx.draw_networkx_edges(
        G=G,
        pos=pos,
        ax=ax,
        node_size=d_n_s,
        edgelist=[k for k in VOFBVP.keys()], 
        #arrowsize=ars,
    )
    # draw one way internal fluxes straight
    nx.draw_networkx_edges(
        G=G,
        pos=pos,
        ax=ax,
        node_size=d_n_s,
        #arrows=True,
        edgelist=[k for k in  one_way_edges], 
        #arrowsize=ars,
    )
    nx.draw_networkx_edge_labels(
        G=G,
        pos=pos,
        ax=ax,
        edge_labels={ k:v for k,v in e_l_d.items() if k in [*one_way_edges,*VIFBVP.keys(),*VOFBVP.keys()]},
        #label_pos=0.7,
        bbox=edge_label_bbox,
        #horizontalalignment='left',
        #verticalalignment="top",
        #verticalalignment="center_baseline",
        font_size=elfs_test*sf2,
        rotate=rotate
        
    )
    # draw internal those fluxes, that have a return flux, curved  
    funcs=[lambda x:x,flipped]
    for f in funcs:
        nx.draw_networkx_edges(
            G=G,
            pos=pos,
            ax=ax,
            node_size=d_n_s,
            arrows=True,
            edgelist=[f(e) for e in  two_way_edges], 
            connectionstyle=f"arc3,rad={rad}"
            #arrowsize=ars,
        )
    
    draw_edge_labels(
        ax,
        pos=pos,
        edge_dict={
            k: v
            for k, v in IntFBS.items()
            if (k in two_way_edges) or (flipped(k) in two_way_edges)
        },
        rotate=rotate,
        fontsize=elfs_test * sf2,
        bbox=edge_label_bbox
    )



##############################################################################

def to_int_keys_1(flux_by_sym, state_vector):
    return {list(state_vector).index(k):v for k,v in flux_by_sym.items()}


def to_int_keys_2(fluxes_by_sym_tup, state_vector):
    return{
        (list(state_vector).index(k[0]),list(state_vector).index(k[1])):v 
        for k,v in fluxes_by_sym_tup.items()
    }

def in_or_out_flux_tuple(
        state_vector,
        in_or_out_fluxes_by_symbol
):
    keys = in_or_out_fluxes_by_symbol.keys()

    def f(ind):
        v = state_vector[ind]
        return in_or_out_fluxes_by_symbol[v] if v in keys else 0

    return Matrix([f(ind) for ind in range(len(state_vector))])



def release_operator_1(
    out_fluxes_by_index, 
    internal_fluxes_by_index,
    state_vector
):
    decomp_rates = []
    for pool in range(len(state_vector)):
        if pool in out_fluxes_by_index.keys():
            decomp_flux = out_fluxes_by_index[pool]
        else:
            decomp_flux = 0
        decomp_flux += sum([flux for (i,j), flux in internal_fluxes_by_index.items() 
                                    if i == pool])
        decomp_rates.append(simplify(decomp_flux/state_vector[pool]))

    R = diag(*decomp_rates)
    return R

def release_operator_2(
    out_fluxes_by_symbol, 
    internal_fluxes_by_symbol,
    state_vector
):  
    return release_operator_1(
        to_int_keys_1(out_fluxes_by_symbol, state_vector),
        to_int_keys_2(internal_fluxes_by_symbol,state_vector),
        state_vector
    )

def tranfer_operator_1(
    out_fluxes_by_index,
    internal_fluxes_by_index,
    state_vector
):
    R = release_operator_1(
        out_fluxes_by_index,
        internal_fluxes_by_index,
        state_vector
    )
    # calculate transition operator
    return transfer_operator_3(
        internal_fluxes_by_index,
        R,
        state_vector
    )

def transfer_operator_2(
    out_fluxes_by_symbol, 
    internal_fluxes_by_symbol,
    state_vector
):  
    return tranfer_operator_1(
        to_int_keys_1( out_fluxes_by_symbol, state_vector),
        to_int_keys_2( internal_fluxes_by_symbol, state_vector),
        state_vector
    )

def transfer_operator_3(
    # this is just a shortcut if we know R already
    internal_fluxes_by_index,
    release_operator,
    state_vector
):
    # calculate transition operator
    T = -eye(len(state_vector))
    for (i, j), flux in internal_fluxes_by_index.items():
        T[j, i] = flux/state_vector[i]/release_operator[i, i]
    return T


def compartmental_matrix_1(
    out_fluxes_by_index,
    internal_fluxes_by_index,
    state_vector
):
    C = -1*release_operator_1(
        out_fluxes_by_index,
        internal_fluxes_by_index,
        state_vector
    )
    for (i, j), flux in internal_fluxes_by_index.items():
        C[j, i] = flux/state_vector[i]
    return C

def compartmental_matrix_2(
    out_fluxes_by_symbol, 
    internal_fluxes_by_symbol,
    state_vector
):  
    return compartmental_matrix_1(
        to_int_keys_1( out_fluxes_by_symbol, state_vector),
        to_int_keys_2( internal_fluxes_by_symbol, state_vector),
        state_vector
    )


def in_fluxes_by_index(state_vector, u):
    return {
        pool_nr: u[pool_nr]
        for pool_nr in range(state_vector.rows)
    }

def in_fluxes_by_symbol(state_vector,u):
    return {
        state_vector[pool_nr]: u[pool_nr]
        for pool_nr in range(state_vector.rows)
        if u[pool_nr] != 0
    }

def out_fluxes_by_index(state_vector,B):
    output_fluxes = dict()
    # calculate outputs
    for pool in range(state_vector.rows):
        outp = -sum(B[:, pool]) * state_vector[pool]
        s_outp = simplify(outp)
        if s_outp:
            output_fluxes[pool] = s_outp
    return output_fluxes

def out_fluxes_by_symbol(state_vector,B):
    fbi = out_fluxes_by_index(state_vector,B)
    return out_fluxes_by_symbol_2(state_vector, fbi)


def out_fluxes_by_symbol_2(state_vector,fbi):
    return {
        state_vector[pool_nr]: flux
        for pool_nr, flux in fbi.items()
    }

def internal_fluxes_by_index(state_vector,B):
    # calculate internal fluxes
    internal_fluxes = dict()
    pipes = [(i,j) for i in range(state_vector.rows) 
                    for j in range(state_vector.rows) if i != j]
    for pool_from, pool_to in pipes:
        flux = B[pool_to, pool_from] * state_vector[pool_from]
        s_flux = simplify(flux)
        if s_flux:
            internal_fluxes[(pool_from, pool_to)] = s_flux

    return internal_fluxes

def internal_fluxes_by_symbol(state_vector,B):
    fbi = internal_fluxes_by_index(state_vector,B)
    return internal_fluxes_by_symbol_2(state_vector, fbi)


def internal_fluxes_by_symbol_2(state_vector,fbi):
    return {
        (state_vector[tup[0]],state_vector[tup[1]]): flux 
        for tup,flux in fbi.items() 
    }

#def fluxes_by_symbol(state_vector, fluxes_by_index):
#    internal_fluxes, out_fluxes = fluxes_by_index


def warning(txt):
    print('############################################')
    calling_frame = inspect.getouterframes(inspect.currentframe(), 2)
    func_name = calling_frame[1][3]
    print("Warning in function {0}:".format(func_name))
    print(txt)


def deprecation_warning(txt):
    print('############################################')
    calling_frame = inspect.getouterframes(inspect.currentframe(), 2)
    func_name = calling_frame[1][3]
    print("The function {0} is deprecated".format(func_name))
    print(txt)


def flux_dict_string(d, indent=0):
    s = ""
    for k, val in d.items():
        s += ' '*indent+str(k)+": "+str(val)+"\n"

    return s


def func_subs(t, Func_expr, func, t0):
    """
    returns the function part_func
    where part_func(_,_,...) =func(_,t=t0,_..) (func partially applied to t0)
    The position of argument t in the argument list is found
    by examining the Func_expression argument.
    Args:
        t (sympy.symbol): the symbol to be replaced by t0
        t0 (value)      : the value to which the function will be applied
        func (function) : a python function
        Func_exprs (sympy.Function) : An expression for an undefined Function

    """
    assert(isinstance(type(Func_expr), UndefinedFunction))
    pos = Func_expr.args.index(t)

    def frozen(*args):
        # tuples are immutable
        L = list(args)
        L.insert(pos, t0)
        new_args = tuple(L)
        return func(*new_args)
    return frozen


def jacobian(vec, state_vec):
    dim1 = vec.rows
    dim2 = state_vec.rows
    return Matrix(dim1, dim2, lambda i, j: diff(vec[i], state_vec[j]))


# fixme: test
def has_pw(expr):
    if expr.is_Matrix:
        for c in list(expr):
            if has_pw(c):
                return True
        return False

    if expr.is_Piecewise:
        return True

    for a in expr.args:
        if has_pw(a):
            return True
    return False


def is_DiracDelta(expr):
    """Check if expr is a Dirac delta function."""
    if len(expr.args) != 1:
        return False

    arg = expr.args[0]
    return DiracDelta(arg) == expr


def parse_input_function(u_i, time_symbol):
    """Return an ordered list of jumps in the input function u.

    Args:
        u (SymPy expression): input function in :math:`\\dot{x} = B\\,x + u`

    Returns:
        ascending list of jumps in u
    """
    impulse_times = []
    pieces = []

    def rek(expr, imp_t, p):
        if hasattr(expr, 'args'):
            for arg in expr.args:
                if is_DiracDelta(arg):
                    dirac_arg = arg.args[0]
                    zeros = solve(dirac_arg)
                    imp_t += zeros

                if arg.is_Piecewise:
                    for pw_arg in arg.args:
                        cond = pw_arg[1]
                        # 'if not cond' led to strange behavior
                        if cond != True:  # noqa: E712
                            atoms = cond.args
                            zeros = solve(atoms[0] - atoms[1])
                            p += zeros

                rek(arg, imp_t, p)

    rek(u_i, impulse_times, pieces)

    impulses = []
    impulse_times = sorted(impulse_times)
    for impulse_time in impulse_times:
        intensity = u_i.coeff(DiracDelta(impulse_time-time_symbol))
        impulses.append({'time': impulse_time, 'intensity': intensity})

    jump_times = sorted(pieces + impulse_times)
    return (impulses, jump_times)


def factor_out_from_matrix(M):
    if has_pw(M):
        return(1)

    try:
        return gcd(list(M))
    except(PolynomialError):
        # print('no factoring out possible')
        # fixme: does not work if a function of X, t is in the expressios,
        # we could make it work...if we really wanted to
        return 1


def numerical_function_from_expression(expr, tup, parameter_dict, func_set):
    # the function returns a function that given numeric arguments
    # returns a numeric result.
    # This is a more specific requirement than a function returned by lambdify
    # which can still return symbolic
    # results if the tuple argument to lambdify does not contain all free
    # symbols of the lambdified expression.
    # To avoid this case here we check this.
    expr_par = sympify(expr).subs(parameter_dict)
    ss_expr = expr_par.free_symbols

    cut_func_set = make_cut_func_set(func_set)
    ss_allowed = set(
            [s for s in tup]
    )
    if not(ss_expr.issubset(ss_allowed)):
        raise Exception(
            """The following free symbols: {1} of the expression: {0} 
            are not arguments.
            """.format(ss_expr, ss_expr.difference(ss_allowed))
        )

    expr_func = lambdify(tup, expr_par, modules=[cut_func_set, 'numpy'])

    # fixme mm 02-25-2023: 
    # this causes errors and seems to be not neccessary
    # any more numpy gives runtime Warnings now as it should.
    # removing it does not cause any test to fail. so it is temporarily
    # commented and considered deprecated
    #def expr_func_safe_0_over_0(*val):
    #    with np.errstate(invalid='raise'):
    #        try:
    #            res = expr_func(*val)
    #        except FloatingPointError as e:
    #            if e.args[0] == 'invalid value encountered in double_scalars':
    #                with np.errstate(invalid='ignore'):
    #                    res = expr_func(*val)
    #                    res = np.nan_to_num(res, copy=False)
    #    return res

    #return expr_func_safe_0_over_0
    return expr_func


def numerical_rhs(
    state_vector,
    time_symbol,
    rhs,
    parameter_dict,
    func_dict
):

    FL = numerical_function_from_expression(
        rhs,
        (time_symbol,)+tuple(state_vector),
        parameter_dict,
        func_dict
    )

    # 2.) Write a wrapper that transformes Matrices to numpy.ndarrays and
    # accepts array instead of the separate arguments for the states)
    def num_rhs(t, X):
        Y = np.array([x for x in X]) # 

        Fval = FL(t, *Y)
        return Fval.reshape(X.shape,)

    return num_rhs

def numerical_func_of_t_and_Xvec(
    state_vector,
    time_symbol, # could also be the iteration symbol
    expr,
    parameter_dict,
    func_dict
):
    FL = numerical_function_from_expression(
        expr,
        (time_symbol,)+tuple(state_vector),
        parameter_dict,
        func_dict
    )

    # 2.) Write a wrapper that transformes Matrices to numpy.ndarrays and
    # accepts array instead of the separate arguments for the state variables)
    def  f_of_t_and_Xvec(t, X):

        if (X.ndim > 1 ) :
            # fixme mm 11-16-2022
            # this is evaluated each time the function is called
            if X.shape[1]!=1:
                print( f"""
                X={0},
                X.shape={1},
                X represents the numeric state vector and can not have more that one
                dimension. 
                This could be caused by adding an array of shape(n,) 
                and one of shape (n,1)
                The numpy broadcasting rules will create a result of shape (n,n)
                """.format(X,X.shape))
                raise
            else:
                X=X.reshape(-1)
        
        #Y = np.array([x for x in X]) # 
        Y = np.array(X).flatten()
        Fval = FL(t, *Y)
        return Fval

    return f_of_t_and_Xvec
    

def numerical_array_func(
    state_vector,
    time_symbol, # could also be the iteration symbol
    expr,
    parameter_dict,
    func_dict
):

    FL = numerical_func_of_t_and_Xvec(
        state_vector,
        time_symbol, # could also be the iteration symbol
        expr,
        parameter_dict,
        func_dict
    )

    def num_arr_fun(t, X):
        Fval = FL(t, X)
        return Fval.reshape(expr.shape)

    return num_arr_fun

def numerical_1d_vector_func(*args, **kwargs):
    
    FL = numerical_func_of_t_and_Xvec(*args,**kwargs)
    def flat_func(t,X):
        return FL(t,X).flatten()

    return flat_func


def numerical_rhs_old(
    state_vector,
    time_symbol,
    rhs,
    parameter_dict,
    func_set,
    times
):

    FL = numerical_function_from_expression(
        rhs,
        tuple(state_vector) + (time_symbol,),
        parameter_dict,
        func_set
    )

    # 2.) Write a wrapper that transformes Matrices numpy.ndarrays and accepts
    # array instead of the separate arguments for the states)
    def num_rhs(X, t):
        Fval = FL(*X, t)
        return Fval.reshape(X.shape,)

    def bounded_num_rhs(X, t):
        # fixme 1:
        # maybe odeint (or another integrator)
        # can be told >>not<< to look outside
        # the interval

        # fixme 2:
        # actually the times vector is not the smallest
        # possible allowed set but the intersection of
        # all the intervals where the
        # time dependent functions are defined
        # this should be tested in init
        t_max = times[-1]

        # fixme: we should die hard here, because now we think we can compute
        # the state transition operator till any time in the future,
        # but it is actually biased by the fact, that we use the last value
        # over and over again
        # and hence assume some "constant" future
        if t > t_max:
            res = num_rhs(X, t_max)
        else:
            res = num_rhs(X, t)

#        print('brhs', 't', t, 'X', X, 'res', res)
#        print('t', t)
        return res

    return bounded_num_rhs


def numsol_symbolic_system_old(
    state_vector,
    time_symbol,
    rhs,
    parameter_dict,
    func_set,
    start_values,
    times
):

    nr_pools = len(state_vector)

    if times[0] == times[-1]:
        return start_values.reshape((1, nr_pools))

    num_rhs = numerical_rhs_old(
        state_vector,
        time_symbol,
        rhs,
        parameter_dict,
        func_set,
        times
    )
    return odeint(num_rhs, start_values, times, mxstep=10000)


def numsol_symbolical_system(
    state_vector,
    time_symbol,
    rhs,
    parameter_dicts,
    func_dicts,
    start_values,
    times,
    disc_times=()
):
    assert(isinstance(parameter_dicts, Iterable))
    assert(isinstance(func_dicts, Iterable))

    nr_pools = len(state_vector)
    t_min = times[0]
    t_max = times[-1]

    if times[0] == times[-1]:
        return start_values.reshape((1, nr_pools))

    num_rhss = tuple(
        numerical_rhs(
            state_vector,
            time_symbol,
            rhs,
            parameter_dict,
            func_dict
        )
        for parameter_dict, func_dict in zip(parameter_dicts, func_dicts)
    )

    res = solve_ivp_pwc(
        rhss=num_rhss,
        t_span=(t_min, t_max),
        y0=start_values,
        t_eval=tuple(times),
        disc_times=disc_times
    )

    # adapt to the old ode_int interface
    # since our code at the moment expects it
    values = np.rollaxis(res.y, -1, 0)

    return (values, res.sol)


def arrange_subplots(n):
    if n <= 3:
        rows = 1
        cols = n
    if n == 4:
        rows = 2
        cols = 2
    if n >= 5:
        rows = n // 3
        if n % 3 != 0:
            rows += 1
        cols = 3

    return (rows, cols)


def melt(ndarr, identifiers=None):
    shape = ndarr.shape

    if identifiers is None:
        identifiers = [range(shape[dim]) for dim in range(len(shape))]

    def rek(struct, ids, melted_list, dim):
        if type(struct) != np.ndarray:
            melted_list.append(ids + [struct])
        else:
            shape = struct.shape
            for k in range(shape[0]):
                rek(struct[k], ids + [identifiers[dim][k]], melted_list, dim+1)

    melted_list = []
    rek(ndarr, [], melted_list, 0)
    rows = len(melted_list)
    cols = len(melted_list[0])
    melted = np.array(melted_list).reshape((rows, cols))

    return melted


# fixme: test
# compute inverse of CDF at u for quantiles or generation of random variables
#def generalized_inverse_CDF(CDF, u, start_dist=1e-4, tol=1e-8):
def generalized_inverse_CDF(CDF, u, x1=0.0, tol=1e-8):
    y1 = -1
    def f(a):
#        print("HR 398", x1, y1, u)
        return u-CDF(a)

    x0 = 0.0

    y1 = f(x1)
    if (y1 <= 0):
        if x1 == 0.0:
#            print("schon fertig", "\n"*200)
            return x1
        else:
            x1 = 0.0
            y1 = f(x1)
            if y1 <= 0:
                return x1

    # go so far to the right such that CDF(x1) > u, the bisect in
    # interval [0, x1]
    while y1 >= 0:
        x0 = x1
        x1 = x1*2 + 0.1
        y1 = f(x1)

    if np.isnan(y1):
        res = np.nan
    else:
        res, root_results = brentq(f, x0, x1, xtol=tol, full_output=True)
        if not root_results.converged:
            print("quantile convegence failed")

#    if f(res) > tol: res = np.nan
#    print('gi_res', res)
#    print('finished', method_f.__name__, 'on [0,', x1, ']')
    return res


# draw a random variable with given CDF
def draw_rv(CDF):
    return generalized_inverse_CDF(CDF, np.random.uniform())


# return function g, such that g(normally distributed sv) is distributed
# according to CDF
def stochastic_collocation_transform(M, CDF):
    # collocation points for normal distribution,
    # taken from Table 10 in Appendix 3 of Grzelak2015SSRN
    cc_data = {
         2: [1],
         3: [0.0, 1.7321],
         4: [0.7420, 2.3344],
         5: [0.0, 1.3556, 2.8570],
         6: [0.6167, 1.8892, 3.3243],
         7: [0.0, 1.1544, 2.3668, 3.7504],
         8: [0.5391, 1.6365, 2.8025, 4.1445],
         9: [0.0, 1.0233, 2.0768, 3.2054, 4.5127],
        10: [0.4849, 1.4660, 2.8463, 3.5818, 4.8595],  # noqa: E131
        11: [0.0, 0.9289, 1.8760, 2.8651, 3.9362, 5.1880]  # noqa: E131
    }

    if M not in cc_data.keys():
        return None
    cc_points = [-x for x in reversed(cc_data[M]) if x != 0.0] + cc_data[M]
    cc_points = np.array(cc_points)
#    print('start computing collocation transform')
    ys = np.array(
        [generalized_inverse_CDF(CDF, norm.cdf(x)) for x in cc_points]
    )
#    print('ys', ys)
#    print('finished computing collocation transform')

    return lagrange(cc_points, ys)


# Metropolis-Hastings sampling for PDFs with nonnegative support
# no thinning, no burn-in period
def MH_sampling(N, PDF, start=1.0):
    xvec = np.ndarray((N,))
    x = start
    PDF_x = PDF(x)
    norm_cdf_x = norm.cdf(x)

    for i in range(N):
        xs = -1.0
        while xs <= 0:
            xs = x + np.random.normal()

        PDF_xs = PDF(xs)
        A1 = PDF_xs/PDF_x
        norm_cdf_xs = norm.cdf(xs)
        A2 = norm_cdf_x/norm_cdf_xs
        A = A1 * A2

        if np.random.uniform() < A:
            x = xs
            PDF_x = PDF_xs
            norm_cdf_x = norm_cdf_xs

        xvec[i] = x

    return xvec


def save_csv(filename, melted, header):
    np.savetxt(
        filename,
        melted,
        header=header,
        delimiter=',',
        fmt="%10.8f",
        comments=''
    )


def load_csv(filename):
    return np.loadtxt(filename, skiprows=1, delimiter=',')


def tup2str(tup):
    # uses for stoichiometric models
    string = Template("${f}_${s}").substitute(f=tup[0], s=tup[1])
    return(string)


# use only every (k_1,k_2,...,k_n)th element of the n-dimensional numpy array
# data,
# strides is a list of k_j of length n
# always inlcude first and last elements
def stride(data, strides):
    if isinstance(strides, int):
        strides = [strides]

    index_list = []
    for dim in range(data.ndim):
        n = data.shape[dim]
        stride = strides[dim]
        ind = np.arange(0, n, stride).tolist()
        if (n-1) % stride != 0:
            ind.append(n-1)

        index_list.append(ind)

    return data[np.ix_(*index_list)]


def is_compartmental(M):
    gen = range(M.shape[0])
    return all(
        [
            M.is_square,
            all([M[j, j] <= 0 for j in gen]),
            all([sum(M[:, j]) <= 0 for j in gen])
        ]
    )


def make_cut_func_set(func_set):
    def unify_index(expr):
        # for the case Function('f'):f_numeric
        if isinstance(expr, UndefinedFunction):
            res = str(expr)
        # for the case {f(x, y): f_numeric} f(x, y)
        elif isinstance(expr, Symbol):
            res = str(expr)
        elif isinstance(expr, Function):
            res = str(type(expr))
        elif isinstance(expr, str):
            expr = sympify(expr)
            res = unify_index(expr)
        else:
            print(type(expr))
            raise(TypeError(
                """
                funcset indices should be indexed by instances of
                sympy.core.functions.UndefinedFunction
                """
            ))
        return res

    cut_func_set = {unify_index(key): val for key, val in func_set.items()}
    return cut_func_set


def f_of_t_maker(sol_funcs, ol):
    def ot(t):
        sv = [sol_funcs[i](t) for i in range(len(sol_funcs))]
        tup = tuple(sv) + (t,)
        res = ol(*tup)
        return res
    return ot


def const_of_t_maker(const):
    def const_arr_fun(possible_vec_arg):
        if isinstance(possible_vec_arg, Number):
            return const  # also a number
        else:
            return const*np.ones_like(possible_vec_arg)
    return const_arr_fun


def x_phi_ode(
    srm,
    parameter_dicts,
    func_dicts,
    x_block_name='x',
    phi_block_name='phi',
    disc_times=()
):
    nr_pools = srm.nr_pools

    sol_rhss = []
    for pd, fd in zip(parameter_dicts, func_dicts):
        sol_rhs = numerical_rhs(
            srm.state_vector,
            srm.time_symbol,
            srm.F,
            pd,
            fd
        )
        sol_rhss.append(sol_rhs)

    B_sym = srm.compartmental_matrix
    tup = (srm.time_symbol,) + tuple(srm.state_vector)

    B_funcs_non_lin = []
    for pd, fd in zip(parameter_dicts, func_dicts):
        B_func_non_lin = numerical_function_from_expression(
            B_sym,
            tup,
            pd,
            fd
        )
        B_funcs_non_lin.append(B_func_non_lin)

    def Phi_rhs_maker(B_func_non_lin):
        def Phi_rhs(t, x, Phi_2d):
            return np.matmul(B_func_non_lin(t, *x), Phi_2d)
        return Phi_rhs

    Phi_rhss = []
    for B_func_non_lin in B_funcs_non_lin:
        Phi_rhss.append(Phi_rhs_maker(B_func_non_lin))

    functionss = []
    for sol_rhs, Phi_rhs in zip(sol_rhss, Phi_rhss):
        functions = [
            (sol_rhs, [srm.time_symbol.name, x_block_name]),
            (Phi_rhs, [srm.time_symbol.name, x_block_name, phi_block_name])
        ]
        functionss.append(functions)

    return BlockOde(
        time_str=srm.time_symbol.name,
        block_names_and_shapes=[
            (x_block_name, (nr_pools,)),
            (phi_block_name, (nr_pools, nr_pools,))
        ],
        functionss=functionss,
        disc_times=disc_times
    )


def integrate_array_func_for_nested_boundaries(
    f: Callable[[float], np.ndarray],
    integrator: Callable[
        [
            Callable[[float], np.ndarray],
            float,
            float
        ],
        np.ndarray
    ],  # e.g. array_quad_result
    tuples: Sequence[Tuple[float, float]]
) -> Sequence[float]:
    # we assume that the first (a,b) tuple contains the second,
    # the second the third and so on from outside to inside
    def compute(f, tuples, results: Sequence[float]):
        (a_out, b_out), *tail = tuples
        if len(tail) == 0:
            # number=quad(f, a_out, b_out)[0]
            arr = integrator(f, a_out, b_out)
        else:
            (a_in, b_in) = tail[0]
            results = compute(f, tail, results)
            arr = (
                integrator(f, a_out, a_in)
                + results[0]
                + integrator(f, b_in, b_out)
            )

        results = [arr] + results
        return results

    return compute(f, tuples, [])


def array_quad_result(
    f: Callable[[float], np.ndarray],
    a: float,
    b: float,
    epsrel=1e-3,  # epsabs would be a dangerous default
    *args,
    **kwargs
) -> np.ndarray:
    # integrates a vectorvalued function of a single argument
    # we transform the result array of the function into a one dimensional
    # vector compute the result for every component
    # and reshape the result to the form of the integrand
    test = f(a)
    n = len(test.flatten())
    vec = np.array(
        [quad(
            lambda t:f(t).reshape(n,)[i],
            a,
            b,
            *args,
            epsrel=epsrel,
            **kwargs
        )[0] for i in range(n)]
    )
    return vec.reshape(test.shape)


def array_integration_by_ode(
    f: Callable[[float], np.ndarray],
    a: float,
    b: float,
    *args,
    **kwargs
) -> np.ndarray:
    # another integrator like array_quad_result
    test = f(a)
    n = len(test.flatten())

    def rhs(tau, X):
        # although we do not need X we have to provide a
        # righthandside s uitable for solve_ivp

        # avoid overshooting if the solver
        # tries to look where the integrand might not be defined
        if tau < a or tau > b:
            return 0
        else:
            return f(tau).flatten()

    ys = solve_ivp_pwc(
        rhss=(rhs,),
        y0=np.zeros(n),
        t_span=(a, b)
    ).y
    val = ys[:, -1].reshape(test.shape)
    return val


def array_integration_by_values(
    f: Callable[[float], np.ndarray],
    taus: np.ndarray,
    *args,
    **kwargs,
) -> np.ndarray:
    # only allow taus as vector
    assert(len(taus.shape) == 1)
    assert(len(taus) > 0)
    test = f(taus[0])
    # create a big 2 dimensional array suitable for trapz
    integrand_vals = np.stack([f(tau).flatten() for tau in taus], 1)
    vec = np.trapz(y=integrand_vals, x=taus)
    return vec.reshape(test.shape)


def x_phi_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name):
    x_s = np.array(x_s)
    nr_pools = len(x_s)

    start_Phi_2d = np.identity(nr_pools)
    start_blocks = [
        (x_block_name, x_s),
        (phi_block_name, start_Phi_2d)
    ]
    blivp = block_ode.blockIvp(start_blocks)
    return blivp


def phi_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name):
    blivp = x_phi_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name)
    phi_func = blivp.block_solve_functions(t_span=(s, t_max))[phi_block_name]

    return phi_func


@lru_cache()
def x_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name):
    blivp = x_phi_tmax(s, t_max, block_ode, x_s, x_block_name, phi_block_name)
    x_func = blivp.block_solve_functions(t_span=(s, t_max))[x_block_name]

    return x_func


_CacheStats = namedtuple(
    'CacheStats',
    ['hitss', 'missess', 'currsizes', 'hitting_ratios']
)


def custom_lru_cache_wrapper(maxsize=None, typed=False, stats=False):
    if stats:
        hitss = []
        missess = []
        currsizes = []
        hitting_ratios = []

    def decorating_function(user_function):
        func = _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo)

        def wrapper(*args, **kwargs):
            nonlocal stats, hitss, missess, currsizes, hitting_ratios

            result = func(*args, **kwargs)
            if stats:
                hitss.append(func.cache_info().hits)
                missess.append(func.cache_info().misses)
                currsizes.append(func.cache_info().currsize)
                hitting_ratios.append(
                    round(hitss[-1]/(hitss[-1]+missess[-1])*100.0, 2)
                )
            return result

        wrapper.cache_info = func.cache_info
        if stats:
            def cache_stats():
                nonlocal hitss, missess, currsizes
                return _CacheStats(hitss, missess, currsizes, hitting_ratios)

            wrapper.cache_stats = cache_stats

            def plot_hitss():
                nonlocal hitss
                plt.plot(range(len(hitss)), hitss)
                plt.title('Hitss')
                plt.show()

            wrapper.plot_hitss = plot_hitss

            def plot_hit_history():
                nonlocal hitss
                plt.scatter(
                    range(len(hitss)-1),
                    np.diff(hitss),
                    s=1,
                    alpha=0.2
                )
                plt.title('Hit history')
                plt.show()

            wrapper.plot_hit_history = plot_hit_history

            def plot_hitting_ratios():
                nonlocal hitss, hitting_ratios
                plt.plot(
                    range(len(hitss)),
                    hitting_ratios
                )
                plt.title('Hitting ratios')
                plt.show()

            wrapper.plot_hitting_ratios = plot_hitting_ratios

            def plot_currsizes():
                nonlocal currsizes
                plt.plot(
                    range(len(currsizes)),
                    currsizes
                )
                plt.title('Currsizes')
                plt.show()

            wrapper.plot_currsizes = plot_currsizes

            def plot_hitting_ratios_over_currsizes():
                nonlocal hitting_ratios, currsizes
                plt.plot(
                    range(len(hitting_ratios)),
                    [hitting_ratios[i]/currsizes[i]
                     for i in range(len(hitting_ratios))]
                )
                plt.title('Hitting ratios over currsizes')
                plt.show()

            wrapper.plot_hitting_ratios_over_currsizes =\
                plot_hitting_ratios_over_currsizes

            def plot_hitting_ratios_vs_currsizes():
                nonlocal hitting_ratios, currsizes
                plt.plot(
                    currsizes,
                    hitting_ratios
                )
                plt.title('Hitting ratios vs currsizes')
                plt.show()

            wrapper.plot_hitting_ratios_vs_currsizes =\
                plot_hitting_ratios_vs_currsizes

        def cache_clear():
            nonlocal hitss, missess, currsizes
            hitss = []
            missess = []
            currsizes = []
            func.cache_clear()

        wrapper.cache_clear = cache_clear
        return wrapper

    return decorating_function


def print_quantile_error_statisctics(qs_ode, qs_pi):
    print('ODE          :', ['{: 7.2f}'.format(v) for v in qs_ode])
    print('Expl.        :', ['{: 7.2f}'.format(v) for v in qs_pi])
    abs_err = np.abs(qs_ode-qs_pi)
    print('abs. err.    :', ['{: 7.2f}'.format(v) for v in abs_err])
    rel_err = np.abs(qs_ode-qs_pi)/qs_pi * 100
    print('rel. err. (%):', ['{: 7.2f}'.format(v) for v in rel_err])
    print()
    print('mean abs. err    :', '{: 6.2f}'.format(abs_err.mean()))
    print('mean rel. err (%):', '{: 6.2f}'.format(rel_err.mean()))
    print('max. abs. err    :', '{: 6.2f}'.format(np.max(abs_err)))
    print('max. rel. err (%):', '{: 6.2f}'.format(np.max(rel_err)))
    print()


def net_Fs_from_discrete_Bs_and_xs(Bs, xs):
    nr_pools = xs.shape[1]
    nt = len(Bs)

    net_Fs = np.zeros((nt, nr_pools, nr_pools))
    for k in range(nt):
        for j in range(nr_pools):
            for i in range(nr_pools):
                if i != j:
                    net_Fs[k, i, j] = Bs[k, i, j] * xs[k, j]

    return net_Fs


def net_Rs_from_discrete_Bs_and_xs(Bs, xs, decay_corr=None):
    nr_pools = xs.shape[1]
    nt = len(Bs)

    if decay_corr is None:
        decay_corr = np.ones((nt,))

    net_Rs = np.zeros((nt, nr_pools))
    # sum(Bs_k[:,j]) is the  column sum of column j of B_k
    # which multiplied by the j component of x_k, x_[k,j] should give
    # the mass leaving pool j towards the outsice

    I=np.eye(nr_pools)
    for k in range(nt):
        for j in range(nr_pools):
            net_Rs[k, j] = (1-sum(Bs[k, :, j])*decay_corr[k]) * xs[k, j]

    return net_Rs


def net_Us_from_discrete_Bs_and_xs(Bs, xs):
    nr_pools = xs.shape[1]
    nt = len(Bs)

    net_Us = np.zeros((nt, nr_pools))
    for k in range(nt):
        net_Us[k] = xs[k+1] - Bs[k] @ xs[k]

    return net_Us


def check_parameter_dict_complete(model, parameter_dict, func_set):
    """Check if the parameter set  the function set are complete
       to enable a model run.

    Args:
        model (:class:`~.smooth_reservoir_model.SmoothReservoirModel`):
            The reservoir model on which the model run bases.
        parameter_dict (dict): ``{x: y}`` with ``x`` being a SymPy symbol
            and ``y`` being a numerical value.
        func_set (dict): ``{f: func}`` with ``f`` being a SymPy symbol and
            ``func`` being a Python function. Defaults to ``dict()``.
    Returns:
        free_symbols (set): set of free symbols, parameter_dict is complete if
                            ``free_symbols`` is the empty set
    """
    free_symbols = model.F.subs(parameter_dict).free_symbols
#    print('fs', free_symbols)
    free_symbols -= {model.time_symbol}
#    print(free_symbols)
    free_symbols -= set(model.state_vector)
#    print(free_symbols)

    # remove function names, are given as strings
    free_names = set([symbol.name for symbol in free_symbols])
    func_names = set([key for key in func_set.keys()])
    free_names = free_names - func_names

    return free_names


def F_Delta_14C(C12, C14, alpha=None):
    if alpha is None:
        alpha = ALPHA_14C

    C12[C12 == 0] = np.nan
    return (C14/C12/alpha - 1) * 1000

def densities_to_distributions(
        densities: Callable[[float],np.ndarray],
        nr_pools: int
    )->Callable[[float],np.ndarray]:
        
        def distributions(A: float )->np.ndarray: 
            return np.array(
                [
                    quad(
                        lambda a:densities(a)[i],
                        -np.inf, 
                        A
                    )[0]
                    for i in range(nr_pools)
                ]
            )

        return distributions    

def pool_wise_bin_densities_from_smooth_densities_and_index(
        densities: Callable[[float],np.ndarray],
        nr_pools: int,
        dt: float,
    )->Callable[[int],np.ndarray]:
    def content(ai:int)->np.ndarray:
        da = dt
        return np.array(
            [
                quad(
                    lambda a:densities(a)[i],
                    ai*da,
                    (ai+1)*da
                )[0] / da
                for i in range(nr_pools)
            ]
        )
    return content     

def pool_wise_bin_densities_from_smooth_densities_and_indices(
        densities: Callable[[float],np.ndarray],
        nr_pools: int,
        dt: float,
    )->Callable[[np.ndarray],np.ndarray]:
    bin_density = pool_wise_bin_densities_from_smooth_densities_and_index(
                densities,
                nr_pools,
                dt
            )
    # vectorize it
    def bin_densities(age_bin_indices: np.ndarray)->np.ndarray:
        return np.stack(
            [
                bin_density(ai)
                for ai in age_bin_indices
            ],
            axis=1
        )
    return bin_densities

def negative_indicies_to_zero(
        f: Callable[[np.ndarray],np.ndarray]
    )->Callable[[np.ndarray],np.ndarray]:

    def wrapper(age_bin_indices):
        arr_true = f(age_bin_indices)
        nr_pools = arr_true.shape[0]
        return np.stack(
            [
                np.where(
                    age_bin_indices >=0,
                    arr_true[ip,:],
                    0
                )
                for ip in range(nr_pools)
            ]
        )

    return wrapper

# make sure that the start age distribution
# yields zero for negative ages or indices 
def p0_maker(
        start_age_densities_of_bin: Callable[[int],np.ndarray],
):
    def p0(ai):
        res = start_age_densities_of_bin(ai)
        if ai >= 0:
            return res
        else:
            return np.zeros_like(res)
    return p0

def discrete_time_dict(
        cont_time: Symbol,
        delta_t: Symbol,
        iteration: Symbol
    )->Dict:
    return {cont_time: delta_t*iteration}
    


def discrete_time_sym(
        sym_cont: Expr, 
        cont_time: Symbol, 
        delta_t: Symbol, 
        iteration: Symbol
    )-> Expr:
    flux_sym_discrete = sym_cont.subs(
       discrete_time_dict(
           cont_time,
           delta_t,
           iteration
       )
    )
    return flux_sym_discrete

# fixme mm 2-11-2022
# this function is identical to discrete_time_sym and should
# be replaced wherever it is called
# this is a wrapper until all calls are removed
def euler_forward_net_u_sym(
        u_sym_cont: Expr, 
        cont_time: Symbol, 
        delta_t: Symbol, 
        iteration: Symbol
    )-> Expr:
    return discrete_time_sym(
        u_sym_cont, 
        cont_time, 
        delta_t, 
        iteration
    )

# fixme mm 2-11-2022
# this function is identical to discrete_time_sym and should
# be replaced wherever it is called
# this is a wrapper until all calls are removed
def euler_forward_B_sym(
        B_sym_cont: Expr, 
        cont_time: Symbol, 
        delta_t: Symbol, 
        iteration: Symbol
    )-> Expr:
    return discrete_time_sym(
        B_sym_cont, 
        cont_time, 
        delta_t, 
        iteration
    )

def partitions(start, stop, step=1):
    # little helper to partition an iterable e.g. to then compute averages
    # of the partitions (avoids fence post errors)
    diff = stop - start
    number_of_steps = int(diff / step)
    last_start = start + number_of_steps * step
    last_tup_l = [(last_start, stop)] if last_start < stop else []
    return [
        (start + step * i, start + step * (i + 1)) for i in range(number_of_steps)
    ] + last_tup_l

def average_iterator(
    iterator, # an iterator whose results support addition
    step: int
    ):
    # this is a generatorwhich takes an iterator
    # and returns a new iterator that yields the averages of iterator
    # over step steps
    
    
    myit,old=it.tee(iterator) 
    while True:
        sl_it = it.islice(myit,step)
        # test,sl_it = it.tee(sl_it)
        # print([v.x for v in test])
        sl_sum = reduce(lambda acc,el: acc+el, sl_it) 
        # this will also advance myit by step steps
        # but this is what we want
        yield sl_sum/step


def average_iterator_from_partitions(
    iterator, # an iterator whose results support addition
    partitions: Iterator[Tuple[int]] # typically a list [(0,5),(5,10),...(100,101)]
    ):
    # the partitions can have uneven length (mostly used to compute the average
    # over the last remaining part if it is smaller than the normal stepsize)
    
    myit, old = it.tee(iterator) #don't consume the original but a copy.. 
    mypart, old = it.tee(partitions) #don't consume the original but a copy.. 
    while True:
        try:
            start, stop = next(mypart)
            step = stop - start
            sl_it = it.islice(myit,step)
            # test,sl_it = it.tee(sl_it)
            # print([v for v in test])
            sl_sum = reduce(lambda acc,el: acc+el, sl_it) 
            # this will also advance myit by step steps
            # but this is what we want
            yield sl_sum/step
        except(StopIteration):
            break # one of the iterators has been exhausted usually the finite partitions
