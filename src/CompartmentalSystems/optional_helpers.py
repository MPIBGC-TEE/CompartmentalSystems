import igraph as ig
import networkx as nx
import matplotlib 
import networkx as nx
import numpy as np
from sympy import (
    Matrix,
    #gcd,
    #diag,
    #sympify,
    #lambdify,
    #DiracDelta,
    #solve,
    Symbol,
    Expr,
    #diff,
    #simplify,
    #eye,
    #ImmutableMatrix,
    latex
)
from frozendict import frozendict
from typing import Callable, Tuple, Sequence, Set, Dict, Iterator
from functools import reduce #, lru_cache, _CacheInfo, _lru_cache_wrapper
from copy import deepcopy

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



