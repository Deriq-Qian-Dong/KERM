# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    This package implement Heterogeneous Graph structure for handling Heterogeneous graph data.
"""

import os
import json
import paddle
import copy
import numpy as np
import pickle as pkl
from collections import defaultdict

from pgl.graph import Graph
from pgl.utils import op

def unbatch(graph):
    """This method disjoint list of graph into a big graph.

    Args:

        graph_list (Graph List): A list of Graphs.

        merged_graph_index: whether to keeped the graph_id that the nodes belongs to.


    .. code-block:: python

        import numpy as np
        import pgl

        num_nodes = 5
        edges = [ (0, 1), (1, 2), (3, 4)]
        graph = pgl.Graph(num_nodes=num_nodes,
                    edges=edges)
        joint_graph = pgl.Graph.disjoint([graph, graph], merged_graph_index=False)
        print(joint_graph.graph_node_id)
        >>> [0, 0, 0, 0, 0, 1, 1, 1, 1 ,1]
        print(joint_graph.num_graph)
        >>> 2

        joint_graph = pgl.Graph.disjoint([graph, graph], merged_graph_index=True)
        print(joint_graph.graph_node_id)
        >>> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print(joint_graph.num_graph)
        >>> 1 
    """

    edges_list = disjoin_edges(graph)
    num_nodes_list = disjoin_nodes(graph)
    node_feat_list = disjoin_feature(graph, mode="node")
    edge_feat_list = disjoin_feature(graph, mode="edge")
    graph_list = []
    for edges, num_nodes, node_feat, edge_feat in zip(edges_list, num_nodes_list, node_feat_list, edge_feat_list):
        graph = Graph(num_nodes=num_nodes,
                    edges=edges,
                    node_feat=node_feat,
                    edge_feat=edge_feat)
        graph_list.append(graph)
    return graph_list

def disjoin_edges(graph):
    """join edges for multiple graph"""
    start_offset_list = graph._graph_node_index[: -1]
    start_list, end_list = graph._graph_edge_index[: -1], graph._graph_edge_index[1: ]

    edges_list = []
    for start, end, start_offset in zip(start_list, end_list, start_offset_list):
        edges = graph.edges[start: end]
        edges -= start_offset
        edges_list.append(edges)
    return edges_list

def disjoin_nodes(graph):
    num_nodes_list = []
    start_list, end_list = graph._graph_node_index[: -1], graph._graph_node_index[1: ]
    for start, end in zip(start_list, end_list):
        num_nodes_list.append(end - start)
    return num_nodes_list

def disjoin_feature(graph, mode="node"):
    """join node features for multiple graph"""
    is_tensor = graph.is_tensor()
    feat_list = []
    if mode == "node":
        start_list, end_list = graph._graph_node_index[: -1], graph._graph_node_index[1: ]
        for start, end in zip(start_list, end_list):
            feat = defaultdict(lambda: [])
            for key in graph.node_feat:
                feat[key].append(graph.node_feat[key][start: end])
            feat_list.append(feat)
    elif mode == "edge":
        start_list, end_list = graph._graph_edge_index[: -1], graph._graph_edge_index[1: ]
        for start, end in zip(start_list, end_list):
            feat = defaultdict(lambda: [])
            for key in graph.edge_feat:
                feat[key].append(graph.edge_feat[key][start: end])
            feat_list.append(feat)
    else:
        raise ValueError(
            "mode must be in ['node', 'edge']. But received model=%s" %
            mode)

    feat_list_temp = []
    for feat in feat_list:
        ret_feat = {}
        for key in feat:
            if len(feat[key]) == 1:
                ret_feat[key] = feat[key][0]
            else:
                if is_tensor:
                    ret_feat[key] = paddle.concat(feat[key], 0)
                else:
                    ret_feat[key] = np.concatenate(feat[key], axis=0)
        feat_list_temp.append(ret_feat)
    return feat_list_temp
