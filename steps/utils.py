import logging
import os
import sys

import pydot_ng as pydot
from IPython.display import Image, display


def view_graph(graph_info):
    """Displays graph in the notebook

    Args:
        graph_info (dict): graph dictionary with 'nodes' and 'edges' defined
    """
    graph = create_graph(graph_info)
    view_pydot(graph)


def save_graph(graph_info, filepath):
    """Saves pydot graph to file

    Args:
        graph_info (dict): graph dictionary with 'nodes' and 'edges' defined
        filepath (str): filepath to which the graph should be saved

    """
    graph = create_graph(graph_info)
    graph.write(filepath, format='png')


def view_pydot(pydot_object):
    """Displays pydot graph in jupyter notebook

    Args:
        pydot_object: pydot.Dot object
    """
    plt = Image(pydot_object.create_png())
    display(plt)


def create_graph(graph_info):
    """Creates pydot graph from the step graph dictionary.

    Args:
        graph_info (dict): graph dictionary with 'nodes' and 'edges' defined

    Returns:
        obj: pydot.Dot object representing the step graph
    """
    dot = pydot.Dot()
    for node in graph_info['nodes']:
        dot.add_node(pydot.Node(node))
    for node1, node2 in graph_info['edges']:
        dot.add_edge(pydot.Edge(node1, node2))
    return dot


def create_filepath(filepath):
    """Creates directory path for the filepath if non-existend

    Makes it easy to created necessary directory for the filepath.

    Args:
        filepath (str): filepath for which directory needs to be created

    """
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)


def initialize_logger():
    """Initialize steps logger

    It creates logger of name 'steps'

    Returns:
        logging.Logger: logger object
    """
    logger = logging.getLogger('steps')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def get_logger():
    """Fetch existing steps logger

    It fetches logger of name 'steps'

    Returns:
        logging.Logger: logger object
    """
    return logging.getLogger('steps')
