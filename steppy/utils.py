import logging
import sys

import pydot_ng as pydot
from IPython.display import Image, display


def initialize_logger():
    """Initialize steppy logger.

    This logger is used throughout the steppy library to report computation progress.

    Example:
    
        Simple use of steppy logger:

        .. code-block:: python
        
            initialize_logger()
            logger = get_logger()
            logger.info('My message inside pipeline')
            
        result looks like this:
        
        .. code::
        
            2018-06-02 12:33:48 steppy >>> My message inside pipeline

    Returns:
        logging.Logger: logger object formatted in the steppy style
    """
    logger = logging.getLogger('steppy')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(console_handler)

    return logger


def get_logger():
    """Fetch existing steppy logger.

    Example:
    
        .. code-block:: python
        
            initialize_logger()
            logger = get_logger()
            logger.info('My message goes here')
            
        result looks like this:
        
        .. code::
        
            2018-06-02 12:33:48 steppy >>> My message inside pipeline

    Returns:
        logging.Logger: logger object formatted in the steppy style
    """
    return logging.getLogger('steppy')


def display_upstream_structure(structure_dict):
    """Displays pipeline structure in the jupyter notebook.

    Args:
        structure_dict (dict): dict returned by
            :func:`~steppy.base.Step.upstream_structure`.
    """
    graph = _create_graph(structure_dict)
    plt = Image(graph.create_png())
    display(plt)


def persist_as_png(structure_dict, filepath):
    """Saves pipeline diagram to disk as png file.

    Args:
        structure_dict (dict): dict returned by
            :func:`~steppy.base.Step.upstream_structure`
        filepath (str): filepath to which the png with pipeline visualization should be persisted
    """
    graph = _create_graph(structure_dict)
    graph.write(filepath, format='png')


def _create_graph(structure_dict):
    """Creates pydot graph from the pipeline structure dict.

    Args:
        structure_dict (dict): dict returned by step.upstream_structure

    Returns:
        graph (pydot.Dot): object representing upstream pipeline structure (with regard to the current Step).
    """
    graph = pydot.Dot()
    for node in structure_dict['nodes']:
        graph.add_node(pydot.Node(node))
    for node1, node2 in structure_dict['edges']:
        graph.add_edge(pydot.Edge(node1, node2))
    return graph
