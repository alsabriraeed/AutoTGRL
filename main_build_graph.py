"""Entry point."""
import argparse
import os
import sys        
     
from build_transductive_graphs.build_textgraph import  buildgraph     
from build_inductive_graphs.build_graph import buildgraph
def build_args():
    parser = argparse.ArgumentParser(description='AutoTGRL')
    register_default_args(parser)
    args = parser.parse_args()

    return args

def register_default_args(parser):
    parser.add_argument("--dataset", type=str, default="mr", required=False,
                        help="The input dataset.")
    parser.add_argument('--model_type', type=str, default='transductive',
                        choices=['transductive', 'inductive'],
                        help='inductive: each docunemt a graph, transductive: a graph for the whole corpus')
def main(args):
    path = sys.path[0]           
    if args.model_type == 'transductive':
        buildgraph(args.dataset, path)       
    else:       
        buildgraph(args.dataset, path)      
        
if __name__ == "__main__":
    args = build_args()
    main(args)
