#!/usr/bin/env python

# coding: utf-8

import yaml
import json
import sys
import os
from elasticsearch import Elasticsearch

sys.path.append('../telemanom') 

class Config:
    '''Loads parameters from config.yaml into global object'''

    def __init__(self, path_to_config):
        
        if os.path.isfile(path_to_config):    
            pass
        else:
            path_to_config = '../%s' %path_to_config 

        setattr(self, "path_to_config", path_to_config)

        dictionary = None
        
        with open(path_to_config, "r") as f:
            dictionary = yaml.load(f.read())
                
        try:
            for k,v in dictionary.items():
                setattr(self, k, v)
        except:
            for k,v in dictionary.iteritems():
                setattr(self, k, v)
            


    def build_group_lookup(self, path_to_groupings):
        
        channel_group_lookup = {}   

        with open(path_to_groupings, "r") as f:
            groupings = json.loads(f.read())

            for subsystem in groupings.keys():
                for subgroup in groupings[subsystem].keys():
                    for chan in groupings[subsystem][subgroup]:
                        channel_group_lookup[chan["key"]] = {}
                        channel_group_lookup[chan["key"]]["subsystem"] = subsystem
                        channel_group_lookup[chan["key"]]["subgroup"] = subgroup

        return channel_group_lookup