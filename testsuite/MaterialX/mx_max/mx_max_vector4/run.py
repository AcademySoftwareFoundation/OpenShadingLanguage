#!/usr/bin/env python
import os
loc = os.environ["OSLHOME"] + os.environ["MATERIALX_OSOS"]+ "/"

def buildCmd(shader, inputs):
	cmd = "--print "
	for pName, pValue in inputs.iteritems():
		cmd += " -param " + pName + " " + pValue	
	cmd += " " + shader + " -o out mx.exr"
	return cmd


inputs = {
	         "in1.x" : "0.1",
	         "in1.y" : "0.2",
	         "in1.z" : "0.3",
	         "in1.w" : "0.4",

	         "in2.x" : "0.5",
	         "in2.y" : "0.6",
	         "in2.z" : "0.7",
	         "in2.w" : "0.8",	         
	         }

shader = loc + os.path.basename(os.getcwd())
command = testshade(buildCmd(shader, inputs))
command = None