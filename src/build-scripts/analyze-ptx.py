#!/usr/bin/env python3

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


import getopt
import graphviz
import itertools
import os
import re
import sys

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict

#-------------------------------------------------------------------------------
# PTX parsing helpers
#-------------------------------------------------------------------------------

# pre-compile some regex patterns to speed up parsing
call_ptn = re.compile(r"^\s*(call\.uni).*$")
dec_ptn  = re.compile(r"^\s*(\.pragma|\.extern|\.visible|\.func|\.local|\.global|\.reg|\.param|LBB[0-9_]+\:)(.*)$")
fn_ptn   = re.compile(r"^(\.visible \.func|\.visible \.entry|\.weak \.func|\.func).*[\s]+([a-zA-Z0-9_]+)[\(\)]*$")
inst_ptn = re.compile(r"^\s*([@!%p0-9 ]*)([a-zA-Z0-9\.]+)\s*([a-zA-Z0-9\$\.,% _\[\]{}+-]+);.*$")
mem_ptn  = re.compile(r"^\s*(st|ld).*;$")
ret_ptn  = re.compile(r"^\s*(ret);.*$")

def get_func_name(line):
    match = fn_ptn.match(line)
    if match:
        return match.group(2)
    else:
        return ""

def is_func_or_decl(line):
    match = fn_ptn.match(line)
    if match:
       return True
    else:
       return False

def is_call_start(line):
    if line.find("call.uni") != -1:
        return True
    else:
        return False

def is_instruction(line):
    match = call_ptn.match(line)
    if match:
        return True
    if len(line) < 4:
        return False
    if not line.endswith(';'):
        return False

    match = dec_ptn.match(line)
    if not match:
       return True
    else:
       return False

def is_mem_inst(line):
    match = mem_ptn.match(line)
    if match:
        return True
    else:
        return False

def parse_inst(line):
    if not is_instruction(line):
        return "", "", ""
    match = call_ptn.match(line)
    if match:
        return "", "call.uni", ""
    match = ret_ptn.match(line)
    if match:
        return "", "ret", ""
    match = inst_ptn.match(line)
    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        return "", "", ""

#-------------------------------------------------------------------------------

@dataclass
class FuncStats:
    func_name:  str = ""
    inst_count: int = 0
    call_count: int = 0
    mem_count:  int = 0
    call_dict:  Dict[str, int] = field(default_factory=dict) # func name, count
    inst_dict:  Dict[str, int] = field(default_factory=dict) # inst, count

@dataclass
class FileStats:
    total_funcs:      int = 0
    total_insts:      int = 0
    total_mem_insts:  int = 0
    total_calls:      int = 0
    extern_funcs:     Dict[str, int] = field(default_factory=dict) # func name, count
    func_calls:       Dict[str, int] = field(default_factory=dict) # func name, count
    summary_dict:     Dict[str, str] = field(default_factory=dict) # func name, summary
    inst_dict:        Dict[str, int] = field(default_factory=dict) # inst, count
    largest_function: str = ""
    largest_size:     int = 0
    csv_string:       str = ""

#-------------------------------------------------------------------------------

def parse_ptx(ptx_filename):
    # read the raw PTX into a string
    raw_ptx = ""
    with open(ptx_filename, 'r') as ptx_file:
        raw_ptx = ptx_file.read()

    # variables to keep track of the parsing state
    begin_func_or_decl = False
    begin_call         = False
    in_func            = False
    brace_count        = 0

    func_stats = FuncStats()
    func_dict  = {}

    for line in raw_ptx.splitlines():
        # strip off comments
        if line.find("//") != -1:
            line = line[:line.find("//")]

        # trim whitespace and trailing commas
        line = line.strip(" \t\n\r,")

        # keep track of when we hit a function declaration or definition
        if is_func_or_decl(line):
            begin_func_or_decl   = True
            func_stats.func_name = get_func_name(line)

        # keep track of braces so we can have some picture of nesting
        brace_count = brace_count + line.count('{')
        brace_count = brace_count - line.count('}')

        # if we hit a call.uni, we need to look for the name of the callee
        # on the next pass
        if is_call_start(line):
            begin_call            = True
            func_stats.call_count = func_stats.call_count + 1
        elif begin_call:
            begin_call                 = False
            func_stats.call_dict[line] = func_stats.call_dict.get(line, 0) + 1

        pred, opcode, args = parse_inst(line)
        if in_func and opcode != "":
            func_stats.inst_count      = func_stats.inst_count + 1
            func_stats.mem_count       = func_stats.mem_count + is_mem_inst(line)
            inst                       = opcode
            func_stats.inst_dict[inst] = func_stats.inst_dict.get((inst), 0) + 1

        if begin_func_or_decl and brace_count == 1:
            in_func            = True
            begin_func_or_decl = False

        # exiting the function -- save off the stats
        if in_func and brace_count == 0:
            func_dict[func_stats.func_name] = func_stats

            # prepare for the next iteration
            func_stats         = FuncStats()
            begin_func_or_decl = False
            begin_call         = False
            in_func            = False
            brace_count        = 0

    return func_dict

#-------------------------------------------------------------------------------

def gather_file_stats(func_dict):
    file_stats             = FileStats()
    file_stats.total_funcs = len(func_dict)
    file_stats.csv_string  = "Function Name, Total Instructions, Memory Instructions, Function Calls\n"

    # print the function statistics
    for key, entry in sorted(func_dict.items()):
        func_name                   = entry.func_name
        file_stats.total_insts     += entry.inst_count
        file_stats.total_mem_insts += entry.mem_count
        file_stats.total_calls     += entry.call_count

        # keep track of the largest function
        if entry.inst_count > file_stats.largest_size:
            file_stats.largest_function = func_name
            file_stats.largest_size     = entry.inst_count

        # clamp the counts to prevent divide-by-zero
        inst_count = max(1, entry.inst_count)
        mem_count  = max(1, entry.mem_count)
        call_count = max(1, entry.call_count)

        summary_string  = ""
        summary_string += " Function:            {}\n".format(func_name)
        summary_string += " Total Instructions:  {}\n".format(inst_count)
        summary_string += " Memory Instructions: {} ({:.2f}%)\n".format(mem_count, 100 * float(mem_count)/inst_count)
        summary_string += " Total Calls:         {}\n".format(entry.call_count)

        summary_string += "\n"
        summary_string += "Top 10 Function Calls (name, count, %):\n"
        for name, count in itertools.islice(sorted(entry.call_dict.items(), key=lambda item: -item[1]), 0, 11):
            summary_string += " {}, {}, {:.2f}%\n".format(name, count, 100 * float(count)/max(1,file_stats.total_calls))

        # print the instruction mix
        summary_string += "\n"
        summary_string += "Instruction Mix (name, count, %):\n"
        for inst, count in sorted(entry.inst_dict.items()):
            summary_string += " {}, {}, {:.2f}%\n".format(inst, count, 100 * float(count)/inst_count)
            file_stats.inst_dict[inst] = file_stats.inst_dict.get((inst), 0) + count

        # print the call mix
        summary_string += "\n"
        summary_string += "Function Calls (name, count, %):\n"
        for name, count in sorted(entry.call_dict.items()):
            file_stats.func_calls[name] = file_stats.func_calls.get((name), 0) + count
            summary_string += " {}, {}, {:.2f}%\n".format(name, count, 100 * float(count)/call_count)
            if not name in func_dict:
                file_stats.extern_funcs[name] = 1

        file_stats.summary_dict[func_name] = summary_string

        file_stats.csv_string += "{},{},{},{}\n".format(func_name, entry.inst_count, entry.mem_count, entry.call_count)

    return file_stats

#-------------------------------------------------------------------------------

def generate_summary(func_dict, file_stats, out_name):
    summary_string  = "Overall Stats\n"
    summary_string += " Total Functions:     {}\n".format(file_stats.total_funcs)
    summary_string += " Total Instructions:  {}\n".format(file_stats.total_insts)
    summary_string += " Memory Instructions: {} ({:.2f}%)\n".format(file_stats.total_mem_insts, 100 * float(file_stats.total_mem_insts)/max(1, file_stats.total_insts))
    summary_string += " Total Calls:         {}\n".format(file_stats.total_calls)
    summary_string += " Biggest Function:    {} ({} instructions)\n".format(file_stats.largest_function, file_stats.largest_size)

    summary_string += "\n"
    summary_string += "Top 10 Function Calls (name, count, %):\n"
    for name, count in itertools.islice(sorted(file_stats.func_calls.items(), key=lambda item: -item[1]), 0, 11):
        summary_string += " {}, {}, {:.2f}%\n".format(name, count, 100 * float(count)/max(1,file_stats.total_calls))

    summary_string += "\n"
    summary_string += "Function Call Counts (name, count, %):\n"
    for name, count in sorted(file_stats.func_calls.items()):
        summary_string += " {}, {}, {:.2f}%\n".format(name, count, 100 * float(count)/max(1,file_stats.total_calls))

    summary_string += "\n"
    summary_string += "Instruction Mix (name, count, %):\n"
    for name, count in sorted(file_stats.inst_dict.items()):
        summary_string += " {}, {}, {:.2f}%\n".format(name, count, 100 * float(count)/max(1,file_stats.total_insts))

    summary_string += "\n"
    summary_string += "Uncalled Functions:\n"
    for name, count in sorted(func_dict.items()):
        if not name in file_stats.func_calls:
            summary_string += " {}\n".format(name)

    summary_string += "\n"
    summary_string += "Extern Functions:\n"
    for name, val in sorted(file_stats.extern_funcs.items()):
        summary_string += " {}\n".format(name)

    with open("{}-summary.txt".format(out_name), "w") as summary_file:
        summary_file.write(summary_string)
        for func_name, summary in sorted(file_stats.summary_dict.items()):
            summary_file.write("\n#---------------------------------------\n")
            summary_file.write(summary)

    with open("{}.csv".format(out_name), "w") as csv_file:
        csv_file.write(file_stats.csv_string)

    return summary_string

#-------------------------------------------------------------------------------

def wrap_name(name):
    wrapped_name = ""
    cur_line     = ""
    last_char    = ''
    wrap_len     = 20
    split_pos    = 0
    line_pos     = 0

    for cur_char in name:
        cur_line = cur_line + cur_char
        line_pos = line_pos + 1

        if not cur_char.isalpha():
            # prefer to split at non-alphabetic characters
            split_pos = line_pos
        elif last_char.isupper() and not cur_char.isupper():
            # otherwise, prefer to split at the first capitalized letter
            # of some group
            split_pos = line_pos - 2

        if line_pos == wrap_len:
            wrapped_name = wrapped_name + cur_line[:split_pos] + '\n'
            cur_line     = cur_line[split_pos:]
            line_pos     = len(cur_line)

        last_char = cur_char

    wrapped_name = wrapped_name + cur_line + '\n'
    return wrapped_name.strip(" \t\n\r,")

#-------------------------------------------------------------------------------

def generate_callgraph(func_dict, file_stats, out_name):
    callgraph = graphviz.Digraph("{} Call Graph".format(out_name),
                                 filename="{}-callgraph.gv".format(out_name),
                                 engine="dot")

    callgraph.attr("graph", rankdir="TB")
    callgraph.attr("graph", splines="true")
    callgraph.attr("graph", concentrate="false")
    callgraph.attr("graph", clusterrank="none")
    callgraph.attr("node",  fontname="Courier New")
    callgraph.attr("node",  shape="square")
    callgraph.attr("node",  nojustify="true")
    callgraph.attr("node",  width="2.5")
    callgraph.attr("node",  height="2.5")
    callgraph.attr("edge",  penwidth="3")
    callgraph.attr("edge",  constrain="false")

    # create a summary node
    with callgraph.subgraph(name="Summary") as c:
        c.attr("graph", rankdir="TB")
        c.attr("graph", splines="true")
        c.attr("graph", concentrate="false")
        c.attr("graph", clusterrank="none")
        c.attr("node",  fontname="Courier New")
        c.attr("node",  shape="square")
        c.attr("node",  nojustify="false")
        c.attr("node",  width="2.5")
        c.attr("node",  height="2.5")
        c.attr("edge",  penwidth="3")
        c.attr("edge",  constrain="false")
        c.node(name="SUMMARY",
               label="Summary\linsts: {}\lfuncs: {}\lcalls: {}\l".format
               (file_stats.total_insts, file_stats.total_funcs, file_stats.total_calls))

    # create the nodes
    for func_name, entry in sorted(func_dict.items()):
        num_calls_to = 0
        if func_name in file_stats.func_calls:
            num_calls_to = file_stats.func_calls[func_name]
        label = "{}\n\ninsts:      {}\lcalls from: {}\lcalls to:   {}\l".format(
            wrap_name(func_name),
            entry.inst_count,
            entry.call_count,
            num_calls_to)
        callgraph.node(name=wrap_name(func_name), label=label)

    # create nodes for the extern functions
    callgraph.attr("node", style="dashed")
    for key, entry in sorted(file_stats.extern_funcs.items()):
        func_name = key
        callgraph.node(wrap_name(func_name))

    # cycle through a handful of edge colors to add a little eye candy
    color_idx   = 0
    edge_colors = [ "#FF000080",  # red
                    "#00800F80",  # green
                    "#FFAF0080",  # orange
                    "#0000FF80",  # blue
                    "#40404080" ] # gray

    # create the edges
    for key, entry in sorted(func_dict.items()):
        func_name = entry.func_name
        callgraph.attr("edge",  color=edge_colors[color_idx])
        for calleeName, count in sorted(entry.call_dict.items()):
            callgraph.attr("edge",  tooltip="{} -> {}".format(func_name, calleeName))
            callgraph.edge(wrap_name(func_name), wrap_name(calleeName))
        color_idx = (color_idx + 1) % len(edge_colors)

    callgraph.render()

    return

#-------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        sys.exit("Wrong number of arguments\nUsage: $ parse-ptx.py <path-to-PTX-file>")

    in_name = sys.argv[1]
    if not os.path.isfile(in_name):
        sys.exit("Unable to open PTX file")

    out_name   = os.path.splitext(os.path.basename(in_name))[0]
    func_dict  = parse_ptx(in_name)
    file_stats = gather_file_stats(func_dict)

    generate_summary(func_dict, file_stats, out_name)
    generate_callgraph(func_dict, file_stats, out_name)

    return

if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------
