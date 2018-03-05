#!/usr/bin/env python3
from __future__ import print_function

# Copyright (c) 2018 Sony Pictures Imageworks Inc., et al.
# BSD 3-clause license.
# Distributed as part of Open Shading Language.
# https://github.com/imageworks/OpenShadingLanguage


# ----------------------------------------------------------------------
# To generate docdeep for docdeep, run this:
#     python3 docdeep.py -d docdeep docdeep.py > docdeep.md.html
# ----------------------------------------------------------------------


### <doc docdeep>
###
###                       **docdeep**
###
### Introduction
### ============
###
### `docdeep` is a utility that extracts documentation from source code and
### turns them into beautiful Markdeep documents.
###
### [Markdeep](https://casual-effects.com/markdeep) is a package by Morgan
### McGuire, an extension of Markdown with a whole bunch of nice features
### that make it extra nice for code, math, and diagrams. Read the Markdeep
### web page and look at the examples for details.
###
### `docdeep` is a little like `Doxygen`. A poor person's Doxygen. A very
### poor person. But without Doxygen's awkward syntax -- you just write
### the comments in Markdeep. And the output is very aesthetically pleasing.
### But it doesn't have any bells and whistles, like cross-referencing.
###
###
###
### Markup controls
### ===============
###
### ## Documentation lines.
###
### A line whose first three non-whitespace characters are either `///` or
### `###` is called a *doc-line*. Any other line is a *non-doc-line*.
###
### The `///` or `###` is the *doc-symbol*. Generally speaking, the `///`
### doc-symbol is used when doc-marking C or C++, and `###` when doc-marking
### Python, shell scripts, or other programs in languages where `#` is the
### comment character. For simplicity, in the rest of this document, we will
### always use `///` in our examples.
###
### ## Doc regions
###
### Doc-lines by themeslves don't do much, unless they are within a
### *doc-region*. The beginning of a doc-region is denoted by a doc-line
### whose first characters after the doc-symbol is
###
### <code>
### /// <doc region-name>
### </code>
###
### A doc region is ended one of three ways:
###
### 1. `&lt;/doc>` ends the active region.
### 2. `&lt;doc newname>` setting a new region name.
### 3. The end of the source file.
###
### When there is an active named region, any other doc lines will be
### appended (after stripping off the doc symbol itself) to the current
### doc region text.
###
### You can have multiple regions with the same name, in entirely separate
### parts of your source code. They will just be concatenated.
###
### Only one main doc region will be output by the `docdeep` program,
### specified with the `-d` command line argument. Any other doc-regions
### will not be included in the documention output of that run.
###
### However, one doc-region may *include* the text of another doc-region
### as follows:
###
### <code>
### /// <inc region-name>
### </code>
###
### ## Doc continuations and code regions
###
### If three dots follow the `doc` directive, like this:
###
### <code>
### /// <doc... region-name>
### </code>
###
### This causes *all* lines (until the end of the doc region) to be
### interpreted as Markdeep documentation, even it if doesn't start with
### the doc-symbol.
###
### Furthermore, denoting any region thusly:
### <code>
### /// <code>
### ...
### /// </code>
### </code>
###
### will designate the contents not only as a doc continuation, but also
### to be formatted as source code (mono space, syntax highlighted).
###
### ## API Explanations
###
### There is a special syntax for a common case: call-by-call explanations
### of API methods and their explanations. Of course, this may constitute
### the bulk of your auto-generated documentation. The two cases we are
### concerned about is *pre-code comments* and *post-code comments*.
###
### For pre-code comments, the contiguous comment region associated with
### a declaration immediately precedes the declaration. Such a comment
### set has its first line start with `///>`.  For post-code comments,
### the documentaiton comments follows the code declaration, and this is
### designated by having its first line start with `///<`.  I like to
### just remember that the `<` or `>` points in the direction of the code
### declaration that the comment applies to.
###
### This is perhaps best explained by example:
###
### <code>
### ///> Do the foo operation. Everybody knows what this is. You can write
### /// any comment you want, with full markdeep formatting! It applies
### /// to the declaration that will immediately follow this comment.
### float foo (float x, float y);
###
### void bar (int n);
### ///< The bar procedure. Note that with post-comments, you explain the
### /// function or method *after* the declaration itself.
### </code>
###
### This will generate the following output:
###
### <code>
### float foo (float x, float y);
### </code>
### Do the foo operation. Everybody knows what this is. You can write
### any comment you want, with full markdeep formatting! It applies
### to the declaration that will immediately follow this comment.
###
### <code>
### void bar (int n);
### </code>
### The bar procedure. Note that with post-comments, you explain the
### function or method *after* the declaration itself.
###
### -----------------------
###
### It is important to keep in mind that **this API explanation is extremely
### stupid**. The comment documentation is just a series of doc-lines with
### no break between them, whose *first* line starts with `///>' or `///>`
### (for pre-code and post-code docs, respectively). And the code that it
### documents is just the immediately preceding or following set of non-doc
### lines that contain *no* whitespace-only lines. There is no syntax
### parsing going on here -- it just takes those non-blank, non-doc lines,
### strips off any semicolons and any characters following the semicolon
### from each line in the block, and deletes anything in the whole block
### at or after the first opening brace (`{`) found in the block. Strangely
### enough, this is almost exactly what I want to document.
###
###
### Command line arguments
### ======================
###
### Run from the command line as follows:
###
### `$ python docdeep.py -d` *region* `-s` *style.css* `input1.h input2.cpp [...] > output.md.html`
###
### Arguments:
###
### `-d` *region*
### :  Specifies the name of the doc-region to generate output for. This is
###    required.
### `-s` *stylesheet*
### :  This optional argument will specify which CSS style sheet to include a
###    reference to.
###
### Any number of filenames may be specified. They will be processed in the
### order that they appear on the command line.
###

import sys
import re
import argparse

# pattern for beginning of special comment
docline_pattern = re.compile ('(^[ \t]*)((///)|(###))[<>]?([ \t]|$)(.*)')
doc_begin_pattern = re.compile ('(^[ \t]*)((///)|(###))[ ]*<doc(\.\.\.)?[ ]*(\w*)[ ]*>')
docendpattern = re.compile ('(^[ \t]*)((///)|(###))[ ]*</doc>')
api_precomment_pattern = re.compile ('(^[ \t]*)((///)|(###))>( )+( )?(.*)')
api_postcomment_pattern = re.compile ('(^[ \t]*)((///)|(###))<( )+( )?(.*)')
inc_pattern = re.compile ('(^[ \t]*)<inc[ ]+([^>]+)>')
code_pattern = re.compile ('(^[ \t]*)((///)|(###))[ ]*<code>')
codeend_pattern = re.compile ('(^[ \t]*)((///)|(###))[ ]*</code>')
trimbrace_pattern = re.compile ("([^{]*)")
trimsemi_pattern = re.compile ("([^;]*)")
blankline_pattern = re.compile ("^[ \t]*$")

region_name = '_'
alldocs = { '_' : '' }
DEBUG = False


# Utility: blob is a string (possibly containing many "lines" separated by
# '\n'). For each line of text in the blob, shave off the first `indent`
# characters. Then reassemble and return.
def shave_each_line (blob, indent) :
    r = ''
    lines = blob.split ('\n')
    for line in lines :
        r += line[indent:] + '\n'
    # strip trailing newline unless the original blob had a trailing newline
    if len(r) and r[-1] == '\n' and blob[-1] != '\n':
        r = r[:-1]
    return r

# Enumerated type for the state machine.
class LineType :
    BLANK = 0
    DOC = 2
    NONDOC = 3

# Append this doc blob to the current region, and clear the blob
def flush_doc_blob (doc_blob) :
    global alldocs, region_name
    if len(doc_blob) :
        alldocs[region_name] += doc_blob + '\n'
        doc_blob = ''

def read_input (filename, file) :
    global region_name, alldocs
    doc_cont_mode = False
    post_api_mode = False
    pre_api_mode = False
    indent = 0    # Amount of indentation we saw on the liast doc line
    lines = file.read().splitlines()
    code_blob = ''  # Running set of contiguous non-doc, non-blank lines
    doc_blob = ''   # Running set of contiguous doc lines

    # state machine
    line_type = LineType.BLANK
    last_line_type = LineType.BLANK

    for line in lines :

        # Figure out what type of line we're on, and remember what type
        # of line we saw last.
        last_line_type = line_type
        if blankline_pattern.match(line) :
            line_type = LineType.BLANK
        elif docline_pattern.match(line) :
            line_type = LineType.DOC
        else :
            line_type = LineType.NONDOC

        # print ('<!-- LINE----- ', line_type, line, ' -->\n')

        # Not a doc line, but we're in "continuation mode": append
        # (unindented) to the doc blob.
        if line_type != LineType.DOC and doc_cont_mode :
            doc_blob += line[indent:] + '\n'
            if DEBUG :
                alldocs[region_name] += '<!-- CONT ' + line[:35] + ' -->\n'
            continue

        # Blank line, not in continuation mode
        if line_type == LineType.BLANK :
            if DEBUG and last_line_type != LineType.BLANK :
                alldocs[region_name] += '<!-- BLANK -->\n'

            # FIXME: does this end a post-declaration API doc?
            if pre_api_mode :  # this blank ends a pre-api comment
                continue       #   ...keep reading

            # If this line ends a post-api comment, output the code blob
            if post_api_mode :
                if DEBUG :
                    alldocs[region_name] += ' <!-- blank ended post-api -->\n'
                # Trim everything past the first ; or { from the code blob
                alllines = code_blob.split('\n')
                # code_blob = ''
                new_code_blob = ''
                for oneline in alllines :
                    m = trimsemi_pattern.search (oneline)
                    oneline = m.group(1)
                    new_code_blob += oneline + '\n'
                new_code_blob = new_code_blob.rstrip(' \n\r')
                m = trimbrace_pattern.match(new_code_blob)
                alldocs[region_name] += ('~~~C\n' +
                                         shave_each_line(m.group(1),indent) +
                                         '\n~~~\n')
                post_api_mode = False

            # If this line ends a doc blob, output it
            if len(doc_blob) :
                alldocs[region_name] += doc_blob + '\n'
                doc_blob = ''
            # Blank lines clear the code blob
            code_blob = ''
            continue

        # Non-blank, non-doc line: append to code blob and move on.
        if line_type == LineType.NONDOC :
            if DEBUG :
                alldocs[region_name] += '<!-- NONDOC ' + line[:35] + '  -->\n'
            code_blob += line + '\n'
            continue

        # Remaining cases are all doc lines!

        # Any doc line resets the pre-doc-symbol indentation level
        m = docline_pattern.match(line)
        indent = len(m.group(1))

        # Handle <doc> and <doc...>  : start of new doc section
        m = doc_begin_pattern.match(line)
        if m :
            flush_doc_blob (doc_blob)
            doc_blob = ''
            # If it led with <doc...> it also starts continuation mode
            doc_cont_mode = (m.group(5) == '...')
            # The <doc> directive gave the region name.
            r = m.group(6)
            r.strip()
            if r == '' :
                r = '_'
            if DEBUG :
                alldocs[region_name] += '<!-- DOCBEGIN ' + r + '  -->\n'
            region_name = r
            if not (region_name in alldocs) :
                alldocs[region_name] = ''
            continue
        #
        # Handle </doc>
        m = docendpattern.match(line)
        if m :
            if DEBUG :
                alldocs[region_name] += '<!-- DOCEND ' + region_name + '-->\n'
            flush_doc_blob (doc_blob)
            doc_blob = ''
            doc_cont_mode = False
            region_name = '_'
            continue
        #
        # Handle start of post-declaration API comment
        m = api_postcomment_pattern.match(line)
        if m :
            if DEBUG :
                alldocs[region_name] += '<!-- start post-decl api ' + line[:35] + '-->\n'
            flush_doc_blob (doc_blob)
            doc_blob = ''
            # Deduce indentation level
            indent = len(m.group(1))
            post_api_mode = True
            remainder = m.group(7)
            doc_blob += remainder + '\n'
            continue
        #
        # Handle start of pre-declaration API comment
        m = api_precomment_pattern.match(line)
        if m :
            flush_doc_blob (doc_blob)
            doc_blob = ''
            indent = len(m.group(1))
            post_api_mode = True
            remainder = m.group(7)
            doc_blob += remainder + '\n'
            continue
        #
        # Handle code section <code> ... </code>
        m = code_pattern.match(line)
        if m :
            flush_doc_blob (doc_blob)
            doc_blob = ''
            if DEBUG :
                doc_blob += '<!-- start code in '+ region_name+ '-->\n'
            indent = len(m.group(1))
            doc_blob += '<script type="preformatted">\n~~~C\n'
            doc_cont_mode = True
            continue
        if codeend_pattern.match(line) :
            doc_cont_mode = False
            doc_blob += '~~~\n</script>\n'
            if DEBUG :
                doc_blob += '<!-- end code in '+ region_name+ '-->\n'
            flush_doc_blob (doc_blob)
            doc_blob = ''
            continue
        #
        # Last case: just a continuing doc line
        m = docline_pattern.match (line)
        if m :
            if DEBUG :
                alldocs[region_name] += '<!-- doc cont ' + line[:35] + '  -->\n'
            contents = m.group(6)
            doc_blob += contents + '\n'
            continue
        print ('<!-- REMAINING CASE:', line, '-->\n')
    flush_doc_blob (doc_blob)
    region_name = '_'


def output_blob (blob) :
    for line in blob.splitlines() :
        m = inc_pattern.match (line)
        if m and (m.group(2) in alldocs) :
            # print ('Want to include', m.group(2))
            output_blob (alldocs[m.group(2)])
        else :
            print (line)


parser = argparse.ArgumentParser (prog='docdeep',
                                  description='Turn source comments into markdeep document')
parser.add_argument ('-d', dest='docname', default='main',
                     help='Name of documentation (default: "main")')
parser.add_argument ('-s', dest='stylesheet', default='',
                     help='Name of style sheet (default: "")')
parser.add_argument ('--debug', dest='DEBUG', action='store_const',
                     const=True, default=False)
parser.add_argument ('filenames', nargs='+')
# parser.add_argument ('-o', dest='output_filename', nargs=1,
#                      help='Output filename (default: stdout)')
args = parser.parse_args()
DEBUG = args.DEBUG

for filename in args.filenames :
    with open(filename) as file:
        read_input (filename, file)

print ('<meta charset="utf-8">\n')
if args.docname in alldocs :
    output_blob (alldocs[args.docname])
else :
    print ("Ick! could not find docs for", args.docname)

print ('<!-- Markdeep: --><style class="fallback">body{visibility:hidden;white-space:pre;font-family:monospace}</style><script src="markdeep.min.js"></script><script src="https://casual-effects.com/markdeep/latest/markdeep.min.js?"></script><script>window.alreadyProcessedMarkdeep||(document.body.style.visibility="visible")</script>')
if args.stylesheet :
    print ('<link rel="stylesheet" href="{}">'.format (args.stylesheet))
