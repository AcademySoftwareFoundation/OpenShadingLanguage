#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#osl_concat
command += testshade("-t 1 -g 64 64 str_concat -od uint8 -o res concat_ref.tif -o res_m concat_m_ref.tif")

#osl_stoi
command += testshade("-t 1 -g 64 64 str_stoi -od uint8 -o res stoi_ref.tif -o res_m stoi_m_ref.tif")

#osl_endswith
command += testshade("-t 1 -g 64 64 str_endswith -od uint8 -o res_t endswith_t_ref.tif -o res_f endswith_f_ref.tif"
                     " -o res_t_m endswith_t_m_ref.tif -o res_f_m endswith_f_m_ref.tif")

#osl_getchar
command += testshade("-t 1 -g 64 64 str_getchar -od uint8 str_getchar  -o res_t1 getchar_t1_ref.tif -o res_t2 getchar_t2_ref.tif"
                     " -o res_f1 getchar_f1_ref.tif -o res_f2 getchar_f2_ref.tif"
                     " -o res_t1_m getchar_t1_m_ref.tif -o res_t2_m getchar_t2_m_ref.tif"
                     " -o res_f1_m getchar_f1_m_ref.tif -o res_f2_m getchar_f2_m_ref.tif") 
#osl_hash
command += testshade("-t 1 -g 64 64 str_hash -od uint8  -o res hash_ref.tif -o res_m hash_m_ref.tif")

#osl_startswith
command += testshade("-t 1 -g 64 64 str_startswith -od uint8 -o res_t startswith_t_ref.tif -o res_f startswith_f_ref.tif"
                     " -o res_t_m startswith_t_m_ref.tif -o res_f_m startswith_f_m_ref.tif")

#osl_stof
command += testshade("-t 1 -g 64 64 str_stof -od uint8 -o res stof_ref.tif -o res_m stof_m_ref.tif")

#osl_strlen
command += testshade("-t 1 -g 64 64 str_strlen -od uint8 -o res strlen_ref.tif -o res_m strlen_m_ref.tif")

#osl_substr
command += testshade("-t 1 -g 64 64 str_substr -od uint8 -o res sub_ref.tif -o res1 sub1_ref.tif -o res2 sub2_ref.tif"
                     " -o res_m sub_m_ref.tif -o res1_m sub1_m_ref.tif -o res2_m sub2_m_ref.tif")
                                                    

outputs = [ 
    "concat_ref.tif",
    "concat_m_ref.tif",
    "stoi_ref.tif",
    "stoi_m_ref.tif",
    "endswith_t_ref.tif",
    "endswith_f_ref.tif",
    "endswith_t_m_ref.tif",
    "endswith_f_m_ref.tif",
    "getchar_t1_ref.tif",
    "getchar_t2_ref.tif",
    "getchar_f1_ref.tif",
    "getchar_f2_ref.tif",
    "getchar_t1_m_ref.tif",
    "getchar_t2_m_ref.tif",
    "getchar_f1_m_ref.tif",
    "getchar_f2_m_ref.tif",
    "hash_ref.tif",
    "hash_m_ref.tif",
    "startswith_t_ref.tif",
    "startswith_f_ref.tif",
    "startswith_t_m_ref.tif",
    "startswith_f_m_ref.tif",
    "stof_ref.tif",
    "stof_m_ref.tif",
    "strlen_ref.tif",
    "strlen_m_ref.tif",
    "sub_ref.tif",
    "sub1_ref.tif",
    "sub2_ref.tif",
    "sub_m_ref.tif",
    "sub1_m_ref.tif",
    "sub2_m_ref.tif",
]

# expect a few LSB failures
failthresh = 0.008
failpercent = 3


