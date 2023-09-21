#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("--res 128 128 --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1 -o result out.tif test")
command += testshade("--res 128 128 --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1 --center -od uint8 -o Cout out2.tif test2")
outputs.append ("out.tif")
outputs.append ("out2.tif")

command += testshade("--res 144 144 --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1 --center -od uint8 --layer srcLayer varying_eq -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.eq.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_ge -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.ge.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_gt -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.gt.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.int.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_le -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.le.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_lt -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.lt.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_ne -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.ne.tif")
outputs.append ("out2.eq.tif")
outputs.append ("out2.ge.tif")
outputs.append ("out2.gt.tif")
outputs.append ("out2.int.tif")
outputs.append ("out2.le.tif")
outputs.append ("out2.lt.tif")
outputs.append ("out2.ne.tif")

command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_not_bool -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.not_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_not_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.not_int.tif")
outputs.append ("out2.not_bool.tif")
outputs.append ("out2.not_int.tif")

command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_xor_bool_bool -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.xor_bool_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_xor_bool_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.xor_bool_int.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_xor_int_bool -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.xor_int_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_xor_int_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.xor_int_int.tif")
outputs.append ("out2.xor_bool_bool.tif")
outputs.append ("out2.xor_bool_int.tif")
outputs.append ("out2.xor_int_bool.tif")
outputs.append ("out2.xor_int_int.tif")


command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitand_bool_bool -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.bitand_bool_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitand_bool_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.bitand_bool_int.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitand_int_bool -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.bitand_int_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitand_int_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.bitand_int_int.tif")
outputs.append ("out2.bitand_bool_bool.tif")
outputs.append ("out2.bitand_bool_int.tif")
outputs.append ("out2.bitand_int_bool.tif")
outputs.append ("out2.bitand_int_int.tif")

command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitor_bool_bool -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.bitor_bool_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitor_bool_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.bitor_bool_int.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitor_int_bool -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.bitor_int_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitor_int_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.bitor_int_int.tif")
outputs.append ("out2.bitor_bool_bool.tif")
outputs.append ("out2.bitor_bool_int.tif")
outputs.append ("out2.bitor_int_bool.tif")
outputs.append ("out2.bitor_int_int.tif")

command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_compl_bool -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.compl_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_compl_int -layer testLayer test2  --connect srcLayer val testLayer blue -o testLayer.Cout out2.compl_int.tif")
outputs.append ("out2.compl_bool.tif")
outputs.append ("out2.compl_int.tif")

command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_gt -layer testLayer test4  --connect srcLayer val testLayer blue_param -o testLayer.Cout out4A.gt.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_compl_bool -layer testLayer test4  --connect srcLayer val testLayer blue_param -o testLayer.Cout out4A.compl_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitand_bool_bool -layer testLayer test4  --connect srcLayer val testLayer blue_param -o testLayer.Cout out4A.bitand_bool_bool.tif")
outputs.append ("out4A.gt.tif")
outputs.append ("out4A.compl_bool.tif")
outputs.append ("out4A.bitand_bool_bool.tif")

# renderer outputs are not allowed to become bools, test that impact when they "could" have been forced to be bools otherwise
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_gt -layer testLayer test4  --connect srcLayer val testLayer blue_param -o testLayer.Cout out4B.gt.tif -o testLayer.blue_param out4_blue_param.gt.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_compl_bool -layer testLayer test4  --connect srcLayer val testLayer blue_param -o testLayer.Cout out4B.compl_bool.tif -o testLayer.blue_param out4_blue_param.compl_bool.tif")
command += testshade("--res 144 144 --center --options opt_batched_analysis=1,dump_forced_llvm_bool_symbols=1  -od uint8 --layer srcLayer varying_bitand_bool_bool -layer testLayer test4  --connect srcLayer val testLayer blue_param -o testLayer.Cout out4B.bitand_bool_bool.tif -o testLayer.blue_param out4_blue_param.bitand_bool_bool.tif")
outputs.append ("out4B.gt.tif")
outputs.append ("out4B.compl_bool.tif")
outputs.append ("out4B.bitand_bool_bool.tif")
outputs.append ("out4_blue_param.gt.tif")
outputs.append ("out4_blue_param.compl_bool.tif")
outputs.append ("out4_blue_param.bitand_bool_bool.tif")
