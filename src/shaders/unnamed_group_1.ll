; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #0

declare ptr @osl_allocate_closure_component(ptr, i32, i32) #1

define void @osl_init_group_unnamed_group_1(ptr %shaderglobals_ptr, ptr %groupdata_ptr, ptr %userdata_base_ptr, ptr %output_base_ptr, i32 %shadeindex, ptr %interactive_params_ptr) #1 !dbg !8 {
bb_osl_init_group_unnamed_group_1_0:
  ret void, !dbg !12
}

define void @osl_layer_group_unnamed_group_1_name_matte_0(ptr %shaderglobals_ptr, ptr %groupdata_ptr, ptr %userdata_base_ptr, ptr %output_base_ptr, i32 %shadeindex, ptr %interactive_params_ptr) #1 !dbg !13 {
bb_osl_layer_group_unnamed_group_1_name_matte_0_1:
  %0 = getelementptr %Groupdata, ptr %groupdata_ptr, i32 0, i32 0, i32 0, !dbg !14
  %1 = call ptr @osl_allocate_closure_component(ptr %shaderglobals_ptr, i32 3, i32 24), !dbg !15
  %2 = icmp ne ptr %1, null, !dbg !15
  br i1 %2, label %bb_non_null_closure_2, label %bb_3, !dbg !15

bb_non_null_closure_2:                            ; preds = %bb_osl_layer_group_unnamed_group_1_name_matte_0_1
  %3 = getelementptr %ClosureComponent, ptr %1, i32 0, i32 2, !dbg !15
  call void @llvm.memset.p0.i64(ptr align 4 %3, i8 0, i64 24, i1 false), !dbg !15
  %4 = getelementptr %ShaderGlobals, ptr %shaderglobals_ptr, i32 0, i32 3, !dbg !15
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %3, ptr align 4 %4, i64 12, i1 false), !dbg !15
  br label %bb_3, !dbg !15

bb_3:                                             ; preds = %bb_non_null_closure_2, %bb_osl_layer_group_unnamed_group_1_name_matte_0_1
  %5 = getelementptr %ShaderGlobals, ptr %shaderglobals_ptr, i32 0, i32 23, !dbg !15
  store ptr %1, ptr %5, align 8, !dbg !15
  ret void, !dbg !15
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

