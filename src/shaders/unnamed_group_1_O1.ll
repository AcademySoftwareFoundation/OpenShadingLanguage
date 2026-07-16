; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #0

declare ptr @osl_allocate_closure_component(ptr, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define void @osl_init_group_unnamed_group_1(ptr nocapture readnone %shaderglobals_ptr, ptr nocapture readnone %groupdata_ptr, ptr nocapture readnone %userdata_base_ptr, ptr nocapture readnone %output_base_ptr, i32 %shadeindex, ptr nocapture readnone %interactive_params_ptr) local_unnamed_addr #2 !dbg !8 {
bb_osl_init_group_unnamed_group_1_0:
  ret void, !dbg !12
}

define void @osl_layer_group_unnamed_group_1_name_matte_0(ptr %shaderglobals_ptr, ptr nocapture readnone %groupdata_ptr, ptr nocapture readnone %userdata_base_ptr, ptr nocapture readnone %output_base_ptr, i32 %shadeindex, ptr nocapture readnone %interactive_params_ptr) local_unnamed_addr #1 !dbg !13 {
bb_osl_layer_group_unnamed_group_1_name_matte_0_1:
  %0 = tail call ptr @osl_allocate_closure_component(ptr %shaderglobals_ptr, i32 3, i32 24), !dbg !14
  %.not = icmp eq ptr %0, null, !dbg !14
  br i1 %.not, label %bb_3, label %bb_non_null_closure_2, !dbg !14

bb_non_null_closure_2:                            ; preds = %bb_osl_layer_group_unnamed_group_1_name_matte_0_1
  %1 = getelementptr %ClosureComponent, ptr %0, i64 0, i32 2, !dbg !14
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(24) %1, i8 0, i64 24, i1 false), !dbg !14
  %2 = getelementptr %ShaderGlobals, ptr %shaderglobals_ptr, i64 0, i32 3, !dbg !14
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(12) %1, ptr noundef nonnull align 4 dereferenceable(12) %2, i64 12, i1 false), !dbg !14
  br label %bb_3, !dbg !14

bb_3:                                             ; preds = %bb_non_null_closure_2, %bb_osl_layer_group_unnamed_group_1_name_matte_0_1
  %3 = getelementptr %ShaderGlobals, ptr %shaderglobals_ptr, i64 0, i32 23, !dbg !14
  store ptr %0, ptr %3, align 8, !dbg !14
  ret void, !dbg !14
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #3

