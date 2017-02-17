; ModuleID = 'llvm_ops.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.OSL::Dual2" = type { float, float, float }
%"class.OSL::Dual2.0" = type { %"class.Imath_2_2::Vec3", %"class.Imath_2_2::Vec3", %"class.Imath_2_2::Vec3" }
%"class.Imath_2_2::Vec3" = type { float, float, float }
%"class.Imath_2_2::Matrix44" = type { [4 x [4 x float]] }
%"class.Imath_2_2::Vec3.1" = type { %"class.OSL::Dual2", %"class.OSL::Dual2", %"class.OSL::Dual2" }
%"class.Iex_2_2::BaseExc" = type { %"class.std::exception", %"class.std::basic_string", %"class.std::basic_string" }
%"class.std::exception" = type { i32 (...)** }
%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%"class.Imath_2_2::SingMatrixExc" = type { %"class.Iex_2_2::MathExc" }
%"class.Iex_2_2::MathExc" = type { %"class.Iex_2_2::BaseExc" }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = global i8* null, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTSN7Iex_2_27MathExcE = linkonce_odr constant [19 x i8] c"N7Iex_2_27MathExcE\00"
@_ZTIN7Iex_2_27BaseExcE = external constant i8*
@_ZTIN7Iex_2_27MathExcE = linkonce_odr constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([19 x i8]* @_ZTSN7Iex_2_27MathExcE, i32 0, i32 0), i8* bitcast (i8** @_ZTIN7Iex_2_27BaseExcE to i8*) }
@.str = private unnamed_addr constant [31 x i8] c"Cannot invert singular matrix.\00", align 1
@_ZTSN9Imath_2_213SingMatrixExcE = linkonce_odr constant [28 x i8] c"N9Imath_2_213SingMatrixExcE\00"
@_ZTIN9Imath_2_213SingMatrixExcE = linkonce_odr constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([28 x i8]* @_ZTSN9Imath_2_213SingMatrixExcE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN7Iex_2_27MathExcE to i8*) }
@_ZTVN9Imath_2_213SingMatrixExcE = linkonce_odr unnamed_addr constant [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN9Imath_2_213SingMatrixExcE to i8*), i8* bitcast (void (%"class.Iex_2_2::BaseExc"*)* @_ZN7Iex_2_27BaseExcD2Ev to i8*), i8* bitcast (void (%"class.Imath_2_2::SingMatrixExc"*)* @_ZN9Imath_2_213SingMatrixExcD0Ev to i8*), i8* bitcast (i8* (%"class.Iex_2_2::BaseExc"*)* @_ZNK7Iex_2_27BaseExc4whatEv to i8*)]
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_llvm_ops.cpp, i8* null }]

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) #1

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: nounwind readnone uwtable
define float @osl_sin_ff(float %a) #3 {
  %1 = fmul float %a, 0x3FD45F3060000000
  %2 = tail call float @copysignf(float 5.000000e-01, float %1) #12
  %3 = fadd float %1, %2
  %4 = fptosi float %3 to i32
  %5 = sitofp i32 %4 to float
  %6 = fmul float %5, -3.140625e+00
  %7 = fadd float %6, %a
  %8 = fmul float %5, 0xBF4FB40000000000
  %9 = fadd float %8, %7
  %10 = fmul float %5, 0xBE84440000000000
  %11 = fadd float %10, %9
  %12 = fmul float %5, 0xBD968C2340000000
  %13 = fadd float %12, %11
  %14 = fsub float 0x3FF921FB60000000, %13
  %15 = fsub float 0x3FF921FB60000000, %14
  %16 = fmul float %15, %15
  %17 = and i32 %4, 1
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %_ZN11OpenImageIO4v1_78fast_sinEf.exit, label %19

; <label>:19                                      ; preds = %0
  %20 = fsub float -0.000000e+00, %15
  br label %_ZN11OpenImageIO4v1_78fast_sinEf.exit

_ZN11OpenImageIO4v1_78fast_sinEf.exit:            ; preds = %0, %19
  %.0.i = phi float [ %20, %19 ], [ %15, %0 ]
  %21 = fmul float %16, 0x3EC5E150E0000000
  %22 = fadd float %21, 0xBF29F75D60000000
  %23 = fmul float %16, %22
  %24 = fadd float %23, 0x3F8110EEE0000000
  %25 = fmul float %16, %24
  %26 = fadd float %25, 0xBFC55554C0000000
  %27 = fmul float %26, %.0.i
  %28 = fmul float %16, %27
  %29 = fadd float %.0.i, %28
  %30 = tail call float @fabsf(float %29) #12
  %31 = fcmp ogt float %30, 1.000000e+00
  %u.0.i = select i1 %31, float 0.000000e+00, float %29
  ret float %u.0.i
}

; Function Attrs: nounwind uwtable
define void @osl_sin_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FD45F3060000000
  %4 = tail call float @copysignf(float 5.000000e-01, float %3) #12
  %5 = fadd float %3, %4
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fmul float %7, -3.140625e+00
  %9 = fadd float %2, %8
  %10 = fmul float %7, 0xBF4FB40000000000
  %11 = fadd float %10, %9
  %12 = fmul float %7, 0xBE84440000000000
  %13 = fadd float %12, %11
  %14 = fmul float %7, 0xBD968C2340000000
  %15 = fadd float %14, %13
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = fmul float %17, %17
  %19 = and i32 %6, 1
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %21, label %23

; <label>:21                                      ; preds = %0
  %22 = fsub float -0.000000e+00, %17
  br label %23

; <label>:23                                      ; preds = %21, %0
  %.0.i.i = phi float [ %22, %21 ], [ %17, %0 ]
  %24 = fmul float %18, 0x3EC5E150E0000000
  %25 = fadd float %24, 0xBF29F75D60000000
  %26 = fmul float %18, %25
  %27 = fadd float %26, 0x3F8110EEE0000000
  %28 = fmul float %18, %27
  %29 = fadd float %28, 0xBFC55554C0000000
  %30 = fmul float %29, %.0.i.i
  %31 = fmul float %18, %30
  %32 = fadd float %.0.i.i, %31
  %33 = fmul float %18, 0xBE923DB120000000
  %34 = fadd float %33, 0x3EFA00F160000000
  %35 = fmul float %18, %34
  %36 = fadd float %35, 0xBF56C16B00000000
  %37 = fmul float %18, %36
  %38 = fadd float %37, 0x3FA5555540000000
  %39 = fmul float %18, %38
  %40 = fadd float %39, -5.000000e-01
  %41 = fmul float %18, %40
  %42 = fadd float %41, 1.000000e+00
  br i1 %20, label %43, label %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit

; <label>:43                                      ; preds = %23
  %44 = fsub float -0.000000e+00, %42
  br label %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit

_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit:            ; preds = %23, %43
  %cu.0.i.i = phi float [ %44, %43 ], [ %42, %23 ]
  %45 = tail call float @fabsf(float %32) #12
  %46 = fcmp ogt float %45, 1.000000e+00
  %su.0.i.i = select i1 %46, float 0.000000e+00, float %32
  %47 = tail call float @fabsf(float %cu.0.i.i) #12
  %48 = fcmp ogt float %47, 1.000000e+00
  %cu.1.i.i = select i1 %48, float 0.000000e+00, float %cu.0.i.i
  %49 = getelementptr inbounds i8* %a, i64 4
  %50 = bitcast i8* %49 to float*
  %51 = load float* %50, align 4, !tbaa !1
  %52 = fmul float %51, %cu.1.i.i
  %53 = getelementptr inbounds i8* %a, i64 8
  %54 = bitcast i8* %53 to float*
  %55 = load float* %54, align 4, !tbaa !1
  %56 = fmul float %cu.1.i.i, %55
  %57 = insertelement <2 x float> undef, float %su.0.i.i, i32 0
  %58 = insertelement <2 x float> %57, float %52, i32 1
  %59 = bitcast i8* %r to <2 x float>*
  store <2 x float> %58, <2 x float>* %59, align 4
  %60 = getelementptr inbounds i8* %r, i64 8
  %61 = bitcast i8* %60 to float*
  store float %56, float* %61, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sin_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FD45F3060000000
  %4 = tail call float @copysignf(float 5.000000e-01, float %3) #12
  %5 = fadd float %3, %4
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fmul float %7, -3.140625e+00
  %9 = fadd float %2, %8
  %10 = fmul float %7, 0xBF4FB40000000000
  %11 = fadd float %10, %9
  %12 = fmul float %7, 0xBE84440000000000
  %13 = fadd float %12, %11
  %14 = fmul float %7, 0xBD968C2340000000
  %15 = fadd float %14, %13
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = fmul float %17, %17
  %19 = and i32 %6, 1
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %_ZN11OpenImageIO4v1_78fast_sinEf.exit3, label %21

; <label>:21                                      ; preds = %0
  %22 = fsub float -0.000000e+00, %17
  br label %_ZN11OpenImageIO4v1_78fast_sinEf.exit3

_ZN11OpenImageIO4v1_78fast_sinEf.exit3:           ; preds = %0, %21
  %.0.i1 = phi float [ %22, %21 ], [ %17, %0 ]
  %23 = fmul float %18, 0x3EC5E150E0000000
  %24 = fadd float %23, 0xBF29F75D60000000
  %25 = fmul float %18, %24
  %26 = fadd float %25, 0x3F8110EEE0000000
  %27 = fmul float %18, %26
  %28 = fadd float %27, 0xBFC55554C0000000
  %29 = fmul float %28, %.0.i1
  %30 = fmul float %18, %29
  %31 = fadd float %.0.i1, %30
  %32 = tail call float @fabsf(float %31) #12
  %33 = fcmp ogt float %32, 1.000000e+00
  %u.0.i2 = select i1 %33, float 0.000000e+00, float %31
  %34 = bitcast i8* %r_ to float*
  store float %u.0.i2, float* %34, align 4, !tbaa !1
  %35 = getelementptr inbounds i8* %a_, i64 4
  %36 = bitcast i8* %35 to float*
  %37 = load float* %36, align 4, !tbaa !1
  %38 = fmul float %37, 0x3FD45F3060000000
  %39 = tail call float @copysignf(float 5.000000e-01, float %38) #12
  %40 = fadd float %38, %39
  %41 = fptosi float %40 to i32
  %42 = sitofp i32 %41 to float
  %43 = fmul float %42, -3.140625e+00
  %44 = fadd float %37, %43
  %45 = fmul float %42, 0xBF4FB40000000000
  %46 = fadd float %45, %44
  %47 = fmul float %42, 0xBE84440000000000
  %48 = fadd float %47, %46
  %49 = fmul float %42, 0xBD968C2340000000
  %50 = fadd float %49, %48
  %51 = fsub float 0x3FF921FB60000000, %50
  %52 = fsub float 0x3FF921FB60000000, %51
  %53 = fmul float %52, %52
  %54 = and i32 %41, 1
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %_ZN11OpenImageIO4v1_78fast_sinEf.exit6, label %56

; <label>:56                                      ; preds = %_ZN11OpenImageIO4v1_78fast_sinEf.exit3
  %57 = fsub float -0.000000e+00, %52
  br label %_ZN11OpenImageIO4v1_78fast_sinEf.exit6

_ZN11OpenImageIO4v1_78fast_sinEf.exit6:           ; preds = %_ZN11OpenImageIO4v1_78fast_sinEf.exit3, %56
  %.0.i4 = phi float [ %57, %56 ], [ %52, %_ZN11OpenImageIO4v1_78fast_sinEf.exit3 ]
  %58 = fmul float %53, 0x3EC5E150E0000000
  %59 = fadd float %58, 0xBF29F75D60000000
  %60 = fmul float %53, %59
  %61 = fadd float %60, 0x3F8110EEE0000000
  %62 = fmul float %53, %61
  %63 = fadd float %62, 0xBFC55554C0000000
  %64 = fmul float %63, %.0.i4
  %65 = fmul float %53, %64
  %66 = fadd float %.0.i4, %65
  %67 = tail call float @fabsf(float %66) #12
  %68 = fcmp ogt float %67, 1.000000e+00
  %u.0.i5 = select i1 %68, float 0.000000e+00, float %66
  %69 = getelementptr inbounds i8* %r_, i64 4
  %70 = bitcast i8* %69 to float*
  store float %u.0.i5, float* %70, align 4, !tbaa !1
  %71 = getelementptr inbounds i8* %a_, i64 8
  %72 = bitcast i8* %71 to float*
  %73 = load float* %72, align 4, !tbaa !1
  %74 = fmul float %73, 0x3FD45F3060000000
  %75 = tail call float @copysignf(float 5.000000e-01, float %74) #12
  %76 = fadd float %74, %75
  %77 = fptosi float %76 to i32
  %78 = sitofp i32 %77 to float
  %79 = fmul float %78, -3.140625e+00
  %80 = fadd float %73, %79
  %81 = fmul float %78, 0xBF4FB40000000000
  %82 = fadd float %81, %80
  %83 = fmul float %78, 0xBE84440000000000
  %84 = fadd float %83, %82
  %85 = fmul float %78, 0xBD968C2340000000
  %86 = fadd float %85, %84
  %87 = fsub float 0x3FF921FB60000000, %86
  %88 = fsub float 0x3FF921FB60000000, %87
  %89 = fmul float %88, %88
  %90 = and i32 %77, 1
  %91 = icmp eq i32 %90, 0
  br i1 %91, label %_ZN11OpenImageIO4v1_78fast_sinEf.exit, label %92

; <label>:92                                      ; preds = %_ZN11OpenImageIO4v1_78fast_sinEf.exit6
  %93 = fsub float -0.000000e+00, %88
  br label %_ZN11OpenImageIO4v1_78fast_sinEf.exit

_ZN11OpenImageIO4v1_78fast_sinEf.exit:            ; preds = %_ZN11OpenImageIO4v1_78fast_sinEf.exit6, %92
  %.0.i = phi float [ %93, %92 ], [ %88, %_ZN11OpenImageIO4v1_78fast_sinEf.exit6 ]
  %94 = fmul float %89, 0x3EC5E150E0000000
  %95 = fadd float %94, 0xBF29F75D60000000
  %96 = fmul float %89, %95
  %97 = fadd float %96, 0x3F8110EEE0000000
  %98 = fmul float %89, %97
  %99 = fadd float %98, 0xBFC55554C0000000
  %100 = fmul float %99, %.0.i
  %101 = fmul float %89, %100
  %102 = fadd float %.0.i, %101
  %103 = tail call float @fabsf(float %102) #12
  %104 = fcmp ogt float %103, 1.000000e+00
  %u.0.i = select i1 %104, float 0.000000e+00, float %102
  %105 = getelementptr inbounds i8* %r_, i64 8
  %106 = bitcast i8* %105 to float*
  store float %u.0.i, float* %106, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sin_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = fmul float %6, 0x3FD45F3060000000
  %10 = tail call float @copysignf(float 5.000000e-01, float %9) #12
  %11 = fadd float %9, %10
  %12 = fptosi float %11 to i32
  %13 = sitofp i32 %12 to float
  %14 = fmul float %13, -3.140625e+00
  %15 = fadd float %6, %14
  %16 = fmul float %13, 0xBF4FB40000000000
  %17 = fadd float %16, %15
  %18 = fmul float %13, 0xBE84440000000000
  %19 = fadd float %18, %17
  %20 = fmul float %13, 0xBD968C2340000000
  %21 = fadd float %20, %19
  %22 = fsub float 0x3FF921FB60000000, %21
  %23 = fsub float 0x3FF921FB60000000, %22
  %24 = fmul float %23, %23
  %25 = and i32 %12, 1
  %26 = icmp ne i32 %25, 0
  br i1 %26, label %27, label %29

; <label>:27                                      ; preds = %0
  %28 = fsub float -0.000000e+00, %23
  br label %29

; <label>:29                                      ; preds = %27, %0
  %.0.i.i16 = phi float [ %28, %27 ], [ %23, %0 ]
  %30 = fmul float %24, 0x3EC5E150E0000000
  %31 = fadd float %30, 0xBF29F75D60000000
  %32 = fmul float %24, %31
  %33 = fadd float %32, 0x3F8110EEE0000000
  %34 = fmul float %24, %33
  %35 = fadd float %34, 0xBFC55554C0000000
  %36 = fmul float %35, %.0.i.i16
  %37 = fmul float %24, %36
  %38 = fadd float %.0.i.i16, %37
  %39 = fmul float %24, 0xBE923DB120000000
  %40 = fadd float %39, 0x3EFA00F160000000
  %41 = fmul float %24, %40
  %42 = fadd float %41, 0xBF56C16B00000000
  %43 = fmul float %24, %42
  %44 = fadd float %43, 0x3FA5555540000000
  %45 = fmul float %24, %44
  %46 = fadd float %45, -5.000000e-01
  %47 = fmul float %24, %46
  %48 = fadd float %47, 1.000000e+00
  br i1 %26, label %49, label %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit20

; <label>:49                                      ; preds = %29
  %50 = fsub float -0.000000e+00, %48
  br label %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit20

_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit20:          ; preds = %29, %49
  %cu.0.i.i17 = phi float [ %50, %49 ], [ %48, %29 ]
  %51 = tail call float @fabsf(float %38) #12
  %52 = fcmp ogt float %51, 1.000000e+00
  %su.0.i.i18 = select i1 %52, float 0.000000e+00, float %38
  %53 = tail call float @fabsf(float %cu.0.i.i17) #12
  %54 = fcmp ogt float %53, 1.000000e+00
  %cu.1.i.i19 = select i1 %54, float 0.000000e+00, float %cu.0.i.i17
  %55 = fmul float %7, %cu.1.i.i19
  %56 = fmul float %8, %cu.1.i.i19
  %57 = getelementptr inbounds i8* %a_, i64 4
  %58 = bitcast i8* %57 to float*
  %59 = getelementptr inbounds i8* %a_, i64 16
  %60 = bitcast i8* %59 to float*
  %61 = getelementptr inbounds i8* %a_, i64 28
  %62 = bitcast i8* %61 to float*
  %63 = load float* %58, align 4, !tbaa !1
  %64 = load float* %60, align 4, !tbaa !1
  %65 = load float* %62, align 4, !tbaa !1
  %66 = fmul float %63, 0x3FD45F3060000000
  %67 = tail call float @copysignf(float 5.000000e-01, float %66) #12
  %68 = fadd float %66, %67
  %69 = fptosi float %68 to i32
  %70 = sitofp i32 %69 to float
  %71 = fmul float %70, -3.140625e+00
  %72 = fadd float %63, %71
  %73 = fmul float %70, 0xBF4FB40000000000
  %74 = fadd float %73, %72
  %75 = fmul float %70, 0xBE84440000000000
  %76 = fadd float %75, %74
  %77 = fmul float %70, 0xBD968C2340000000
  %78 = fadd float %77, %76
  %79 = fsub float 0x3FF921FB60000000, %78
  %80 = fsub float 0x3FF921FB60000000, %79
  %81 = fmul float %80, %80
  %82 = and i32 %69, 1
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %84, label %86

; <label>:84                                      ; preds = %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit20
  %85 = fsub float -0.000000e+00, %80
  br label %86

; <label>:86                                      ; preds = %84, %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit20
  %.0.i.i11 = phi float [ %85, %84 ], [ %80, %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit20 ]
  %87 = fmul float %81, 0x3EC5E150E0000000
  %88 = fadd float %87, 0xBF29F75D60000000
  %89 = fmul float %81, %88
  %90 = fadd float %89, 0x3F8110EEE0000000
  %91 = fmul float %81, %90
  %92 = fadd float %91, 0xBFC55554C0000000
  %93 = fmul float %92, %.0.i.i11
  %94 = fmul float %81, %93
  %95 = fadd float %.0.i.i11, %94
  %96 = fmul float %81, 0xBE923DB120000000
  %97 = fadd float %96, 0x3EFA00F160000000
  %98 = fmul float %81, %97
  %99 = fadd float %98, 0xBF56C16B00000000
  %100 = fmul float %81, %99
  %101 = fadd float %100, 0x3FA5555540000000
  %102 = fmul float %81, %101
  %103 = fadd float %102, -5.000000e-01
  %104 = fmul float %81, %103
  %105 = fadd float %104, 1.000000e+00
  br i1 %83, label %106, label %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit15

; <label>:106                                     ; preds = %86
  %107 = fsub float -0.000000e+00, %105
  br label %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit15

_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit15:          ; preds = %86, %106
  %cu.0.i.i12 = phi float [ %107, %106 ], [ %105, %86 ]
  %108 = tail call float @fabsf(float %95) #12
  %109 = fcmp ogt float %108, 1.000000e+00
  %su.0.i.i13 = select i1 %109, float 0.000000e+00, float %95
  %110 = tail call float @fabsf(float %cu.0.i.i12) #12
  %111 = fcmp ogt float %110, 1.000000e+00
  %cu.1.i.i14 = select i1 %111, float 0.000000e+00, float %cu.0.i.i12
  %112 = fmul float %64, %cu.1.i.i14
  %113 = fmul float %65, %cu.1.i.i14
  %114 = getelementptr inbounds i8* %a_, i64 8
  %115 = bitcast i8* %114 to float*
  %116 = getelementptr inbounds i8* %a_, i64 20
  %117 = bitcast i8* %116 to float*
  %118 = getelementptr inbounds i8* %a_, i64 32
  %119 = bitcast i8* %118 to float*
  %120 = load float* %115, align 4, !tbaa !1
  %121 = load float* %117, align 4, !tbaa !1
  %122 = load float* %119, align 4, !tbaa !1
  %123 = fmul float %120, 0x3FD45F3060000000
  %124 = tail call float @copysignf(float 5.000000e-01, float %123) #12
  %125 = fadd float %123, %124
  %126 = fptosi float %125 to i32
  %127 = sitofp i32 %126 to float
  %128 = fmul float %127, -3.140625e+00
  %129 = fadd float %120, %128
  %130 = fmul float %127, 0xBF4FB40000000000
  %131 = fadd float %130, %129
  %132 = fmul float %127, 0xBE84440000000000
  %133 = fadd float %132, %131
  %134 = fmul float %127, 0xBD968C2340000000
  %135 = fadd float %134, %133
  %136 = fsub float 0x3FF921FB60000000, %135
  %137 = fsub float 0x3FF921FB60000000, %136
  %138 = fmul float %137, %137
  %139 = and i32 %126, 1
  %140 = icmp ne i32 %139, 0
  br i1 %140, label %141, label %143

; <label>:141                                     ; preds = %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit15
  %142 = fsub float -0.000000e+00, %137
  br label %143

; <label>:143                                     ; preds = %141, %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit15
  %.0.i.i = phi float [ %142, %141 ], [ %137, %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit15 ]
  %144 = fmul float %138, 0x3EC5E150E0000000
  %145 = fadd float %144, 0xBF29F75D60000000
  %146 = fmul float %138, %145
  %147 = fadd float %146, 0x3F8110EEE0000000
  %148 = fmul float %138, %147
  %149 = fadd float %148, 0xBFC55554C0000000
  %150 = fmul float %149, %.0.i.i
  %151 = fmul float %138, %150
  %152 = fadd float %.0.i.i, %151
  %153 = fmul float %138, 0xBE923DB120000000
  %154 = fadd float %153, 0x3EFA00F160000000
  %155 = fmul float %138, %154
  %156 = fadd float %155, 0xBF56C16B00000000
  %157 = fmul float %138, %156
  %158 = fadd float %157, 0x3FA5555540000000
  %159 = fmul float %138, %158
  %160 = fadd float %159, -5.000000e-01
  %161 = fmul float %138, %160
  %162 = fadd float %161, 1.000000e+00
  br i1 %140, label %163, label %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit

; <label>:163                                     ; preds = %143
  %164 = fsub float -0.000000e+00, %162
  br label %_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit

_ZN3OSL8fast_sinERKNS_5Dual2IfEE.exit:            ; preds = %143, %163
  %cu.0.i.i = phi float [ %164, %163 ], [ %162, %143 ]
  %165 = tail call float @fabsf(float %152) #12
  %166 = fcmp ogt float %165, 1.000000e+00
  %su.0.i.i = select i1 %166, float 0.000000e+00, float %152
  %167 = tail call float @fabsf(float %cu.0.i.i) #12
  %168 = fcmp ogt float %167, 1.000000e+00
  %cu.1.i.i = select i1 %168, float 0.000000e+00, float %cu.0.i.i
  %169 = fmul float %121, %cu.1.i.i
  %170 = fmul float %122, %cu.1.i.i
  %171 = bitcast i8* %r_ to float*
  store float %su.0.i.i18, float* %171, align 4, !tbaa !5
  %172 = getelementptr inbounds i8* %r_, i64 4
  %173 = bitcast i8* %172 to float*
  store float %su.0.i.i13, float* %173, align 4, !tbaa !7
  %174 = getelementptr inbounds i8* %r_, i64 8
  %175 = bitcast i8* %174 to float*
  store float %su.0.i.i, float* %175, align 4, !tbaa !8
  %176 = getelementptr inbounds i8* %r_, i64 12
  %177 = bitcast i8* %176 to float*
  store float %55, float* %177, align 4, !tbaa !5
  %178 = getelementptr inbounds i8* %r_, i64 16
  %179 = bitcast i8* %178 to float*
  store float %112, float* %179, align 4, !tbaa !7
  %180 = getelementptr inbounds i8* %r_, i64 20
  %181 = bitcast i8* %180 to float*
  store float %169, float* %181, align 4, !tbaa !8
  %182 = getelementptr inbounds i8* %r_, i64 24
  %183 = bitcast i8* %182 to float*
  store float %56, float* %183, align 4, !tbaa !5
  %184 = getelementptr inbounds i8* %r_, i64 28
  %185 = bitcast i8* %184 to float*
  store float %113, float* %185, align 4, !tbaa !7
  %186 = getelementptr inbounds i8* %r_, i64 32
  %187 = bitcast i8* %186 to float*
  store float %170, float* %187, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_cos_ff(float %a) #3 {
  %1 = fmul float %a, 0x3FD45F3060000000
  %2 = tail call float @copysignf(float 5.000000e-01, float %1) #12
  %3 = fadd float %1, %2
  %4 = fptosi float %3 to i32
  %5 = sitofp i32 %4 to float
  %6 = fmul float %5, -3.140625e+00
  %7 = fadd float %6, %a
  %8 = fmul float %5, 0xBF4FB40000000000
  %9 = fadd float %8, %7
  %10 = fmul float %5, 0xBE84440000000000
  %11 = fadd float %10, %9
  %12 = fmul float %5, 0xBD968C2340000000
  %13 = fadd float %12, %11
  %14 = fsub float 0x3FF921FB60000000, %13
  %15 = fsub float 0x3FF921FB60000000, %14
  %16 = fmul float %15, %15
  %17 = fmul float %16, 0xBE923DB120000000
  %18 = fadd float %17, 0x3EFA00F160000000
  %19 = fmul float %16, %18
  %20 = fadd float %19, 0xBF56C16B00000000
  %21 = fmul float %16, %20
  %22 = fadd float %21, 0x3FA5555540000000
  %23 = fmul float %16, %22
  %24 = fadd float %23, -5.000000e-01
  %25 = fmul float %16, %24
  %26 = fadd float %25, 1.000000e+00
  %27 = and i32 %4, 1
  %28 = icmp eq i32 %27, 0
  br i1 %28, label %_ZN11OpenImageIO4v1_78fast_cosEf.exit, label %29

; <label>:29                                      ; preds = %0
  %30 = fsub float -0.000000e+00, %26
  br label %_ZN11OpenImageIO4v1_78fast_cosEf.exit

_ZN11OpenImageIO4v1_78fast_cosEf.exit:            ; preds = %0, %29
  %u.0.i = phi float [ %30, %29 ], [ %26, %0 ]
  %31 = tail call float @fabsf(float %u.0.i) #12
  %32 = fcmp ogt float %31, 1.000000e+00
  %u.1.i = select i1 %32, float 0.000000e+00, float %u.0.i
  ret float %u.1.i
}

; Function Attrs: nounwind uwtable
define void @osl_cos_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FD45F3060000000
  %4 = tail call float @copysignf(float 5.000000e-01, float %3) #12
  %5 = fadd float %3, %4
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fmul float %7, -3.140625e+00
  %9 = fadd float %2, %8
  %10 = fmul float %7, 0xBF4FB40000000000
  %11 = fadd float %10, %9
  %12 = fmul float %7, 0xBE84440000000000
  %13 = fadd float %12, %11
  %14 = fmul float %7, 0xBD968C2340000000
  %15 = fadd float %14, %13
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = fmul float %17, %17
  %19 = and i32 %6, 1
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %21, label %23

; <label>:21                                      ; preds = %0
  %22 = fsub float -0.000000e+00, %17
  br label %23

; <label>:23                                      ; preds = %21, %0
  %.0.i.i = phi float [ %22, %21 ], [ %17, %0 ]
  %24 = fmul float %18, 0x3EC5E150E0000000
  %25 = fadd float %24, 0xBF29F75D60000000
  %26 = fmul float %18, %25
  %27 = fadd float %26, 0x3F8110EEE0000000
  %28 = fmul float %18, %27
  %29 = fadd float %28, 0xBFC55554C0000000
  %30 = fmul float %29, %.0.i.i
  %31 = fmul float %18, %30
  %32 = fadd float %.0.i.i, %31
  %33 = fmul float %18, 0xBE923DB120000000
  %34 = fadd float %33, 0x3EFA00F160000000
  %35 = fmul float %18, %34
  %36 = fadd float %35, 0xBF56C16B00000000
  %37 = fmul float %18, %36
  %38 = fadd float %37, 0x3FA5555540000000
  %39 = fmul float %18, %38
  %40 = fadd float %39, -5.000000e-01
  %41 = fmul float %18, %40
  %42 = fadd float %41, 1.000000e+00
  br i1 %20, label %43, label %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit

; <label>:43                                      ; preds = %23
  %44 = fsub float -0.000000e+00, %42
  br label %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit

_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit:            ; preds = %23, %43
  %cu.0.i.i = phi float [ %44, %43 ], [ %42, %23 ]
  %45 = tail call float @fabsf(float %32) #12
  %46 = fcmp ogt float %45, 1.000000e+00
  %su.0.i.i = select i1 %46, float 0.000000e+00, float %32
  %47 = tail call float @fabsf(float %cu.0.i.i) #12
  %48 = fcmp ogt float %47, 1.000000e+00
  %cu.1.i.i = select i1 %48, float 0.000000e+00, float %cu.0.i.i
  %49 = getelementptr inbounds i8* %a, i64 4
  %50 = bitcast i8* %49 to float*
  %51 = load float* %50, align 4, !tbaa !1
  %52 = fmul float %su.0.i.i, %51
  %53 = fsub float -0.000000e+00, %52
  %54 = getelementptr inbounds i8* %a, i64 8
  %55 = bitcast i8* %54 to float*
  %56 = load float* %55, align 4, !tbaa !1
  %57 = fmul float %su.0.i.i, %56
  %58 = fsub float -0.000000e+00, %57
  %59 = insertelement <2 x float> undef, float %cu.1.i.i, i32 0
  %60 = insertelement <2 x float> %59, float %53, i32 1
  %61 = bitcast i8* %r to <2 x float>*
  store <2 x float> %60, <2 x float>* %61, align 4
  %62 = getelementptr inbounds i8* %r, i64 8
  %63 = bitcast i8* %62 to float*
  store float %58, float* %63, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cos_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FD45F3060000000
  %4 = tail call float @copysignf(float 5.000000e-01, float %3) #12
  %5 = fadd float %3, %4
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fmul float %7, -3.140625e+00
  %9 = fadd float %2, %8
  %10 = fmul float %7, 0xBF4FB40000000000
  %11 = fadd float %10, %9
  %12 = fmul float %7, 0xBE84440000000000
  %13 = fadd float %12, %11
  %14 = fmul float %7, 0xBD968C2340000000
  %15 = fadd float %14, %13
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = fmul float %17, %17
  %19 = fmul float %18, 0xBE923DB120000000
  %20 = fadd float %19, 0x3EFA00F160000000
  %21 = fmul float %18, %20
  %22 = fadd float %21, 0xBF56C16B00000000
  %23 = fmul float %18, %22
  %24 = fadd float %23, 0x3FA5555540000000
  %25 = fmul float %18, %24
  %26 = fadd float %25, -5.000000e-01
  %27 = fmul float %18, %26
  %28 = fadd float %27, 1.000000e+00
  %29 = and i32 %6, 1
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %_ZN11OpenImageIO4v1_78fast_cosEf.exit3, label %31

; <label>:31                                      ; preds = %0
  %32 = fsub float -0.000000e+00, %28
  br label %_ZN11OpenImageIO4v1_78fast_cosEf.exit3

_ZN11OpenImageIO4v1_78fast_cosEf.exit3:           ; preds = %0, %31
  %u.0.i1 = phi float [ %32, %31 ], [ %28, %0 ]
  %33 = tail call float @fabsf(float %u.0.i1) #12
  %34 = fcmp ogt float %33, 1.000000e+00
  %u.1.i2 = select i1 %34, float 0.000000e+00, float %u.0.i1
  %35 = bitcast i8* %r_ to float*
  store float %u.1.i2, float* %35, align 4, !tbaa !1
  %36 = getelementptr inbounds i8* %a_, i64 4
  %37 = bitcast i8* %36 to float*
  %38 = load float* %37, align 4, !tbaa !1
  %39 = fmul float %38, 0x3FD45F3060000000
  %40 = tail call float @copysignf(float 5.000000e-01, float %39) #12
  %41 = fadd float %39, %40
  %42 = fptosi float %41 to i32
  %43 = sitofp i32 %42 to float
  %44 = fmul float %43, -3.140625e+00
  %45 = fadd float %38, %44
  %46 = fmul float %43, 0xBF4FB40000000000
  %47 = fadd float %46, %45
  %48 = fmul float %43, 0xBE84440000000000
  %49 = fadd float %48, %47
  %50 = fmul float %43, 0xBD968C2340000000
  %51 = fadd float %50, %49
  %52 = fsub float 0x3FF921FB60000000, %51
  %53 = fsub float 0x3FF921FB60000000, %52
  %54 = fmul float %53, %53
  %55 = fmul float %54, 0xBE923DB120000000
  %56 = fadd float %55, 0x3EFA00F160000000
  %57 = fmul float %54, %56
  %58 = fadd float %57, 0xBF56C16B00000000
  %59 = fmul float %54, %58
  %60 = fadd float %59, 0x3FA5555540000000
  %61 = fmul float %54, %60
  %62 = fadd float %61, -5.000000e-01
  %63 = fmul float %54, %62
  %64 = fadd float %63, 1.000000e+00
  %65 = and i32 %42, 1
  %66 = icmp eq i32 %65, 0
  br i1 %66, label %_ZN11OpenImageIO4v1_78fast_cosEf.exit6, label %67

; <label>:67                                      ; preds = %_ZN11OpenImageIO4v1_78fast_cosEf.exit3
  %68 = fsub float -0.000000e+00, %64
  br label %_ZN11OpenImageIO4v1_78fast_cosEf.exit6

_ZN11OpenImageIO4v1_78fast_cosEf.exit6:           ; preds = %_ZN11OpenImageIO4v1_78fast_cosEf.exit3, %67
  %u.0.i4 = phi float [ %68, %67 ], [ %64, %_ZN11OpenImageIO4v1_78fast_cosEf.exit3 ]
  %69 = tail call float @fabsf(float %u.0.i4) #12
  %70 = fcmp ogt float %69, 1.000000e+00
  %u.1.i5 = select i1 %70, float 0.000000e+00, float %u.0.i4
  %71 = getelementptr inbounds i8* %r_, i64 4
  %72 = bitcast i8* %71 to float*
  store float %u.1.i5, float* %72, align 4, !tbaa !1
  %73 = getelementptr inbounds i8* %a_, i64 8
  %74 = bitcast i8* %73 to float*
  %75 = load float* %74, align 4, !tbaa !1
  %76 = fmul float %75, 0x3FD45F3060000000
  %77 = tail call float @copysignf(float 5.000000e-01, float %76) #12
  %78 = fadd float %76, %77
  %79 = fptosi float %78 to i32
  %80 = sitofp i32 %79 to float
  %81 = fmul float %80, -3.140625e+00
  %82 = fadd float %75, %81
  %83 = fmul float %80, 0xBF4FB40000000000
  %84 = fadd float %83, %82
  %85 = fmul float %80, 0xBE84440000000000
  %86 = fadd float %85, %84
  %87 = fmul float %80, 0xBD968C2340000000
  %88 = fadd float %87, %86
  %89 = fsub float 0x3FF921FB60000000, %88
  %90 = fsub float 0x3FF921FB60000000, %89
  %91 = fmul float %90, %90
  %92 = fmul float %91, 0xBE923DB120000000
  %93 = fadd float %92, 0x3EFA00F160000000
  %94 = fmul float %91, %93
  %95 = fadd float %94, 0xBF56C16B00000000
  %96 = fmul float %91, %95
  %97 = fadd float %96, 0x3FA5555540000000
  %98 = fmul float %91, %97
  %99 = fadd float %98, -5.000000e-01
  %100 = fmul float %91, %99
  %101 = fadd float %100, 1.000000e+00
  %102 = and i32 %79, 1
  %103 = icmp eq i32 %102, 0
  br i1 %103, label %_ZN11OpenImageIO4v1_78fast_cosEf.exit, label %104

; <label>:104                                     ; preds = %_ZN11OpenImageIO4v1_78fast_cosEf.exit6
  %105 = fsub float -0.000000e+00, %101
  br label %_ZN11OpenImageIO4v1_78fast_cosEf.exit

_ZN11OpenImageIO4v1_78fast_cosEf.exit:            ; preds = %_ZN11OpenImageIO4v1_78fast_cosEf.exit6, %104
  %u.0.i = phi float [ %105, %104 ], [ %101, %_ZN11OpenImageIO4v1_78fast_cosEf.exit6 ]
  %106 = tail call float @fabsf(float %u.0.i) #12
  %107 = fcmp ogt float %106, 1.000000e+00
  %u.1.i = select i1 %107, float 0.000000e+00, float %u.0.i
  %108 = getelementptr inbounds i8* %r_, i64 8
  %109 = bitcast i8* %108 to float*
  store float %u.1.i, float* %109, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cos_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = fmul float %6, 0x3FD45F3060000000
  %10 = tail call float @copysignf(float 5.000000e-01, float %9) #12
  %11 = fadd float %9, %10
  %12 = fptosi float %11 to i32
  %13 = sitofp i32 %12 to float
  %14 = fmul float %13, -3.140625e+00
  %15 = fadd float %6, %14
  %16 = fmul float %13, 0xBF4FB40000000000
  %17 = fadd float %16, %15
  %18 = fmul float %13, 0xBE84440000000000
  %19 = fadd float %18, %17
  %20 = fmul float %13, 0xBD968C2340000000
  %21 = fadd float %20, %19
  %22 = fsub float 0x3FF921FB60000000, %21
  %23 = fsub float 0x3FF921FB60000000, %22
  %24 = fmul float %23, %23
  %25 = and i32 %12, 1
  %26 = icmp ne i32 %25, 0
  br i1 %26, label %27, label %29

; <label>:27                                      ; preds = %0
  %28 = fsub float -0.000000e+00, %23
  br label %29

; <label>:29                                      ; preds = %27, %0
  %.0.i.i16 = phi float [ %28, %27 ], [ %23, %0 ]
  %30 = fmul float %24, 0x3EC5E150E0000000
  %31 = fadd float %30, 0xBF29F75D60000000
  %32 = fmul float %24, %31
  %33 = fadd float %32, 0x3F8110EEE0000000
  %34 = fmul float %24, %33
  %35 = fadd float %34, 0xBFC55554C0000000
  %36 = fmul float %35, %.0.i.i16
  %37 = fmul float %24, %36
  %38 = fadd float %.0.i.i16, %37
  %39 = fmul float %24, 0xBE923DB120000000
  %40 = fadd float %39, 0x3EFA00F160000000
  %41 = fmul float %24, %40
  %42 = fadd float %41, 0xBF56C16B00000000
  %43 = fmul float %24, %42
  %44 = fadd float %43, 0x3FA5555540000000
  %45 = fmul float %24, %44
  %46 = fadd float %45, -5.000000e-01
  %47 = fmul float %24, %46
  %48 = fadd float %47, 1.000000e+00
  br i1 %26, label %49, label %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit20

; <label>:49                                      ; preds = %29
  %50 = fsub float -0.000000e+00, %48
  br label %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit20

_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit20:          ; preds = %29, %49
  %cu.0.i.i17 = phi float [ %50, %49 ], [ %48, %29 ]
  %51 = tail call float @fabsf(float %38) #12
  %52 = fcmp ogt float %51, 1.000000e+00
  %su.0.i.i18 = select i1 %52, float 0.000000e+00, float %38
  %53 = tail call float @fabsf(float %cu.0.i.i17) #12
  %54 = fcmp ogt float %53, 1.000000e+00
  %cu.1.i.i19 = select i1 %54, float 0.000000e+00, float %cu.0.i.i17
  %55 = fmul float %7, %su.0.i.i18
  %56 = fsub float -0.000000e+00, %55
  %57 = fmul float %8, %su.0.i.i18
  %58 = fsub float -0.000000e+00, %57
  %59 = getelementptr inbounds i8* %a_, i64 4
  %60 = bitcast i8* %59 to float*
  %61 = getelementptr inbounds i8* %a_, i64 16
  %62 = bitcast i8* %61 to float*
  %63 = getelementptr inbounds i8* %a_, i64 28
  %64 = bitcast i8* %63 to float*
  %65 = load float* %60, align 4, !tbaa !1
  %66 = load float* %62, align 4, !tbaa !1
  %67 = load float* %64, align 4, !tbaa !1
  %68 = fmul float %65, 0x3FD45F3060000000
  %69 = tail call float @copysignf(float 5.000000e-01, float %68) #12
  %70 = fadd float %68, %69
  %71 = fptosi float %70 to i32
  %72 = sitofp i32 %71 to float
  %73 = fmul float %72, -3.140625e+00
  %74 = fadd float %65, %73
  %75 = fmul float %72, 0xBF4FB40000000000
  %76 = fadd float %75, %74
  %77 = fmul float %72, 0xBE84440000000000
  %78 = fadd float %77, %76
  %79 = fmul float %72, 0xBD968C2340000000
  %80 = fadd float %79, %78
  %81 = fsub float 0x3FF921FB60000000, %80
  %82 = fsub float 0x3FF921FB60000000, %81
  %83 = fmul float %82, %82
  %84 = and i32 %71, 1
  %85 = icmp ne i32 %84, 0
  br i1 %85, label %86, label %88

; <label>:86                                      ; preds = %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit20
  %87 = fsub float -0.000000e+00, %82
  br label %88

; <label>:88                                      ; preds = %86, %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit20
  %.0.i.i11 = phi float [ %87, %86 ], [ %82, %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit20 ]
  %89 = fmul float %83, 0x3EC5E150E0000000
  %90 = fadd float %89, 0xBF29F75D60000000
  %91 = fmul float %83, %90
  %92 = fadd float %91, 0x3F8110EEE0000000
  %93 = fmul float %83, %92
  %94 = fadd float %93, 0xBFC55554C0000000
  %95 = fmul float %94, %.0.i.i11
  %96 = fmul float %83, %95
  %97 = fadd float %.0.i.i11, %96
  %98 = fmul float %83, 0xBE923DB120000000
  %99 = fadd float %98, 0x3EFA00F160000000
  %100 = fmul float %83, %99
  %101 = fadd float %100, 0xBF56C16B00000000
  %102 = fmul float %83, %101
  %103 = fadd float %102, 0x3FA5555540000000
  %104 = fmul float %83, %103
  %105 = fadd float %104, -5.000000e-01
  %106 = fmul float %83, %105
  %107 = fadd float %106, 1.000000e+00
  br i1 %85, label %108, label %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit15

; <label>:108                                     ; preds = %88
  %109 = fsub float -0.000000e+00, %107
  br label %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit15

_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit15:          ; preds = %88, %108
  %cu.0.i.i12 = phi float [ %109, %108 ], [ %107, %88 ]
  %110 = tail call float @fabsf(float %97) #12
  %111 = fcmp ogt float %110, 1.000000e+00
  %su.0.i.i13 = select i1 %111, float 0.000000e+00, float %97
  %112 = tail call float @fabsf(float %cu.0.i.i12) #12
  %113 = fcmp ogt float %112, 1.000000e+00
  %cu.1.i.i14 = select i1 %113, float 0.000000e+00, float %cu.0.i.i12
  %114 = fmul float %66, %su.0.i.i13
  %115 = fsub float -0.000000e+00, %114
  %116 = fmul float %67, %su.0.i.i13
  %117 = fsub float -0.000000e+00, %116
  %118 = getelementptr inbounds i8* %a_, i64 8
  %119 = bitcast i8* %118 to float*
  %120 = getelementptr inbounds i8* %a_, i64 20
  %121 = bitcast i8* %120 to float*
  %122 = getelementptr inbounds i8* %a_, i64 32
  %123 = bitcast i8* %122 to float*
  %124 = load float* %119, align 4, !tbaa !1
  %125 = load float* %121, align 4, !tbaa !1
  %126 = load float* %123, align 4, !tbaa !1
  %127 = fmul float %124, 0x3FD45F3060000000
  %128 = tail call float @copysignf(float 5.000000e-01, float %127) #12
  %129 = fadd float %127, %128
  %130 = fptosi float %129 to i32
  %131 = sitofp i32 %130 to float
  %132 = fmul float %131, -3.140625e+00
  %133 = fadd float %124, %132
  %134 = fmul float %131, 0xBF4FB40000000000
  %135 = fadd float %134, %133
  %136 = fmul float %131, 0xBE84440000000000
  %137 = fadd float %136, %135
  %138 = fmul float %131, 0xBD968C2340000000
  %139 = fadd float %138, %137
  %140 = fsub float 0x3FF921FB60000000, %139
  %141 = fsub float 0x3FF921FB60000000, %140
  %142 = fmul float %141, %141
  %143 = and i32 %130, 1
  %144 = icmp ne i32 %143, 0
  br i1 %144, label %145, label %147

; <label>:145                                     ; preds = %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit15
  %146 = fsub float -0.000000e+00, %141
  br label %147

; <label>:147                                     ; preds = %145, %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit15
  %.0.i.i = phi float [ %146, %145 ], [ %141, %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit15 ]
  %148 = fmul float %142, 0x3EC5E150E0000000
  %149 = fadd float %148, 0xBF29F75D60000000
  %150 = fmul float %142, %149
  %151 = fadd float %150, 0x3F8110EEE0000000
  %152 = fmul float %142, %151
  %153 = fadd float %152, 0xBFC55554C0000000
  %154 = fmul float %153, %.0.i.i
  %155 = fmul float %142, %154
  %156 = fadd float %.0.i.i, %155
  %157 = fmul float %142, 0xBE923DB120000000
  %158 = fadd float %157, 0x3EFA00F160000000
  %159 = fmul float %142, %158
  %160 = fadd float %159, 0xBF56C16B00000000
  %161 = fmul float %142, %160
  %162 = fadd float %161, 0x3FA5555540000000
  %163 = fmul float %142, %162
  %164 = fadd float %163, -5.000000e-01
  %165 = fmul float %142, %164
  %166 = fadd float %165, 1.000000e+00
  br i1 %144, label %167, label %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit

; <label>:167                                     ; preds = %147
  %168 = fsub float -0.000000e+00, %166
  br label %_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit

_ZN3OSL8fast_cosERKNS_5Dual2IfEE.exit:            ; preds = %147, %167
  %cu.0.i.i = phi float [ %168, %167 ], [ %166, %147 ]
  %169 = tail call float @fabsf(float %156) #12
  %170 = fcmp ogt float %169, 1.000000e+00
  %su.0.i.i = select i1 %170, float 0.000000e+00, float %156
  %171 = tail call float @fabsf(float %cu.0.i.i) #12
  %172 = fcmp ogt float %171, 1.000000e+00
  %cu.1.i.i = select i1 %172, float 0.000000e+00, float %cu.0.i.i
  %173 = fmul float %125, %su.0.i.i
  %174 = fsub float -0.000000e+00, %173
  %175 = fmul float %126, %su.0.i.i
  %176 = fsub float -0.000000e+00, %175
  %177 = bitcast i8* %r_ to float*
  store float %cu.1.i.i19, float* %177, align 4, !tbaa !5
  %178 = getelementptr inbounds i8* %r_, i64 4
  %179 = bitcast i8* %178 to float*
  store float %cu.1.i.i14, float* %179, align 4, !tbaa !7
  %180 = getelementptr inbounds i8* %r_, i64 8
  %181 = bitcast i8* %180 to float*
  store float %cu.1.i.i, float* %181, align 4, !tbaa !8
  %182 = getelementptr inbounds i8* %r_, i64 12
  %183 = bitcast i8* %182 to float*
  store float %56, float* %183, align 4, !tbaa !5
  %184 = getelementptr inbounds i8* %r_, i64 16
  %185 = bitcast i8* %184 to float*
  store float %115, float* %185, align 4, !tbaa !7
  %186 = getelementptr inbounds i8* %r_, i64 20
  %187 = bitcast i8* %186 to float*
  store float %174, float* %187, align 4, !tbaa !8
  %188 = getelementptr inbounds i8* %r_, i64 24
  %189 = bitcast i8* %188 to float*
  store float %58, float* %189, align 4, !tbaa !5
  %190 = getelementptr inbounds i8* %r_, i64 28
  %191 = bitcast i8* %190 to float*
  store float %117, float* %191, align 4, !tbaa !7
  %192 = getelementptr inbounds i8* %r_, i64 32
  %193 = bitcast i8* %192 to float*
  store float %176, float* %193, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_tan_ff(float %a) #3 {
  %1 = fmul float %a, 0x3FE45F3060000000
  %2 = tail call float @copysignf(float 5.000000e-01, float %1) #12
  %3 = fadd float %1, %2
  %4 = fptosi float %3 to i32
  %5 = sitofp i32 %4 to float
  %6 = fmul float %5, 0xBFF9200000000000
  %7 = fadd float %6, %a
  %8 = fmul float %5, 0xBF3FB40000000000
  %9 = fadd float %8, %7
  %10 = fmul float %5, 0xBE74440000000000
  %11 = fadd float %10, %9
  %12 = fmul float %5, 0xBD868C2340000000
  %13 = fadd float %12, %11
  %14 = and i32 %4, 1
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %16, label %19

; <label>:16                                      ; preds = %0
  %17 = fsub float 0x3FE921FB60000000, %13
  %18 = fsub float 0x3FE921FB60000000, %17
  br label %19

; <label>:19                                      ; preds = %16, %0
  %.0.i = phi float [ %18, %16 ], [ %13, %0 ]
  %20 = fmul float %.0.i, %.0.i
  %21 = fmul float %20, 0x3F82FD7040000000
  %22 = fadd float %21, 0x3F6B323AE0000000
  %23 = fmul float %20, %22
  %24 = fadd float %23, 0x3F98E20C80000000
  %25 = fmul float %20, %24
  %26 = fadd float %25, 0x3FAB5DBCA0000000
  %27 = fmul float %20, %26
  %28 = fadd float %27, 0x3FC112B1C0000000
  %29 = fmul float %20, %28
  %30 = fadd float %29, 0x3FD5554F20000000
  %31 = fmul float %.0.i, %30
  %32 = fmul float %20, %31
  %33 = fadd float %.0.i, %32
  br i1 %15, label %_ZN11OpenImageIO4v1_78fast_tanEf.exit, label %34

; <label>:34                                      ; preds = %19
  %35 = fdiv float -1.000000e+00, %33
  br label %_ZN11OpenImageIO4v1_78fast_tanEf.exit

_ZN11OpenImageIO4v1_78fast_tanEf.exit:            ; preds = %19, %34
  %u.0.i = phi float [ %35, %34 ], [ %33, %19 ]
  ret float %u.0.i
}

; Function Attrs: nounwind uwtable
define void @osl_tan_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to %"class.OSL::Dual2"*
  %2 = tail call { <2 x float>, float } @_ZN3OSL8fast_tanERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %1)
  %3 = extractvalue { <2 x float>, float } %2, 0
  %4 = extractvalue { <2 x float>, float } %2, 1
  %5 = bitcast i8* %r to <2 x float>*
  store <2 x float> %3, <2 x float>* %5, align 4
  %6 = getelementptr inbounds i8* %r, i64 8
  %7 = bitcast i8* %6 to float*
  store float %4, float* %7, align 4
  ret void
}

; Function Attrs: inlinehint nounwind readonly uwtable
define linkonce_odr { <2 x float>, float } @_ZN3OSL8fast_tanERKNS_5Dual2IfEE(%"class.OSL::Dual2"* nocapture readonly dereferenceable(12) %a) #5 {
  %1 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 0
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FE45F3060000000
  %4 = tail call float @copysignf(float 5.000000e-01, float %3) #12
  %5 = fadd float %3, %4
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fmul float %7, 0xBFF9200000000000
  %9 = fadd float %2, %8
  %10 = fmul float %7, 0xBF3FB40000000000
  %11 = fadd float %10, %9
  %12 = fmul float %7, 0xBE74440000000000
  %13 = fadd float %12, %11
  %14 = fmul float %7, 0xBD868C2340000000
  %15 = fadd float %14, %13
  %16 = and i32 %6, 1
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %18, label %21

; <label>:18                                      ; preds = %0
  %19 = fsub float 0x3FE921FB60000000, %15
  %20 = fsub float 0x3FE921FB60000000, %19
  br label %21

; <label>:21                                      ; preds = %18, %0
  %.0.i = phi float [ %20, %18 ], [ %15, %0 ]
  %22 = fmul float %.0.i, %.0.i
  %23 = fmul float %22, 0x3F82FD7040000000
  %24 = fadd float %23, 0x3F6B323AE0000000
  %25 = fmul float %22, %24
  %26 = fadd float %25, 0x3F98E20C80000000
  %27 = fmul float %22, %26
  %28 = fadd float %27, 0x3FAB5DBCA0000000
  %29 = fmul float %22, %28
  %30 = fadd float %29, 0x3FC112B1C0000000
  %31 = fmul float %22, %30
  %32 = fadd float %31, 0x3FD5554F20000000
  %33 = fmul float %.0.i, %32
  %34 = fmul float %22, %33
  %35 = fadd float %.0.i, %34
  br i1 %17, label %_ZN11OpenImageIO4v1_78fast_tanEf.exit, label %36

; <label>:36                                      ; preds = %21
  %37 = fdiv float -1.000000e+00, %35
  br label %_ZN11OpenImageIO4v1_78fast_tanEf.exit

_ZN11OpenImageIO4v1_78fast_tanEf.exit:            ; preds = %21, %36
  %u.0.i1 = phi float [ %37, %36 ], [ %35, %21 ]
  %38 = fmul float %2, 0x3FD45F3060000000
  %39 = tail call float @copysignf(float 5.000000e-01, float %38) #12
  %40 = fadd float %38, %39
  %41 = fptosi float %40 to i32
  %42 = sitofp i32 %41 to float
  %43 = fmul float %42, -3.140625e+00
  %44 = fadd float %2, %43
  %45 = fmul float %42, 0xBF4FB40000000000
  %46 = fadd float %45, %44
  %47 = fmul float %42, 0xBE84440000000000
  %48 = fadd float %47, %46
  %49 = fmul float %42, 0xBD968C2340000000
  %50 = fadd float %49, %48
  %51 = fsub float 0x3FF921FB60000000, %50
  %52 = fsub float 0x3FF921FB60000000, %51
  %53 = fmul float %52, %52
  %54 = fmul float %53, 0xBE923DB120000000
  %55 = fadd float %54, 0x3EFA00F160000000
  %56 = fmul float %53, %55
  %57 = fadd float %56, 0xBF56C16B00000000
  %58 = fmul float %53, %57
  %59 = fadd float %58, 0x3FA5555540000000
  %60 = fmul float %53, %59
  %61 = fadd float %60, -5.000000e-01
  %62 = fmul float %53, %61
  %63 = fadd float %62, 1.000000e+00
  %64 = and i32 %41, 1
  %65 = icmp eq i32 %64, 0
  br i1 %65, label %_ZN11OpenImageIO4v1_78fast_cosEf.exit, label %66

; <label>:66                                      ; preds = %_ZN11OpenImageIO4v1_78fast_tanEf.exit
  %67 = fsub float -0.000000e+00, %63
  br label %_ZN11OpenImageIO4v1_78fast_cosEf.exit

_ZN11OpenImageIO4v1_78fast_cosEf.exit:            ; preds = %_ZN11OpenImageIO4v1_78fast_tanEf.exit, %66
  %u.0.i = phi float [ %67, %66 ], [ %63, %_ZN11OpenImageIO4v1_78fast_tanEf.exit ]
  %68 = tail call float @fabsf(float %u.0.i) #12
  %69 = fcmp ogt float %68, 1.000000e+00
  %u.1.i = select i1 %69, float 0.000000e+00, float %u.0.i
  %70 = fmul float %u.1.i, %u.1.i
  %71 = fdiv float 1.000000e+00, %70
  %72 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 1
  %73 = load float* %72, align 4, !tbaa !1
  %74 = fmul float %71, %73
  %75 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 2
  %76 = load float* %75, align 4, !tbaa !1
  %77 = fmul float %71, %76
  %78 = insertelement <2 x float> undef, float %u.0.i1, i32 0
  %79 = insertelement <2 x float> %78, float %74, i32 1
  %80 = insertvalue { <2 x float>, float } undef, <2 x float> %79, 0
  %81 = insertvalue { <2 x float>, float } %80, float %77, 1
  ret { <2 x float>, float } %81
}

; Function Attrs: nounwind uwtable
define void @osl_tan_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FE45F3060000000
  %4 = tail call float @copysignf(float 5.000000e-01, float %3) #12
  %5 = fadd float %3, %4
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fmul float %7, 0xBFF9200000000000
  %9 = fadd float %2, %8
  %10 = fmul float %7, 0xBF3FB40000000000
  %11 = fadd float %10, %9
  %12 = fmul float %7, 0xBE74440000000000
  %13 = fadd float %12, %11
  %14 = fmul float %7, 0xBD868C2340000000
  %15 = fadd float %14, %13
  %16 = and i32 %6, 1
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %18, label %21

; <label>:18                                      ; preds = %0
  %19 = fsub float 0x3FE921FB60000000, %15
  %20 = fsub float 0x3FE921FB60000000, %19
  br label %21

; <label>:21                                      ; preds = %18, %0
  %.0.i1 = phi float [ %20, %18 ], [ %15, %0 ]
  %22 = fmul float %.0.i1, %.0.i1
  %23 = fmul float %22, 0x3F82FD7040000000
  %24 = fadd float %23, 0x3F6B323AE0000000
  %25 = fmul float %22, %24
  %26 = fadd float %25, 0x3F98E20C80000000
  %27 = fmul float %22, %26
  %28 = fadd float %27, 0x3FAB5DBCA0000000
  %29 = fmul float %22, %28
  %30 = fadd float %29, 0x3FC112B1C0000000
  %31 = fmul float %22, %30
  %32 = fadd float %31, 0x3FD5554F20000000
  %33 = fmul float %.0.i1, %32
  %34 = fmul float %22, %33
  %35 = fadd float %.0.i1, %34
  br i1 %17, label %_ZN11OpenImageIO4v1_78fast_tanEf.exit3, label %36

; <label>:36                                      ; preds = %21
  %37 = fdiv float -1.000000e+00, %35
  br label %_ZN11OpenImageIO4v1_78fast_tanEf.exit3

_ZN11OpenImageIO4v1_78fast_tanEf.exit3:           ; preds = %21, %36
  %u.0.i2 = phi float [ %37, %36 ], [ %35, %21 ]
  %38 = bitcast i8* %r_ to float*
  store float %u.0.i2, float* %38, align 4, !tbaa !1
  %39 = getelementptr inbounds i8* %a_, i64 4
  %40 = bitcast i8* %39 to float*
  %41 = load float* %40, align 4, !tbaa !1
  %42 = fmul float %41, 0x3FE45F3060000000
  %43 = tail call float @copysignf(float 5.000000e-01, float %42) #12
  %44 = fadd float %42, %43
  %45 = fptosi float %44 to i32
  %46 = sitofp i32 %45 to float
  %47 = fmul float %46, 0xBFF9200000000000
  %48 = fadd float %41, %47
  %49 = fmul float %46, 0xBF3FB40000000000
  %50 = fadd float %49, %48
  %51 = fmul float %46, 0xBE74440000000000
  %52 = fadd float %51, %50
  %53 = fmul float %46, 0xBD868C2340000000
  %54 = fadd float %53, %52
  %55 = and i32 %45, 1
  %56 = icmp eq i32 %55, 0
  br i1 %56, label %57, label %60

; <label>:57                                      ; preds = %_ZN11OpenImageIO4v1_78fast_tanEf.exit3
  %58 = fsub float 0x3FE921FB60000000, %54
  %59 = fsub float 0x3FE921FB60000000, %58
  br label %60

; <label>:60                                      ; preds = %57, %_ZN11OpenImageIO4v1_78fast_tanEf.exit3
  %.0.i4 = phi float [ %59, %57 ], [ %54, %_ZN11OpenImageIO4v1_78fast_tanEf.exit3 ]
  %61 = fmul float %.0.i4, %.0.i4
  %62 = fmul float %61, 0x3F82FD7040000000
  %63 = fadd float %62, 0x3F6B323AE0000000
  %64 = fmul float %61, %63
  %65 = fadd float %64, 0x3F98E20C80000000
  %66 = fmul float %61, %65
  %67 = fadd float %66, 0x3FAB5DBCA0000000
  %68 = fmul float %61, %67
  %69 = fadd float %68, 0x3FC112B1C0000000
  %70 = fmul float %61, %69
  %71 = fadd float %70, 0x3FD5554F20000000
  %72 = fmul float %.0.i4, %71
  %73 = fmul float %61, %72
  %74 = fadd float %.0.i4, %73
  br i1 %56, label %_ZN11OpenImageIO4v1_78fast_tanEf.exit6, label %75

; <label>:75                                      ; preds = %60
  %76 = fdiv float -1.000000e+00, %74
  br label %_ZN11OpenImageIO4v1_78fast_tanEf.exit6

_ZN11OpenImageIO4v1_78fast_tanEf.exit6:           ; preds = %60, %75
  %u.0.i5 = phi float [ %76, %75 ], [ %74, %60 ]
  %77 = getelementptr inbounds i8* %r_, i64 4
  %78 = bitcast i8* %77 to float*
  store float %u.0.i5, float* %78, align 4, !tbaa !1
  %79 = getelementptr inbounds i8* %a_, i64 8
  %80 = bitcast i8* %79 to float*
  %81 = load float* %80, align 4, !tbaa !1
  %82 = fmul float %81, 0x3FE45F3060000000
  %83 = tail call float @copysignf(float 5.000000e-01, float %82) #12
  %84 = fadd float %82, %83
  %85 = fptosi float %84 to i32
  %86 = sitofp i32 %85 to float
  %87 = fmul float %86, 0xBFF9200000000000
  %88 = fadd float %81, %87
  %89 = fmul float %86, 0xBF3FB40000000000
  %90 = fadd float %89, %88
  %91 = fmul float %86, 0xBE74440000000000
  %92 = fadd float %91, %90
  %93 = fmul float %86, 0xBD868C2340000000
  %94 = fadd float %93, %92
  %95 = and i32 %85, 1
  %96 = icmp eq i32 %95, 0
  br i1 %96, label %97, label %100

; <label>:97                                      ; preds = %_ZN11OpenImageIO4v1_78fast_tanEf.exit6
  %98 = fsub float 0x3FE921FB60000000, %94
  %99 = fsub float 0x3FE921FB60000000, %98
  br label %100

; <label>:100                                     ; preds = %97, %_ZN11OpenImageIO4v1_78fast_tanEf.exit6
  %.0.i = phi float [ %99, %97 ], [ %94, %_ZN11OpenImageIO4v1_78fast_tanEf.exit6 ]
  %101 = fmul float %.0.i, %.0.i
  %102 = fmul float %101, 0x3F82FD7040000000
  %103 = fadd float %102, 0x3F6B323AE0000000
  %104 = fmul float %101, %103
  %105 = fadd float %104, 0x3F98E20C80000000
  %106 = fmul float %101, %105
  %107 = fadd float %106, 0x3FAB5DBCA0000000
  %108 = fmul float %101, %107
  %109 = fadd float %108, 0x3FC112B1C0000000
  %110 = fmul float %101, %109
  %111 = fadd float %110, 0x3FD5554F20000000
  %112 = fmul float %.0.i, %111
  %113 = fmul float %101, %112
  %114 = fadd float %.0.i, %113
  br i1 %96, label %_ZN11OpenImageIO4v1_78fast_tanEf.exit, label %115

; <label>:115                                     ; preds = %100
  %116 = fdiv float -1.000000e+00, %114
  br label %_ZN11OpenImageIO4v1_78fast_tanEf.exit

_ZN11OpenImageIO4v1_78fast_tanEf.exit:            ; preds = %100, %115
  %u.0.i = phi float [ %116, %115 ], [ %114, %100 ]
  %117 = getelementptr inbounds i8* %r_, i64 8
  %118 = bitcast i8* %117 to float*
  store float %u.0.i, float* %118, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_tan_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = alloca %"class.OSL::Dual2", align 4
  %2 = alloca %"class.OSL::Dual2", align 4
  %3 = alloca %"class.OSL::Dual2", align 4
  %4 = bitcast i8* %a_ to float*
  %5 = getelementptr inbounds i8* %a_, i64 12
  %6 = bitcast i8* %5 to float*
  %7 = getelementptr inbounds i8* %a_, i64 24
  %8 = bitcast i8* %7 to float*
  %9 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 0
  %10 = load float* %4, align 4, !tbaa !1
  store float %10, float* %9, align 4, !tbaa !9
  %11 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 1
  %12 = load float* %6, align 4, !tbaa !1
  store float %12, float* %11, align 4, !tbaa !11
  %13 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 2
  %14 = load float* %8, align 4, !tbaa !1
  store float %14, float* %13, align 4, !tbaa !12
  %15 = call { <2 x float>, float } @_ZN3OSL8fast_tanERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %1)
  %16 = extractvalue { <2 x float>, float } %15, 0
  %17 = extractvalue { <2 x float>, float } %15, 1
  %18 = getelementptr inbounds i8* %a_, i64 4
  %19 = bitcast i8* %18 to float*
  %20 = getelementptr inbounds i8* %a_, i64 16
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8* %a_, i64 28
  %23 = bitcast i8* %22 to float*
  %24 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 0
  %25 = load float* %19, align 4, !tbaa !1
  store float %25, float* %24, align 4, !tbaa !9
  %26 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 1
  %27 = load float* %21, align 4, !tbaa !1
  store float %27, float* %26, align 4, !tbaa !11
  %28 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 2
  %29 = load float* %23, align 4, !tbaa !1
  store float %29, float* %28, align 4, !tbaa !12
  %30 = call { <2 x float>, float } @_ZN3OSL8fast_tanERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %2)
  %31 = extractvalue { <2 x float>, float } %30, 0
  %32 = extractvalue { <2 x float>, float } %30, 1
  %33 = getelementptr inbounds i8* %a_, i64 8
  %34 = bitcast i8* %33 to float*
  %35 = getelementptr inbounds i8* %a_, i64 20
  %36 = bitcast i8* %35 to float*
  %37 = getelementptr inbounds i8* %a_, i64 32
  %38 = bitcast i8* %37 to float*
  %39 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 0
  %40 = load float* %34, align 4, !tbaa !1
  store float %40, float* %39, align 4, !tbaa !9
  %41 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 1
  %42 = load float* %36, align 4, !tbaa !1
  store float %42, float* %41, align 4, !tbaa !11
  %43 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 2
  %44 = load float* %38, align 4, !tbaa !1
  store float %44, float* %43, align 4, !tbaa !12
  %45 = call { <2 x float>, float } @_ZN3OSL8fast_tanERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %3)
  %46 = extractvalue { <2 x float>, float } %45, 0
  %47 = extractvalue { <2 x float>, float } %45, 1
  %48 = extractelement <2 x float> %16, i32 0
  %49 = extractelement <2 x float> %31, i32 0
  %50 = extractelement <2 x float> %46, i32 0
  %51 = extractelement <2 x float> %16, i32 1
  %52 = extractelement <2 x float> %31, i32 1
  %53 = extractelement <2 x float> %46, i32 1
  %54 = bitcast i8* %r_ to float*
  store float %48, float* %54, align 4, !tbaa !5
  %55 = getelementptr inbounds i8* %r_, i64 4
  %56 = bitcast i8* %55 to float*
  store float %49, float* %56, align 4, !tbaa !7
  %57 = getelementptr inbounds i8* %r_, i64 8
  %58 = bitcast i8* %57 to float*
  store float %50, float* %58, align 4, !tbaa !8
  %59 = getelementptr inbounds i8* %r_, i64 12
  %60 = bitcast i8* %59 to float*
  store float %51, float* %60, align 4, !tbaa !5
  %61 = getelementptr inbounds i8* %r_, i64 16
  %62 = bitcast i8* %61 to float*
  store float %52, float* %62, align 4, !tbaa !7
  %63 = getelementptr inbounds i8* %r_, i64 20
  %64 = bitcast i8* %63 to float*
  store float %53, float* %64, align 4, !tbaa !8
  %65 = getelementptr inbounds i8* %r_, i64 24
  %66 = bitcast i8* %65 to float*
  store float %17, float* %66, align 4, !tbaa !5
  %67 = getelementptr inbounds i8* %r_, i64 28
  %68 = bitcast i8* %67 to float*
  store float %32, float* %68, align 4, !tbaa !7
  %69 = getelementptr inbounds i8* %r_, i64 32
  %70 = bitcast i8* %69 to float*
  store float %47, float* %70, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_asin_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fcmp olt float %1, 1.000000e+00
  br i1 %2, label %3, label %_ZN11OpenImageIO4v1_79fast_asinEf.exit

; <label>:3                                       ; preds = %0
  %4 = fsub float 1.000000e+00, %1
  %5 = fsub float 1.000000e+00, %4
  br label %_ZN11OpenImageIO4v1_79fast_asinEf.exit

_ZN11OpenImageIO4v1_79fast_asinEf.exit:           ; preds = %0, %3
  %6 = phi float [ %5, %3 ], [ 1.000000e+00, %0 ]
  %7 = fsub float 1.000000e+00, %6
  %8 = tail call float @sqrtf(float %7) #12
  %9 = fmul float %6, 0xBF96290BA0000000
  %10 = fadd float %9, 0x3FB3F68760000000
  %11 = fmul float %6, %10
  %12 = fadd float %11, 0xBFCB4D7260000000
  %13 = fmul float %6, %12
  %14 = fadd float %13, 0x3FF921FB60000000
  %15 = fmul float %8, %14
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = tail call float @copysignf(float %16, float %a) #12
  ret float %17
}

; Function Attrs: nounwind uwtable
define void @osl_asin_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp olt float %3, 1.000000e+00
  br i1 %4, label %5, label %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i

; <label>:5                                       ; preds = %0
  %6 = fsub float 1.000000e+00, %3
  %7 = fsub float 1.000000e+00, %6
  br label %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i

_ZN11OpenImageIO4v1_79fast_asinEf.exit.i:         ; preds = %5, %0
  %8 = phi float [ %7, %5 ], [ 1.000000e+00, %0 ]
  %9 = fsub float 1.000000e+00, %8
  %10 = tail call float @sqrtf(float %9) #12
  %11 = fmul float %8, 0xBF96290BA0000000
  %12 = fadd float %11, 0x3FB3F68760000000
  %13 = fmul float %8, %12
  %14 = fadd float %13, 0xBFCB4D7260000000
  %15 = fmul float %8, %14
  %16 = fadd float %15, 0x3FF921FB60000000
  %17 = fmul float %10, %16
  %18 = fsub float 0x3FF921FB60000000, %17
  %19 = tail call float @copysignf(float %18, float %2) #12
  br i1 %4, label %20, label %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit

; <label>:20                                      ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i
  %21 = fmul float %2, %2
  %22 = fsub float 1.000000e+00, %21
  %23 = tail call float @sqrtf(float %22) #12
  %24 = fdiv float 1.000000e+00, %23
  br label %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i, %20
  %25 = phi float [ %24, %20 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i ]
  %26 = getelementptr inbounds i8* %a, i64 4
  %27 = bitcast i8* %26 to float*
  %28 = load float* %27, align 4, !tbaa !1
  %29 = fmul float %25, %28
  %30 = getelementptr inbounds i8* %a, i64 8
  %31 = bitcast i8* %30 to float*
  %32 = load float* %31, align 4, !tbaa !1
  %33 = fmul float %25, %32
  %34 = insertelement <2 x float> undef, float %19, i32 0
  %35 = insertelement <2 x float> %34, float %29, i32 1
  %36 = bitcast i8* %r to <2 x float>*
  store <2 x float> %35, <2 x float>* %36, align 4
  %37 = getelementptr inbounds i8* %r, i64 8
  %38 = bitcast i8* %37 to float*
  store float %33, float* %38, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_asin_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp olt float %3, 1.000000e+00
  br i1 %4, label %5, label %_ZN11OpenImageIO4v1_79fast_asinEf.exit1

; <label>:5                                       ; preds = %0
  %6 = fsub float 1.000000e+00, %3
  %7 = fsub float 1.000000e+00, %6
  br label %_ZN11OpenImageIO4v1_79fast_asinEf.exit1

_ZN11OpenImageIO4v1_79fast_asinEf.exit1:          ; preds = %0, %5
  %8 = phi float [ %7, %5 ], [ 1.000000e+00, %0 ]
  %9 = fsub float 1.000000e+00, %8
  %10 = tail call float @sqrtf(float %9) #12
  %11 = fmul float %8, 0xBF96290BA0000000
  %12 = fadd float %11, 0x3FB3F68760000000
  %13 = fmul float %8, %12
  %14 = fadd float %13, 0xBFCB4D7260000000
  %15 = fmul float %8, %14
  %16 = fadd float %15, 0x3FF921FB60000000
  %17 = fmul float %10, %16
  %18 = fsub float 0x3FF921FB60000000, %17
  %19 = tail call float @copysignf(float %18, float %2) #12
  %20 = bitcast i8* %r_ to float*
  store float %19, float* %20, align 4, !tbaa !1
  %21 = getelementptr inbounds i8* %a_, i64 4
  %22 = bitcast i8* %21 to float*
  %23 = load float* %22, align 4, !tbaa !1
  %24 = tail call float @fabsf(float %23) #12
  %25 = fcmp olt float %24, 1.000000e+00
  br i1 %25, label %26, label %_ZN11OpenImageIO4v1_79fast_asinEf.exit2

; <label>:26                                      ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit1
  %27 = fsub float 1.000000e+00, %24
  %28 = fsub float 1.000000e+00, %27
  br label %_ZN11OpenImageIO4v1_79fast_asinEf.exit2

_ZN11OpenImageIO4v1_79fast_asinEf.exit2:          ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit1, %26
  %29 = phi float [ %28, %26 ], [ 1.000000e+00, %_ZN11OpenImageIO4v1_79fast_asinEf.exit1 ]
  %30 = fsub float 1.000000e+00, %29
  %31 = tail call float @sqrtf(float %30) #12
  %32 = fmul float %29, 0xBF96290BA0000000
  %33 = fadd float %32, 0x3FB3F68760000000
  %34 = fmul float %29, %33
  %35 = fadd float %34, 0xBFCB4D7260000000
  %36 = fmul float %29, %35
  %37 = fadd float %36, 0x3FF921FB60000000
  %38 = fmul float %31, %37
  %39 = fsub float 0x3FF921FB60000000, %38
  %40 = tail call float @copysignf(float %39, float %23) #12
  %41 = getelementptr inbounds i8* %r_, i64 4
  %42 = bitcast i8* %41 to float*
  store float %40, float* %42, align 4, !tbaa !1
  %43 = getelementptr inbounds i8* %a_, i64 8
  %44 = bitcast i8* %43 to float*
  %45 = load float* %44, align 4, !tbaa !1
  %46 = tail call float @fabsf(float %45) #12
  %47 = fcmp olt float %46, 1.000000e+00
  br i1 %47, label %48, label %_ZN11OpenImageIO4v1_79fast_asinEf.exit

; <label>:48                                      ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit2
  %49 = fsub float 1.000000e+00, %46
  %50 = fsub float 1.000000e+00, %49
  br label %_ZN11OpenImageIO4v1_79fast_asinEf.exit

_ZN11OpenImageIO4v1_79fast_asinEf.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit2, %48
  %51 = phi float [ %50, %48 ], [ 1.000000e+00, %_ZN11OpenImageIO4v1_79fast_asinEf.exit2 ]
  %52 = fsub float 1.000000e+00, %51
  %53 = tail call float @sqrtf(float %52) #12
  %54 = fmul float %51, 0xBF96290BA0000000
  %55 = fadd float %54, 0x3FB3F68760000000
  %56 = fmul float %51, %55
  %57 = fadd float %56, 0xBFCB4D7260000000
  %58 = fmul float %51, %57
  %59 = fadd float %58, 0x3FF921FB60000000
  %60 = fmul float %53, %59
  %61 = fsub float 0x3FF921FB60000000, %60
  %62 = tail call float @copysignf(float %61, float %45) #12
  %63 = getelementptr inbounds i8* %r_, i64 8
  %64 = bitcast i8* %63 to float*
  store float %62, float* %64, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_asin_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = tail call float @fabsf(float %6) #12
  %10 = fcmp olt float %9, 1.000000e+00
  br i1 %10, label %11, label %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i13

; <label>:11                                      ; preds = %0
  %12 = fsub float 1.000000e+00, %9
  %13 = fsub float 1.000000e+00, %12
  br label %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i13

_ZN11OpenImageIO4v1_79fast_asinEf.exit.i13:       ; preds = %11, %0
  %14 = phi float [ %13, %11 ], [ 1.000000e+00, %0 ]
  %15 = fsub float 1.000000e+00, %14
  %16 = tail call float @sqrtf(float %15) #12
  %17 = fmul float %14, 0xBF96290BA0000000
  %18 = fadd float %17, 0x3FB3F68760000000
  %19 = fmul float %14, %18
  %20 = fadd float %19, 0xBFCB4D7260000000
  %21 = fmul float %14, %20
  %22 = fadd float %21, 0x3FF921FB60000000
  %23 = fmul float %16, %22
  %24 = fsub float 0x3FF921FB60000000, %23
  %25 = tail call float @copysignf(float %24, float %6) #12
  br i1 %10, label %26, label %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit14

; <label>:26                                      ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i13
  %27 = fmul float %6, %6
  %28 = fsub float 1.000000e+00, %27
  %29 = tail call float @sqrtf(float %28) #12
  %30 = fdiv float 1.000000e+00, %29
  br label %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit14

_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit14:         ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i13, %26
  %31 = phi float [ %30, %26 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i13 ]
  %32 = fmul float %7, %31
  %33 = fmul float %8, %31
  %34 = getelementptr inbounds i8* %a_, i64 4
  %35 = bitcast i8* %34 to float*
  %36 = getelementptr inbounds i8* %a_, i64 16
  %37 = bitcast i8* %36 to float*
  %38 = getelementptr inbounds i8* %a_, i64 28
  %39 = bitcast i8* %38 to float*
  %40 = load float* %35, align 4, !tbaa !1
  %41 = load float* %37, align 4, !tbaa !1
  %42 = load float* %39, align 4, !tbaa !1
  %43 = tail call float @fabsf(float %40) #12
  %44 = fcmp olt float %43, 1.000000e+00
  br i1 %44, label %45, label %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i11

; <label>:45                                      ; preds = %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit14
  %46 = fsub float 1.000000e+00, %43
  %47 = fsub float 1.000000e+00, %46
  br label %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i11

_ZN11OpenImageIO4v1_79fast_asinEf.exit.i11:       ; preds = %45, %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit14
  %48 = phi float [ %47, %45 ], [ 1.000000e+00, %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit14 ]
  %49 = fsub float 1.000000e+00, %48
  %50 = tail call float @sqrtf(float %49) #12
  %51 = fmul float %48, 0xBF96290BA0000000
  %52 = fadd float %51, 0x3FB3F68760000000
  %53 = fmul float %48, %52
  %54 = fadd float %53, 0xBFCB4D7260000000
  %55 = fmul float %48, %54
  %56 = fadd float %55, 0x3FF921FB60000000
  %57 = fmul float %50, %56
  %58 = fsub float 0x3FF921FB60000000, %57
  %59 = tail call float @copysignf(float %58, float %40) #12
  br i1 %44, label %60, label %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit12

; <label>:60                                      ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i11
  %61 = fmul float %40, %40
  %62 = fsub float 1.000000e+00, %61
  %63 = tail call float @sqrtf(float %62) #12
  %64 = fdiv float 1.000000e+00, %63
  br label %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit12

_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit12:         ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i11, %60
  %65 = phi float [ %64, %60 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i11 ]
  %66 = fmul float %41, %65
  %67 = fmul float %42, %65
  %68 = getelementptr inbounds i8* %a_, i64 8
  %69 = bitcast i8* %68 to float*
  %70 = getelementptr inbounds i8* %a_, i64 20
  %71 = bitcast i8* %70 to float*
  %72 = getelementptr inbounds i8* %a_, i64 32
  %73 = bitcast i8* %72 to float*
  %74 = load float* %69, align 4, !tbaa !1
  %75 = load float* %71, align 4, !tbaa !1
  %76 = load float* %73, align 4, !tbaa !1
  %77 = tail call float @fabsf(float %74) #12
  %78 = fcmp olt float %77, 1.000000e+00
  br i1 %78, label %79, label %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i

; <label>:79                                      ; preds = %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit12
  %80 = fsub float 1.000000e+00, %77
  %81 = fsub float 1.000000e+00, %80
  br label %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i

_ZN11OpenImageIO4v1_79fast_asinEf.exit.i:         ; preds = %79, %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit12
  %82 = phi float [ %81, %79 ], [ 1.000000e+00, %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit12 ]
  %83 = fsub float 1.000000e+00, %82
  %84 = tail call float @sqrtf(float %83) #12
  %85 = fmul float %82, 0xBF96290BA0000000
  %86 = fadd float %85, 0x3FB3F68760000000
  %87 = fmul float %82, %86
  %88 = fadd float %87, 0xBFCB4D7260000000
  %89 = fmul float %82, %88
  %90 = fadd float %89, 0x3FF921FB60000000
  %91 = fmul float %84, %90
  %92 = fsub float 0x3FF921FB60000000, %91
  %93 = tail call float @copysignf(float %92, float %74) #12
  br i1 %78, label %94, label %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit

; <label>:94                                      ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i
  %95 = fmul float %74, %74
  %96 = fsub float 1.000000e+00, %95
  %97 = tail call float @sqrtf(float %96) #12
  %98 = fdiv float 1.000000e+00, %97
  br label %_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_asinERKNS_5Dual2IfEE.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i, %94
  %99 = phi float [ %98, %94 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_asinEf.exit.i ]
  %100 = fmul float %75, %99
  %101 = fmul float %76, %99
  %102 = bitcast i8* %r_ to float*
  store float %25, float* %102, align 4, !tbaa !5
  %103 = getelementptr inbounds i8* %r_, i64 4
  %104 = bitcast i8* %103 to float*
  store float %59, float* %104, align 4, !tbaa !7
  %105 = getelementptr inbounds i8* %r_, i64 8
  %106 = bitcast i8* %105 to float*
  store float %93, float* %106, align 4, !tbaa !8
  %107 = getelementptr inbounds i8* %r_, i64 12
  %108 = bitcast i8* %107 to float*
  store float %32, float* %108, align 4, !tbaa !5
  %109 = getelementptr inbounds i8* %r_, i64 16
  %110 = bitcast i8* %109 to float*
  store float %66, float* %110, align 4, !tbaa !7
  %111 = getelementptr inbounds i8* %r_, i64 20
  %112 = bitcast i8* %111 to float*
  store float %100, float* %112, align 4, !tbaa !8
  %113 = getelementptr inbounds i8* %r_, i64 24
  %114 = bitcast i8* %113 to float*
  store float %33, float* %114, align 4, !tbaa !5
  %115 = getelementptr inbounds i8* %r_, i64 28
  %116 = bitcast i8* %115 to float*
  store float %67, float* %116, align 4, !tbaa !7
  %117 = getelementptr inbounds i8* %r_, i64 32
  %118 = bitcast i8* %117 to float*
  store float %101, float* %118, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_acos_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fcmp olt float %1, 1.000000e+00
  br i1 %2, label %3, label %6

; <label>:3                                       ; preds = %0
  %4 = fsub float 1.000000e+00, %1
  %5 = fsub float 1.000000e+00, %4
  br label %6

; <label>:6                                       ; preds = %3, %0
  %7 = phi float [ %5, %3 ], [ 1.000000e+00, %0 ]
  %8 = fsub float 1.000000e+00, %7
  %9 = tail call float @sqrtf(float %8) #12
  %10 = fmul float %7, 0xBF96290BA0000000
  %11 = fadd float %10, 0x3FB3F68760000000
  %12 = fmul float %7, %11
  %13 = fadd float %12, 0xBFCB4D7260000000
  %14 = fmul float %7, %13
  %15 = fadd float %14, 0x3FF921FB60000000
  %16 = fmul float %9, %15
  %17 = fcmp olt float %a, 0.000000e+00
  br i1 %17, label %18, label %_ZN11OpenImageIO4v1_79fast_acosEf.exit

; <label>:18                                      ; preds = %6
  %19 = fsub float 0x400921FB60000000, %16
  br label %_ZN11OpenImageIO4v1_79fast_acosEf.exit

_ZN11OpenImageIO4v1_79fast_acosEf.exit:           ; preds = %6, %18
  %20 = phi float [ %19, %18 ], [ %16, %6 ]
  ret float %20
}

; Function Attrs: nounwind uwtable
define void @osl_acos_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp olt float %3, 1.000000e+00
  br i1 %4, label %5, label %8

; <label>:5                                       ; preds = %0
  %6 = fsub float 1.000000e+00, %3
  %7 = fsub float 1.000000e+00, %6
  br label %8

; <label>:8                                       ; preds = %5, %0
  %9 = phi float [ %7, %5 ], [ 1.000000e+00, %0 ]
  %10 = fsub float 1.000000e+00, %9
  %11 = tail call float @sqrtf(float %10) #12
  %12 = fmul float %9, 0xBF96290BA0000000
  %13 = fadd float %12, 0x3FB3F68760000000
  %14 = fmul float %9, %13
  %15 = fadd float %14, 0xBFCB4D7260000000
  %16 = fmul float %9, %15
  %17 = fadd float %16, 0x3FF921FB60000000
  %18 = fmul float %11, %17
  %19 = fcmp olt float %2, 0.000000e+00
  br i1 %19, label %20, label %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i

; <label>:20                                      ; preds = %8
  %21 = fsub float 0x400921FB60000000, %18
  br label %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i

_ZN11OpenImageIO4v1_79fast_acosEf.exit.i:         ; preds = %20, %8
  %22 = phi float [ %21, %20 ], [ %18, %8 ]
  br i1 %4, label %23, label %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit

; <label>:23                                      ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i
  %24 = fmul float %2, %2
  %25 = fsub float 1.000000e+00, %24
  %26 = tail call float @sqrtf(float %25) #12
  %27 = fdiv float -1.000000e+00, %26
  br label %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i, %23
  %28 = phi float [ %27, %23 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i ]
  %29 = getelementptr inbounds i8* %a, i64 4
  %30 = bitcast i8* %29 to float*
  %31 = load float* %30, align 4, !tbaa !1
  %32 = fmul float %28, %31
  %33 = getelementptr inbounds i8* %a, i64 8
  %34 = bitcast i8* %33 to float*
  %35 = load float* %34, align 4, !tbaa !1
  %36 = fmul float %28, %35
  %37 = insertelement <2 x float> undef, float %22, i32 0
  %38 = insertelement <2 x float> %37, float %32, i32 1
  %39 = bitcast i8* %r to <2 x float>*
  store <2 x float> %38, <2 x float>* %39, align 4
  %40 = getelementptr inbounds i8* %r, i64 8
  %41 = bitcast i8* %40 to float*
  store float %36, float* %41, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_acos_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp olt float %3, 1.000000e+00
  br i1 %4, label %5, label %8

; <label>:5                                       ; preds = %0
  %6 = fsub float 1.000000e+00, %3
  %7 = fsub float 1.000000e+00, %6
  br label %8

; <label>:8                                       ; preds = %5, %0
  %9 = phi float [ %7, %5 ], [ 1.000000e+00, %0 ]
  %10 = fsub float 1.000000e+00, %9
  %11 = tail call float @sqrtf(float %10) #12
  %12 = fmul float %9, 0xBF96290BA0000000
  %13 = fadd float %12, 0x3FB3F68760000000
  %14 = fmul float %9, %13
  %15 = fadd float %14, 0xBFCB4D7260000000
  %16 = fmul float %9, %15
  %17 = fadd float %16, 0x3FF921FB60000000
  %18 = fmul float %11, %17
  %19 = fcmp olt float %2, 0.000000e+00
  br i1 %19, label %20, label %_ZN11OpenImageIO4v1_79fast_acosEf.exit1

; <label>:20                                      ; preds = %8
  %21 = fsub float 0x400921FB60000000, %18
  br label %_ZN11OpenImageIO4v1_79fast_acosEf.exit1

_ZN11OpenImageIO4v1_79fast_acosEf.exit1:          ; preds = %8, %20
  %22 = phi float [ %21, %20 ], [ %18, %8 ]
  %23 = bitcast i8* %r_ to float*
  store float %22, float* %23, align 4, !tbaa !1
  %24 = getelementptr inbounds i8* %a_, i64 4
  %25 = bitcast i8* %24 to float*
  %26 = load float* %25, align 4, !tbaa !1
  %27 = tail call float @fabsf(float %26) #12
  %28 = fcmp olt float %27, 1.000000e+00
  br i1 %28, label %29, label %32

; <label>:29                                      ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit1
  %30 = fsub float 1.000000e+00, %27
  %31 = fsub float 1.000000e+00, %30
  br label %32

; <label>:32                                      ; preds = %29, %_ZN11OpenImageIO4v1_79fast_acosEf.exit1
  %33 = phi float [ %31, %29 ], [ 1.000000e+00, %_ZN11OpenImageIO4v1_79fast_acosEf.exit1 ]
  %34 = fsub float 1.000000e+00, %33
  %35 = tail call float @sqrtf(float %34) #12
  %36 = fmul float %33, 0xBF96290BA0000000
  %37 = fadd float %36, 0x3FB3F68760000000
  %38 = fmul float %33, %37
  %39 = fadd float %38, 0xBFCB4D7260000000
  %40 = fmul float %33, %39
  %41 = fadd float %40, 0x3FF921FB60000000
  %42 = fmul float %35, %41
  %43 = fcmp olt float %26, 0.000000e+00
  br i1 %43, label %44, label %_ZN11OpenImageIO4v1_79fast_acosEf.exit2

; <label>:44                                      ; preds = %32
  %45 = fsub float 0x400921FB60000000, %42
  br label %_ZN11OpenImageIO4v1_79fast_acosEf.exit2

_ZN11OpenImageIO4v1_79fast_acosEf.exit2:          ; preds = %32, %44
  %46 = phi float [ %45, %44 ], [ %42, %32 ]
  %47 = getelementptr inbounds i8* %r_, i64 4
  %48 = bitcast i8* %47 to float*
  store float %46, float* %48, align 4, !tbaa !1
  %49 = getelementptr inbounds i8* %a_, i64 8
  %50 = bitcast i8* %49 to float*
  %51 = load float* %50, align 4, !tbaa !1
  %52 = tail call float @fabsf(float %51) #12
  %53 = fcmp olt float %52, 1.000000e+00
  br i1 %53, label %54, label %57

; <label>:54                                      ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit2
  %55 = fsub float 1.000000e+00, %52
  %56 = fsub float 1.000000e+00, %55
  br label %57

; <label>:57                                      ; preds = %54, %_ZN11OpenImageIO4v1_79fast_acosEf.exit2
  %58 = phi float [ %56, %54 ], [ 1.000000e+00, %_ZN11OpenImageIO4v1_79fast_acosEf.exit2 ]
  %59 = fsub float 1.000000e+00, %58
  %60 = tail call float @sqrtf(float %59) #12
  %61 = fmul float %58, 0xBF96290BA0000000
  %62 = fadd float %61, 0x3FB3F68760000000
  %63 = fmul float %58, %62
  %64 = fadd float %63, 0xBFCB4D7260000000
  %65 = fmul float %58, %64
  %66 = fadd float %65, 0x3FF921FB60000000
  %67 = fmul float %60, %66
  %68 = fcmp olt float %51, 0.000000e+00
  br i1 %68, label %69, label %_ZN11OpenImageIO4v1_79fast_acosEf.exit

; <label>:69                                      ; preds = %57
  %70 = fsub float 0x400921FB60000000, %67
  br label %_ZN11OpenImageIO4v1_79fast_acosEf.exit

_ZN11OpenImageIO4v1_79fast_acosEf.exit:           ; preds = %57, %69
  %71 = phi float [ %70, %69 ], [ %67, %57 ]
  %72 = getelementptr inbounds i8* %r_, i64 8
  %73 = bitcast i8* %72 to float*
  store float %71, float* %73, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_acos_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = tail call float @fabsf(float %6) #12
  %10 = fcmp olt float %9, 1.000000e+00
  br i1 %10, label %11, label %14

; <label>:11                                      ; preds = %0
  %12 = fsub float 1.000000e+00, %9
  %13 = fsub float 1.000000e+00, %12
  br label %14

; <label>:14                                      ; preds = %11, %0
  %15 = phi float [ %13, %11 ], [ 1.000000e+00, %0 ]
  %16 = fsub float 1.000000e+00, %15
  %17 = tail call float @sqrtf(float %16) #12
  %18 = fmul float %15, 0xBF96290BA0000000
  %19 = fadd float %18, 0x3FB3F68760000000
  %20 = fmul float %15, %19
  %21 = fadd float %20, 0xBFCB4D7260000000
  %22 = fmul float %15, %21
  %23 = fadd float %22, 0x3FF921FB60000000
  %24 = fmul float %17, %23
  %25 = fcmp olt float %6, 0.000000e+00
  br i1 %25, label %26, label %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i13

; <label>:26                                      ; preds = %14
  %27 = fsub float 0x400921FB60000000, %24
  br label %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i13

_ZN11OpenImageIO4v1_79fast_acosEf.exit.i13:       ; preds = %26, %14
  %28 = phi float [ %27, %26 ], [ %24, %14 ]
  br i1 %10, label %29, label %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit14

; <label>:29                                      ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i13
  %30 = fmul float %6, %6
  %31 = fsub float 1.000000e+00, %30
  %32 = tail call float @sqrtf(float %31) #12
  %33 = fdiv float -1.000000e+00, %32
  br label %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit14

_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit14:         ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i13, %29
  %34 = phi float [ %33, %29 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i13 ]
  %35 = fmul float %7, %34
  %36 = fmul float %8, %34
  %37 = getelementptr inbounds i8* %a_, i64 4
  %38 = bitcast i8* %37 to float*
  %39 = getelementptr inbounds i8* %a_, i64 16
  %40 = bitcast i8* %39 to float*
  %41 = getelementptr inbounds i8* %a_, i64 28
  %42 = bitcast i8* %41 to float*
  %43 = load float* %38, align 4, !tbaa !1
  %44 = load float* %40, align 4, !tbaa !1
  %45 = load float* %42, align 4, !tbaa !1
  %46 = tail call float @fabsf(float %43) #12
  %47 = fcmp olt float %46, 1.000000e+00
  br i1 %47, label %48, label %51

; <label>:48                                      ; preds = %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit14
  %49 = fsub float 1.000000e+00, %46
  %50 = fsub float 1.000000e+00, %49
  br label %51

; <label>:51                                      ; preds = %48, %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit14
  %52 = phi float [ %50, %48 ], [ 1.000000e+00, %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit14 ]
  %53 = fsub float 1.000000e+00, %52
  %54 = tail call float @sqrtf(float %53) #12
  %55 = fmul float %52, 0xBF96290BA0000000
  %56 = fadd float %55, 0x3FB3F68760000000
  %57 = fmul float %52, %56
  %58 = fadd float %57, 0xBFCB4D7260000000
  %59 = fmul float %52, %58
  %60 = fadd float %59, 0x3FF921FB60000000
  %61 = fmul float %54, %60
  %62 = fcmp olt float %43, 0.000000e+00
  br i1 %62, label %63, label %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i11

; <label>:63                                      ; preds = %51
  %64 = fsub float 0x400921FB60000000, %61
  br label %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i11

_ZN11OpenImageIO4v1_79fast_acosEf.exit.i11:       ; preds = %63, %51
  %65 = phi float [ %64, %63 ], [ %61, %51 ]
  br i1 %47, label %66, label %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit12

; <label>:66                                      ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i11
  %67 = fmul float %43, %43
  %68 = fsub float 1.000000e+00, %67
  %69 = tail call float @sqrtf(float %68) #12
  %70 = fdiv float -1.000000e+00, %69
  br label %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit12

_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit12:         ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i11, %66
  %71 = phi float [ %70, %66 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i11 ]
  %72 = fmul float %44, %71
  %73 = fmul float %45, %71
  %74 = getelementptr inbounds i8* %a_, i64 8
  %75 = bitcast i8* %74 to float*
  %76 = getelementptr inbounds i8* %a_, i64 20
  %77 = bitcast i8* %76 to float*
  %78 = getelementptr inbounds i8* %a_, i64 32
  %79 = bitcast i8* %78 to float*
  %80 = load float* %75, align 4, !tbaa !1
  %81 = load float* %77, align 4, !tbaa !1
  %82 = load float* %79, align 4, !tbaa !1
  %83 = tail call float @fabsf(float %80) #12
  %84 = fcmp olt float %83, 1.000000e+00
  br i1 %84, label %85, label %88

; <label>:85                                      ; preds = %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit12
  %86 = fsub float 1.000000e+00, %83
  %87 = fsub float 1.000000e+00, %86
  br label %88

; <label>:88                                      ; preds = %85, %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit12
  %89 = phi float [ %87, %85 ], [ 1.000000e+00, %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit12 ]
  %90 = fsub float 1.000000e+00, %89
  %91 = tail call float @sqrtf(float %90) #12
  %92 = fmul float %89, 0xBF96290BA0000000
  %93 = fadd float %92, 0x3FB3F68760000000
  %94 = fmul float %89, %93
  %95 = fadd float %94, 0xBFCB4D7260000000
  %96 = fmul float %89, %95
  %97 = fadd float %96, 0x3FF921FB60000000
  %98 = fmul float %91, %97
  %99 = fcmp olt float %80, 0.000000e+00
  br i1 %99, label %100, label %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i

; <label>:100                                     ; preds = %88
  %101 = fsub float 0x400921FB60000000, %98
  br label %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i

_ZN11OpenImageIO4v1_79fast_acosEf.exit.i:         ; preds = %100, %88
  %102 = phi float [ %101, %100 ], [ %98, %88 ]
  br i1 %84, label %103, label %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit

; <label>:103                                     ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i
  %104 = fmul float %80, %80
  %105 = fsub float 1.000000e+00, %104
  %106 = tail call float @sqrtf(float %105) #12
  %107 = fdiv float -1.000000e+00, %106
  br label %_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_acosERKNS_5Dual2IfEE.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i, %103
  %108 = phi float [ %107, %103 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_acosEf.exit.i ]
  %109 = fmul float %81, %108
  %110 = fmul float %82, %108
  %111 = bitcast i8* %r_ to float*
  store float %28, float* %111, align 4, !tbaa !5
  %112 = getelementptr inbounds i8* %r_, i64 4
  %113 = bitcast i8* %112 to float*
  store float %65, float* %113, align 4, !tbaa !7
  %114 = getelementptr inbounds i8* %r_, i64 8
  %115 = bitcast i8* %114 to float*
  store float %102, float* %115, align 4, !tbaa !8
  %116 = getelementptr inbounds i8* %r_, i64 12
  %117 = bitcast i8* %116 to float*
  store float %35, float* %117, align 4, !tbaa !5
  %118 = getelementptr inbounds i8* %r_, i64 16
  %119 = bitcast i8* %118 to float*
  store float %72, float* %119, align 4, !tbaa !7
  %120 = getelementptr inbounds i8* %r_, i64 20
  %121 = bitcast i8* %120 to float*
  store float %109, float* %121, align 4, !tbaa !8
  %122 = getelementptr inbounds i8* %r_, i64 24
  %123 = bitcast i8* %122 to float*
  store float %36, float* %123, align 4, !tbaa !5
  %124 = getelementptr inbounds i8* %r_, i64 28
  %125 = bitcast i8* %124 to float*
  store float %73, float* %125, align 4, !tbaa !7
  %126 = getelementptr inbounds i8* %r_, i64 32
  %127 = bitcast i8* %126 to float*
  store float %110, float* %127, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_atan_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fcmp ogt float %1, 1.000000e+00
  br i1 %2, label %3, label %5

; <label>:3                                       ; preds = %0
  %4 = fdiv float 1.000000e+00, %1
  br label %5

; <label>:5                                       ; preds = %3, %0
  %6 = phi float [ %4, %3 ], [ %1, %0 ]
  %7 = fsub float 1.000000e+00, %6
  %8 = fsub float 1.000000e+00, %7
  %9 = fmul float %8, %8
  %10 = fmul float %9, 0x3FDB9F00A0000000
  %11 = fadd float %10, 1.000000e+00
  %12 = fmul float %8, %11
  %13 = fmul float %9, 0x3FADDC09A0000000
  %14 = fadd float %13, 0x3FE87649C0000000
  %15 = fmul float %9, %14
  %16 = fadd float %15, 1.000000e+00
  %17 = fdiv float %12, %16
  br i1 %2, label %18, label %_ZN11OpenImageIO4v1_79fast_atanEf.exit

; <label>:18                                      ; preds = %5
  %19 = fsub float 0x3FF921FB60000000, %17
  br label %_ZN11OpenImageIO4v1_79fast_atanEf.exit

_ZN11OpenImageIO4v1_79fast_atanEf.exit:           ; preds = %5, %18
  %r.0.i = phi float [ %19, %18 ], [ %17, %5 ]
  %20 = tail call float @copysignf(float %r.0.i, float %a) #12
  ret float %20
}

; Function Attrs: nounwind uwtable
define void @osl_atan_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp ogt float %3, 1.000000e+00
  br i1 %4, label %5, label %7

; <label>:5                                       ; preds = %0
  %6 = fdiv float 1.000000e+00, %3
  br label %7

; <label>:7                                       ; preds = %5, %0
  %8 = phi float [ %6, %5 ], [ %3, %0 ]
  %9 = fsub float 1.000000e+00, %8
  %10 = fsub float 1.000000e+00, %9
  %11 = fmul float %10, %10
  %12 = fmul float %11, 0x3FDB9F00A0000000
  %13 = fadd float %12, 1.000000e+00
  %14 = fmul float %10, %13
  %15 = fmul float %11, 0x3FADDC09A0000000
  %16 = fadd float %15, 0x3FE87649C0000000
  %17 = fmul float %11, %16
  %18 = fadd float %17, 1.000000e+00
  %19 = fdiv float %14, %18
  br i1 %4, label %20, label %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit

; <label>:20                                      ; preds = %7
  %21 = fsub float 0x3FF921FB60000000, %19
  br label %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit:           ; preds = %7, %20
  %r.0.i.i = phi float [ %21, %20 ], [ %19, %7 ]
  %22 = tail call float @copysignf(float %r.0.i.i, float %2) #12
  %23 = fmul float %2, %2
  %24 = fadd float %23, 1.000000e+00
  %25 = fdiv float 1.000000e+00, %24
  %26 = getelementptr inbounds i8* %a, i64 4
  %27 = bitcast i8* %26 to float*
  %28 = load float* %27, align 4, !tbaa !1
  %29 = fmul float %25, %28
  %30 = getelementptr inbounds i8* %a, i64 8
  %31 = bitcast i8* %30 to float*
  %32 = load float* %31, align 4, !tbaa !1
  %33 = fmul float %25, %32
  %34 = insertelement <2 x float> undef, float %22, i32 0
  %35 = insertelement <2 x float> %34, float %29, i32 1
  %36 = bitcast i8* %r to <2 x float>*
  store <2 x float> %35, <2 x float>* %36, align 4
  %37 = getelementptr inbounds i8* %r, i64 8
  %38 = bitcast i8* %37 to float*
  store float %33, float* %38, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp ogt float %3, 1.000000e+00
  br i1 %4, label %5, label %7

; <label>:5                                       ; preds = %0
  %6 = fdiv float 1.000000e+00, %3
  br label %7

; <label>:7                                       ; preds = %5, %0
  %8 = phi float [ %6, %5 ], [ %3, %0 ]
  %9 = fsub float 1.000000e+00, %8
  %10 = fsub float 1.000000e+00, %9
  %11 = fmul float %10, %10
  %12 = fmul float %11, 0x3FDB9F00A0000000
  %13 = fadd float %12, 1.000000e+00
  %14 = fmul float %10, %13
  %15 = fmul float %11, 0x3FADDC09A0000000
  %16 = fadd float %15, 0x3FE87649C0000000
  %17 = fmul float %11, %16
  %18 = fadd float %17, 1.000000e+00
  %19 = fdiv float %14, %18
  br i1 %4, label %20, label %_ZN11OpenImageIO4v1_79fast_atanEf.exit2

; <label>:20                                      ; preds = %7
  %21 = fsub float 0x3FF921FB60000000, %19
  br label %_ZN11OpenImageIO4v1_79fast_atanEf.exit2

_ZN11OpenImageIO4v1_79fast_atanEf.exit2:          ; preds = %7, %20
  %r.0.i1 = phi float [ %21, %20 ], [ %19, %7 ]
  %22 = tail call float @copysignf(float %r.0.i1, float %2) #12
  %23 = bitcast i8* %r_ to float*
  store float %22, float* %23, align 4, !tbaa !1
  %24 = getelementptr inbounds i8* %a_, i64 4
  %25 = bitcast i8* %24 to float*
  %26 = load float* %25, align 4, !tbaa !1
  %27 = tail call float @fabsf(float %26) #12
  %28 = fcmp ogt float %27, 1.000000e+00
  br i1 %28, label %29, label %31

; <label>:29                                      ; preds = %_ZN11OpenImageIO4v1_79fast_atanEf.exit2
  %30 = fdiv float 1.000000e+00, %27
  br label %31

; <label>:31                                      ; preds = %29, %_ZN11OpenImageIO4v1_79fast_atanEf.exit2
  %32 = phi float [ %30, %29 ], [ %27, %_ZN11OpenImageIO4v1_79fast_atanEf.exit2 ]
  %33 = fsub float 1.000000e+00, %32
  %34 = fsub float 1.000000e+00, %33
  %35 = fmul float %34, %34
  %36 = fmul float %35, 0x3FDB9F00A0000000
  %37 = fadd float %36, 1.000000e+00
  %38 = fmul float %34, %37
  %39 = fmul float %35, 0x3FADDC09A0000000
  %40 = fadd float %39, 0x3FE87649C0000000
  %41 = fmul float %35, %40
  %42 = fadd float %41, 1.000000e+00
  %43 = fdiv float %38, %42
  br i1 %28, label %44, label %_ZN11OpenImageIO4v1_79fast_atanEf.exit4

; <label>:44                                      ; preds = %31
  %45 = fsub float 0x3FF921FB60000000, %43
  br label %_ZN11OpenImageIO4v1_79fast_atanEf.exit4

_ZN11OpenImageIO4v1_79fast_atanEf.exit4:          ; preds = %31, %44
  %r.0.i3 = phi float [ %45, %44 ], [ %43, %31 ]
  %46 = tail call float @copysignf(float %r.0.i3, float %26) #12
  %47 = getelementptr inbounds i8* %r_, i64 4
  %48 = bitcast i8* %47 to float*
  store float %46, float* %48, align 4, !tbaa !1
  %49 = getelementptr inbounds i8* %a_, i64 8
  %50 = bitcast i8* %49 to float*
  %51 = load float* %50, align 4, !tbaa !1
  %52 = tail call float @fabsf(float %51) #12
  %53 = fcmp ogt float %52, 1.000000e+00
  br i1 %53, label %54, label %56

; <label>:54                                      ; preds = %_ZN11OpenImageIO4v1_79fast_atanEf.exit4
  %55 = fdiv float 1.000000e+00, %52
  br label %56

; <label>:56                                      ; preds = %54, %_ZN11OpenImageIO4v1_79fast_atanEf.exit4
  %57 = phi float [ %55, %54 ], [ %52, %_ZN11OpenImageIO4v1_79fast_atanEf.exit4 ]
  %58 = fsub float 1.000000e+00, %57
  %59 = fsub float 1.000000e+00, %58
  %60 = fmul float %59, %59
  %61 = fmul float %60, 0x3FDB9F00A0000000
  %62 = fadd float %61, 1.000000e+00
  %63 = fmul float %59, %62
  %64 = fmul float %60, 0x3FADDC09A0000000
  %65 = fadd float %64, 0x3FE87649C0000000
  %66 = fmul float %60, %65
  %67 = fadd float %66, 1.000000e+00
  %68 = fdiv float %63, %67
  br i1 %53, label %69, label %_ZN11OpenImageIO4v1_79fast_atanEf.exit

; <label>:69                                      ; preds = %56
  %70 = fsub float 0x3FF921FB60000000, %68
  br label %_ZN11OpenImageIO4v1_79fast_atanEf.exit

_ZN11OpenImageIO4v1_79fast_atanEf.exit:           ; preds = %56, %69
  %r.0.i = phi float [ %70, %69 ], [ %68, %56 ]
  %71 = tail call float @copysignf(float %r.0.i, float %51) #12
  %72 = getelementptr inbounds i8* %r_, i64 8
  %73 = bitcast i8* %72 to float*
  store float %71, float* %73, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = tail call float @fabsf(float %6) #12
  %10 = fcmp ogt float %9, 1.000000e+00
  br i1 %10, label %11, label %13

; <label>:11                                      ; preds = %0
  %12 = fdiv float 1.000000e+00, %9
  br label %13

; <label>:13                                      ; preds = %11, %0
  %14 = phi float [ %12, %11 ], [ %9, %0 ]
  %15 = fsub float 1.000000e+00, %14
  %16 = fsub float 1.000000e+00, %15
  %17 = fmul float %16, %16
  %18 = fmul float %17, 0x3FDB9F00A0000000
  %19 = fadd float %18, 1.000000e+00
  %20 = fmul float %16, %19
  %21 = fmul float %17, 0x3FADDC09A0000000
  %22 = fadd float %21, 0x3FE87649C0000000
  %23 = fmul float %17, %22
  %24 = fadd float %23, 1.000000e+00
  %25 = fdiv float %20, %24
  br i1 %10, label %26, label %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit14

; <label>:26                                      ; preds = %13
  %27 = fsub float 0x3FF921FB60000000, %25
  br label %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit14

_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit14:         ; preds = %13, %26
  %r.0.i.i13 = phi float [ %27, %26 ], [ %25, %13 ]
  %28 = tail call float @copysignf(float %r.0.i.i13, float %6) #12
  %29 = fmul float %6, %6
  %30 = fadd float %29, 1.000000e+00
  %31 = fdiv float 1.000000e+00, %30
  %32 = fmul float %7, %31
  %33 = fmul float %8, %31
  %34 = getelementptr inbounds i8* %a_, i64 4
  %35 = bitcast i8* %34 to float*
  %36 = getelementptr inbounds i8* %a_, i64 16
  %37 = bitcast i8* %36 to float*
  %38 = getelementptr inbounds i8* %a_, i64 28
  %39 = bitcast i8* %38 to float*
  %40 = load float* %35, align 4, !tbaa !1
  %41 = load float* %37, align 4, !tbaa !1
  %42 = load float* %39, align 4, !tbaa !1
  %43 = tail call float @fabsf(float %40) #12
  %44 = fcmp ogt float %43, 1.000000e+00
  br i1 %44, label %45, label %47

; <label>:45                                      ; preds = %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit14
  %46 = fdiv float 1.000000e+00, %43
  br label %47

; <label>:47                                      ; preds = %45, %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit14
  %48 = phi float [ %46, %45 ], [ %43, %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit14 ]
  %49 = fsub float 1.000000e+00, %48
  %50 = fsub float 1.000000e+00, %49
  %51 = fmul float %50, %50
  %52 = fmul float %51, 0x3FDB9F00A0000000
  %53 = fadd float %52, 1.000000e+00
  %54 = fmul float %50, %53
  %55 = fmul float %51, 0x3FADDC09A0000000
  %56 = fadd float %55, 0x3FE87649C0000000
  %57 = fmul float %51, %56
  %58 = fadd float %57, 1.000000e+00
  %59 = fdiv float %54, %58
  br i1 %44, label %60, label %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit12

; <label>:60                                      ; preds = %47
  %61 = fsub float 0x3FF921FB60000000, %59
  br label %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit12

_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit12:         ; preds = %47, %60
  %r.0.i.i11 = phi float [ %61, %60 ], [ %59, %47 ]
  %62 = tail call float @copysignf(float %r.0.i.i11, float %40) #12
  %63 = fmul float %40, %40
  %64 = fadd float %63, 1.000000e+00
  %65 = fdiv float 1.000000e+00, %64
  %66 = fmul float %41, %65
  %67 = fmul float %42, %65
  %68 = getelementptr inbounds i8* %a_, i64 8
  %69 = bitcast i8* %68 to float*
  %70 = getelementptr inbounds i8* %a_, i64 20
  %71 = bitcast i8* %70 to float*
  %72 = getelementptr inbounds i8* %a_, i64 32
  %73 = bitcast i8* %72 to float*
  %74 = load float* %69, align 4, !tbaa !1
  %75 = load float* %71, align 4, !tbaa !1
  %76 = load float* %73, align 4, !tbaa !1
  %77 = tail call float @fabsf(float %74) #12
  %78 = fcmp ogt float %77, 1.000000e+00
  br i1 %78, label %79, label %81

; <label>:79                                      ; preds = %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit12
  %80 = fdiv float 1.000000e+00, %77
  br label %81

; <label>:81                                      ; preds = %79, %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit12
  %82 = phi float [ %80, %79 ], [ %77, %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit12 ]
  %83 = fsub float 1.000000e+00, %82
  %84 = fsub float 1.000000e+00, %83
  %85 = fmul float %84, %84
  %86 = fmul float %85, 0x3FDB9F00A0000000
  %87 = fadd float %86, 1.000000e+00
  %88 = fmul float %84, %87
  %89 = fmul float %85, 0x3FADDC09A0000000
  %90 = fadd float %89, 0x3FE87649C0000000
  %91 = fmul float %85, %90
  %92 = fadd float %91, 1.000000e+00
  %93 = fdiv float %88, %92
  br i1 %78, label %94, label %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit

; <label>:94                                      ; preds = %81
  %95 = fsub float 0x3FF921FB60000000, %93
  br label %_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_atanERKNS_5Dual2IfEE.exit:           ; preds = %81, %94
  %r.0.i.i = phi float [ %95, %94 ], [ %93, %81 ]
  %96 = tail call float @copysignf(float %r.0.i.i, float %74) #12
  %97 = fmul float %74, %74
  %98 = fadd float %97, 1.000000e+00
  %99 = fdiv float 1.000000e+00, %98
  %100 = fmul float %75, %99
  %101 = fmul float %76, %99
  %102 = bitcast i8* %r_ to float*
  store float %28, float* %102, align 4, !tbaa !5
  %103 = getelementptr inbounds i8* %r_, i64 4
  %104 = bitcast i8* %103 to float*
  store float %62, float* %104, align 4, !tbaa !7
  %105 = getelementptr inbounds i8* %r_, i64 8
  %106 = bitcast i8* %105 to float*
  store float %96, float* %106, align 4, !tbaa !8
  %107 = getelementptr inbounds i8* %r_, i64 12
  %108 = bitcast i8* %107 to float*
  store float %32, float* %108, align 4, !tbaa !5
  %109 = getelementptr inbounds i8* %r_, i64 16
  %110 = bitcast i8* %109 to float*
  store float %66, float* %110, align 4, !tbaa !7
  %111 = getelementptr inbounds i8* %r_, i64 20
  %112 = bitcast i8* %111 to float*
  store float %100, float* %112, align 4, !tbaa !8
  %113 = getelementptr inbounds i8* %r_, i64 24
  %114 = bitcast i8* %113 to float*
  store float %33, float* %114, align 4, !tbaa !5
  %115 = getelementptr inbounds i8* %r_, i64 28
  %116 = bitcast i8* %115 to float*
  store float %67, float* %116, align 4, !tbaa !7
  %117 = getelementptr inbounds i8* %r_, i64 32
  %118 = bitcast i8* %117 to float*
  store float %101, float* %118, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_atan2_fff(float %a, float %b) #3 {
  %1 = tail call float @fabsf(float %b) #12
  %2 = tail call float @fabsf(float %a) #12
  %3 = fcmp oeq float %a, 0.000000e+00
  br i1 %3, label %12, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp oeq float %1, %2
  br i1 %5, label %12, label %6

; <label>:6                                       ; preds = %4
  %7 = fcmp ogt float %2, %1
  br i1 %7, label %8, label %10

; <label>:8                                       ; preds = %6
  %9 = fdiv float %1, %2
  br label %12

; <label>:10                                      ; preds = %6
  %11 = fdiv float %2, %1
  br label %12

; <label>:12                                      ; preds = %10, %8, %4, %0
  %13 = phi float [ 0.000000e+00, %0 ], [ 1.000000e+00, %4 ], [ %9, %8 ], [ %11, %10 ]
  %14 = fsub float 1.000000e+00, %13
  %15 = fsub float 1.000000e+00, %14
  %16 = fmul float %15, %15
  %17 = fmul float %16, 0x3FDB9F00A0000000
  %18 = fadd float %17, 1.000000e+00
  %19 = fmul float %15, %18
  %20 = fmul float %16, 0x3FADDC09A0000000
  %21 = fadd float %20, 0x3FE87649C0000000
  %22 = fmul float %16, %21
  %23 = fadd float %22, 1.000000e+00
  %24 = fdiv float %19, %23
  %25 = fcmp ogt float %2, %1
  br i1 %25, label %26, label %28

; <label>:26                                      ; preds = %12
  %27 = fsub float 0x3FF921FB60000000, %24
  br label %28

; <label>:28                                      ; preds = %26, %12
  %r.0.i = phi float [ %27, %26 ], [ %24, %12 ]
  %29 = bitcast float %b to i32
  %30 = icmp slt i32 %29, 0
  br i1 %30, label %31, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit

; <label>:31                                      ; preds = %28
  %32 = fsub float 0x400921FB60000000, %r.0.i
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit:        ; preds = %28, %31
  %r.1.i = phi float [ %32, %31 ], [ %r.0.i, %28 ]
  %33 = tail call float @copysignf(float %r.1.i, float %a) #12
  ret float %33
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dfdfdf(i8* nocapture %r, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = bitcast i8* %b to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #12
  %6 = tail call float @fabsf(float %2) #12
  %7 = fcmp oeq float %2, 0.000000e+00
  br i1 %7, label %16, label %8

; <label>:8                                       ; preds = %0
  %9 = fcmp oeq float %5, %6
  br i1 %9, label %16, label %10

; <label>:10                                      ; preds = %8
  %11 = fcmp ogt float %6, %5
  br i1 %11, label %12, label %14

; <label>:12                                      ; preds = %10
  %13 = fdiv float %5, %6
  br label %16

; <label>:14                                      ; preds = %10
  %15 = fdiv float %6, %5
  br label %16

; <label>:16                                      ; preds = %14, %12, %8, %0
  %17 = phi float [ 0.000000e+00, %0 ], [ 1.000000e+00, %8 ], [ %13, %12 ], [ %15, %14 ]
  %18 = fsub float 1.000000e+00, %17
  %19 = fsub float 1.000000e+00, %18
  %20 = fmul float %19, %19
  %21 = fmul float %20, 0x3FDB9F00A0000000
  %22 = fadd float %21, 1.000000e+00
  %23 = fmul float %19, %22
  %24 = fmul float %20, 0x3FADDC09A0000000
  %25 = fadd float %24, 0x3FE87649C0000000
  %26 = fmul float %20, %25
  %27 = fadd float %26, 1.000000e+00
  %28 = fdiv float %23, %27
  %29 = fcmp ogt float %6, %5
  br i1 %29, label %30, label %32

; <label>:30                                      ; preds = %16
  %31 = fsub float 0x3FF921FB60000000, %28
  br label %32

; <label>:32                                      ; preds = %30, %16
  %r.0.i.i = phi float [ %31, %30 ], [ %28, %16 ]
  %33 = bitcast float %4 to i32
  %34 = icmp slt i32 %33, 0
  br i1 %34, label %35, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i

; <label>:35                                      ; preds = %32
  %36 = fsub float 0x400921FB60000000, %r.0.i.i
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i:      ; preds = %35, %32
  %r.1.i.i = phi float [ %36, %35 ], [ %r.0.i.i, %32 ]
  %37 = tail call float @copysignf(float %r.1.i.i, float %2) #12
  %38 = fcmp oeq float %4, 0.000000e+00
  %or.cond.i = and i1 %38, %7
  br i1 %or.cond.i, label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit, label %39

; <label>:39                                      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i
  %40 = fmul float %4, %4
  %41 = fmul float %2, %2
  %42 = fadd float %41, %40
  %43 = fdiv float 1.000000e+00, %42
  br label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit

_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit:      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i, %39
  %44 = phi float [ %43, %39 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i ]
  %45 = getelementptr inbounds i8* %b, i64 4
  %46 = bitcast i8* %45 to float*
  %47 = load float* %46, align 4, !tbaa !1
  %48 = fmul float %2, %47
  %49 = getelementptr inbounds i8* %a, i64 4
  %50 = bitcast i8* %49 to float*
  %51 = load float* %50, align 4, !tbaa !1
  %52 = fmul float %4, %51
  %53 = fsub float %48, %52
  %54 = fmul float %44, %53
  %55 = getelementptr inbounds i8* %b, i64 8
  %56 = bitcast i8* %55 to float*
  %57 = load float* %56, align 4, !tbaa !1
  %58 = fmul float %2, %57
  %59 = getelementptr inbounds i8* %a, i64 8
  %60 = bitcast i8* %59 to float*
  %61 = load float* %60, align 4, !tbaa !1
  %62 = fmul float %4, %61
  %63 = fsub float %58, %62
  %64 = fmul float %44, %63
  %65 = insertelement <2 x float> undef, float %37, i32 0
  %66 = insertelement <2 x float> %65, float %54, i32 1
  %67 = bitcast i8* %r to <2 x float>*
  store <2 x float> %66, <2 x float>* %67, align 4
  %68 = getelementptr inbounds i8* %r, i64 8
  %69 = bitcast i8* %68 to float*
  store float %64, float* %69, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dffdf(i8* nocapture %r, float %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %b to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = tail call float @fabsf(float %a) #12
  %5 = fcmp oeq float %a, 0.000000e+00
  br i1 %5, label %14, label %6

; <label>:6                                       ; preds = %0
  %7 = fcmp oeq float %3, %4
  br i1 %7, label %14, label %8

; <label>:8                                       ; preds = %6
  %9 = fcmp ogt float %4, %3
  br i1 %9, label %10, label %12

; <label>:10                                      ; preds = %8
  %11 = fdiv float %3, %4
  br label %14

; <label>:12                                      ; preds = %8
  %13 = fdiv float %4, %3
  br label %14

; <label>:14                                      ; preds = %12, %10, %6, %0
  %15 = phi float [ 0.000000e+00, %0 ], [ 1.000000e+00, %6 ], [ %11, %10 ], [ %13, %12 ]
  %16 = fsub float 1.000000e+00, %15
  %17 = fsub float 1.000000e+00, %16
  %18 = fmul float %17, %17
  %19 = fmul float %18, 0x3FDB9F00A0000000
  %20 = fadd float %19, 1.000000e+00
  %21 = fmul float %17, %20
  %22 = fmul float %18, 0x3FADDC09A0000000
  %23 = fadd float %22, 0x3FE87649C0000000
  %24 = fmul float %18, %23
  %25 = fadd float %24, 1.000000e+00
  %26 = fdiv float %21, %25
  %27 = fcmp ogt float %4, %3
  br i1 %27, label %28, label %30

; <label>:28                                      ; preds = %14
  %29 = fsub float 0x3FF921FB60000000, %26
  br label %30

; <label>:30                                      ; preds = %28, %14
  %r.0.i.i = phi float [ %29, %28 ], [ %26, %14 ]
  %31 = bitcast float %2 to i32
  %32 = icmp slt i32 %31, 0
  br i1 %32, label %33, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i

; <label>:33                                      ; preds = %30
  %34 = fsub float 0x400921FB60000000, %r.0.i.i
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i:      ; preds = %33, %30
  %r.1.i.i = phi float [ %34, %33 ], [ %r.0.i.i, %30 ]
  %35 = tail call float @copysignf(float %r.1.i.i, float %a) #12
  %36 = fcmp oeq float %2, 0.000000e+00
  %or.cond.i = and i1 %36, %5
  br i1 %or.cond.i, label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit, label %37

; <label>:37                                      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i
  %38 = fmul float %2, %2
  %39 = fmul float %a, %a
  %40 = fadd float %39, %38
  %41 = fdiv float 1.000000e+00, %40
  br label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit

_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit:      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i, %37
  %42 = phi float [ %41, %37 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i ]
  %43 = getelementptr inbounds i8* %b, i64 4
  %44 = bitcast i8* %43 to float*
  %45 = load float* %44, align 4, !tbaa !1
  %46 = fmul float %45, %a
  %47 = fmul float %2, 0.000000e+00
  %48 = fsub float %46, %47
  %49 = fmul float %42, %48
  %50 = getelementptr inbounds i8* %b, i64 8
  %51 = bitcast i8* %50 to float*
  %52 = load float* %51, align 4, !tbaa !1
  %53 = fmul float %52, %a
  %54 = fsub float %53, %47
  %55 = fmul float %42, %54
  %56 = insertelement <2 x float> undef, float %35, i32 0
  %57 = insertelement <2 x float> %56, float %49, i32 1
  %58 = bitcast i8* %r to <2 x float>*
  store <2 x float> %57, <2 x float>* %58, align 4
  %59 = getelementptr inbounds i8* %r, i64 8
  %60 = bitcast i8* %59 to float*
  store float %55, float* %60, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dfdff(i8* nocapture %r, i8* nocapture readonly %a, float %b) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %b) #12
  %4 = tail call float @fabsf(float %2) #12
  %5 = fcmp oeq float %2, 0.000000e+00
  br i1 %5, label %14, label %6

; <label>:6                                       ; preds = %0
  %7 = fcmp oeq float %3, %4
  br i1 %7, label %14, label %8

; <label>:8                                       ; preds = %6
  %9 = fcmp ogt float %4, %3
  br i1 %9, label %10, label %12

; <label>:10                                      ; preds = %8
  %11 = fdiv float %3, %4
  br label %14

; <label>:12                                      ; preds = %8
  %13 = fdiv float %4, %3
  br label %14

; <label>:14                                      ; preds = %12, %10, %6, %0
  %15 = phi float [ 0.000000e+00, %0 ], [ 1.000000e+00, %6 ], [ %11, %10 ], [ %13, %12 ]
  %16 = fsub float 1.000000e+00, %15
  %17 = fsub float 1.000000e+00, %16
  %18 = fmul float %17, %17
  %19 = fmul float %18, 0x3FDB9F00A0000000
  %20 = fadd float %19, 1.000000e+00
  %21 = fmul float %17, %20
  %22 = fmul float %18, 0x3FADDC09A0000000
  %23 = fadd float %22, 0x3FE87649C0000000
  %24 = fmul float %18, %23
  %25 = fadd float %24, 1.000000e+00
  %26 = fdiv float %21, %25
  %27 = fcmp ogt float %4, %3
  br i1 %27, label %28, label %30

; <label>:28                                      ; preds = %14
  %29 = fsub float 0x3FF921FB60000000, %26
  br label %30

; <label>:30                                      ; preds = %28, %14
  %r.0.i.i = phi float [ %29, %28 ], [ %26, %14 ]
  %31 = bitcast float %b to i32
  %32 = icmp slt i32 %31, 0
  br i1 %32, label %33, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i

; <label>:33                                      ; preds = %30
  %34 = fsub float 0x400921FB60000000, %r.0.i.i
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i:      ; preds = %33, %30
  %r.1.i.i = phi float [ %34, %33 ], [ %r.0.i.i, %30 ]
  %35 = tail call float @copysignf(float %r.1.i.i, float %2) #12
  %36 = fcmp oeq float %b, 0.000000e+00
  %or.cond.i = and i1 %36, %5
  br i1 %or.cond.i, label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit, label %37

; <label>:37                                      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i
  %38 = fmul float %b, %b
  %39 = fmul float %2, %2
  %40 = fadd float %38, %39
  %41 = fdiv float 1.000000e+00, %40
  br label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit

_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit:      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i, %37
  %42 = phi float [ %41, %37 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i ]
  %43 = fmul float %2, 0.000000e+00
  %44 = getelementptr inbounds i8* %a, i64 4
  %45 = bitcast i8* %44 to float*
  %46 = load float* %45, align 4, !tbaa !1
  %47 = fmul float %46, %b
  %48 = fsub float %43, %47
  %49 = fmul float %42, %48
  %50 = getelementptr inbounds i8* %a, i64 8
  %51 = bitcast i8* %50 to float*
  %52 = load float* %51, align 4, !tbaa !1
  %53 = fmul float %52, %b
  %54 = fsub float %43, %53
  %55 = fmul float %42, %54
  %56 = insertelement <2 x float> undef, float %35, i32 0
  %57 = insertelement <2 x float> %56, float %49, i32 1
  %58 = bitcast i8* %r to <2 x float>*
  store <2 x float> %57, <2 x float>* %58, align 4
  %59 = getelementptr inbounds i8* %r, i64 8
  %60 = bitcast i8* %59 to float*
  store float %55, float* %60, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_vvv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = bitcast i8* %b_ to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #12
  %6 = tail call float @fabsf(float %2) #12
  %7 = fcmp oeq float %2, 0.000000e+00
  br i1 %7, label %16, label %8

; <label>:8                                       ; preds = %0
  %9 = fcmp oeq float %5, %6
  br i1 %9, label %16, label %10

; <label>:10                                      ; preds = %8
  %11 = fcmp ogt float %6, %5
  br i1 %11, label %12, label %14

; <label>:12                                      ; preds = %10
  %13 = fdiv float %5, %6
  br label %16

; <label>:14                                      ; preds = %10
  %15 = fdiv float %6, %5
  br label %16

; <label>:16                                      ; preds = %14, %12, %8, %0
  %17 = phi float [ 0.000000e+00, %0 ], [ 1.000000e+00, %8 ], [ %13, %12 ], [ %15, %14 ]
  %18 = fsub float 1.000000e+00, %17
  %19 = fsub float 1.000000e+00, %18
  %20 = fmul float %19, %19
  %21 = fmul float %20, 0x3FDB9F00A0000000
  %22 = fadd float %21, 1.000000e+00
  %23 = fmul float %19, %22
  %24 = fmul float %20, 0x3FADDC09A0000000
  %25 = fadd float %24, 0x3FE87649C0000000
  %26 = fmul float %20, %25
  %27 = fadd float %26, 1.000000e+00
  %28 = fdiv float %23, %27
  %29 = fcmp ogt float %6, %5
  br i1 %29, label %30, label %32

; <label>:30                                      ; preds = %16
  %31 = fsub float 0x3FF921FB60000000, %28
  br label %32

; <label>:32                                      ; preds = %30, %16
  %r.0.i1 = phi float [ %31, %30 ], [ %28, %16 ]
  %33 = bitcast float %4 to i32
  %34 = icmp slt i32 %33, 0
  br i1 %34, label %35, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit3

; <label>:35                                      ; preds = %32
  %36 = fsub float 0x400921FB60000000, %r.0.i1
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit3

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit3:       ; preds = %32, %35
  %r.1.i2 = phi float [ %36, %35 ], [ %r.0.i1, %32 ]
  %37 = tail call float @copysignf(float %r.1.i2, float %2) #12
  %38 = bitcast i8* %r_ to float*
  store float %37, float* %38, align 4, !tbaa !1
  %39 = getelementptr inbounds i8* %a_, i64 4
  %40 = bitcast i8* %39 to float*
  %41 = load float* %40, align 4, !tbaa !1
  %42 = getelementptr inbounds i8* %b_, i64 4
  %43 = bitcast i8* %42 to float*
  %44 = load float* %43, align 4, !tbaa !1
  %45 = tail call float @fabsf(float %44) #12
  %46 = tail call float @fabsf(float %41) #12
  %47 = fcmp oeq float %41, 0.000000e+00
  br i1 %47, label %56, label %48

; <label>:48                                      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit3
  %49 = fcmp oeq float %45, %46
  br i1 %49, label %56, label %50

; <label>:50                                      ; preds = %48
  %51 = fcmp ogt float %46, %45
  br i1 %51, label %52, label %54

; <label>:52                                      ; preds = %50
  %53 = fdiv float %45, %46
  br label %56

; <label>:54                                      ; preds = %50
  %55 = fdiv float %46, %45
  br label %56

; <label>:56                                      ; preds = %54, %52, %48, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit3
  %57 = phi float [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit3 ], [ 1.000000e+00, %48 ], [ %53, %52 ], [ %55, %54 ]
  %58 = fsub float 1.000000e+00, %57
  %59 = fsub float 1.000000e+00, %58
  %60 = fmul float %59, %59
  %61 = fmul float %60, 0x3FDB9F00A0000000
  %62 = fadd float %61, 1.000000e+00
  %63 = fmul float %59, %62
  %64 = fmul float %60, 0x3FADDC09A0000000
  %65 = fadd float %64, 0x3FE87649C0000000
  %66 = fmul float %60, %65
  %67 = fadd float %66, 1.000000e+00
  %68 = fdiv float %63, %67
  %69 = fcmp ogt float %46, %45
  br i1 %69, label %70, label %72

; <label>:70                                      ; preds = %56
  %71 = fsub float 0x3FF921FB60000000, %68
  br label %72

; <label>:72                                      ; preds = %70, %56
  %r.0.i4 = phi float [ %71, %70 ], [ %68, %56 ]
  %73 = bitcast float %44 to i32
  %74 = icmp slt i32 %73, 0
  br i1 %74, label %75, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit6

; <label>:75                                      ; preds = %72
  %76 = fsub float 0x400921FB60000000, %r.0.i4
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit6

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit6:       ; preds = %72, %75
  %r.1.i5 = phi float [ %76, %75 ], [ %r.0.i4, %72 ]
  %77 = tail call float @copysignf(float %r.1.i5, float %41) #12
  %78 = getelementptr inbounds i8* %r_, i64 4
  %79 = bitcast i8* %78 to float*
  store float %77, float* %79, align 4, !tbaa !1
  %80 = getelementptr inbounds i8* %a_, i64 8
  %81 = bitcast i8* %80 to float*
  %82 = load float* %81, align 4, !tbaa !1
  %83 = getelementptr inbounds i8* %b_, i64 8
  %84 = bitcast i8* %83 to float*
  %85 = load float* %84, align 4, !tbaa !1
  %86 = tail call float @fabsf(float %85) #12
  %87 = tail call float @fabsf(float %82) #12
  %88 = fcmp oeq float %82, 0.000000e+00
  br i1 %88, label %97, label %89

; <label>:89                                      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit6
  %90 = fcmp oeq float %86, %87
  br i1 %90, label %97, label %91

; <label>:91                                      ; preds = %89
  %92 = fcmp ogt float %87, %86
  br i1 %92, label %93, label %95

; <label>:93                                      ; preds = %91
  %94 = fdiv float %86, %87
  br label %97

; <label>:95                                      ; preds = %91
  %96 = fdiv float %87, %86
  br label %97

; <label>:97                                      ; preds = %95, %93, %89, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit6
  %98 = phi float [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit6 ], [ 1.000000e+00, %89 ], [ %94, %93 ], [ %96, %95 ]
  %99 = fsub float 1.000000e+00, %98
  %100 = fsub float 1.000000e+00, %99
  %101 = fmul float %100, %100
  %102 = fmul float %101, 0x3FDB9F00A0000000
  %103 = fadd float %102, 1.000000e+00
  %104 = fmul float %100, %103
  %105 = fmul float %101, 0x3FADDC09A0000000
  %106 = fadd float %105, 0x3FE87649C0000000
  %107 = fmul float %101, %106
  %108 = fadd float %107, 1.000000e+00
  %109 = fdiv float %104, %108
  %110 = fcmp ogt float %87, %86
  br i1 %110, label %111, label %113

; <label>:111                                     ; preds = %97
  %112 = fsub float 0x3FF921FB60000000, %109
  br label %113

; <label>:113                                     ; preds = %111, %97
  %r.0.i = phi float [ %112, %111 ], [ %109, %97 ]
  %114 = bitcast float %85 to i32
  %115 = icmp slt i32 %114, 0
  br i1 %115, label %116, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit

; <label>:116                                     ; preds = %113
  %117 = fsub float 0x400921FB60000000, %r.0.i
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit:        ; preds = %113, %116
  %r.1.i = phi float [ %117, %116 ], [ %r.0.i, %113 ]
  %118 = tail call float @copysignf(float %r.1.i, float %82) #12
  %119 = getelementptr inbounds i8* %r_, i64 8
  %120 = bitcast i8* %119 to float*
  store float %118, float* %120, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dvdvdv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = bitcast i8* %b_ to float*
  %10 = getelementptr inbounds i8* %b_, i64 12
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8* %b_, i64 24
  %13 = bitcast i8* %12 to float*
  %14 = load float* %9, align 4, !tbaa !1
  %15 = load float* %11, align 4, !tbaa !1
  %16 = load float* %13, align 4, !tbaa !1
  %17 = tail call float @fabsf(float %14) #12
  %18 = tail call float @fabsf(float %6) #12
  %19 = fcmp oeq float %6, 0.000000e+00
  br i1 %19, label %28, label %20

; <label>:20                                      ; preds = %0
  %21 = fcmp oeq float %17, %18
  br i1 %21, label %28, label %22

; <label>:22                                      ; preds = %20
  %23 = fcmp ogt float %18, %17
  br i1 %23, label %24, label %26

; <label>:24                                      ; preds = %22
  %25 = fdiv float %17, %18
  br label %28

; <label>:26                                      ; preds = %22
  %27 = fdiv float %18, %17
  br label %28

; <label>:28                                      ; preds = %26, %24, %20, %0
  %29 = phi float [ 0.000000e+00, %0 ], [ 1.000000e+00, %20 ], [ %25, %24 ], [ %27, %26 ]
  %30 = fsub float 1.000000e+00, %29
  %31 = fsub float 1.000000e+00, %30
  %32 = fmul float %31, %31
  %33 = fmul float %32, 0x3FDB9F00A0000000
  %34 = fadd float %33, 1.000000e+00
  %35 = fmul float %31, %34
  %36 = fmul float %32, 0x3FADDC09A0000000
  %37 = fadd float %36, 0x3FE87649C0000000
  %38 = fmul float %32, %37
  %39 = fadd float %38, 1.000000e+00
  %40 = fdiv float %35, %39
  %41 = fcmp ogt float %18, %17
  br i1 %41, label %42, label %44

; <label>:42                                      ; preds = %28
  %43 = fsub float 0x3FF921FB60000000, %40
  br label %44

; <label>:44                                      ; preds = %42, %28
  %r.0.i.i16 = phi float [ %43, %42 ], [ %40, %28 ]
  %45 = bitcast float %14 to i32
  %46 = icmp slt i32 %45, 0
  br i1 %46, label %47, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i19

; <label>:47                                      ; preds = %44
  %48 = fsub float 0x400921FB60000000, %r.0.i.i16
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i19

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i19:    ; preds = %47, %44
  %r.1.i.i17 = phi float [ %48, %47 ], [ %r.0.i.i16, %44 ]
  %49 = tail call float @copysignf(float %r.1.i.i17, float %6) #12
  %50 = fcmp oeq float %14, 0.000000e+00
  %or.cond.i18 = and i1 %50, %19
  br i1 %or.cond.i18, label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit20, label %51

; <label>:51                                      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i19
  %52 = fmul float %14, %14
  %53 = fmul float %6, %6
  %54 = fadd float %53, %52
  %55 = fdiv float 1.000000e+00, %54
  br label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit20

_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit20:    ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i19, %51
  %56 = phi float [ %55, %51 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i19 ]
  %57 = fmul float %6, %15
  %58 = fmul float %7, %14
  %59 = fsub float %57, %58
  %60 = fmul float %59, %56
  %61 = fmul float %6, %16
  %62 = fmul float %8, %14
  %63 = fsub float %61, %62
  %64 = fmul float %63, %56
  %65 = getelementptr inbounds i8* %a_, i64 4
  %66 = bitcast i8* %65 to float*
  %67 = getelementptr inbounds i8* %a_, i64 16
  %68 = bitcast i8* %67 to float*
  %69 = getelementptr inbounds i8* %a_, i64 28
  %70 = bitcast i8* %69 to float*
  %71 = load float* %66, align 4, !tbaa !1
  %72 = load float* %68, align 4, !tbaa !1
  %73 = load float* %70, align 4, !tbaa !1
  %74 = getelementptr inbounds i8* %b_, i64 4
  %75 = bitcast i8* %74 to float*
  %76 = getelementptr inbounds i8* %b_, i64 16
  %77 = bitcast i8* %76 to float*
  %78 = getelementptr inbounds i8* %b_, i64 28
  %79 = bitcast i8* %78 to float*
  %80 = load float* %75, align 4, !tbaa !1
  %81 = load float* %77, align 4, !tbaa !1
  %82 = load float* %79, align 4, !tbaa !1
  %83 = tail call float @fabsf(float %80) #12
  %84 = tail call float @fabsf(float %71) #12
  %85 = fcmp oeq float %71, 0.000000e+00
  br i1 %85, label %94, label %86

; <label>:86                                      ; preds = %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit20
  %87 = fcmp oeq float %83, %84
  br i1 %87, label %94, label %88

; <label>:88                                      ; preds = %86
  %89 = fcmp ogt float %84, %83
  br i1 %89, label %90, label %92

; <label>:90                                      ; preds = %88
  %91 = fdiv float %83, %84
  br label %94

; <label>:92                                      ; preds = %88
  %93 = fdiv float %84, %83
  br label %94

; <label>:94                                      ; preds = %92, %90, %86, %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit20
  %95 = phi float [ 0.000000e+00, %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit20 ], [ 1.000000e+00, %86 ], [ %91, %90 ], [ %93, %92 ]
  %96 = fsub float 1.000000e+00, %95
  %97 = fsub float 1.000000e+00, %96
  %98 = fmul float %97, %97
  %99 = fmul float %98, 0x3FDB9F00A0000000
  %100 = fadd float %99, 1.000000e+00
  %101 = fmul float %97, %100
  %102 = fmul float %98, 0x3FADDC09A0000000
  %103 = fadd float %102, 0x3FE87649C0000000
  %104 = fmul float %98, %103
  %105 = fadd float %104, 1.000000e+00
  %106 = fdiv float %101, %105
  %107 = fcmp ogt float %84, %83
  br i1 %107, label %108, label %110

; <label>:108                                     ; preds = %94
  %109 = fsub float 0x3FF921FB60000000, %106
  br label %110

; <label>:110                                     ; preds = %108, %94
  %r.0.i.i11 = phi float [ %109, %108 ], [ %106, %94 ]
  %111 = bitcast float %80 to i32
  %112 = icmp slt i32 %111, 0
  br i1 %112, label %113, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i14

; <label>:113                                     ; preds = %110
  %114 = fsub float 0x400921FB60000000, %r.0.i.i11
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i14

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i14:    ; preds = %113, %110
  %r.1.i.i12 = phi float [ %114, %113 ], [ %r.0.i.i11, %110 ]
  %115 = tail call float @copysignf(float %r.1.i.i12, float %71) #12
  %116 = fcmp oeq float %80, 0.000000e+00
  %or.cond.i13 = and i1 %116, %85
  br i1 %or.cond.i13, label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit15, label %117

; <label>:117                                     ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i14
  %118 = fmul float %80, %80
  %119 = fmul float %71, %71
  %120 = fadd float %119, %118
  %121 = fdiv float 1.000000e+00, %120
  br label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit15

_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit15:    ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i14, %117
  %122 = phi float [ %121, %117 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i14 ]
  %123 = fmul float %71, %81
  %124 = fmul float %72, %80
  %125 = fsub float %123, %124
  %126 = fmul float %125, %122
  %127 = fmul float %71, %82
  %128 = fmul float %73, %80
  %129 = fsub float %127, %128
  %130 = fmul float %129, %122
  %131 = getelementptr inbounds i8* %a_, i64 8
  %132 = bitcast i8* %131 to float*
  %133 = getelementptr inbounds i8* %a_, i64 20
  %134 = bitcast i8* %133 to float*
  %135 = getelementptr inbounds i8* %a_, i64 32
  %136 = bitcast i8* %135 to float*
  %137 = load float* %132, align 4, !tbaa !1
  %138 = load float* %134, align 4, !tbaa !1
  %139 = load float* %136, align 4, !tbaa !1
  %140 = getelementptr inbounds i8* %b_, i64 8
  %141 = bitcast i8* %140 to float*
  %142 = getelementptr inbounds i8* %b_, i64 20
  %143 = bitcast i8* %142 to float*
  %144 = getelementptr inbounds i8* %b_, i64 32
  %145 = bitcast i8* %144 to float*
  %146 = load float* %141, align 4, !tbaa !1
  %147 = load float* %143, align 4, !tbaa !1
  %148 = load float* %145, align 4, !tbaa !1
  %149 = tail call float @fabsf(float %146) #12
  %150 = tail call float @fabsf(float %137) #12
  %151 = fcmp oeq float %137, 0.000000e+00
  br i1 %151, label %160, label %152

; <label>:152                                     ; preds = %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit15
  %153 = fcmp oeq float %149, %150
  br i1 %153, label %160, label %154

; <label>:154                                     ; preds = %152
  %155 = fcmp ogt float %150, %149
  br i1 %155, label %156, label %158

; <label>:156                                     ; preds = %154
  %157 = fdiv float %149, %150
  br label %160

; <label>:158                                     ; preds = %154
  %159 = fdiv float %150, %149
  br label %160

; <label>:160                                     ; preds = %158, %156, %152, %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit15
  %161 = phi float [ 0.000000e+00, %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit15 ], [ 1.000000e+00, %152 ], [ %157, %156 ], [ %159, %158 ]
  %162 = fsub float 1.000000e+00, %161
  %163 = fsub float 1.000000e+00, %162
  %164 = fmul float %163, %163
  %165 = fmul float %164, 0x3FDB9F00A0000000
  %166 = fadd float %165, 1.000000e+00
  %167 = fmul float %163, %166
  %168 = fmul float %164, 0x3FADDC09A0000000
  %169 = fadd float %168, 0x3FE87649C0000000
  %170 = fmul float %164, %169
  %171 = fadd float %170, 1.000000e+00
  %172 = fdiv float %167, %171
  %173 = fcmp ogt float %150, %149
  br i1 %173, label %174, label %176

; <label>:174                                     ; preds = %160
  %175 = fsub float 0x3FF921FB60000000, %172
  br label %176

; <label>:176                                     ; preds = %174, %160
  %r.0.i.i = phi float [ %175, %174 ], [ %172, %160 ]
  %177 = bitcast float %146 to i32
  %178 = icmp slt i32 %177, 0
  br i1 %178, label %179, label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i

; <label>:179                                     ; preds = %176
  %180 = fsub float 0x400921FB60000000, %r.0.i.i
  br label %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i

_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i:      ; preds = %179, %176
  %r.1.i.i = phi float [ %180, %179 ], [ %r.0.i.i, %176 ]
  %181 = tail call float @copysignf(float %r.1.i.i, float %137) #12
  %182 = fcmp oeq float %146, 0.000000e+00
  %or.cond.i = and i1 %182, %151
  br i1 %or.cond.i, label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit, label %183

; <label>:183                                     ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i
  %184 = fmul float %146, %146
  %185 = fmul float %137, %137
  %186 = fadd float %185, %184
  %187 = fdiv float 1.000000e+00, %186
  br label %_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit

_ZN3OSL10fast_atan2ERKNS_5Dual2IfEES3_.exit:      ; preds = %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i, %183
  %188 = phi float [ %187, %183 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_atan2Eff.exit.i ]
  %189 = fmul float %137, %147
  %190 = fmul float %138, %146
  %191 = fsub float %189, %190
  %192 = fmul float %191, %188
  %193 = fmul float %137, %148
  %194 = fmul float %139, %146
  %195 = fsub float %193, %194
  %196 = fmul float %195, %188
  %197 = bitcast i8* %r_ to float*
  store float %49, float* %197, align 4, !tbaa !5
  %198 = getelementptr inbounds i8* %r_, i64 4
  %199 = bitcast i8* %198 to float*
  store float %115, float* %199, align 4, !tbaa !7
  %200 = getelementptr inbounds i8* %r_, i64 8
  %201 = bitcast i8* %200 to float*
  store float %181, float* %201, align 4, !tbaa !8
  %202 = getelementptr inbounds i8* %r_, i64 12
  %203 = bitcast i8* %202 to float*
  store float %60, float* %203, align 4, !tbaa !5
  %204 = getelementptr inbounds i8* %r_, i64 16
  %205 = bitcast i8* %204 to float*
  store float %126, float* %205, align 4, !tbaa !7
  %206 = getelementptr inbounds i8* %r_, i64 20
  %207 = bitcast i8* %206 to float*
  store float %192, float* %207, align 4, !tbaa !8
  %208 = getelementptr inbounds i8* %r_, i64 24
  %209 = bitcast i8* %208 to float*
  store float %64, float* %209, align 4, !tbaa !5
  %210 = getelementptr inbounds i8* %r_, i64 28
  %211 = bitcast i8* %210 to float*
  store float %130, float* %211, align 4, !tbaa !7
  %212 = getelementptr inbounds i8* %r_, i64 32
  %213 = bitcast i8* %212 to float*
  store float %196, float* %213, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dvvdv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %a = alloca %"class.OSL::Dual2.0", align 4
  %1 = bitcast %"class.OSL::Dual2.0"* %a to i8*
  call void @llvm.lifetime.start(i64 36, i8* %1) #2
  %2 = bitcast i8* %a_ to float*
  %3 = load float* %2, align 4, !tbaa !5
  %4 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 0
  store float %3, float* %4, align 4, !tbaa !5
  %5 = getelementptr inbounds i8* %a_, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4, !tbaa !7
  %8 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 1
  store float %7, float* %8, align 4, !tbaa !7
  %9 = getelementptr inbounds i8* %a_, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !8
  %12 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 2
  store float %11, float* %12, align 4, !tbaa !8
  %13 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 1, i32 0
  %14 = bitcast float* %13 to i8*
  call void @llvm.memset.p0i8.i64(i8* %14, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_atan2_dvdvdv(i8* %r_, i8* %1, i8* %b_)
  call void @llvm.lifetime.end(i64 36, i8* %1) #2
  ret void
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #2

; Function Attrs: nounwind uwtable
define void @osl_atan2_dvdvv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %b = alloca %"class.OSL::Dual2.0", align 4
  %1 = bitcast %"class.OSL::Dual2.0"* %b to i8*
  call void @llvm.lifetime.start(i64 36, i8* %1) #2
  %2 = bitcast i8* %b_ to float*
  %3 = load float* %2, align 4, !tbaa !5
  %4 = getelementptr inbounds %"class.OSL::Dual2.0"* %b, i64 0, i32 0, i32 0
  store float %3, float* %4, align 4, !tbaa !5
  %5 = getelementptr inbounds i8* %b_, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4, !tbaa !7
  %8 = getelementptr inbounds %"class.OSL::Dual2.0"* %b, i64 0, i32 0, i32 1
  store float %7, float* %8, align 4, !tbaa !7
  %9 = getelementptr inbounds i8* %b_, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !8
  %12 = getelementptr inbounds %"class.OSL::Dual2.0"* %b, i64 0, i32 0, i32 2
  store float %11, float* %12, align 4, !tbaa !8
  %13 = getelementptr inbounds %"class.OSL::Dual2.0"* %b, i64 0, i32 1, i32 0
  %14 = bitcast float* %13 to i8*
  call void @llvm.memset.p0i8.i64(i8* %14, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_atan2_dvdvdv(i8* %r_, i8* %a_, i8* %1)
  call void @llvm.lifetime.end(i64 36, i8* %1) #2
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_sinh_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fcmp ogt float %1, 1.000000e+00
  br i1 %2, label %3, label %32

; <label>:3                                       ; preds = %0
  %4 = fmul float %1, 0x3FF7154760000000
  %5 = fcmp olt float %4, -1.260000e+02
  br i1 %5, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, label %6

; <label>:6                                       ; preds = %3
  %7 = fcmp ogt float %4, 1.260000e+02
  %8 = select i1 %7, float 1.260000e+02, float %4
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i: ; preds = %6, %3
  %9 = phi float [ %8, %6 ], [ -1.260000e+02, %3 ]
  %10 = fptosi float %9 to i32
  %11 = sitofp i32 %10 to float
  %12 = fsub float %9, %11
  %13 = fsub float 1.000000e+00, %12
  %14 = fsub float 1.000000e+00, %13
  %15 = fmul float %14, 0x3F55D889C0000000
  %16 = fadd float %15, 0x3F84177340000000
  %17 = fmul float %14, %16
  %18 = fadd float %17, 0x3FAC6CE660000000
  %19 = fmul float %14, %18
  %20 = fadd float %19, 0x3FCEBE3240000000
  %21 = fmul float %14, %20
  %22 = fadd float %21, 0x3FE62E3E20000000
  %23 = fmul float %14, %22
  %24 = fadd float %23, 1.000000e+00
  %25 = bitcast float %24 to i32
  %26 = shl i32 %10, 23
  %27 = add i32 %25, %26
  %28 = bitcast i32 %27 to float
  %29 = fmul float %28, 5.000000e-01
  %30 = fdiv float 5.000000e-01, %28
  %31 = fsub float %29, %30
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit

; <label>:32                                      ; preds = %0
  %33 = fsub float 1.000000e+00, %1
  %34 = fsub float 1.000000e+00, %33
  %35 = fmul float %34, %34
  %36 = fmul float %35, 0x3F2ABB46A0000000
  %37 = fadd float %36, 0x3F810F44A0000000
  %38 = fmul float %35, %37
  %39 = fadd float %38, 0x3FC5555B00000000
  %40 = fmul float %34, %39
  %41 = fmul float %35, %40
  %42 = fadd float %34, %41
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit

_ZN11OpenImageIO4v1_79fast_sinhEf.exit:           ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, %32
  %.sink.i = phi float [ %31, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i ], [ %42, %32 ]
  %43 = tail call float @copysignf(float %.sink.i, float %a) #12
  ret float %43
}

; Function Attrs: nounwind uwtable
define void @osl_sinh_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to %"class.OSL::Dual2"*
  %2 = tail call { <2 x float>, float } @_ZN3OSL9fast_sinhERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %1)
  %3 = extractvalue { <2 x float>, float } %2, 0
  %4 = extractvalue { <2 x float>, float } %2, 1
  %5 = bitcast i8* %r to <2 x float>*
  store <2 x float> %3, <2 x float>* %5, align 4
  %6 = getelementptr inbounds i8* %r, i64 8
  %7 = bitcast i8* %6 to float*
  store float %4, float* %7, align 4
  ret void
}

; Function Attrs: inlinehint nounwind readonly uwtable
define linkonce_odr { <2 x float>, float } @_ZN3OSL9fast_sinhERKNS_5Dual2IfEE(%"class.OSL::Dual2"* nocapture readonly dereferenceable(12) %a) #5 {
  %1 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 0
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fmul float %3, 0x3FF7154760000000
  %5 = fcmp olt float %4, -1.260000e+02
  br i1 %5, label %_ZN11OpenImageIO4v1_79fast_coshEf.exit, label %6

; <label>:6                                       ; preds = %0
  %7 = fcmp ogt float %4, 1.260000e+02
  %8 = select i1 %7, float 1.260000e+02, float %4
  br label %_ZN11OpenImageIO4v1_79fast_coshEf.exit

_ZN11OpenImageIO4v1_79fast_coshEf.exit:           ; preds = %0, %6
  %9 = phi float [ %8, %6 ], [ -1.260000e+02, %0 ]
  %10 = fptosi float %9 to i32
  %11 = sitofp i32 %10 to float
  %12 = fsub float %9, %11
  %13 = fsub float 1.000000e+00, %12
  %14 = fsub float 1.000000e+00, %13
  %15 = fmul float %14, 0x3F55D889C0000000
  %16 = fadd float %15, 0x3F84177340000000
  %17 = fmul float %14, %16
  %18 = fadd float %17, 0x3FAC6CE660000000
  %19 = fmul float %14, %18
  %20 = fadd float %19, 0x3FCEBE3240000000
  %21 = fmul float %14, %20
  %22 = fadd float %21, 0x3FE62E3E20000000
  %23 = fmul float %14, %22
  %24 = fadd float %23, 1.000000e+00
  %25 = bitcast float %24 to i32
  %26 = shl i32 %10, 23
  %27 = add i32 %25, %26
  %28 = bitcast i32 %27 to float
  %29 = fmul float %28, 5.000000e-01
  %30 = fdiv float 5.000000e-01, %28
  %31 = fadd float %30, %29
  %32 = fcmp ogt float %3, 1.000000e+00
  br i1 %32, label %33, label %60

; <label>:33                                      ; preds = %_ZN11OpenImageIO4v1_79fast_coshEf.exit
  br i1 %5, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, label %34

; <label>:34                                      ; preds = %33
  %35 = fcmp ogt float %4, 1.260000e+02
  %36 = select i1 %35, float 1.260000e+02, float %4
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i: ; preds = %34, %33
  %37 = phi float [ %36, %34 ], [ -1.260000e+02, %33 ]
  %38 = fptosi float %37 to i32
  %39 = sitofp i32 %38 to float
  %40 = fsub float %37, %39
  %41 = fsub float 1.000000e+00, %40
  %42 = fsub float 1.000000e+00, %41
  %43 = fmul float %42, 0x3F55D889C0000000
  %44 = fadd float %43, 0x3F84177340000000
  %45 = fmul float %42, %44
  %46 = fadd float %45, 0x3FAC6CE660000000
  %47 = fmul float %42, %46
  %48 = fadd float %47, 0x3FCEBE3240000000
  %49 = fmul float %42, %48
  %50 = fadd float %49, 0x3FE62E3E20000000
  %51 = fmul float %42, %50
  %52 = fadd float %51, 1.000000e+00
  %53 = bitcast float %52 to i32
  %54 = shl i32 %38, 23
  %55 = add i32 %53, %54
  %56 = bitcast i32 %55 to float
  %57 = fmul float %56, 5.000000e-01
  %58 = fdiv float 5.000000e-01, %56
  %59 = fsub float %57, %58
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit

; <label>:60                                      ; preds = %_ZN11OpenImageIO4v1_79fast_coshEf.exit
  %61 = fsub float 1.000000e+00, %3
  %62 = fsub float 1.000000e+00, %61
  %63 = fmul float %62, %62
  %64 = fmul float %63, 0x3F2ABB46A0000000
  %65 = fadd float %64, 0x3F810F44A0000000
  %66 = fmul float %63, %65
  %67 = fadd float %66, 0x3FC5555B00000000
  %68 = fmul float %62, %67
  %69 = fmul float %63, %68
  %70 = fadd float %62, %69
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit

_ZN11OpenImageIO4v1_79fast_sinhEf.exit:           ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, %60
  %.sink.i = phi float [ %59, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i ], [ %70, %60 ]
  %71 = tail call float @copysignf(float %.sink.i, float %2) #12
  %72 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 1
  %73 = load float* %72, align 4, !tbaa !1
  %74 = fmul float %31, %73
  %75 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 2
  %76 = load float* %75, align 4, !tbaa !1
  %77 = fmul float %31, %76
  %78 = insertelement <2 x float> undef, float %71, i32 0
  %79 = insertelement <2 x float> %78, float %74, i32 1
  %80 = insertvalue { <2 x float>, float } undef, <2 x float> %79, 0
  %81 = insertvalue { <2 x float>, float } %80, float %77, 1
  ret { <2 x float>, float } %81
}

; Function Attrs: nounwind uwtable
define void @osl_sinh_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp ogt float %3, 1.000000e+00
  br i1 %4, label %5, label %34

; <label>:5                                       ; preds = %0
  %6 = fmul float %3, 0x3FF7154760000000
  %7 = fcmp olt float %6, -1.260000e+02
  br i1 %7, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1, label %8

; <label>:8                                       ; preds = %5
  %9 = fcmp ogt float %6, 1.260000e+02
  %10 = select i1 %9, float 1.260000e+02, float %6
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1: ; preds = %8, %5
  %11 = phi float [ %10, %8 ], [ -1.260000e+02, %5 ]
  %12 = fptosi float %11 to i32
  %13 = sitofp i32 %12 to float
  %14 = fsub float %11, %13
  %15 = fsub float 1.000000e+00, %14
  %16 = fsub float 1.000000e+00, %15
  %17 = fmul float %16, 0x3F55D889C0000000
  %18 = fadd float %17, 0x3F84177340000000
  %19 = fmul float %16, %18
  %20 = fadd float %19, 0x3FAC6CE660000000
  %21 = fmul float %16, %20
  %22 = fadd float %21, 0x3FCEBE3240000000
  %23 = fmul float %16, %22
  %24 = fadd float %23, 0x3FE62E3E20000000
  %25 = fmul float %16, %24
  %26 = fadd float %25, 1.000000e+00
  %27 = bitcast float %26 to i32
  %28 = shl i32 %12, 23
  %29 = add i32 %27, %28
  %30 = bitcast i32 %29 to float
  %31 = fmul float %30, 5.000000e-01
  %32 = fdiv float 5.000000e-01, %30
  %33 = fsub float %31, %32
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit3

; <label>:34                                      ; preds = %0
  %35 = fsub float 1.000000e+00, %3
  %36 = fsub float 1.000000e+00, %35
  %37 = fmul float %36, %36
  %38 = fmul float %37, 0x3F2ABB46A0000000
  %39 = fadd float %38, 0x3F810F44A0000000
  %40 = fmul float %37, %39
  %41 = fadd float %40, 0x3FC5555B00000000
  %42 = fmul float %36, %41
  %43 = fmul float %37, %42
  %44 = fadd float %36, %43
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit3

_ZN11OpenImageIO4v1_79fast_sinhEf.exit3:          ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1, %34
  %.sink.i2 = phi float [ %33, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1 ], [ %44, %34 ]
  %45 = tail call float @copysignf(float %.sink.i2, float %2) #12
  %46 = bitcast i8* %r_ to float*
  store float %45, float* %46, align 4, !tbaa !1
  %47 = getelementptr inbounds i8* %a_, i64 4
  %48 = bitcast i8* %47 to float*
  %49 = load float* %48, align 4, !tbaa !1
  %50 = tail call float @fabsf(float %49) #12
  %51 = fcmp ogt float %50, 1.000000e+00
  br i1 %51, label %52, label %81

; <label>:52                                      ; preds = %_ZN11OpenImageIO4v1_79fast_sinhEf.exit3
  %53 = fmul float %50, 0x3FF7154760000000
  %54 = fcmp olt float %53, -1.260000e+02
  br i1 %54, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4, label %55

; <label>:55                                      ; preds = %52
  %56 = fcmp ogt float %53, 1.260000e+02
  %57 = select i1 %56, float 1.260000e+02, float %53
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4: ; preds = %55, %52
  %58 = phi float [ %57, %55 ], [ -1.260000e+02, %52 ]
  %59 = fptosi float %58 to i32
  %60 = sitofp i32 %59 to float
  %61 = fsub float %58, %60
  %62 = fsub float 1.000000e+00, %61
  %63 = fsub float 1.000000e+00, %62
  %64 = fmul float %63, 0x3F55D889C0000000
  %65 = fadd float %64, 0x3F84177340000000
  %66 = fmul float %63, %65
  %67 = fadd float %66, 0x3FAC6CE660000000
  %68 = fmul float %63, %67
  %69 = fadd float %68, 0x3FCEBE3240000000
  %70 = fmul float %63, %69
  %71 = fadd float %70, 0x3FE62E3E20000000
  %72 = fmul float %63, %71
  %73 = fadd float %72, 1.000000e+00
  %74 = bitcast float %73 to i32
  %75 = shl i32 %59, 23
  %76 = add i32 %74, %75
  %77 = bitcast i32 %76 to float
  %78 = fmul float %77, 5.000000e-01
  %79 = fdiv float 5.000000e-01, %77
  %80 = fsub float %78, %79
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit6

; <label>:81                                      ; preds = %_ZN11OpenImageIO4v1_79fast_sinhEf.exit3
  %82 = fsub float 1.000000e+00, %50
  %83 = fsub float 1.000000e+00, %82
  %84 = fmul float %83, %83
  %85 = fmul float %84, 0x3F2ABB46A0000000
  %86 = fadd float %85, 0x3F810F44A0000000
  %87 = fmul float %84, %86
  %88 = fadd float %87, 0x3FC5555B00000000
  %89 = fmul float %83, %88
  %90 = fmul float %84, %89
  %91 = fadd float %83, %90
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit6

_ZN11OpenImageIO4v1_79fast_sinhEf.exit6:          ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4, %81
  %.sink.i5 = phi float [ %80, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4 ], [ %91, %81 ]
  %92 = tail call float @copysignf(float %.sink.i5, float %49) #12
  %93 = getelementptr inbounds i8* %r_, i64 4
  %94 = bitcast i8* %93 to float*
  store float %92, float* %94, align 4, !tbaa !1
  %95 = getelementptr inbounds i8* %a_, i64 8
  %96 = bitcast i8* %95 to float*
  %97 = load float* %96, align 4, !tbaa !1
  %98 = tail call float @fabsf(float %97) #12
  %99 = fcmp ogt float %98, 1.000000e+00
  br i1 %99, label %100, label %129

; <label>:100                                     ; preds = %_ZN11OpenImageIO4v1_79fast_sinhEf.exit6
  %101 = fmul float %98, 0x3FF7154760000000
  %102 = fcmp olt float %101, -1.260000e+02
  br i1 %102, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, label %103

; <label>:103                                     ; preds = %100
  %104 = fcmp ogt float %101, 1.260000e+02
  %105 = select i1 %104, float 1.260000e+02, float %101
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i: ; preds = %103, %100
  %106 = phi float [ %105, %103 ], [ -1.260000e+02, %100 ]
  %107 = fptosi float %106 to i32
  %108 = sitofp i32 %107 to float
  %109 = fsub float %106, %108
  %110 = fsub float 1.000000e+00, %109
  %111 = fsub float 1.000000e+00, %110
  %112 = fmul float %111, 0x3F55D889C0000000
  %113 = fadd float %112, 0x3F84177340000000
  %114 = fmul float %111, %113
  %115 = fadd float %114, 0x3FAC6CE660000000
  %116 = fmul float %111, %115
  %117 = fadd float %116, 0x3FCEBE3240000000
  %118 = fmul float %111, %117
  %119 = fadd float %118, 0x3FE62E3E20000000
  %120 = fmul float %111, %119
  %121 = fadd float %120, 1.000000e+00
  %122 = bitcast float %121 to i32
  %123 = shl i32 %107, 23
  %124 = add i32 %122, %123
  %125 = bitcast i32 %124 to float
  %126 = fmul float %125, 5.000000e-01
  %127 = fdiv float 5.000000e-01, %125
  %128 = fsub float %126, %127
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit

; <label>:129                                     ; preds = %_ZN11OpenImageIO4v1_79fast_sinhEf.exit6
  %130 = fsub float 1.000000e+00, %98
  %131 = fsub float 1.000000e+00, %130
  %132 = fmul float %131, %131
  %133 = fmul float %132, 0x3F2ABB46A0000000
  %134 = fadd float %133, 0x3F810F44A0000000
  %135 = fmul float %132, %134
  %136 = fadd float %135, 0x3FC5555B00000000
  %137 = fmul float %131, %136
  %138 = fmul float %132, %137
  %139 = fadd float %131, %138
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit

_ZN11OpenImageIO4v1_79fast_sinhEf.exit:           ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, %129
  %.sink.i = phi float [ %128, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i ], [ %139, %129 ]
  %140 = tail call float @copysignf(float %.sink.i, float %97) #12
  %141 = getelementptr inbounds i8* %r_, i64 8
  %142 = bitcast i8* %141 to float*
  store float %140, float* %142, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sinh_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = alloca %"class.OSL::Dual2", align 4
  %2 = alloca %"class.OSL::Dual2", align 4
  %3 = alloca %"class.OSL::Dual2", align 4
  %4 = bitcast i8* %a_ to float*
  %5 = getelementptr inbounds i8* %a_, i64 12
  %6 = bitcast i8* %5 to float*
  %7 = getelementptr inbounds i8* %a_, i64 24
  %8 = bitcast i8* %7 to float*
  %9 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 0
  %10 = load float* %4, align 4, !tbaa !1
  store float %10, float* %9, align 4, !tbaa !9
  %11 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 1
  %12 = load float* %6, align 4, !tbaa !1
  store float %12, float* %11, align 4, !tbaa !11
  %13 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 2
  %14 = load float* %8, align 4, !tbaa !1
  store float %14, float* %13, align 4, !tbaa !12
  %15 = call { <2 x float>, float } @_ZN3OSL9fast_sinhERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %1)
  %16 = extractvalue { <2 x float>, float } %15, 0
  %17 = extractvalue { <2 x float>, float } %15, 1
  %18 = getelementptr inbounds i8* %a_, i64 4
  %19 = bitcast i8* %18 to float*
  %20 = getelementptr inbounds i8* %a_, i64 16
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8* %a_, i64 28
  %23 = bitcast i8* %22 to float*
  %24 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 0
  %25 = load float* %19, align 4, !tbaa !1
  store float %25, float* %24, align 4, !tbaa !9
  %26 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 1
  %27 = load float* %21, align 4, !tbaa !1
  store float %27, float* %26, align 4, !tbaa !11
  %28 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 2
  %29 = load float* %23, align 4, !tbaa !1
  store float %29, float* %28, align 4, !tbaa !12
  %30 = call { <2 x float>, float } @_ZN3OSL9fast_sinhERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %2)
  %31 = extractvalue { <2 x float>, float } %30, 0
  %32 = extractvalue { <2 x float>, float } %30, 1
  %33 = getelementptr inbounds i8* %a_, i64 8
  %34 = bitcast i8* %33 to float*
  %35 = getelementptr inbounds i8* %a_, i64 20
  %36 = bitcast i8* %35 to float*
  %37 = getelementptr inbounds i8* %a_, i64 32
  %38 = bitcast i8* %37 to float*
  %39 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 0
  %40 = load float* %34, align 4, !tbaa !1
  store float %40, float* %39, align 4, !tbaa !9
  %41 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 1
  %42 = load float* %36, align 4, !tbaa !1
  store float %42, float* %41, align 4, !tbaa !11
  %43 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 2
  %44 = load float* %38, align 4, !tbaa !1
  store float %44, float* %43, align 4, !tbaa !12
  %45 = call { <2 x float>, float } @_ZN3OSL9fast_sinhERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %3)
  %46 = extractvalue { <2 x float>, float } %45, 0
  %47 = extractvalue { <2 x float>, float } %45, 1
  %48 = extractelement <2 x float> %16, i32 0
  %49 = extractelement <2 x float> %31, i32 0
  %50 = extractelement <2 x float> %46, i32 0
  %51 = extractelement <2 x float> %16, i32 1
  %52 = extractelement <2 x float> %31, i32 1
  %53 = extractelement <2 x float> %46, i32 1
  %54 = bitcast i8* %r_ to float*
  store float %48, float* %54, align 4, !tbaa !5
  %55 = getelementptr inbounds i8* %r_, i64 4
  %56 = bitcast i8* %55 to float*
  store float %49, float* %56, align 4, !tbaa !7
  %57 = getelementptr inbounds i8* %r_, i64 8
  %58 = bitcast i8* %57 to float*
  store float %50, float* %58, align 4, !tbaa !8
  %59 = getelementptr inbounds i8* %r_, i64 12
  %60 = bitcast i8* %59 to float*
  store float %51, float* %60, align 4, !tbaa !5
  %61 = getelementptr inbounds i8* %r_, i64 16
  %62 = bitcast i8* %61 to float*
  store float %52, float* %62, align 4, !tbaa !7
  %63 = getelementptr inbounds i8* %r_, i64 20
  %64 = bitcast i8* %63 to float*
  store float %53, float* %64, align 4, !tbaa !8
  %65 = getelementptr inbounds i8* %r_, i64 24
  %66 = bitcast i8* %65 to float*
  store float %17, float* %66, align 4, !tbaa !5
  %67 = getelementptr inbounds i8* %r_, i64 28
  %68 = bitcast i8* %67 to float*
  store float %32, float* %68, align 4, !tbaa !7
  %69 = getelementptr inbounds i8* %r_, i64 32
  %70 = bitcast i8* %69 to float*
  store float %47, float* %70, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_cosh_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fmul float %1, 0x3FF7154760000000
  %3 = fcmp olt float %2, -1.260000e+02
  br i1 %3, label %_ZN11OpenImageIO4v1_79fast_coshEf.exit, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 1.260000e+02
  %6 = select i1 %5, float 1.260000e+02, float %2
  br label %_ZN11OpenImageIO4v1_79fast_coshEf.exit

_ZN11OpenImageIO4v1_79fast_coshEf.exit:           ; preds = %0, %4
  %7 = phi float [ %6, %4 ], [ -1.260000e+02, %0 ]
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fsub float %7, %9
  %11 = fsub float 1.000000e+00, %10
  %12 = fsub float 1.000000e+00, %11
  %13 = fmul float %12, 0x3F55D889C0000000
  %14 = fadd float %13, 0x3F84177340000000
  %15 = fmul float %12, %14
  %16 = fadd float %15, 0x3FAC6CE660000000
  %17 = fmul float %12, %16
  %18 = fadd float %17, 0x3FCEBE3240000000
  %19 = fmul float %12, %18
  %20 = fadd float %19, 0x3FE62E3E20000000
  %21 = fmul float %12, %20
  %22 = fadd float %21, 1.000000e+00
  %23 = bitcast float %22 to i32
  %24 = shl i32 %8, 23
  %25 = add i32 %23, %24
  %26 = bitcast i32 %25 to float
  %27 = fmul float %26, 5.000000e-01
  %28 = fdiv float 5.000000e-01, %26
  %29 = fadd float %28, %27
  ret float %29
}

; Function Attrs: nounwind uwtable
define void @osl_cosh_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to %"class.OSL::Dual2"*
  %2 = tail call { <2 x float>, float } @_ZN3OSL9fast_coshERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %1)
  %3 = extractvalue { <2 x float>, float } %2, 0
  %4 = extractvalue { <2 x float>, float } %2, 1
  %5 = bitcast i8* %r to <2 x float>*
  store <2 x float> %3, <2 x float>* %5, align 4
  %6 = getelementptr inbounds i8* %r, i64 8
  %7 = bitcast i8* %6 to float*
  store float %4, float* %7, align 4
  ret void
}

; Function Attrs: inlinehint nounwind readonly uwtable
define linkonce_odr { <2 x float>, float } @_ZN3OSL9fast_coshERKNS_5Dual2IfEE(%"class.OSL::Dual2"* nocapture readonly dereferenceable(12) %a) #5 {
  %1 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 0
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fmul float %3, 0x3FF7154760000000
  %5 = fcmp olt float %4, -1.260000e+02
  br i1 %5, label %_ZN11OpenImageIO4v1_79fast_coshEf.exit, label %6

; <label>:6                                       ; preds = %0
  %7 = fcmp ogt float %4, 1.260000e+02
  %8 = select i1 %7, float 1.260000e+02, float %4
  br label %_ZN11OpenImageIO4v1_79fast_coshEf.exit

_ZN11OpenImageIO4v1_79fast_coshEf.exit:           ; preds = %0, %6
  %9 = phi float [ %8, %6 ], [ -1.260000e+02, %0 ]
  %10 = fptosi float %9 to i32
  %11 = sitofp i32 %10 to float
  %12 = fsub float %9, %11
  %13 = fsub float 1.000000e+00, %12
  %14 = fsub float 1.000000e+00, %13
  %15 = fmul float %14, 0x3F55D889C0000000
  %16 = fadd float %15, 0x3F84177340000000
  %17 = fmul float %14, %16
  %18 = fadd float %17, 0x3FAC6CE660000000
  %19 = fmul float %14, %18
  %20 = fadd float %19, 0x3FCEBE3240000000
  %21 = fmul float %14, %20
  %22 = fadd float %21, 0x3FE62E3E20000000
  %23 = fmul float %14, %22
  %24 = fadd float %23, 1.000000e+00
  %25 = bitcast float %24 to i32
  %26 = shl i32 %10, 23
  %27 = add i32 %25, %26
  %28 = bitcast i32 %27 to float
  %29 = fmul float %28, 5.000000e-01
  %30 = fdiv float 5.000000e-01, %28
  %31 = fadd float %30, %29
  %32 = fcmp ogt float %3, 1.000000e+00
  br i1 %32, label %33, label %60

; <label>:33                                      ; preds = %_ZN11OpenImageIO4v1_79fast_coshEf.exit
  br i1 %5, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, label %34

; <label>:34                                      ; preds = %33
  %35 = fcmp ogt float %4, 1.260000e+02
  %36 = select i1 %35, float 1.260000e+02, float %4
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i: ; preds = %34, %33
  %37 = phi float [ %36, %34 ], [ -1.260000e+02, %33 ]
  %38 = fptosi float %37 to i32
  %39 = sitofp i32 %38 to float
  %40 = fsub float %37, %39
  %41 = fsub float 1.000000e+00, %40
  %42 = fsub float 1.000000e+00, %41
  %43 = fmul float %42, 0x3F55D889C0000000
  %44 = fadd float %43, 0x3F84177340000000
  %45 = fmul float %42, %44
  %46 = fadd float %45, 0x3FAC6CE660000000
  %47 = fmul float %42, %46
  %48 = fadd float %47, 0x3FCEBE3240000000
  %49 = fmul float %42, %48
  %50 = fadd float %49, 0x3FE62E3E20000000
  %51 = fmul float %42, %50
  %52 = fadd float %51, 1.000000e+00
  %53 = bitcast float %52 to i32
  %54 = shl i32 %38, 23
  %55 = add i32 %53, %54
  %56 = bitcast i32 %55 to float
  %57 = fmul float %56, 5.000000e-01
  %58 = fdiv float 5.000000e-01, %56
  %59 = fsub float %57, %58
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit

; <label>:60                                      ; preds = %_ZN11OpenImageIO4v1_79fast_coshEf.exit
  %61 = fsub float 1.000000e+00, %3
  %62 = fsub float 1.000000e+00, %61
  %63 = fmul float %62, %62
  %64 = fmul float %63, 0x3F2ABB46A0000000
  %65 = fadd float %64, 0x3F810F44A0000000
  %66 = fmul float %63, %65
  %67 = fadd float %66, 0x3FC5555B00000000
  %68 = fmul float %62, %67
  %69 = fmul float %63, %68
  %70 = fadd float %62, %69
  br label %_ZN11OpenImageIO4v1_79fast_sinhEf.exit

_ZN11OpenImageIO4v1_79fast_sinhEf.exit:           ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, %60
  %.sink.i = phi float [ %59, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i ], [ %70, %60 ]
  %71 = tail call float @copysignf(float %.sink.i, float %2) #12
  %72 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 1
  %73 = load float* %72, align 4, !tbaa !1
  %74 = fmul float %71, %73
  %75 = getelementptr inbounds %"class.OSL::Dual2"* %a, i64 0, i32 2
  %76 = load float* %75, align 4, !tbaa !1
  %77 = fmul float %71, %76
  %78 = insertelement <2 x float> undef, float %31, i32 0
  %79 = insertelement <2 x float> %78, float %74, i32 1
  %80 = insertvalue { <2 x float>, float } undef, <2 x float> %79, 0
  %81 = insertvalue { <2 x float>, float } %80, float %77, 1
  ret { <2 x float>, float } %81
}

; Function Attrs: nounwind uwtable
define void @osl_cosh_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fmul float %3, 0x3FF7154760000000
  %5 = fcmp olt float %4, -1.260000e+02
  br i1 %5, label %_ZN11OpenImageIO4v1_79fast_coshEf.exit1, label %6

; <label>:6                                       ; preds = %0
  %7 = fcmp ogt float %4, 1.260000e+02
  %8 = select i1 %7, float 1.260000e+02, float %4
  br label %_ZN11OpenImageIO4v1_79fast_coshEf.exit1

_ZN11OpenImageIO4v1_79fast_coshEf.exit1:          ; preds = %0, %6
  %9 = phi float [ %8, %6 ], [ -1.260000e+02, %0 ]
  %10 = fptosi float %9 to i32
  %11 = sitofp i32 %10 to float
  %12 = fsub float %9, %11
  %13 = fsub float 1.000000e+00, %12
  %14 = fsub float 1.000000e+00, %13
  %15 = fmul float %14, 0x3F55D889C0000000
  %16 = fadd float %15, 0x3F84177340000000
  %17 = fmul float %14, %16
  %18 = fadd float %17, 0x3FAC6CE660000000
  %19 = fmul float %14, %18
  %20 = fadd float %19, 0x3FCEBE3240000000
  %21 = fmul float %14, %20
  %22 = fadd float %21, 0x3FE62E3E20000000
  %23 = fmul float %14, %22
  %24 = fadd float %23, 1.000000e+00
  %25 = bitcast float %24 to i32
  %26 = shl i32 %10, 23
  %27 = add i32 %25, %26
  %28 = bitcast i32 %27 to float
  %29 = fmul float %28, 5.000000e-01
  %30 = fdiv float 5.000000e-01, %28
  %31 = fadd float %30, %29
  %32 = bitcast i8* %r_ to float*
  store float %31, float* %32, align 4, !tbaa !1
  %33 = getelementptr inbounds i8* %a_, i64 4
  %34 = bitcast i8* %33 to float*
  %35 = load float* %34, align 4, !tbaa !1
  %36 = tail call float @fabsf(float %35) #12
  %37 = fmul float %36, 0x3FF7154760000000
  %38 = fcmp olt float %37, -1.260000e+02
  br i1 %38, label %_ZN11OpenImageIO4v1_79fast_coshEf.exit2, label %39

; <label>:39                                      ; preds = %_ZN11OpenImageIO4v1_79fast_coshEf.exit1
  %40 = fcmp ogt float %37, 1.260000e+02
  %41 = select i1 %40, float 1.260000e+02, float %37
  br label %_ZN11OpenImageIO4v1_79fast_coshEf.exit2

_ZN11OpenImageIO4v1_79fast_coshEf.exit2:          ; preds = %_ZN11OpenImageIO4v1_79fast_coshEf.exit1, %39
  %42 = phi float [ %41, %39 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_coshEf.exit1 ]
  %43 = fptosi float %42 to i32
  %44 = sitofp i32 %43 to float
  %45 = fsub float %42, %44
  %46 = fsub float 1.000000e+00, %45
  %47 = fsub float 1.000000e+00, %46
  %48 = fmul float %47, 0x3F55D889C0000000
  %49 = fadd float %48, 0x3F84177340000000
  %50 = fmul float %47, %49
  %51 = fadd float %50, 0x3FAC6CE660000000
  %52 = fmul float %47, %51
  %53 = fadd float %52, 0x3FCEBE3240000000
  %54 = fmul float %47, %53
  %55 = fadd float %54, 0x3FE62E3E20000000
  %56 = fmul float %47, %55
  %57 = fadd float %56, 1.000000e+00
  %58 = bitcast float %57 to i32
  %59 = shl i32 %43, 23
  %60 = add i32 %58, %59
  %61 = bitcast i32 %60 to float
  %62 = fmul float %61, 5.000000e-01
  %63 = fdiv float 5.000000e-01, %61
  %64 = fadd float %63, %62
  %65 = getelementptr inbounds i8* %r_, i64 4
  %66 = bitcast i8* %65 to float*
  store float %64, float* %66, align 4, !tbaa !1
  %67 = getelementptr inbounds i8* %a_, i64 8
  %68 = bitcast i8* %67 to float*
  %69 = load float* %68, align 4, !tbaa !1
  %70 = tail call float @fabsf(float %69) #12
  %71 = fmul float %70, 0x3FF7154760000000
  %72 = fcmp olt float %71, -1.260000e+02
  br i1 %72, label %_ZN11OpenImageIO4v1_79fast_coshEf.exit, label %73

; <label>:73                                      ; preds = %_ZN11OpenImageIO4v1_79fast_coshEf.exit2
  %74 = fcmp ogt float %71, 1.260000e+02
  %75 = select i1 %74, float 1.260000e+02, float %71
  br label %_ZN11OpenImageIO4v1_79fast_coshEf.exit

_ZN11OpenImageIO4v1_79fast_coshEf.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_coshEf.exit2, %73
  %76 = phi float [ %75, %73 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_coshEf.exit2 ]
  %77 = fptosi float %76 to i32
  %78 = sitofp i32 %77 to float
  %79 = fsub float %76, %78
  %80 = fsub float 1.000000e+00, %79
  %81 = fsub float 1.000000e+00, %80
  %82 = fmul float %81, 0x3F55D889C0000000
  %83 = fadd float %82, 0x3F84177340000000
  %84 = fmul float %81, %83
  %85 = fadd float %84, 0x3FAC6CE660000000
  %86 = fmul float %81, %85
  %87 = fadd float %86, 0x3FCEBE3240000000
  %88 = fmul float %81, %87
  %89 = fadd float %88, 0x3FE62E3E20000000
  %90 = fmul float %81, %89
  %91 = fadd float %90, 1.000000e+00
  %92 = bitcast float %91 to i32
  %93 = shl i32 %77, 23
  %94 = add i32 %92, %93
  %95 = bitcast i32 %94 to float
  %96 = fmul float %95, 5.000000e-01
  %97 = fdiv float 5.000000e-01, %95
  %98 = fadd float %97, %96
  %99 = getelementptr inbounds i8* %r_, i64 8
  %100 = bitcast i8* %99 to float*
  store float %98, float* %100, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cosh_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = alloca %"class.OSL::Dual2", align 4
  %2 = alloca %"class.OSL::Dual2", align 4
  %3 = alloca %"class.OSL::Dual2", align 4
  %4 = bitcast i8* %a_ to float*
  %5 = getelementptr inbounds i8* %a_, i64 12
  %6 = bitcast i8* %5 to float*
  %7 = getelementptr inbounds i8* %a_, i64 24
  %8 = bitcast i8* %7 to float*
  %9 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 0
  %10 = load float* %4, align 4, !tbaa !1
  store float %10, float* %9, align 4, !tbaa !9
  %11 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 1
  %12 = load float* %6, align 4, !tbaa !1
  store float %12, float* %11, align 4, !tbaa !11
  %13 = getelementptr inbounds %"class.OSL::Dual2"* %1, i64 0, i32 2
  %14 = load float* %8, align 4, !tbaa !1
  store float %14, float* %13, align 4, !tbaa !12
  %15 = call { <2 x float>, float } @_ZN3OSL9fast_coshERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %1)
  %16 = extractvalue { <2 x float>, float } %15, 0
  %17 = extractvalue { <2 x float>, float } %15, 1
  %18 = getelementptr inbounds i8* %a_, i64 4
  %19 = bitcast i8* %18 to float*
  %20 = getelementptr inbounds i8* %a_, i64 16
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8* %a_, i64 28
  %23 = bitcast i8* %22 to float*
  %24 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 0
  %25 = load float* %19, align 4, !tbaa !1
  store float %25, float* %24, align 4, !tbaa !9
  %26 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 1
  %27 = load float* %21, align 4, !tbaa !1
  store float %27, float* %26, align 4, !tbaa !11
  %28 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 0, i32 2
  %29 = load float* %23, align 4, !tbaa !1
  store float %29, float* %28, align 4, !tbaa !12
  %30 = call { <2 x float>, float } @_ZN3OSL9fast_coshERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %2)
  %31 = extractvalue { <2 x float>, float } %30, 0
  %32 = extractvalue { <2 x float>, float } %30, 1
  %33 = getelementptr inbounds i8* %a_, i64 8
  %34 = bitcast i8* %33 to float*
  %35 = getelementptr inbounds i8* %a_, i64 20
  %36 = bitcast i8* %35 to float*
  %37 = getelementptr inbounds i8* %a_, i64 32
  %38 = bitcast i8* %37 to float*
  %39 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 0
  %40 = load float* %34, align 4, !tbaa !1
  store float %40, float* %39, align 4, !tbaa !9
  %41 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 1
  %42 = load float* %36, align 4, !tbaa !1
  store float %42, float* %41, align 4, !tbaa !11
  %43 = getelementptr inbounds %"class.OSL::Dual2"* %3, i64 0, i32 2
  %44 = load float* %38, align 4, !tbaa !1
  store float %44, float* %43, align 4, !tbaa !12
  %45 = call { <2 x float>, float } @_ZN3OSL9fast_coshERKNS_5Dual2IfEE(%"class.OSL::Dual2"* dereferenceable(12) %3)
  %46 = extractvalue { <2 x float>, float } %45, 0
  %47 = extractvalue { <2 x float>, float } %45, 1
  %48 = extractelement <2 x float> %16, i32 0
  %49 = extractelement <2 x float> %31, i32 0
  %50 = extractelement <2 x float> %46, i32 0
  %51 = extractelement <2 x float> %16, i32 1
  %52 = extractelement <2 x float> %31, i32 1
  %53 = extractelement <2 x float> %46, i32 1
  %54 = bitcast i8* %r_ to float*
  store float %48, float* %54, align 4, !tbaa !5
  %55 = getelementptr inbounds i8* %r_, i64 4
  %56 = bitcast i8* %55 to float*
  store float %49, float* %56, align 4, !tbaa !7
  %57 = getelementptr inbounds i8* %r_, i64 8
  %58 = bitcast i8* %57 to float*
  store float %50, float* %58, align 4, !tbaa !8
  %59 = getelementptr inbounds i8* %r_, i64 12
  %60 = bitcast i8* %59 to float*
  store float %51, float* %60, align 4, !tbaa !5
  %61 = getelementptr inbounds i8* %r_, i64 16
  %62 = bitcast i8* %61 to float*
  store float %52, float* %62, align 4, !tbaa !7
  %63 = getelementptr inbounds i8* %r_, i64 20
  %64 = bitcast i8* %63 to float*
  store float %53, float* %64, align 4, !tbaa !8
  %65 = getelementptr inbounds i8* %r_, i64 24
  %66 = bitcast i8* %65 to float*
  store float %17, float* %66, align 4, !tbaa !5
  %67 = getelementptr inbounds i8* %r_, i64 28
  %68 = bitcast i8* %67 to float*
  store float %32, float* %68, align 4, !tbaa !7
  %69 = getelementptr inbounds i8* %r_, i64 32
  %70 = bitcast i8* %69 to float*
  store float %47, float* %70, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_tanh_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fmul float %1, 2.000000e+00
  %3 = fmul float %2, 0x3FF7154760000000
  %4 = fcmp olt float %3, -1.260000e+02
  br i1 %4, label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit, label %5

; <label>:5                                       ; preds = %0
  %6 = fcmp ogt float %3, 1.260000e+02
  %7 = select i1 %6, float 1.260000e+02, float %3
  br label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit

_ZN11OpenImageIO4v1_79fast_tanhEf.exit:           ; preds = %0, %5
  %8 = phi float [ %7, %5 ], [ -1.260000e+02, %0 ]
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fsub float %8, %10
  %12 = fsub float 1.000000e+00, %11
  %13 = fsub float 1.000000e+00, %12
  %14 = fmul float %13, 0x3F55D889C0000000
  %15 = fadd float %14, 0x3F84177340000000
  %16 = fmul float %13, %15
  %17 = fadd float %16, 0x3FAC6CE660000000
  %18 = fmul float %13, %17
  %19 = fadd float %18, 0x3FCEBE3240000000
  %20 = fmul float %13, %19
  %21 = fadd float %20, 0x3FE62E3E20000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 1.000000e+00
  %24 = bitcast float %23 to i32
  %25 = shl i32 %9, 23
  %26 = add i32 %24, %25
  %27 = bitcast i32 %26 to float
  %28 = fadd float %27, 1.000000e+00
  %29 = fdiv float 2.000000e+00, %28
  %30 = fsub float 1.000000e+00, %29
  %31 = tail call float @copysignf(float %30, float %a) #12
  ret float %31
}

; Function Attrs: nounwind uwtable
define void @osl_tanh_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fmul float %3, 2.000000e+00
  %5 = fmul float %4, 0x3FF7154760000000
  %6 = fcmp olt float %5, -1.260000e+02
  br i1 %6, label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i, label %7

; <label>:7                                       ; preds = %0
  %8 = fcmp ogt float %5, 1.260000e+02
  %9 = select i1 %8, float 1.260000e+02, float %5
  br label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i

_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i:         ; preds = %7, %0
  %10 = phi float [ %9, %7 ], [ -1.260000e+02, %0 ]
  %11 = fptosi float %10 to i32
  %12 = sitofp i32 %11 to float
  %13 = fsub float %10, %12
  %14 = fsub float 1.000000e+00, %13
  %15 = fsub float 1.000000e+00, %14
  %16 = fmul float %15, 0x3F55D889C0000000
  %17 = fadd float %16, 0x3F84177340000000
  %18 = fmul float %15, %17
  %19 = fadd float %18, 0x3FAC6CE660000000
  %20 = fmul float %15, %19
  %21 = fadd float %20, 0x3FCEBE3240000000
  %22 = fmul float %15, %21
  %23 = fadd float %22, 0x3FE62E3E20000000
  %24 = fmul float %15, %23
  %25 = fadd float %24, 1.000000e+00
  %26 = bitcast float %25 to i32
  %27 = shl i32 %11, 23
  %28 = add i32 %26, %27
  %29 = bitcast i32 %28 to float
  %30 = fadd float %29, 1.000000e+00
  %31 = fdiv float 2.000000e+00, %30
  %32 = fsub float 1.000000e+00, %31
  %33 = tail call float @copysignf(float %32, float %2) #12
  %34 = fmul float %3, 0x3FF7154760000000
  %35 = fcmp olt float %34, -1.260000e+02
  br i1 %35, label %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit, label %36

; <label>:36                                      ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i
  %37 = fcmp ogt float %34, 1.260000e+02
  %38 = select i1 %37, float 1.260000e+02, float %34
  br label %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i, %36
  %39 = phi float [ %38, %36 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i ]
  %40 = fptosi float %39 to i32
  %41 = sitofp i32 %40 to float
  %42 = fsub float %39, %41
  %43 = fsub float 1.000000e+00, %42
  %44 = fsub float 1.000000e+00, %43
  %45 = fmul float %44, 0x3F55D889C0000000
  %46 = fadd float %45, 0x3F84177340000000
  %47 = fmul float %44, %46
  %48 = fadd float %47, 0x3FAC6CE660000000
  %49 = fmul float %44, %48
  %50 = fadd float %49, 0x3FCEBE3240000000
  %51 = fmul float %44, %50
  %52 = fadd float %51, 0x3FE62E3E20000000
  %53 = fmul float %44, %52
  %54 = fadd float %53, 1.000000e+00
  %55 = bitcast float %54 to i32
  %56 = shl i32 %40, 23
  %57 = add i32 %55, %56
  %58 = bitcast i32 %57 to float
  %59 = fmul float %58, 5.000000e-01
  %60 = fdiv float 5.000000e-01, %58
  %61 = fadd float %60, %59
  %62 = fmul float %61, %61
  %63 = fdiv float 1.000000e+00, %62
  %64 = getelementptr inbounds i8* %a, i64 4
  %65 = bitcast i8* %64 to float*
  %66 = load float* %65, align 4, !tbaa !1
  %67 = fmul float %63, %66
  %68 = getelementptr inbounds i8* %a, i64 8
  %69 = bitcast i8* %68 to float*
  %70 = load float* %69, align 4, !tbaa !1
  %71 = fmul float %63, %70
  %72 = insertelement <2 x float> undef, float %33, i32 0
  %73 = insertelement <2 x float> %72, float %67, i32 1
  %74 = bitcast i8* %r to <2 x float>*
  store <2 x float> %73, <2 x float>* %74, align 4
  %75 = getelementptr inbounds i8* %r, i64 8
  %76 = bitcast i8* %75 to float*
  store float %71, float* %76, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_tanh_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fmul float %3, 2.000000e+00
  %5 = fmul float %4, 0x3FF7154760000000
  %6 = fcmp olt float %5, -1.260000e+02
  br i1 %6, label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit1, label %7

; <label>:7                                       ; preds = %0
  %8 = fcmp ogt float %5, 1.260000e+02
  %9 = select i1 %8, float 1.260000e+02, float %5
  br label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit1

_ZN11OpenImageIO4v1_79fast_tanhEf.exit1:          ; preds = %0, %7
  %10 = phi float [ %9, %7 ], [ -1.260000e+02, %0 ]
  %11 = fptosi float %10 to i32
  %12 = sitofp i32 %11 to float
  %13 = fsub float %10, %12
  %14 = fsub float 1.000000e+00, %13
  %15 = fsub float 1.000000e+00, %14
  %16 = fmul float %15, 0x3F55D889C0000000
  %17 = fadd float %16, 0x3F84177340000000
  %18 = fmul float %15, %17
  %19 = fadd float %18, 0x3FAC6CE660000000
  %20 = fmul float %15, %19
  %21 = fadd float %20, 0x3FCEBE3240000000
  %22 = fmul float %15, %21
  %23 = fadd float %22, 0x3FE62E3E20000000
  %24 = fmul float %15, %23
  %25 = fadd float %24, 1.000000e+00
  %26 = bitcast float %25 to i32
  %27 = shl i32 %11, 23
  %28 = add i32 %26, %27
  %29 = bitcast i32 %28 to float
  %30 = fadd float %29, 1.000000e+00
  %31 = fdiv float 2.000000e+00, %30
  %32 = fsub float 1.000000e+00, %31
  %33 = tail call float @copysignf(float %32, float %2) #12
  %34 = bitcast i8* %r_ to float*
  store float %33, float* %34, align 4, !tbaa !1
  %35 = getelementptr inbounds i8* %a_, i64 4
  %36 = bitcast i8* %35 to float*
  %37 = load float* %36, align 4, !tbaa !1
  %38 = tail call float @fabsf(float %37) #12
  %39 = fmul float %38, 2.000000e+00
  %40 = fmul float %39, 0x3FF7154760000000
  %41 = fcmp olt float %40, -1.260000e+02
  br i1 %41, label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit2, label %42

; <label>:42                                      ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit1
  %43 = fcmp ogt float %40, 1.260000e+02
  %44 = select i1 %43, float 1.260000e+02, float %40
  br label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit2

_ZN11OpenImageIO4v1_79fast_tanhEf.exit2:          ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit1, %42
  %45 = phi float [ %44, %42 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_tanhEf.exit1 ]
  %46 = fptosi float %45 to i32
  %47 = sitofp i32 %46 to float
  %48 = fsub float %45, %47
  %49 = fsub float 1.000000e+00, %48
  %50 = fsub float 1.000000e+00, %49
  %51 = fmul float %50, 0x3F55D889C0000000
  %52 = fadd float %51, 0x3F84177340000000
  %53 = fmul float %50, %52
  %54 = fadd float %53, 0x3FAC6CE660000000
  %55 = fmul float %50, %54
  %56 = fadd float %55, 0x3FCEBE3240000000
  %57 = fmul float %50, %56
  %58 = fadd float %57, 0x3FE62E3E20000000
  %59 = fmul float %50, %58
  %60 = fadd float %59, 1.000000e+00
  %61 = bitcast float %60 to i32
  %62 = shl i32 %46, 23
  %63 = add i32 %61, %62
  %64 = bitcast i32 %63 to float
  %65 = fadd float %64, 1.000000e+00
  %66 = fdiv float 2.000000e+00, %65
  %67 = fsub float 1.000000e+00, %66
  %68 = tail call float @copysignf(float %67, float %37) #12
  %69 = getelementptr inbounds i8* %r_, i64 4
  %70 = bitcast i8* %69 to float*
  store float %68, float* %70, align 4, !tbaa !1
  %71 = getelementptr inbounds i8* %a_, i64 8
  %72 = bitcast i8* %71 to float*
  %73 = load float* %72, align 4, !tbaa !1
  %74 = tail call float @fabsf(float %73) #12
  %75 = fmul float %74, 2.000000e+00
  %76 = fmul float %75, 0x3FF7154760000000
  %77 = fcmp olt float %76, -1.260000e+02
  br i1 %77, label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit, label %78

; <label>:78                                      ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit2
  %79 = fcmp ogt float %76, 1.260000e+02
  %80 = select i1 %79, float 1.260000e+02, float %76
  br label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit

_ZN11OpenImageIO4v1_79fast_tanhEf.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit2, %78
  %81 = phi float [ %80, %78 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_tanhEf.exit2 ]
  %82 = fptosi float %81 to i32
  %83 = sitofp i32 %82 to float
  %84 = fsub float %81, %83
  %85 = fsub float 1.000000e+00, %84
  %86 = fsub float 1.000000e+00, %85
  %87 = fmul float %86, 0x3F55D889C0000000
  %88 = fadd float %87, 0x3F84177340000000
  %89 = fmul float %86, %88
  %90 = fadd float %89, 0x3FAC6CE660000000
  %91 = fmul float %86, %90
  %92 = fadd float %91, 0x3FCEBE3240000000
  %93 = fmul float %86, %92
  %94 = fadd float %93, 0x3FE62E3E20000000
  %95 = fmul float %86, %94
  %96 = fadd float %95, 1.000000e+00
  %97 = bitcast float %96 to i32
  %98 = shl i32 %82, 23
  %99 = add i32 %97, %98
  %100 = bitcast i32 %99 to float
  %101 = fadd float %100, 1.000000e+00
  %102 = fdiv float 2.000000e+00, %101
  %103 = fsub float 1.000000e+00, %102
  %104 = tail call float @copysignf(float %103, float %73) #12
  %105 = getelementptr inbounds i8* %r_, i64 8
  %106 = bitcast i8* %105 to float*
  store float %104, float* %106, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_tanh_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = tail call float @fabsf(float %6) #12
  %10 = fmul float %9, 2.000000e+00
  %11 = fmul float %10, 0x3FF7154760000000
  %12 = fcmp olt float %11, -1.260000e+02
  br i1 %12, label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i13, label %13

; <label>:13                                      ; preds = %0
  %14 = fcmp ogt float %11, 1.260000e+02
  %15 = select i1 %14, float 1.260000e+02, float %11
  br label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i13

_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i13:       ; preds = %13, %0
  %16 = phi float [ %15, %13 ], [ -1.260000e+02, %0 ]
  %17 = fptosi float %16 to i32
  %18 = sitofp i32 %17 to float
  %19 = fsub float %16, %18
  %20 = fsub float 1.000000e+00, %19
  %21 = fsub float 1.000000e+00, %20
  %22 = fmul float %21, 0x3F55D889C0000000
  %23 = fadd float %22, 0x3F84177340000000
  %24 = fmul float %21, %23
  %25 = fadd float %24, 0x3FAC6CE660000000
  %26 = fmul float %21, %25
  %27 = fadd float %26, 0x3FCEBE3240000000
  %28 = fmul float %21, %27
  %29 = fadd float %28, 0x3FE62E3E20000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 1.000000e+00
  %32 = bitcast float %31 to i32
  %33 = shl i32 %17, 23
  %34 = add i32 %32, %33
  %35 = bitcast i32 %34 to float
  %36 = fadd float %35, 1.000000e+00
  %37 = fdiv float 2.000000e+00, %36
  %38 = fsub float 1.000000e+00, %37
  %39 = tail call float @copysignf(float %38, float %6) #12
  %40 = fmul float %9, 0x3FF7154760000000
  %41 = fcmp olt float %40, -1.260000e+02
  br i1 %41, label %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit14, label %42

; <label>:42                                      ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i13
  %43 = fcmp ogt float %40, 1.260000e+02
  %44 = select i1 %43, float 1.260000e+02, float %40
  br label %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit14

_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit14:         ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i13, %42
  %45 = phi float [ %44, %42 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i13 ]
  %46 = fptosi float %45 to i32
  %47 = sitofp i32 %46 to float
  %48 = fsub float %45, %47
  %49 = fsub float 1.000000e+00, %48
  %50 = fsub float 1.000000e+00, %49
  %51 = fmul float %50, 0x3F55D889C0000000
  %52 = fadd float %51, 0x3F84177340000000
  %53 = fmul float %50, %52
  %54 = fadd float %53, 0x3FAC6CE660000000
  %55 = fmul float %50, %54
  %56 = fadd float %55, 0x3FCEBE3240000000
  %57 = fmul float %50, %56
  %58 = fadd float %57, 0x3FE62E3E20000000
  %59 = fmul float %50, %58
  %60 = fadd float %59, 1.000000e+00
  %61 = bitcast float %60 to i32
  %62 = shl i32 %46, 23
  %63 = add i32 %61, %62
  %64 = bitcast i32 %63 to float
  %65 = fmul float %64, 5.000000e-01
  %66 = fdiv float 5.000000e-01, %64
  %67 = fadd float %66, %65
  %68 = fmul float %67, %67
  %69 = fdiv float 1.000000e+00, %68
  %70 = fmul float %7, %69
  %71 = fmul float %8, %69
  %72 = getelementptr inbounds i8* %a_, i64 4
  %73 = bitcast i8* %72 to float*
  %74 = getelementptr inbounds i8* %a_, i64 16
  %75 = bitcast i8* %74 to float*
  %76 = getelementptr inbounds i8* %a_, i64 28
  %77 = bitcast i8* %76 to float*
  %78 = load float* %73, align 4, !tbaa !1
  %79 = load float* %75, align 4, !tbaa !1
  %80 = load float* %77, align 4, !tbaa !1
  %81 = tail call float @fabsf(float %78) #12
  %82 = fmul float %81, 2.000000e+00
  %83 = fmul float %82, 0x3FF7154760000000
  %84 = fcmp olt float %83, -1.260000e+02
  br i1 %84, label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i11, label %85

; <label>:85                                      ; preds = %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit14
  %86 = fcmp ogt float %83, 1.260000e+02
  %87 = select i1 %86, float 1.260000e+02, float %83
  br label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i11

_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i11:       ; preds = %85, %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit14
  %88 = phi float [ %87, %85 ], [ -1.260000e+02, %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit14 ]
  %89 = fptosi float %88 to i32
  %90 = sitofp i32 %89 to float
  %91 = fsub float %88, %90
  %92 = fsub float 1.000000e+00, %91
  %93 = fsub float 1.000000e+00, %92
  %94 = fmul float %93, 0x3F55D889C0000000
  %95 = fadd float %94, 0x3F84177340000000
  %96 = fmul float %93, %95
  %97 = fadd float %96, 0x3FAC6CE660000000
  %98 = fmul float %93, %97
  %99 = fadd float %98, 0x3FCEBE3240000000
  %100 = fmul float %93, %99
  %101 = fadd float %100, 0x3FE62E3E20000000
  %102 = fmul float %93, %101
  %103 = fadd float %102, 1.000000e+00
  %104 = bitcast float %103 to i32
  %105 = shl i32 %89, 23
  %106 = add i32 %104, %105
  %107 = bitcast i32 %106 to float
  %108 = fadd float %107, 1.000000e+00
  %109 = fdiv float 2.000000e+00, %108
  %110 = fsub float 1.000000e+00, %109
  %111 = tail call float @copysignf(float %110, float %78) #12
  %112 = fmul float %81, 0x3FF7154760000000
  %113 = fcmp olt float %112, -1.260000e+02
  br i1 %113, label %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit12, label %114

; <label>:114                                     ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i11
  %115 = fcmp ogt float %112, 1.260000e+02
  %116 = select i1 %115, float 1.260000e+02, float %112
  br label %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit12

_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit12:         ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i11, %114
  %117 = phi float [ %116, %114 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i11 ]
  %118 = fptosi float %117 to i32
  %119 = sitofp i32 %118 to float
  %120 = fsub float %117, %119
  %121 = fsub float 1.000000e+00, %120
  %122 = fsub float 1.000000e+00, %121
  %123 = fmul float %122, 0x3F55D889C0000000
  %124 = fadd float %123, 0x3F84177340000000
  %125 = fmul float %122, %124
  %126 = fadd float %125, 0x3FAC6CE660000000
  %127 = fmul float %122, %126
  %128 = fadd float %127, 0x3FCEBE3240000000
  %129 = fmul float %122, %128
  %130 = fadd float %129, 0x3FE62E3E20000000
  %131 = fmul float %122, %130
  %132 = fadd float %131, 1.000000e+00
  %133 = bitcast float %132 to i32
  %134 = shl i32 %118, 23
  %135 = add i32 %133, %134
  %136 = bitcast i32 %135 to float
  %137 = fmul float %136, 5.000000e-01
  %138 = fdiv float 5.000000e-01, %136
  %139 = fadd float %138, %137
  %140 = fmul float %139, %139
  %141 = fdiv float 1.000000e+00, %140
  %142 = fmul float %79, %141
  %143 = fmul float %80, %141
  %144 = getelementptr inbounds i8* %a_, i64 8
  %145 = bitcast i8* %144 to float*
  %146 = getelementptr inbounds i8* %a_, i64 20
  %147 = bitcast i8* %146 to float*
  %148 = getelementptr inbounds i8* %a_, i64 32
  %149 = bitcast i8* %148 to float*
  %150 = load float* %145, align 4, !tbaa !1
  %151 = load float* %147, align 4, !tbaa !1
  %152 = load float* %149, align 4, !tbaa !1
  %153 = tail call float @fabsf(float %150) #12
  %154 = fmul float %153, 2.000000e+00
  %155 = fmul float %154, 0x3FF7154760000000
  %156 = fcmp olt float %155, -1.260000e+02
  br i1 %156, label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i, label %157

; <label>:157                                     ; preds = %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit12
  %158 = fcmp ogt float %155, 1.260000e+02
  %159 = select i1 %158, float 1.260000e+02, float %155
  br label %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i

_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i:         ; preds = %157, %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit12
  %160 = phi float [ %159, %157 ], [ -1.260000e+02, %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit12 ]
  %161 = fptosi float %160 to i32
  %162 = sitofp i32 %161 to float
  %163 = fsub float %160, %162
  %164 = fsub float 1.000000e+00, %163
  %165 = fsub float 1.000000e+00, %164
  %166 = fmul float %165, 0x3F55D889C0000000
  %167 = fadd float %166, 0x3F84177340000000
  %168 = fmul float %165, %167
  %169 = fadd float %168, 0x3FAC6CE660000000
  %170 = fmul float %165, %169
  %171 = fadd float %170, 0x3FCEBE3240000000
  %172 = fmul float %165, %171
  %173 = fadd float %172, 0x3FE62E3E20000000
  %174 = fmul float %165, %173
  %175 = fadd float %174, 1.000000e+00
  %176 = bitcast float %175 to i32
  %177 = shl i32 %161, 23
  %178 = add i32 %176, %177
  %179 = bitcast i32 %178 to float
  %180 = fadd float %179, 1.000000e+00
  %181 = fdiv float 2.000000e+00, %180
  %182 = fsub float 1.000000e+00, %181
  %183 = tail call float @copysignf(float %182, float %150) #12
  %184 = fmul float %153, 0x3FF7154760000000
  %185 = fcmp olt float %184, -1.260000e+02
  br i1 %185, label %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit, label %186

; <label>:186                                     ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i
  %187 = fcmp ogt float %184, 1.260000e+02
  %188 = select i1 %187, float 1.260000e+02, float %184
  br label %_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_tanhERKNS_5Dual2IfEE.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i, %186
  %189 = phi float [ %188, %186 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_tanhEf.exit.i ]
  %190 = fptosi float %189 to i32
  %191 = sitofp i32 %190 to float
  %192 = fsub float %189, %191
  %193 = fsub float 1.000000e+00, %192
  %194 = fsub float 1.000000e+00, %193
  %195 = fmul float %194, 0x3F55D889C0000000
  %196 = fadd float %195, 0x3F84177340000000
  %197 = fmul float %194, %196
  %198 = fadd float %197, 0x3FAC6CE660000000
  %199 = fmul float %194, %198
  %200 = fadd float %199, 0x3FCEBE3240000000
  %201 = fmul float %194, %200
  %202 = fadd float %201, 0x3FE62E3E20000000
  %203 = fmul float %194, %202
  %204 = fadd float %203, 1.000000e+00
  %205 = bitcast float %204 to i32
  %206 = shl i32 %190, 23
  %207 = add i32 %205, %206
  %208 = bitcast i32 %207 to float
  %209 = fmul float %208, 5.000000e-01
  %210 = fdiv float 5.000000e-01, %208
  %211 = fadd float %210, %209
  %212 = fmul float %211, %211
  %213 = fdiv float 1.000000e+00, %212
  %214 = fmul float %151, %213
  %215 = fmul float %152, %213
  %216 = bitcast i8* %r_ to float*
  store float %39, float* %216, align 4, !tbaa !5
  %217 = getelementptr inbounds i8* %r_, i64 4
  %218 = bitcast i8* %217 to float*
  store float %111, float* %218, align 4, !tbaa !7
  %219 = getelementptr inbounds i8* %r_, i64 8
  %220 = bitcast i8* %219 to float*
  store float %183, float* %220, align 4, !tbaa !8
  %221 = getelementptr inbounds i8* %r_, i64 12
  %222 = bitcast i8* %221 to float*
  store float %70, float* %222, align 4, !tbaa !5
  %223 = getelementptr inbounds i8* %r_, i64 16
  %224 = bitcast i8* %223 to float*
  store float %142, float* %224, align 4, !tbaa !7
  %225 = getelementptr inbounds i8* %r_, i64 20
  %226 = bitcast i8* %225 to float*
  store float %214, float* %226, align 4, !tbaa !8
  %227 = getelementptr inbounds i8* %r_, i64 24
  %228 = bitcast i8* %227 to float*
  store float %71, float* %228, align 4, !tbaa !5
  %229 = getelementptr inbounds i8* %r_, i64 28
  %230 = bitcast i8* %229 to float*
  store float %143, float* %230, align 4, !tbaa !7
  %231 = getelementptr inbounds i8* %r_, i64 32
  %232 = bitcast i8* %231 to float*
  store float %215, float* %232, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_fff(float %x, i8* nocapture %s_, i8* nocapture %c_) #4 {
  %1 = bitcast i8* %s_ to float*
  %2 = bitcast i8* %c_ to float*
  %3 = fmul float %x, 0x3FD45F3060000000
  %4 = tail call float @copysignf(float 5.000000e-01, float %3) #12
  %5 = fadd float %3, %4
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fmul float %7, -3.140625e+00
  %9 = fadd float %8, %x
  %10 = fmul float %7, 0xBF4FB40000000000
  %11 = fadd float %10, %9
  %12 = fmul float %7, 0xBE84440000000000
  %13 = fadd float %12, %11
  %14 = fmul float %7, 0xBD968C2340000000
  %15 = fadd float %14, %13
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = fmul float %17, %17
  %19 = and i32 %6, 1
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %21, label %23

; <label>:21                                      ; preds = %0
  %22 = fsub float -0.000000e+00, %17
  br label %23

; <label>:23                                      ; preds = %21, %0
  %.0.i = phi float [ %22, %21 ], [ %17, %0 ]
  %24 = fmul float %18, 0x3EC5E150E0000000
  %25 = fadd float %24, 0xBF29F75D60000000
  %26 = fmul float %18, %25
  %27 = fadd float %26, 0x3F8110EEE0000000
  %28 = fmul float %18, %27
  %29 = fadd float %28, 0xBFC55554C0000000
  %30 = fmul float %29, %.0.i
  %31 = fmul float %18, %30
  %32 = fadd float %.0.i, %31
  %33 = fmul float %18, 0xBE923DB120000000
  %34 = fadd float %33, 0x3EFA00F160000000
  %35 = fmul float %18, %34
  %36 = fadd float %35, 0xBF56C16B00000000
  %37 = fmul float %18, %36
  %38 = fadd float %37, 0x3FA5555540000000
  %39 = fmul float %18, %38
  %40 = fadd float %39, -5.000000e-01
  %41 = fmul float %18, %40
  %42 = fadd float %41, 1.000000e+00
  br i1 %20, label %43, label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

; <label>:43                                      ; preds = %23
  %44 = fsub float -0.000000e+00, %42
  br label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit:   ; preds = %23, %43
  %cu.0.i = phi float [ %44, %43 ], [ %42, %23 ]
  %45 = tail call float @fabsf(float %32) #12
  %46 = fcmp ogt float %45, 1.000000e+00
  %su.0.i = select i1 %46, float 0.000000e+00, float %32
  %47 = tail call float @fabsf(float %cu.0.i) #12
  %48 = fcmp ogt float %47, 1.000000e+00
  %cu.1.i = select i1 %48, float 0.000000e+00, float %cu.0.i
  store float %su.0.i, float* %1, align 4, !tbaa !1
  store float %cu.1.i, float* %2, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dfdff(i8* nocapture readonly %x_, i8* nocapture %s_, i8* nocapture %c_) #4 {
  %1 = bitcast i8* %c_ to float*
  %2 = bitcast i8* %x_ to float*
  %3 = load float* %2, align 4, !tbaa !1
  %4 = fmul float %3, 0x3FD45F3060000000
  %5 = tail call float @copysignf(float 5.000000e-01, float %4) #12
  %6 = fadd float %4, %5
  %7 = fptosi float %6 to i32
  %8 = sitofp i32 %7 to float
  %9 = fmul float %8, -3.140625e+00
  %10 = fadd float %3, %9
  %11 = fmul float %8, 0xBF4FB40000000000
  %12 = fadd float %11, %10
  %13 = fmul float %8, 0xBE84440000000000
  %14 = fadd float %13, %12
  %15 = fmul float %8, 0xBD968C2340000000
  %16 = fadd float %15, %14
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = fsub float 0x3FF921FB60000000, %17
  %19 = fmul float %18, %18
  %20 = and i32 %7, 1
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %24

; <label>:22                                      ; preds = %0
  %23 = fsub float -0.000000e+00, %18
  br label %24

; <label>:24                                      ; preds = %22, %0
  %.0.i = phi float [ %23, %22 ], [ %18, %0 ]
  %25 = fmul float %19, 0x3EC5E150E0000000
  %26 = fadd float %25, 0xBF29F75D60000000
  %27 = fmul float %19, %26
  %28 = fadd float %27, 0x3F8110EEE0000000
  %29 = fmul float %19, %28
  %30 = fadd float %29, 0xBFC55554C0000000
  %31 = fmul float %30, %.0.i
  %32 = fmul float %19, %31
  %33 = fadd float %.0.i, %32
  %34 = fmul float %19, 0xBE923DB120000000
  %35 = fadd float %34, 0x3EFA00F160000000
  %36 = fmul float %19, %35
  %37 = fadd float %36, 0xBF56C16B00000000
  %38 = fmul float %19, %37
  %39 = fadd float %38, 0x3FA5555540000000
  %40 = fmul float %19, %39
  %41 = fadd float %40, -5.000000e-01
  %42 = fmul float %19, %41
  %43 = fadd float %42, 1.000000e+00
  br i1 %21, label %44, label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

; <label>:44                                      ; preds = %24
  %45 = fsub float -0.000000e+00, %43
  br label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit:   ; preds = %24, %44
  %cu.0.i = phi float [ %45, %44 ], [ %43, %24 ]
  %46 = tail call float @fabsf(float %33) #12
  %47 = fcmp ogt float %46, 1.000000e+00
  %su.0.i = select i1 %47, float 0.000000e+00, float %33
  %48 = tail call float @fabsf(float %cu.0.i) #12
  %49 = fcmp ogt float %48, 1.000000e+00
  %cu.1.i = select i1 %49, float 0.000000e+00, float %cu.0.i
  %50 = getelementptr inbounds i8* %x_, i64 4
  %51 = bitcast i8* %50 to float*
  %52 = load float* %51, align 4, !tbaa !1
  %53 = getelementptr inbounds i8* %x_, i64 8
  %54 = bitcast i8* %53 to float*
  %55 = load float* %54, align 4, !tbaa !1
  %56 = fmul float %52, %cu.1.i
  %57 = fmul float %cu.1.i, %55
  %58 = bitcast i8* %s_ to float*
  store float %su.0.i, float* %58, align 4
  %59 = getelementptr inbounds i8* %s_, i64 4
  %60 = bitcast i8* %59 to float*
  store float %56, float* %60, align 4
  %61 = getelementptr inbounds i8* %s_, i64 8
  %62 = bitcast i8* %61 to float*
  store float %57, float* %62, align 4
  store float %cu.1.i, float* %1, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dffdf(i8* nocapture readonly %x_, i8* nocapture %s_, i8* nocapture %c_) #4 {
  %1 = bitcast i8* %s_ to float*
  %2 = bitcast i8* %x_ to float*
  %3 = load float* %2, align 4, !tbaa !1
  %4 = fmul float %3, 0x3FD45F3060000000
  %5 = tail call float @copysignf(float 5.000000e-01, float %4) #12
  %6 = fadd float %4, %5
  %7 = fptosi float %6 to i32
  %8 = sitofp i32 %7 to float
  %9 = fmul float %8, -3.140625e+00
  %10 = fadd float %3, %9
  %11 = fmul float %8, 0xBF4FB40000000000
  %12 = fadd float %11, %10
  %13 = fmul float %8, 0xBE84440000000000
  %14 = fadd float %13, %12
  %15 = fmul float %8, 0xBD968C2340000000
  %16 = fadd float %15, %14
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = fsub float 0x3FF921FB60000000, %17
  %19 = fmul float %18, %18
  %20 = and i32 %7, 1
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %24

; <label>:22                                      ; preds = %0
  %23 = fsub float -0.000000e+00, %18
  br label %24

; <label>:24                                      ; preds = %22, %0
  %.0.i = phi float [ %23, %22 ], [ %18, %0 ]
  %25 = fmul float %19, 0x3EC5E150E0000000
  %26 = fadd float %25, 0xBF29F75D60000000
  %27 = fmul float %19, %26
  %28 = fadd float %27, 0x3F8110EEE0000000
  %29 = fmul float %19, %28
  %30 = fadd float %29, 0xBFC55554C0000000
  %31 = fmul float %30, %.0.i
  %32 = fmul float %19, %31
  %33 = fadd float %.0.i, %32
  %34 = fmul float %19, 0xBE923DB120000000
  %35 = fadd float %34, 0x3EFA00F160000000
  %36 = fmul float %19, %35
  %37 = fadd float %36, 0xBF56C16B00000000
  %38 = fmul float %19, %37
  %39 = fadd float %38, 0x3FA5555540000000
  %40 = fmul float %19, %39
  %41 = fadd float %40, -5.000000e-01
  %42 = fmul float %19, %41
  %43 = fadd float %42, 1.000000e+00
  br i1 %21, label %44, label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

; <label>:44                                      ; preds = %24
  %45 = fsub float -0.000000e+00, %43
  br label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit:   ; preds = %24, %44
  %cu.0.i = phi float [ %45, %44 ], [ %43, %24 ]
  %46 = tail call float @fabsf(float %33) #12
  %47 = fcmp ogt float %46, 1.000000e+00
  %su.0.i = select i1 %47, float 0.000000e+00, float %33
  %48 = tail call float @fabsf(float %cu.0.i) #12
  %49 = fcmp ogt float %48, 1.000000e+00
  %cu.1.i = select i1 %49, float 0.000000e+00, float %cu.0.i
  %50 = getelementptr inbounds i8* %x_, i64 4
  %51 = bitcast i8* %50 to float*
  %52 = load float* %51, align 4, !tbaa !1
  %53 = getelementptr inbounds i8* %x_, i64 8
  %54 = bitcast i8* %53 to float*
  %55 = load float* %54, align 4, !tbaa !1
  store float %su.0.i, float* %1, align 4, !tbaa !1
  %56 = fmul float %su.0.i, %52
  %57 = fsub float -0.000000e+00, %56
  %58 = fmul float %su.0.i, %55
  %59 = fsub float -0.000000e+00, %58
  %60 = bitcast i8* %c_ to float*
  store float %cu.1.i, float* %60, align 4
  %61 = getelementptr inbounds i8* %c_, i64 4
  %62 = bitcast i8* %61 to float*
  store float %57, float* %62, align 4
  %63 = getelementptr inbounds i8* %c_, i64 8
  %64 = bitcast i8* %63 to float*
  store float %59, float* %64, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dfdfdf(i8* nocapture readonly %x_, i8* nocapture %s_, i8* nocapture %c_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FD45F3060000000
  %4 = tail call float @copysignf(float 5.000000e-01, float %3) #12
  %5 = fadd float %3, %4
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fmul float %7, -3.140625e+00
  %9 = fadd float %2, %8
  %10 = fmul float %7, 0xBF4FB40000000000
  %11 = fadd float %10, %9
  %12 = fmul float %7, 0xBE84440000000000
  %13 = fadd float %12, %11
  %14 = fmul float %7, 0xBD968C2340000000
  %15 = fadd float %14, %13
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = fmul float %17, %17
  %19 = and i32 %6, 1
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %21, label %23

; <label>:21                                      ; preds = %0
  %22 = fsub float -0.000000e+00, %17
  br label %23

; <label>:23                                      ; preds = %21, %0
  %.0.i = phi float [ %22, %21 ], [ %17, %0 ]
  %24 = fmul float %18, 0x3EC5E150E0000000
  %25 = fadd float %24, 0xBF29F75D60000000
  %26 = fmul float %18, %25
  %27 = fadd float %26, 0x3F8110EEE0000000
  %28 = fmul float %18, %27
  %29 = fadd float %28, 0xBFC55554C0000000
  %30 = fmul float %29, %.0.i
  %31 = fmul float %18, %30
  %32 = fadd float %.0.i, %31
  %33 = fmul float %18, 0xBE923DB120000000
  %34 = fadd float %33, 0x3EFA00F160000000
  %35 = fmul float %18, %34
  %36 = fadd float %35, 0xBF56C16B00000000
  %37 = fmul float %18, %36
  %38 = fadd float %37, 0x3FA5555540000000
  %39 = fmul float %18, %38
  %40 = fadd float %39, -5.000000e-01
  %41 = fmul float %18, %40
  %42 = fadd float %41, 1.000000e+00
  br i1 %20, label %43, label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

; <label>:43                                      ; preds = %23
  %44 = fsub float -0.000000e+00, %42
  br label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit:   ; preds = %23, %43
  %cu.0.i = phi float [ %44, %43 ], [ %42, %23 ]
  %45 = tail call float @fabsf(float %32) #12
  %46 = fcmp ogt float %45, 1.000000e+00
  %su.0.i = select i1 %46, float 0.000000e+00, float %32
  %47 = tail call float @fabsf(float %cu.0.i) #12
  %48 = fcmp ogt float %47, 1.000000e+00
  %cu.1.i = select i1 %48, float 0.000000e+00, float %cu.0.i
  %49 = getelementptr inbounds i8* %x_, i64 4
  %50 = bitcast i8* %49 to float*
  %51 = load float* %50, align 4, !tbaa !1
  %52 = getelementptr inbounds i8* %x_, i64 8
  %53 = bitcast i8* %52 to float*
  %54 = load float* %53, align 4, !tbaa !1
  %55 = fmul float %51, %cu.1.i
  %56 = fmul float %cu.1.i, %54
  %57 = bitcast i8* %s_ to float*
  store float %su.0.i, float* %57, align 4
  %58 = getelementptr inbounds i8* %s_, i64 4
  %59 = bitcast i8* %58 to float*
  store float %55, float* %59, align 4
  %60 = getelementptr inbounds i8* %s_, i64 8
  %61 = bitcast i8* %60 to float*
  store float %56, float* %61, align 4
  %62 = fmul float %su.0.i, %51
  %63 = fsub float -0.000000e+00, %62
  %64 = fmul float %su.0.i, %54
  %65 = fsub float -0.000000e+00, %64
  %66 = bitcast i8* %c_ to float*
  store float %cu.1.i, float* %66, align 4
  %67 = getelementptr inbounds i8* %c_, i64 4
  %68 = bitcast i8* %67 to float*
  store float %63, float* %68, align 4
  %69 = getelementptr inbounds i8* %c_, i64 8
  %70 = bitcast i8* %69 to float*
  store float %65, float* %70, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_vvv(i8* nocapture readonly %x_, i8* nocapture %s_, i8* nocapture %c_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = bitcast i8* %s_ to float*
  %3 = bitcast i8* %c_ to float*
  br label %4

; <label>:4                                       ; preds = %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit ]
  %5 = getelementptr inbounds float* %1, i64 %indvars.iv
  %6 = load float* %5, align 4, !tbaa !1
  %7 = getelementptr inbounds float* %2, i64 %indvars.iv
  %8 = getelementptr inbounds float* %3, i64 %indvars.iv
  %9 = fmul float %6, 0x3FD45F3060000000
  %10 = tail call float @copysignf(float 5.000000e-01, float %9) #12
  %11 = fadd float %9, %10
  %12 = fptosi float %11 to i32
  %13 = sitofp i32 %12 to float
  %14 = fmul float %13, -3.140625e+00
  %15 = fadd float %6, %14
  %16 = fmul float %13, 0xBF4FB40000000000
  %17 = fadd float %16, %15
  %18 = fmul float %13, 0xBE84440000000000
  %19 = fadd float %18, %17
  %20 = fmul float %13, 0xBD968C2340000000
  %21 = fadd float %20, %19
  %22 = fsub float 0x3FF921FB60000000, %21
  %23 = fsub float 0x3FF921FB60000000, %22
  %24 = fmul float %23, %23
  %25 = and i32 %12, 1
  %26 = icmp ne i32 %25, 0
  br i1 %26, label %27, label %29

; <label>:27                                      ; preds = %4
  %28 = fsub float -0.000000e+00, %23
  br label %29

; <label>:29                                      ; preds = %27, %4
  %.0.i = phi float [ %28, %27 ], [ %23, %4 ]
  %30 = fmul float %24, 0x3EC5E150E0000000
  %31 = fadd float %30, 0xBF29F75D60000000
  %32 = fmul float %24, %31
  %33 = fadd float %32, 0x3F8110EEE0000000
  %34 = fmul float %24, %33
  %35 = fadd float %34, 0xBFC55554C0000000
  %36 = fmul float %35, %.0.i
  %37 = fmul float %24, %36
  %38 = fadd float %.0.i, %37
  %39 = fmul float %24, 0xBE923DB120000000
  %40 = fadd float %39, 0x3EFA00F160000000
  %41 = fmul float %24, %40
  %42 = fadd float %41, 0xBF56C16B00000000
  %43 = fmul float %24, %42
  %44 = fadd float %43, 0x3FA5555540000000
  %45 = fmul float %24, %44
  %46 = fadd float %45, -5.000000e-01
  %47 = fmul float %24, %46
  %48 = fadd float %47, 1.000000e+00
  br i1 %26, label %49, label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

; <label>:49                                      ; preds = %29
  %50 = fsub float -0.000000e+00, %48
  br label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit:   ; preds = %29, %49
  %cu.0.i = phi float [ %50, %49 ], [ %48, %29 ]
  %51 = tail call float @fabsf(float %38) #12
  %52 = fcmp ogt float %51, 1.000000e+00
  %su.0.i = select i1 %52, float 0.000000e+00, float %38
  %53 = tail call float @fabsf(float %cu.0.i) #12
  %54 = fcmp ogt float %53, 1.000000e+00
  %cu.1.i = select i1 %54, float 0.000000e+00, float %cu.0.i
  store float %su.0.i, float* %7, align 4, !tbaa !1
  store float %cu.1.i, float* %8, align 4, !tbaa !1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 3
  br i1 %exitcond, label %55, label %4

; <label>:55                                      ; preds = %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dvdvv(i8* nocapture readonly %x_, i8* nocapture %s_, i8* nocapture %c_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = getelementptr inbounds i8* %x_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %x_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = bitcast i8* %s_ to float*
  %7 = getelementptr inbounds i8* %s_, i64 12
  %8 = bitcast i8* %7 to float*
  %9 = getelementptr inbounds i8* %s_, i64 24
  %10 = bitcast i8* %9 to float*
  %11 = bitcast i8* %c_ to float*
  br label %12

; <label>:12                                      ; preds = %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit ]
  %13 = getelementptr inbounds float* %1, i64 %indvars.iv
  %14 = load float* %13, align 4, !tbaa !1
  %15 = fmul float %14, 0x3FD45F3060000000
  %16 = tail call float @copysignf(float 5.000000e-01, float %15) #12
  %17 = fadd float %15, %16
  %18 = fptosi float %17 to i32
  %19 = sitofp i32 %18 to float
  %20 = fmul float %19, -3.140625e+00
  %21 = fadd float %14, %20
  %22 = fmul float %19, 0xBF4FB40000000000
  %23 = fadd float %22, %21
  %24 = fmul float %19, 0xBE84440000000000
  %25 = fadd float %24, %23
  %26 = fmul float %19, 0xBD968C2340000000
  %27 = fadd float %26, %25
  %28 = fsub float 0x3FF921FB60000000, %27
  %29 = fsub float 0x3FF921FB60000000, %28
  %30 = fmul float %29, %29
  %31 = and i32 %18, 1
  %32 = icmp ne i32 %31, 0
  br i1 %32, label %33, label %35

; <label>:33                                      ; preds = %12
  %34 = fsub float -0.000000e+00, %29
  br label %35

; <label>:35                                      ; preds = %33, %12
  %.0.i = phi float [ %34, %33 ], [ %29, %12 ]
  %36 = fmul float %30, 0x3EC5E150E0000000
  %37 = fadd float %36, 0xBF29F75D60000000
  %38 = fmul float %30, %37
  %39 = fadd float %38, 0x3F8110EEE0000000
  %40 = fmul float %30, %39
  %41 = fadd float %40, 0xBFC55554C0000000
  %42 = fmul float %41, %.0.i
  %43 = fmul float %30, %42
  %44 = fadd float %.0.i, %43
  %45 = fmul float %30, 0xBE923DB120000000
  %46 = fadd float %45, 0x3EFA00F160000000
  %47 = fmul float %30, %46
  %48 = fadd float %47, 0xBF56C16B00000000
  %49 = fmul float %30, %48
  %50 = fadd float %49, 0x3FA5555540000000
  %51 = fmul float %30, %50
  %52 = fadd float %51, -5.000000e-01
  %53 = fmul float %30, %52
  %54 = fadd float %53, 1.000000e+00
  br i1 %32, label %55, label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

; <label>:55                                      ; preds = %35
  %56 = fsub float -0.000000e+00, %54
  br label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit:   ; preds = %35, %55
  %cu.0.i = phi float [ %56, %55 ], [ %54, %35 ]
  %57 = tail call float @fabsf(float %44) #12
  %58 = fcmp ogt float %57, 1.000000e+00
  %su.0.i = select i1 %58, float 0.000000e+00, float %44
  %59 = tail call float @fabsf(float %cu.0.i) #12
  %60 = fcmp ogt float %59, 1.000000e+00
  %cu.1.i = select i1 %60, float 0.000000e+00, float %cu.0.i
  %61 = getelementptr inbounds float* %3, i64 %indvars.iv
  %62 = load float* %61, align 4, !tbaa !1
  %63 = getelementptr inbounds float* %5, i64 %indvars.iv
  %64 = load float* %63, align 4, !tbaa !1
  %65 = getelementptr inbounds float* %6, i64 %indvars.iv
  store float %su.0.i, float* %65, align 4, !tbaa !1
  %66 = fmul float %62, %cu.1.i
  %67 = getelementptr inbounds float* %8, i64 %indvars.iv
  store float %66, float* %67, align 4, !tbaa !1
  %68 = fmul float %cu.1.i, %64
  %69 = getelementptr inbounds float* %10, i64 %indvars.iv
  store float %68, float* %69, align 4, !tbaa !1
  %70 = getelementptr inbounds float* %11, i64 %indvars.iv
  store float %cu.1.i, float* %70, align 4, !tbaa !1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 3
  br i1 %exitcond, label %71, label %12

; <label>:71                                      ; preds = %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dvvdv(i8* nocapture readonly %x_, i8* nocapture %s_, i8* nocapture %c_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = getelementptr inbounds i8* %x_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %x_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = bitcast i8* %s_ to float*
  %7 = bitcast i8* %c_ to float*
  %8 = getelementptr inbounds i8* %c_, i64 12
  %9 = bitcast i8* %8 to float*
  %10 = getelementptr inbounds i8* %c_, i64 24
  %11 = bitcast i8* %10 to float*
  br label %12

; <label>:12                                      ; preds = %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit ]
  %13 = getelementptr inbounds float* %1, i64 %indvars.iv
  %14 = load float* %13, align 4, !tbaa !1
  %15 = fmul float %14, 0x3FD45F3060000000
  %16 = tail call float @copysignf(float 5.000000e-01, float %15) #12
  %17 = fadd float %15, %16
  %18 = fptosi float %17 to i32
  %19 = sitofp i32 %18 to float
  %20 = fmul float %19, -3.140625e+00
  %21 = fadd float %14, %20
  %22 = fmul float %19, 0xBF4FB40000000000
  %23 = fadd float %22, %21
  %24 = fmul float %19, 0xBE84440000000000
  %25 = fadd float %24, %23
  %26 = fmul float %19, 0xBD968C2340000000
  %27 = fadd float %26, %25
  %28 = fsub float 0x3FF921FB60000000, %27
  %29 = fsub float 0x3FF921FB60000000, %28
  %30 = fmul float %29, %29
  %31 = and i32 %18, 1
  %32 = icmp ne i32 %31, 0
  br i1 %32, label %33, label %35

; <label>:33                                      ; preds = %12
  %34 = fsub float -0.000000e+00, %29
  br label %35

; <label>:35                                      ; preds = %33, %12
  %.0.i = phi float [ %34, %33 ], [ %29, %12 ]
  %36 = fmul float %30, 0x3EC5E150E0000000
  %37 = fadd float %36, 0xBF29F75D60000000
  %38 = fmul float %30, %37
  %39 = fadd float %38, 0x3F8110EEE0000000
  %40 = fmul float %30, %39
  %41 = fadd float %40, 0xBFC55554C0000000
  %42 = fmul float %41, %.0.i
  %43 = fmul float %30, %42
  %44 = fadd float %.0.i, %43
  %45 = fmul float %30, 0xBE923DB120000000
  %46 = fadd float %45, 0x3EFA00F160000000
  %47 = fmul float %30, %46
  %48 = fadd float %47, 0xBF56C16B00000000
  %49 = fmul float %30, %48
  %50 = fadd float %49, 0x3FA5555540000000
  %51 = fmul float %30, %50
  %52 = fadd float %51, -5.000000e-01
  %53 = fmul float %30, %52
  %54 = fadd float %53, 1.000000e+00
  br i1 %32, label %55, label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

; <label>:55                                      ; preds = %35
  %56 = fsub float -0.000000e+00, %54
  br label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit:   ; preds = %35, %55
  %cu.0.i = phi float [ %56, %55 ], [ %54, %35 ]
  %57 = tail call float @fabsf(float %44) #12
  %58 = fcmp ogt float %57, 1.000000e+00
  %su.0.i = select i1 %58, float 0.000000e+00, float %44
  %59 = tail call float @fabsf(float %cu.0.i) #12
  %60 = fcmp ogt float %59, 1.000000e+00
  %cu.1.i = select i1 %60, float 0.000000e+00, float %cu.0.i
  %61 = getelementptr inbounds float* %3, i64 %indvars.iv
  %62 = load float* %61, align 4, !tbaa !1
  %63 = getelementptr inbounds float* %5, i64 %indvars.iv
  %64 = load float* %63, align 4, !tbaa !1
  %65 = getelementptr inbounds float* %6, i64 %indvars.iv
  store float %su.0.i, float* %65, align 4, !tbaa !1
  %66 = getelementptr inbounds float* %7, i64 %indvars.iv
  store float %cu.1.i, float* %66, align 4, !tbaa !1
  %67 = fmul float %su.0.i, %62
  %68 = fsub float -0.000000e+00, %67
  %69 = getelementptr inbounds float* %9, i64 %indvars.iv
  store float %68, float* %69, align 4, !tbaa !1
  %70 = fmul float %su.0.i, %64
  %71 = fsub float -0.000000e+00, %70
  %72 = getelementptr inbounds float* %11, i64 %indvars.iv
  store float %71, float* %72, align 4, !tbaa !1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 3
  br i1 %exitcond, label %73, label %12

; <label>:73                                      ; preds = %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dvdvdv(i8* nocapture readonly %x_, i8* nocapture %s_, i8* nocapture %c_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = getelementptr inbounds i8* %x_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %x_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = bitcast i8* %s_ to float*
  %7 = getelementptr inbounds i8* %s_, i64 12
  %8 = bitcast i8* %7 to float*
  %9 = getelementptr inbounds i8* %s_, i64 24
  %10 = bitcast i8* %9 to float*
  %11 = bitcast i8* %c_ to float*
  %12 = getelementptr inbounds i8* %c_, i64 12
  %13 = bitcast i8* %12 to float*
  %14 = getelementptr inbounds i8* %c_, i64 24
  %15 = bitcast i8* %14 to float*
  br label %16

; <label>:16                                      ; preds = %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit ]
  %17 = getelementptr inbounds float* %1, i64 %indvars.iv
  %18 = load float* %17, align 4, !tbaa !1
  %19 = fmul float %18, 0x3FD45F3060000000
  %20 = tail call float @copysignf(float 5.000000e-01, float %19) #12
  %21 = fadd float %19, %20
  %22 = fptosi float %21 to i32
  %23 = sitofp i32 %22 to float
  %24 = fmul float %23, -3.140625e+00
  %25 = fadd float %18, %24
  %26 = fmul float %23, 0xBF4FB40000000000
  %27 = fadd float %26, %25
  %28 = fmul float %23, 0xBE84440000000000
  %29 = fadd float %28, %27
  %30 = fmul float %23, 0xBD968C2340000000
  %31 = fadd float %30, %29
  %32 = fsub float 0x3FF921FB60000000, %31
  %33 = fsub float 0x3FF921FB60000000, %32
  %34 = fmul float %33, %33
  %35 = and i32 %22, 1
  %36 = icmp ne i32 %35, 0
  br i1 %36, label %37, label %39

; <label>:37                                      ; preds = %16
  %38 = fsub float -0.000000e+00, %33
  br label %39

; <label>:39                                      ; preds = %37, %16
  %.0.i = phi float [ %38, %37 ], [ %33, %16 ]
  %40 = fmul float %34, 0x3EC5E150E0000000
  %41 = fadd float %40, 0xBF29F75D60000000
  %42 = fmul float %34, %41
  %43 = fadd float %42, 0x3F8110EEE0000000
  %44 = fmul float %34, %43
  %45 = fadd float %44, 0xBFC55554C0000000
  %46 = fmul float %45, %.0.i
  %47 = fmul float %34, %46
  %48 = fadd float %.0.i, %47
  %49 = fmul float %34, 0xBE923DB120000000
  %50 = fadd float %49, 0x3EFA00F160000000
  %51 = fmul float %34, %50
  %52 = fadd float %51, 0xBF56C16B00000000
  %53 = fmul float %34, %52
  %54 = fadd float %53, 0x3FA5555540000000
  %55 = fmul float %34, %54
  %56 = fadd float %55, -5.000000e-01
  %57 = fmul float %34, %56
  %58 = fadd float %57, 1.000000e+00
  br i1 %36, label %59, label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

; <label>:59                                      ; preds = %39
  %60 = fsub float -0.000000e+00, %58
  br label %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit

_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit:   ; preds = %39, %59
  %cu.0.i = phi float [ %60, %59 ], [ %58, %39 ]
  %61 = tail call float @fabsf(float %48) #12
  %62 = fcmp ogt float %61, 1.000000e+00
  %su.0.i = select i1 %62, float 0.000000e+00, float %48
  %63 = tail call float @fabsf(float %cu.0.i) #12
  %64 = fcmp ogt float %63, 1.000000e+00
  %cu.1.i = select i1 %64, float 0.000000e+00, float %cu.0.i
  %65 = getelementptr inbounds float* %3, i64 %indvars.iv
  %66 = load float* %65, align 4, !tbaa !1
  %67 = getelementptr inbounds float* %5, i64 %indvars.iv
  %68 = load float* %67, align 4, !tbaa !1
  %69 = getelementptr inbounds float* %6, i64 %indvars.iv
  store float %su.0.i, float* %69, align 4, !tbaa !1
  %70 = fmul float %66, %cu.1.i
  %71 = getelementptr inbounds float* %8, i64 %indvars.iv
  store float %70, float* %71, align 4, !tbaa !1
  %72 = fmul float %cu.1.i, %68
  %73 = getelementptr inbounds float* %10, i64 %indvars.iv
  store float %72, float* %73, align 4, !tbaa !1
  %74 = getelementptr inbounds float* %11, i64 %indvars.iv
  store float %cu.1.i, float* %74, align 4, !tbaa !1
  %75 = fmul float %su.0.i, %66
  %76 = fsub float -0.000000e+00, %75
  %77 = getelementptr inbounds float* %13, i64 %indvars.iv
  store float %76, float* %77, align 4, !tbaa !1
  %78 = fmul float %su.0.i, %68
  %79 = fsub float -0.000000e+00, %78
  %80 = getelementptr inbounds float* %15, i64 %indvars.iv
  store float %79, float* %80, align 4, !tbaa !1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 3
  br i1 %exitcond, label %81, label %16

; <label>:81                                      ; preds = %_ZN11OpenImageIO4v1_711fast_sincosEfPfS1_.exit
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_log_ff(float %a) #3 {
  %1 = fcmp olt float %a, 0x3810000000000000
  br i1 %1, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit, label %2

; <label>:2                                       ; preds = %0
  %3 = fcmp ogt float %a, 0x47EFFFFFE0000000
  %4 = bitcast float %a to i32
  %phitmp.i.i = select i1 %3, i32 2139095039, i32 %4
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit:   ; preds = %0, %2
  %5 = phi i32 [ %phitmp.i.i, %2 ], [ 8388608, %0 ]
  %6 = lshr i32 %5, 23
  %7 = add nsw i32 %6, -127
  %8 = and i32 %5, 8388607
  %9 = or i32 %8, 1065353216
  %10 = bitcast i32 %9 to float
  %11 = fadd float %10, -1.000000e+00
  %12 = fmul float %11, %11
  %13 = fmul float %12, %12
  %14 = fmul float %11, 0xBF831161A0000000
  %15 = fadd float %14, 0x3FAAA83920000000
  %16 = fmul float %11, 0x3FDEA2C5A0000000
  %17 = fadd float %16, 0xBFE713CA80000000
  %18 = fmul float %11, %15
  %19 = fadd float %18, 0xBFC19A9FA0000000
  %20 = fmul float %11, %19
  %21 = fadd float %20, 0x3FCEF5B7A0000000
  %22 = fmul float %11, %21
  %23 = fadd float %22, 0xBFD63A40C0000000
  %24 = fmul float %11, %17
  %25 = fadd float %24, 0x3FF7154200000000
  %26 = fmul float %13, %23
  %27 = fmul float %11, %25
  %28 = fadd float %27, %26
  %29 = sitofp i32 %7 to float
  %30 = fadd float %29, %28
  %31 = fmul float %30, 0x3FE62E4300000000
  ret float %31
}

; Function Attrs: nounwind uwtable
define void @osl_log_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, 0x3810000000000000
  br i1 %3, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 0x47EFFFFFE0000000
  %6 = bitcast float %2 to i32
  %phitmp.i.i.i = select i1 %5, i32 2139095039, i32 %6
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i: ; preds = %4, %0
  %7 = phi i32 [ %phitmp.i.i.i, %4 ], [ 8388608, %0 ]
  %8 = lshr i32 %7, 23
  %9 = add nsw i32 %8, -127
  %10 = and i32 %7, 8388607
  %11 = or i32 %10, 1065353216
  %12 = bitcast i32 %11 to float
  %13 = fadd float %12, -1.000000e+00
  %14 = fmul float %13, %13
  %15 = fmul float %14, %14
  %16 = fmul float %13, 0xBF831161A0000000
  %17 = fadd float %16, 0x3FAAA83920000000
  %18 = fmul float %13, 0x3FDEA2C5A0000000
  %19 = fadd float %18, 0xBFE713CA80000000
  %20 = fmul float %13, %17
  %21 = fadd float %20, 0xBFC19A9FA0000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 0x3FCEF5B7A0000000
  %24 = fmul float %13, %23
  %25 = fadd float %24, 0xBFD63A40C0000000
  %26 = fmul float %13, %19
  %27 = fadd float %26, 0x3FF7154200000000
  %28 = fmul float %15, %25
  %29 = fmul float %13, %27
  %30 = fadd float %29, %28
  %31 = sitofp i32 %9 to float
  %32 = fadd float %31, %30
  %33 = fmul float %32, 0x3FE62E4300000000
  br i1 %3, label %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit, label %34

; <label>:34                                      ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i
  %35 = fdiv float 1.000000e+00, %2
  br label %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit

_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit:            ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, %34
  %36 = phi float [ %35, %34 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i ]
  %37 = getelementptr inbounds i8* %a, i64 4
  %38 = bitcast i8* %37 to float*
  %39 = load float* %38, align 4, !tbaa !1
  %40 = fmul float %36, %39
  %41 = getelementptr inbounds i8* %a, i64 8
  %42 = bitcast i8* %41 to float*
  %43 = load float* %42, align 4, !tbaa !1
  %44 = fmul float %36, %43
  %45 = insertelement <2 x float> undef, float %33, i32 0
  %46 = insertelement <2 x float> %45, float %40, i32 1
  %47 = bitcast i8* %r to <2 x float>*
  store <2 x float> %46, <2 x float>* %47, align 4
  %48 = getelementptr inbounds i8* %r, i64 8
  %49 = bitcast i8* %48 to float*
  store float %44, float* %49, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, 0x3810000000000000
  br i1 %3, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit4, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 0x47EFFFFFE0000000
  %6 = bitcast float %2 to i32
  %phitmp.i.i3 = select i1 %5, i32 2139095039, i32 %6
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit4

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit4:  ; preds = %0, %4
  %7 = phi i32 [ %phitmp.i.i3, %4 ], [ 8388608, %0 ]
  %8 = lshr i32 %7, 23
  %9 = add nsw i32 %8, -127
  %10 = and i32 %7, 8388607
  %11 = or i32 %10, 1065353216
  %12 = bitcast i32 %11 to float
  %13 = fadd float %12, -1.000000e+00
  %14 = fmul float %13, %13
  %15 = fmul float %14, %14
  %16 = fmul float %13, 0xBF831161A0000000
  %17 = fadd float %16, 0x3FAAA83920000000
  %18 = fmul float %13, 0x3FDEA2C5A0000000
  %19 = fadd float %18, 0xBFE713CA80000000
  %20 = fmul float %13, %17
  %21 = fadd float %20, 0xBFC19A9FA0000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 0x3FCEF5B7A0000000
  %24 = fmul float %13, %23
  %25 = fadd float %24, 0xBFD63A40C0000000
  %26 = fmul float %13, %19
  %27 = fadd float %26, 0x3FF7154200000000
  %28 = fmul float %15, %25
  %29 = fmul float %13, %27
  %30 = fadd float %29, %28
  %31 = sitofp i32 %9 to float
  %32 = fadd float %31, %30
  %33 = fmul float %32, 0x3FE62E4300000000
  %34 = bitcast i8* %r_ to float*
  store float %33, float* %34, align 4, !tbaa !1
  %35 = getelementptr inbounds i8* %a_, i64 4
  %36 = bitcast i8* %35 to float*
  %37 = load float* %36, align 4, !tbaa !1
  %38 = fcmp olt float %37, 0x3810000000000000
  br i1 %38, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit2, label %39

; <label>:39                                      ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit4
  %40 = fcmp ogt float %37, 0x47EFFFFFE0000000
  %41 = bitcast float %37 to i32
  %phitmp.i.i1 = select i1 %40, i32 2139095039, i32 %41
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit2

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit2:  ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit4, %39
  %42 = phi i32 [ %phitmp.i.i1, %39 ], [ 8388608, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit4 ]
  %43 = lshr i32 %42, 23
  %44 = add nsw i32 %43, -127
  %45 = and i32 %42, 8388607
  %46 = or i32 %45, 1065353216
  %47 = bitcast i32 %46 to float
  %48 = fadd float %47, -1.000000e+00
  %49 = fmul float %48, %48
  %50 = fmul float %49, %49
  %51 = fmul float %48, 0xBF831161A0000000
  %52 = fadd float %51, 0x3FAAA83920000000
  %53 = fmul float %48, 0x3FDEA2C5A0000000
  %54 = fadd float %53, 0xBFE713CA80000000
  %55 = fmul float %48, %52
  %56 = fadd float %55, 0xBFC19A9FA0000000
  %57 = fmul float %48, %56
  %58 = fadd float %57, 0x3FCEF5B7A0000000
  %59 = fmul float %48, %58
  %60 = fadd float %59, 0xBFD63A40C0000000
  %61 = fmul float %48, %54
  %62 = fadd float %61, 0x3FF7154200000000
  %63 = fmul float %50, %60
  %64 = fmul float %48, %62
  %65 = fadd float %64, %63
  %66 = sitofp i32 %44 to float
  %67 = fadd float %66, %65
  %68 = fmul float %67, 0x3FE62E4300000000
  %69 = getelementptr inbounds i8* %r_, i64 4
  %70 = bitcast i8* %69 to float*
  store float %68, float* %70, align 4, !tbaa !1
  %71 = getelementptr inbounds i8* %a_, i64 8
  %72 = bitcast i8* %71 to float*
  %73 = load float* %72, align 4, !tbaa !1
  %74 = fcmp olt float %73, 0x3810000000000000
  br i1 %74, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit, label %75

; <label>:75                                      ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit2
  %76 = fcmp ogt float %73, 0x47EFFFFFE0000000
  %77 = bitcast float %73 to i32
  %phitmp.i.i = select i1 %76, i32 2139095039, i32 %77
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit:   ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit2, %75
  %78 = phi i32 [ %phitmp.i.i, %75 ], [ 8388608, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit2 ]
  %79 = lshr i32 %78, 23
  %80 = add nsw i32 %79, -127
  %81 = and i32 %78, 8388607
  %82 = or i32 %81, 1065353216
  %83 = bitcast i32 %82 to float
  %84 = fadd float %83, -1.000000e+00
  %85 = fmul float %84, %84
  %86 = fmul float %85, %85
  %87 = fmul float %84, 0xBF831161A0000000
  %88 = fadd float %87, 0x3FAAA83920000000
  %89 = fmul float %84, 0x3FDEA2C5A0000000
  %90 = fadd float %89, 0xBFE713CA80000000
  %91 = fmul float %84, %88
  %92 = fadd float %91, 0xBFC19A9FA0000000
  %93 = fmul float %84, %92
  %94 = fadd float %93, 0x3FCEF5B7A0000000
  %95 = fmul float %84, %94
  %96 = fadd float %95, 0xBFD63A40C0000000
  %97 = fmul float %84, %90
  %98 = fadd float %97, 0x3FF7154200000000
  %99 = fmul float %86, %96
  %100 = fmul float %84, %98
  %101 = fadd float %100, %99
  %102 = sitofp i32 %80 to float
  %103 = fadd float %102, %101
  %104 = fmul float %103, 0x3FE62E4300000000
  %105 = getelementptr inbounds i8* %r_, i64 8
  %106 = bitcast i8* %105 to float*
  store float %104, float* %106, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = fcmp olt float %6, 0x3810000000000000
  br i1 %9, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15, label %10

; <label>:10                                      ; preds = %0
  %11 = fcmp ogt float %6, 0x47EFFFFFE0000000
  %12 = bitcast float %6 to i32
  %phitmp.i.i.i14 = select i1 %11, i32 2139095039, i32 %12
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15: ; preds = %10, %0
  %13 = phi i32 [ %phitmp.i.i.i14, %10 ], [ 8388608, %0 ]
  %14 = lshr i32 %13, 23
  %15 = add nsw i32 %14, -127
  %16 = and i32 %13, 8388607
  %17 = or i32 %16, 1065353216
  %18 = bitcast i32 %17 to float
  %19 = fadd float %18, -1.000000e+00
  %20 = fmul float %19, %19
  %21 = fmul float %20, %20
  %22 = fmul float %19, 0xBF831161A0000000
  %23 = fadd float %22, 0x3FAAA83920000000
  %24 = fmul float %19, 0x3FDEA2C5A0000000
  %25 = fadd float %24, 0xBFE713CA80000000
  %26 = fmul float %19, %23
  %27 = fadd float %26, 0xBFC19A9FA0000000
  %28 = fmul float %19, %27
  %29 = fadd float %28, 0x3FCEF5B7A0000000
  %30 = fmul float %19, %29
  %31 = fadd float %30, 0xBFD63A40C0000000
  %32 = fmul float %19, %25
  %33 = fadd float %32, 0x3FF7154200000000
  %34 = fmul float %21, %31
  %35 = fmul float %19, %33
  %36 = fadd float %35, %34
  %37 = sitofp i32 %15 to float
  %38 = fadd float %37, %36
  %39 = fmul float %38, 0x3FE62E4300000000
  br i1 %9, label %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit16, label %40

; <label>:40                                      ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15
  %41 = fdiv float 1.000000e+00, %6
  br label %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit16

_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit16:          ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15, %40
  %42 = phi float [ %41, %40 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15 ]
  %43 = fmul float %7, %42
  %44 = fmul float %8, %42
  %45 = getelementptr inbounds i8* %a_, i64 4
  %46 = bitcast i8* %45 to float*
  %47 = getelementptr inbounds i8* %a_, i64 16
  %48 = bitcast i8* %47 to float*
  %49 = getelementptr inbounds i8* %a_, i64 28
  %50 = bitcast i8* %49 to float*
  %51 = load float* %46, align 4, !tbaa !1
  %52 = load float* %48, align 4, !tbaa !1
  %53 = load float* %50, align 4, !tbaa !1
  %54 = fcmp olt float %51, 0x3810000000000000
  br i1 %54, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12, label %55

; <label>:55                                      ; preds = %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit16
  %56 = fcmp ogt float %51, 0x47EFFFFFE0000000
  %57 = bitcast float %51 to i32
  %phitmp.i.i.i11 = select i1 %56, i32 2139095039, i32 %57
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12: ; preds = %55, %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit16
  %58 = phi i32 [ %phitmp.i.i.i11, %55 ], [ 8388608, %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit16 ]
  %59 = lshr i32 %58, 23
  %60 = add nsw i32 %59, -127
  %61 = and i32 %58, 8388607
  %62 = or i32 %61, 1065353216
  %63 = bitcast i32 %62 to float
  %64 = fadd float %63, -1.000000e+00
  %65 = fmul float %64, %64
  %66 = fmul float %65, %65
  %67 = fmul float %64, 0xBF831161A0000000
  %68 = fadd float %67, 0x3FAAA83920000000
  %69 = fmul float %64, 0x3FDEA2C5A0000000
  %70 = fadd float %69, 0xBFE713CA80000000
  %71 = fmul float %64, %68
  %72 = fadd float %71, 0xBFC19A9FA0000000
  %73 = fmul float %64, %72
  %74 = fadd float %73, 0x3FCEF5B7A0000000
  %75 = fmul float %64, %74
  %76 = fadd float %75, 0xBFD63A40C0000000
  %77 = fmul float %64, %70
  %78 = fadd float %77, 0x3FF7154200000000
  %79 = fmul float %66, %76
  %80 = fmul float %64, %78
  %81 = fadd float %80, %79
  %82 = sitofp i32 %60 to float
  %83 = fadd float %82, %81
  %84 = fmul float %83, 0x3FE62E4300000000
  br i1 %54, label %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit13, label %85

; <label>:85                                      ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12
  %86 = fdiv float 1.000000e+00, %51
  br label %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit13

_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit13:          ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12, %85
  %87 = phi float [ %86, %85 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12 ]
  %88 = fmul float %52, %87
  %89 = fmul float %53, %87
  %90 = getelementptr inbounds i8* %a_, i64 8
  %91 = bitcast i8* %90 to float*
  %92 = getelementptr inbounds i8* %a_, i64 20
  %93 = bitcast i8* %92 to float*
  %94 = getelementptr inbounds i8* %a_, i64 32
  %95 = bitcast i8* %94 to float*
  %96 = load float* %91, align 4, !tbaa !1
  %97 = load float* %93, align 4, !tbaa !1
  %98 = load float* %95, align 4, !tbaa !1
  %99 = fcmp olt float %96, 0x3810000000000000
  br i1 %99, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, label %100

; <label>:100                                     ; preds = %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit13
  %101 = fcmp ogt float %96, 0x47EFFFFFE0000000
  %102 = bitcast float %96 to i32
  %phitmp.i.i.i = select i1 %101, i32 2139095039, i32 %102
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i: ; preds = %100, %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit13
  %103 = phi i32 [ %phitmp.i.i.i, %100 ], [ 8388608, %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit13 ]
  %104 = lshr i32 %103, 23
  %105 = add nsw i32 %104, -127
  %106 = and i32 %103, 8388607
  %107 = or i32 %106, 1065353216
  %108 = bitcast i32 %107 to float
  %109 = fadd float %108, -1.000000e+00
  %110 = fmul float %109, %109
  %111 = fmul float %110, %110
  %112 = fmul float %109, 0xBF831161A0000000
  %113 = fadd float %112, 0x3FAAA83920000000
  %114 = fmul float %109, 0x3FDEA2C5A0000000
  %115 = fadd float %114, 0xBFE713CA80000000
  %116 = fmul float %109, %113
  %117 = fadd float %116, 0xBFC19A9FA0000000
  %118 = fmul float %109, %117
  %119 = fadd float %118, 0x3FCEF5B7A0000000
  %120 = fmul float %109, %119
  %121 = fadd float %120, 0xBFD63A40C0000000
  %122 = fmul float %109, %115
  %123 = fadd float %122, 0x3FF7154200000000
  %124 = fmul float %111, %121
  %125 = fmul float %109, %123
  %126 = fadd float %125, %124
  %127 = sitofp i32 %105 to float
  %128 = fadd float %127, %126
  %129 = fmul float %128, 0x3FE62E4300000000
  br i1 %99, label %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit, label %130

; <label>:130                                     ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i
  %131 = fdiv float 1.000000e+00, %96
  br label %_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit

_ZN3OSL8fast_logERKNS_5Dual2IfEE.exit:            ; preds = %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, %130
  %132 = phi float [ %131, %130 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i ]
  %133 = fmul float %97, %132
  %134 = fmul float %98, %132
  %135 = bitcast i8* %r_ to float*
  store float %39, float* %135, align 4, !tbaa !5
  %136 = getelementptr inbounds i8* %r_, i64 4
  %137 = bitcast i8* %136 to float*
  store float %84, float* %137, align 4, !tbaa !7
  %138 = getelementptr inbounds i8* %r_, i64 8
  %139 = bitcast i8* %138 to float*
  store float %129, float* %139, align 4, !tbaa !8
  %140 = getelementptr inbounds i8* %r_, i64 12
  %141 = bitcast i8* %140 to float*
  store float %43, float* %141, align 4, !tbaa !5
  %142 = getelementptr inbounds i8* %r_, i64 16
  %143 = bitcast i8* %142 to float*
  store float %88, float* %143, align 4, !tbaa !7
  %144 = getelementptr inbounds i8* %r_, i64 20
  %145 = bitcast i8* %144 to float*
  store float %133, float* %145, align 4, !tbaa !8
  %146 = getelementptr inbounds i8* %r_, i64 24
  %147 = bitcast i8* %146 to float*
  store float %44, float* %147, align 4, !tbaa !5
  %148 = getelementptr inbounds i8* %r_, i64 28
  %149 = bitcast i8* %148 to float*
  store float %89, float* %149, align 4, !tbaa !7
  %150 = getelementptr inbounds i8* %r_, i64 32
  %151 = bitcast i8* %150 to float*
  store float %134, float* %151, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_log2_ff(float %a) #3 {
  %1 = fcmp olt float %a, 0x3810000000000000
  br i1 %1, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit, label %2

; <label>:2                                       ; preds = %0
  %3 = fcmp ogt float %a, 0x47EFFFFFE0000000
  %4 = bitcast float %a to i32
  %phitmp.i = select i1 %3, i32 2139095039, i32 %4
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit:  ; preds = %0, %2
  %5 = phi i32 [ %phitmp.i, %2 ], [ 8388608, %0 ]
  %6 = lshr i32 %5, 23
  %7 = add nsw i32 %6, -127
  %8 = and i32 %5, 8388607
  %9 = or i32 %8, 1065353216
  %10 = bitcast i32 %9 to float
  %11 = fadd float %10, -1.000000e+00
  %12 = fmul float %11, %11
  %13 = fmul float %12, %12
  %14 = fmul float %11, 0xBF831161A0000000
  %15 = fadd float %14, 0x3FAAA83920000000
  %16 = fmul float %11, 0x3FDEA2C5A0000000
  %17 = fadd float %16, 0xBFE713CA80000000
  %18 = fmul float %11, %15
  %19 = fadd float %18, 0xBFC19A9FA0000000
  %20 = fmul float %11, %19
  %21 = fadd float %20, 0x3FCEF5B7A0000000
  %22 = fmul float %11, %21
  %23 = fadd float %22, 0xBFD63A40C0000000
  %24 = fmul float %11, %17
  %25 = fadd float %24, 0x3FF7154200000000
  %26 = fmul float %13, %23
  %27 = fmul float %11, %25
  %28 = fadd float %27, %26
  %29 = sitofp i32 %7 to float
  %30 = fadd float %29, %28
  ret float %30
}

; Function Attrs: nounwind uwtable
define void @osl_log2_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, 0x3810000000000000
  br i1 %3, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 0x47EFFFFFE0000000
  %6 = bitcast float %2 to i32
  %phitmp.i.i = select i1 %5, i32 2139095039, i32 %6
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i: ; preds = %4, %0
  %7 = phi i32 [ %phitmp.i.i, %4 ], [ 8388608, %0 ]
  %8 = lshr i32 %7, 23
  %9 = add nsw i32 %8, -127
  %10 = and i32 %7, 8388607
  %11 = or i32 %10, 1065353216
  %12 = bitcast i32 %11 to float
  %13 = fadd float %12, -1.000000e+00
  %14 = fmul float %13, %13
  %15 = fmul float %14, %14
  %16 = fmul float %13, 0xBF831161A0000000
  %17 = fadd float %16, 0x3FAAA83920000000
  %18 = fmul float %13, 0x3FDEA2C5A0000000
  %19 = fadd float %18, 0xBFE713CA80000000
  %20 = fmul float %13, %17
  %21 = fadd float %20, 0xBFC19A9FA0000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 0x3FCEF5B7A0000000
  %24 = fmul float %13, %23
  %25 = fadd float %24, 0xBFD63A40C0000000
  %26 = fmul float %13, %19
  %27 = fadd float %26, 0x3FF7154200000000
  %28 = fmul float %15, %25
  %29 = fmul float %13, %27
  %30 = fadd float %29, %28
  %31 = sitofp i32 %9 to float
  %32 = fadd float %31, %30
  %33 = fmul float %2, 0x3FE62E4300000000
  %34 = fcmp olt float %33, 0x3810000000000000
  br i1 %34, label %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit, label %35

; <label>:35                                      ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i
  %36 = fdiv float 1.000000e+00, %33
  br label %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i, %35
  %37 = phi float [ %36, %35 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i ]
  %38 = getelementptr inbounds i8* %a, i64 4
  %39 = bitcast i8* %38 to float*
  %40 = load float* %39, align 4, !tbaa !1
  %41 = fmul float %37, %40
  %42 = getelementptr inbounds i8* %a, i64 8
  %43 = bitcast i8* %42 to float*
  %44 = load float* %43, align 4, !tbaa !1
  %45 = fmul float %37, %44
  %46 = insertelement <2 x float> undef, float %32, i32 0
  %47 = insertelement <2 x float> %46, float %41, i32 1
  %48 = bitcast i8* %r to <2 x float>*
  store <2 x float> %47, <2 x float>* %48, align 4
  %49 = getelementptr inbounds i8* %r, i64 8
  %50 = bitcast i8* %49 to float*
  store float %45, float* %50, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log2_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, 0x3810000000000000
  br i1 %3, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit4, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 0x47EFFFFFE0000000
  %6 = bitcast float %2 to i32
  %phitmp.i3 = select i1 %5, i32 2139095039, i32 %6
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit4

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit4: ; preds = %0, %4
  %7 = phi i32 [ %phitmp.i3, %4 ], [ 8388608, %0 ]
  %8 = lshr i32 %7, 23
  %9 = add nsw i32 %8, -127
  %10 = and i32 %7, 8388607
  %11 = or i32 %10, 1065353216
  %12 = bitcast i32 %11 to float
  %13 = fadd float %12, -1.000000e+00
  %14 = fmul float %13, %13
  %15 = fmul float %14, %14
  %16 = fmul float %13, 0xBF831161A0000000
  %17 = fadd float %16, 0x3FAAA83920000000
  %18 = fmul float %13, 0x3FDEA2C5A0000000
  %19 = fadd float %18, 0xBFE713CA80000000
  %20 = fmul float %13, %17
  %21 = fadd float %20, 0xBFC19A9FA0000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 0x3FCEF5B7A0000000
  %24 = fmul float %13, %23
  %25 = fadd float %24, 0xBFD63A40C0000000
  %26 = fmul float %13, %19
  %27 = fadd float %26, 0x3FF7154200000000
  %28 = fmul float %15, %25
  %29 = fmul float %13, %27
  %30 = fadd float %29, %28
  %31 = sitofp i32 %9 to float
  %32 = fadd float %31, %30
  %33 = bitcast i8* %r_ to float*
  store float %32, float* %33, align 4, !tbaa !1
  %34 = getelementptr inbounds i8* %a_, i64 4
  %35 = bitcast i8* %34 to float*
  %36 = load float* %35, align 4, !tbaa !1
  %37 = fcmp olt float %36, 0x3810000000000000
  br i1 %37, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit2, label %38

; <label>:38                                      ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit4
  %39 = fcmp ogt float %36, 0x47EFFFFFE0000000
  %40 = bitcast float %36 to i32
  %phitmp.i1 = select i1 %39, i32 2139095039, i32 %40
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit2

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit2: ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit4, %38
  %41 = phi i32 [ %phitmp.i1, %38 ], [ 8388608, %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit4 ]
  %42 = lshr i32 %41, 23
  %43 = add nsw i32 %42, -127
  %44 = and i32 %41, 8388607
  %45 = or i32 %44, 1065353216
  %46 = bitcast i32 %45 to float
  %47 = fadd float %46, -1.000000e+00
  %48 = fmul float %47, %47
  %49 = fmul float %48, %48
  %50 = fmul float %47, 0xBF831161A0000000
  %51 = fadd float %50, 0x3FAAA83920000000
  %52 = fmul float %47, 0x3FDEA2C5A0000000
  %53 = fadd float %52, 0xBFE713CA80000000
  %54 = fmul float %47, %51
  %55 = fadd float %54, 0xBFC19A9FA0000000
  %56 = fmul float %47, %55
  %57 = fadd float %56, 0x3FCEF5B7A0000000
  %58 = fmul float %47, %57
  %59 = fadd float %58, 0xBFD63A40C0000000
  %60 = fmul float %47, %53
  %61 = fadd float %60, 0x3FF7154200000000
  %62 = fmul float %49, %59
  %63 = fmul float %47, %61
  %64 = fadd float %63, %62
  %65 = sitofp i32 %43 to float
  %66 = fadd float %65, %64
  %67 = getelementptr inbounds i8* %r_, i64 4
  %68 = bitcast i8* %67 to float*
  store float %66, float* %68, align 4, !tbaa !1
  %69 = getelementptr inbounds i8* %a_, i64 8
  %70 = bitcast i8* %69 to float*
  %71 = load float* %70, align 4, !tbaa !1
  %72 = fcmp olt float %71, 0x3810000000000000
  br i1 %72, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit, label %73

; <label>:73                                      ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit2
  %74 = fcmp ogt float %71, 0x47EFFFFFE0000000
  %75 = bitcast float %71 to i32
  %phitmp.i = select i1 %74, i32 2139095039, i32 %75
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit:  ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit2, %73
  %76 = phi i32 [ %phitmp.i, %73 ], [ 8388608, %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit2 ]
  %77 = lshr i32 %76, 23
  %78 = add nsw i32 %77, -127
  %79 = and i32 %76, 8388607
  %80 = or i32 %79, 1065353216
  %81 = bitcast i32 %80 to float
  %82 = fadd float %81, -1.000000e+00
  %83 = fmul float %82, %82
  %84 = fmul float %83, %83
  %85 = fmul float %82, 0xBF831161A0000000
  %86 = fadd float %85, 0x3FAAA83920000000
  %87 = fmul float %82, 0x3FDEA2C5A0000000
  %88 = fadd float %87, 0xBFE713CA80000000
  %89 = fmul float %82, %86
  %90 = fadd float %89, 0xBFC19A9FA0000000
  %91 = fmul float %82, %90
  %92 = fadd float %91, 0x3FCEF5B7A0000000
  %93 = fmul float %82, %92
  %94 = fadd float %93, 0xBFD63A40C0000000
  %95 = fmul float %82, %88
  %96 = fadd float %95, 0x3FF7154200000000
  %97 = fmul float %84, %94
  %98 = fmul float %82, %96
  %99 = fadd float %98, %97
  %100 = sitofp i32 %78 to float
  %101 = fadd float %100, %99
  %102 = getelementptr inbounds i8* %r_, i64 8
  %103 = bitcast i8* %102 to float*
  store float %101, float* %103, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log2_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = fcmp olt float %6, 0x3810000000000000
  br i1 %9, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i15, label %10

; <label>:10                                      ; preds = %0
  %11 = fcmp ogt float %6, 0x47EFFFFFE0000000
  %12 = bitcast float %6 to i32
  %phitmp.i.i14 = select i1 %11, i32 2139095039, i32 %12
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i15

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i15: ; preds = %10, %0
  %13 = phi i32 [ %phitmp.i.i14, %10 ], [ 8388608, %0 ]
  %14 = lshr i32 %13, 23
  %15 = add nsw i32 %14, -127
  %16 = and i32 %13, 8388607
  %17 = or i32 %16, 1065353216
  %18 = bitcast i32 %17 to float
  %19 = fadd float %18, -1.000000e+00
  %20 = fmul float %19, %19
  %21 = fmul float %20, %20
  %22 = fmul float %19, 0xBF831161A0000000
  %23 = fadd float %22, 0x3FAAA83920000000
  %24 = fmul float %19, 0x3FDEA2C5A0000000
  %25 = fadd float %24, 0xBFE713CA80000000
  %26 = fmul float %19, %23
  %27 = fadd float %26, 0xBFC19A9FA0000000
  %28 = fmul float %19, %27
  %29 = fadd float %28, 0x3FCEF5B7A0000000
  %30 = fmul float %19, %29
  %31 = fadd float %30, 0xBFD63A40C0000000
  %32 = fmul float %19, %25
  %33 = fadd float %32, 0x3FF7154200000000
  %34 = fmul float %21, %31
  %35 = fmul float %19, %33
  %36 = fadd float %35, %34
  %37 = sitofp i32 %15 to float
  %38 = fadd float %37, %36
  %39 = fmul float %6, 0x3FE62E4300000000
  %40 = fcmp olt float %39, 0x3810000000000000
  br i1 %40, label %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit16, label %41

; <label>:41                                      ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i15
  %42 = fdiv float 1.000000e+00, %39
  br label %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit16

_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit16:         ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i15, %41
  %43 = phi float [ %42, %41 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i15 ]
  %44 = fmul float %7, %43
  %45 = fmul float %8, %43
  %46 = getelementptr inbounds i8* %a_, i64 4
  %47 = bitcast i8* %46 to float*
  %48 = getelementptr inbounds i8* %a_, i64 16
  %49 = bitcast i8* %48 to float*
  %50 = getelementptr inbounds i8* %a_, i64 28
  %51 = bitcast i8* %50 to float*
  %52 = load float* %47, align 4, !tbaa !1
  %53 = load float* %49, align 4, !tbaa !1
  %54 = load float* %51, align 4, !tbaa !1
  %55 = fcmp olt float %52, 0x3810000000000000
  br i1 %55, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i12, label %56

; <label>:56                                      ; preds = %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit16
  %57 = fcmp ogt float %52, 0x47EFFFFFE0000000
  %58 = bitcast float %52 to i32
  %phitmp.i.i11 = select i1 %57, i32 2139095039, i32 %58
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i12

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i12: ; preds = %56, %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit16
  %59 = phi i32 [ %phitmp.i.i11, %56 ], [ 8388608, %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit16 ]
  %60 = lshr i32 %59, 23
  %61 = add nsw i32 %60, -127
  %62 = and i32 %59, 8388607
  %63 = or i32 %62, 1065353216
  %64 = bitcast i32 %63 to float
  %65 = fadd float %64, -1.000000e+00
  %66 = fmul float %65, %65
  %67 = fmul float %66, %66
  %68 = fmul float %65, 0xBF831161A0000000
  %69 = fadd float %68, 0x3FAAA83920000000
  %70 = fmul float %65, 0x3FDEA2C5A0000000
  %71 = fadd float %70, 0xBFE713CA80000000
  %72 = fmul float %65, %69
  %73 = fadd float %72, 0xBFC19A9FA0000000
  %74 = fmul float %65, %73
  %75 = fadd float %74, 0x3FCEF5B7A0000000
  %76 = fmul float %65, %75
  %77 = fadd float %76, 0xBFD63A40C0000000
  %78 = fmul float %65, %71
  %79 = fadd float %78, 0x3FF7154200000000
  %80 = fmul float %67, %77
  %81 = fmul float %65, %79
  %82 = fadd float %81, %80
  %83 = sitofp i32 %61 to float
  %84 = fadd float %83, %82
  %85 = fmul float %52, 0x3FE62E4300000000
  %86 = fcmp olt float %85, 0x3810000000000000
  br i1 %86, label %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit13, label %87

; <label>:87                                      ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i12
  %88 = fdiv float 1.000000e+00, %85
  br label %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit13

_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit13:         ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i12, %87
  %89 = phi float [ %88, %87 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i12 ]
  %90 = fmul float %53, %89
  %91 = fmul float %54, %89
  %92 = getelementptr inbounds i8* %a_, i64 8
  %93 = bitcast i8* %92 to float*
  %94 = getelementptr inbounds i8* %a_, i64 20
  %95 = bitcast i8* %94 to float*
  %96 = getelementptr inbounds i8* %a_, i64 32
  %97 = bitcast i8* %96 to float*
  %98 = load float* %93, align 4, !tbaa !1
  %99 = load float* %95, align 4, !tbaa !1
  %100 = load float* %97, align 4, !tbaa !1
  %101 = fcmp olt float %98, 0x3810000000000000
  br i1 %101, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i, label %102

; <label>:102                                     ; preds = %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit13
  %103 = fcmp ogt float %98, 0x47EFFFFFE0000000
  %104 = bitcast float %98 to i32
  %phitmp.i.i = select i1 %103, i32 2139095039, i32 %104
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i: ; preds = %102, %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit13
  %105 = phi i32 [ %phitmp.i.i, %102 ], [ 8388608, %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit13 ]
  %106 = lshr i32 %105, 23
  %107 = add nsw i32 %106, -127
  %108 = and i32 %105, 8388607
  %109 = or i32 %108, 1065353216
  %110 = bitcast i32 %109 to float
  %111 = fadd float %110, -1.000000e+00
  %112 = fmul float %111, %111
  %113 = fmul float %112, %112
  %114 = fmul float %111, 0xBF831161A0000000
  %115 = fadd float %114, 0x3FAAA83920000000
  %116 = fmul float %111, 0x3FDEA2C5A0000000
  %117 = fadd float %116, 0xBFE713CA80000000
  %118 = fmul float %111, %115
  %119 = fadd float %118, 0xBFC19A9FA0000000
  %120 = fmul float %111, %119
  %121 = fadd float %120, 0x3FCEF5B7A0000000
  %122 = fmul float %111, %121
  %123 = fadd float %122, 0xBFD63A40C0000000
  %124 = fmul float %111, %117
  %125 = fadd float %124, 0x3FF7154200000000
  %126 = fmul float %113, %123
  %127 = fmul float %111, %125
  %128 = fadd float %127, %126
  %129 = sitofp i32 %107 to float
  %130 = fadd float %129, %128
  %131 = fmul float %98, 0x3FE62E4300000000
  %132 = fcmp olt float %131, 0x3810000000000000
  br i1 %132, label %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit, label %133

; <label>:133                                     ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i
  %134 = fdiv float 1.000000e+00, %131
  br label %_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_log2ERKNS_5Dual2IfEE.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i, %133
  %135 = phi float [ %134, %133 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit.i ]
  %136 = fmul float %99, %135
  %137 = fmul float %100, %135
  %138 = bitcast i8* %r_ to float*
  store float %38, float* %138, align 4, !tbaa !5
  %139 = getelementptr inbounds i8* %r_, i64 4
  %140 = bitcast i8* %139 to float*
  store float %84, float* %140, align 4, !tbaa !7
  %141 = getelementptr inbounds i8* %r_, i64 8
  %142 = bitcast i8* %141 to float*
  store float %130, float* %142, align 4, !tbaa !8
  %143 = getelementptr inbounds i8* %r_, i64 12
  %144 = bitcast i8* %143 to float*
  store float %44, float* %144, align 4, !tbaa !5
  %145 = getelementptr inbounds i8* %r_, i64 16
  %146 = bitcast i8* %145 to float*
  store float %90, float* %146, align 4, !tbaa !7
  %147 = getelementptr inbounds i8* %r_, i64 20
  %148 = bitcast i8* %147 to float*
  store float %136, float* %148, align 4, !tbaa !8
  %149 = getelementptr inbounds i8* %r_, i64 24
  %150 = bitcast i8* %149 to float*
  store float %45, float* %150, align 4, !tbaa !5
  %151 = getelementptr inbounds i8* %r_, i64 28
  %152 = bitcast i8* %151 to float*
  store float %91, float* %152, align 4, !tbaa !7
  %153 = getelementptr inbounds i8* %r_, i64 32
  %154 = bitcast i8* %153 to float*
  store float %137, float* %154, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_log10_ff(float %a) #3 {
  %1 = fcmp olt float %a, 0x3810000000000000
  br i1 %1, label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit, label %2

; <label>:2                                       ; preds = %0
  %3 = fcmp ogt float %a, 0x47EFFFFFE0000000
  %4 = bitcast float %a to i32
  %phitmp.i.i = select i1 %3, i32 2139095039, i32 %4
  br label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit

_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit: ; preds = %0, %2
  %5 = phi i32 [ %phitmp.i.i, %2 ], [ 8388608, %0 ]
  %6 = lshr i32 %5, 23
  %7 = add nsw i32 %6, -127
  %8 = and i32 %5, 8388607
  %9 = or i32 %8, 1065353216
  %10 = bitcast i32 %9 to float
  %11 = fadd float %10, -1.000000e+00
  %12 = fmul float %11, %11
  %13 = fmul float %12, %12
  %14 = fmul float %11, 0xBF831161A0000000
  %15 = fadd float %14, 0x3FAAA83920000000
  %16 = fmul float %11, 0x3FDEA2C5A0000000
  %17 = fadd float %16, 0xBFE713CA80000000
  %18 = fmul float %11, %15
  %19 = fadd float %18, 0xBFC19A9FA0000000
  %20 = fmul float %11, %19
  %21 = fadd float %20, 0x3FCEF5B7A0000000
  %22 = fmul float %11, %21
  %23 = fadd float %22, 0xBFD63A40C0000000
  %24 = fmul float %11, %17
  %25 = fadd float %24, 0x3FF7154200000000
  %26 = fmul float %13, %23
  %27 = fmul float %11, %25
  %28 = fadd float %27, %26
  %29 = sitofp i32 %7 to float
  %30 = fadd float %29, %28
  %31 = fmul float %30, 0x3FD3441360000000
  ret float %31
}

; Function Attrs: nounwind uwtable
define void @osl_log10_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, 0x3810000000000000
  br i1 %3, label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 0x47EFFFFFE0000000
  %6 = bitcast float %2 to i32
  %phitmp.i.i.i = select i1 %5, i32 2139095039, i32 %6
  br label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i: ; preds = %4, %0
  %7 = phi i32 [ %phitmp.i.i.i, %4 ], [ 8388608, %0 ]
  %8 = lshr i32 %7, 23
  %9 = add nsw i32 %8, -127
  %10 = and i32 %7, 8388607
  %11 = or i32 %10, 1065353216
  %12 = bitcast i32 %11 to float
  %13 = fadd float %12, -1.000000e+00
  %14 = fmul float %13, %13
  %15 = fmul float %14, %14
  %16 = fmul float %13, 0xBF831161A0000000
  %17 = fadd float %16, 0x3FAAA83920000000
  %18 = fmul float %13, 0x3FDEA2C5A0000000
  %19 = fadd float %18, 0xBFE713CA80000000
  %20 = fmul float %13, %17
  %21 = fadd float %20, 0xBFC19A9FA0000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 0x3FCEF5B7A0000000
  %24 = fmul float %13, %23
  %25 = fadd float %24, 0xBFD63A40C0000000
  %26 = fmul float %13, %19
  %27 = fadd float %26, 0x3FF7154200000000
  %28 = fmul float %15, %25
  %29 = fmul float %13, %27
  %30 = fadd float %29, %28
  %31 = sitofp i32 %9 to float
  %32 = fadd float %31, %30
  %33 = fmul float %32, 0x3FD3441360000000
  %34 = fmul float %2, 0x40026BB1C0000000
  %35 = fcmp olt float %34, 0x3810000000000000
  br i1 %35, label %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit, label %36

; <label>:36                                      ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i
  %37 = fdiv float 1.000000e+00, %34
  br label %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit

_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit:         ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i, %36
  %38 = phi float [ %37, %36 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i ]
  %39 = getelementptr inbounds i8* %a, i64 4
  %40 = bitcast i8* %39 to float*
  %41 = load float* %40, align 4, !tbaa !1
  %42 = fmul float %38, %41
  %43 = getelementptr inbounds i8* %a, i64 8
  %44 = bitcast i8* %43 to float*
  %45 = load float* %44, align 4, !tbaa !1
  %46 = fmul float %38, %45
  %47 = insertelement <2 x float> undef, float %33, i32 0
  %48 = insertelement <2 x float> %47, float %42, i32 1
  %49 = bitcast i8* %r to <2 x float>*
  store <2 x float> %48, <2 x float>* %49, align 4
  %50 = getelementptr inbounds i8* %r, i64 8
  %51 = bitcast i8* %50 to float*
  store float %46, float* %51, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log10_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, 0x3810000000000000
  br i1 %3, label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit4, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 0x47EFFFFFE0000000
  %6 = bitcast float %2 to i32
  %phitmp.i.i3 = select i1 %5, i32 2139095039, i32 %6
  br label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit4

_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit4: ; preds = %0, %4
  %7 = phi i32 [ %phitmp.i.i3, %4 ], [ 8388608, %0 ]
  %8 = lshr i32 %7, 23
  %9 = add nsw i32 %8, -127
  %10 = and i32 %7, 8388607
  %11 = or i32 %10, 1065353216
  %12 = bitcast i32 %11 to float
  %13 = fadd float %12, -1.000000e+00
  %14 = fmul float %13, %13
  %15 = fmul float %14, %14
  %16 = fmul float %13, 0xBF831161A0000000
  %17 = fadd float %16, 0x3FAAA83920000000
  %18 = fmul float %13, 0x3FDEA2C5A0000000
  %19 = fadd float %18, 0xBFE713CA80000000
  %20 = fmul float %13, %17
  %21 = fadd float %20, 0xBFC19A9FA0000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 0x3FCEF5B7A0000000
  %24 = fmul float %13, %23
  %25 = fadd float %24, 0xBFD63A40C0000000
  %26 = fmul float %13, %19
  %27 = fadd float %26, 0x3FF7154200000000
  %28 = fmul float %15, %25
  %29 = fmul float %13, %27
  %30 = fadd float %29, %28
  %31 = sitofp i32 %9 to float
  %32 = fadd float %31, %30
  %33 = fmul float %32, 0x3FD3441360000000
  %34 = bitcast i8* %r_ to float*
  store float %33, float* %34, align 4, !tbaa !1
  %35 = getelementptr inbounds i8* %a_, i64 4
  %36 = bitcast i8* %35 to float*
  %37 = load float* %36, align 4, !tbaa !1
  %38 = fcmp olt float %37, 0x3810000000000000
  br i1 %38, label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit2, label %39

; <label>:39                                      ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit4
  %40 = fcmp ogt float %37, 0x47EFFFFFE0000000
  %41 = bitcast float %37 to i32
  %phitmp.i.i1 = select i1 %40, i32 2139095039, i32 %41
  br label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit2

_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit2: ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit4, %39
  %42 = phi i32 [ %phitmp.i.i1, %39 ], [ 8388608, %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit4 ]
  %43 = lshr i32 %42, 23
  %44 = add nsw i32 %43, -127
  %45 = and i32 %42, 8388607
  %46 = or i32 %45, 1065353216
  %47 = bitcast i32 %46 to float
  %48 = fadd float %47, -1.000000e+00
  %49 = fmul float %48, %48
  %50 = fmul float %49, %49
  %51 = fmul float %48, 0xBF831161A0000000
  %52 = fadd float %51, 0x3FAAA83920000000
  %53 = fmul float %48, 0x3FDEA2C5A0000000
  %54 = fadd float %53, 0xBFE713CA80000000
  %55 = fmul float %48, %52
  %56 = fadd float %55, 0xBFC19A9FA0000000
  %57 = fmul float %48, %56
  %58 = fadd float %57, 0x3FCEF5B7A0000000
  %59 = fmul float %48, %58
  %60 = fadd float %59, 0xBFD63A40C0000000
  %61 = fmul float %48, %54
  %62 = fadd float %61, 0x3FF7154200000000
  %63 = fmul float %50, %60
  %64 = fmul float %48, %62
  %65 = fadd float %64, %63
  %66 = sitofp i32 %44 to float
  %67 = fadd float %66, %65
  %68 = fmul float %67, 0x3FD3441360000000
  %69 = getelementptr inbounds i8* %r_, i64 4
  %70 = bitcast i8* %69 to float*
  store float %68, float* %70, align 4, !tbaa !1
  %71 = getelementptr inbounds i8* %a_, i64 8
  %72 = bitcast i8* %71 to float*
  %73 = load float* %72, align 4, !tbaa !1
  %74 = fcmp olt float %73, 0x3810000000000000
  br i1 %74, label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit, label %75

; <label>:75                                      ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit2
  %76 = fcmp ogt float %73, 0x47EFFFFFE0000000
  %77 = bitcast float %73 to i32
  %phitmp.i.i = select i1 %76, i32 2139095039, i32 %77
  br label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit

_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit: ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit2, %75
  %78 = phi i32 [ %phitmp.i.i, %75 ], [ 8388608, %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit2 ]
  %79 = lshr i32 %78, 23
  %80 = add nsw i32 %79, -127
  %81 = and i32 %78, 8388607
  %82 = or i32 %81, 1065353216
  %83 = bitcast i32 %82 to float
  %84 = fadd float %83, -1.000000e+00
  %85 = fmul float %84, %84
  %86 = fmul float %85, %85
  %87 = fmul float %84, 0xBF831161A0000000
  %88 = fadd float %87, 0x3FAAA83920000000
  %89 = fmul float %84, 0x3FDEA2C5A0000000
  %90 = fadd float %89, 0xBFE713CA80000000
  %91 = fmul float %84, %88
  %92 = fadd float %91, 0xBFC19A9FA0000000
  %93 = fmul float %84, %92
  %94 = fadd float %93, 0x3FCEF5B7A0000000
  %95 = fmul float %84, %94
  %96 = fadd float %95, 0xBFD63A40C0000000
  %97 = fmul float %84, %90
  %98 = fadd float %97, 0x3FF7154200000000
  %99 = fmul float %86, %96
  %100 = fmul float %84, %98
  %101 = fadd float %100, %99
  %102 = sitofp i32 %80 to float
  %103 = fadd float %102, %101
  %104 = fmul float %103, 0x3FD3441360000000
  %105 = getelementptr inbounds i8* %r_, i64 8
  %106 = bitcast i8* %105 to float*
  store float %104, float* %106, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log10_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = fcmp olt float %6, 0x3810000000000000
  br i1 %9, label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i15, label %10

; <label>:10                                      ; preds = %0
  %11 = fcmp ogt float %6, 0x47EFFFFFE0000000
  %12 = bitcast float %6 to i32
  %phitmp.i.i.i14 = select i1 %11, i32 2139095039, i32 %12
  br label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i15

_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i15: ; preds = %10, %0
  %13 = phi i32 [ %phitmp.i.i.i14, %10 ], [ 8388608, %0 ]
  %14 = lshr i32 %13, 23
  %15 = add nsw i32 %14, -127
  %16 = and i32 %13, 8388607
  %17 = or i32 %16, 1065353216
  %18 = bitcast i32 %17 to float
  %19 = fadd float %18, -1.000000e+00
  %20 = fmul float %19, %19
  %21 = fmul float %20, %20
  %22 = fmul float %19, 0xBF831161A0000000
  %23 = fadd float %22, 0x3FAAA83920000000
  %24 = fmul float %19, 0x3FDEA2C5A0000000
  %25 = fadd float %24, 0xBFE713CA80000000
  %26 = fmul float %19, %23
  %27 = fadd float %26, 0xBFC19A9FA0000000
  %28 = fmul float %19, %27
  %29 = fadd float %28, 0x3FCEF5B7A0000000
  %30 = fmul float %19, %29
  %31 = fadd float %30, 0xBFD63A40C0000000
  %32 = fmul float %19, %25
  %33 = fadd float %32, 0x3FF7154200000000
  %34 = fmul float %21, %31
  %35 = fmul float %19, %33
  %36 = fadd float %35, %34
  %37 = sitofp i32 %15 to float
  %38 = fadd float %37, %36
  %39 = fmul float %38, 0x3FD3441360000000
  %40 = fmul float %6, 0x40026BB1C0000000
  %41 = fcmp olt float %40, 0x3810000000000000
  br i1 %41, label %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit16, label %42

; <label>:42                                      ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i15
  %43 = fdiv float 1.000000e+00, %40
  br label %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit16

_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit16:       ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i15, %42
  %44 = phi float [ %43, %42 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i15 ]
  %45 = fmul float %7, %44
  %46 = fmul float %8, %44
  %47 = getelementptr inbounds i8* %a_, i64 4
  %48 = bitcast i8* %47 to float*
  %49 = getelementptr inbounds i8* %a_, i64 16
  %50 = bitcast i8* %49 to float*
  %51 = getelementptr inbounds i8* %a_, i64 28
  %52 = bitcast i8* %51 to float*
  %53 = load float* %48, align 4, !tbaa !1
  %54 = load float* %50, align 4, !tbaa !1
  %55 = load float* %52, align 4, !tbaa !1
  %56 = fcmp olt float %53, 0x3810000000000000
  br i1 %56, label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i12, label %57

; <label>:57                                      ; preds = %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit16
  %58 = fcmp ogt float %53, 0x47EFFFFFE0000000
  %59 = bitcast float %53 to i32
  %phitmp.i.i.i11 = select i1 %58, i32 2139095039, i32 %59
  br label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i12

_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i12: ; preds = %57, %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit16
  %60 = phi i32 [ %phitmp.i.i.i11, %57 ], [ 8388608, %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit16 ]
  %61 = lshr i32 %60, 23
  %62 = add nsw i32 %61, -127
  %63 = and i32 %60, 8388607
  %64 = or i32 %63, 1065353216
  %65 = bitcast i32 %64 to float
  %66 = fadd float %65, -1.000000e+00
  %67 = fmul float %66, %66
  %68 = fmul float %67, %67
  %69 = fmul float %66, 0xBF831161A0000000
  %70 = fadd float %69, 0x3FAAA83920000000
  %71 = fmul float %66, 0x3FDEA2C5A0000000
  %72 = fadd float %71, 0xBFE713CA80000000
  %73 = fmul float %66, %70
  %74 = fadd float %73, 0xBFC19A9FA0000000
  %75 = fmul float %66, %74
  %76 = fadd float %75, 0x3FCEF5B7A0000000
  %77 = fmul float %66, %76
  %78 = fadd float %77, 0xBFD63A40C0000000
  %79 = fmul float %66, %72
  %80 = fadd float %79, 0x3FF7154200000000
  %81 = fmul float %68, %78
  %82 = fmul float %66, %80
  %83 = fadd float %82, %81
  %84 = sitofp i32 %62 to float
  %85 = fadd float %84, %83
  %86 = fmul float %85, 0x3FD3441360000000
  %87 = fmul float %53, 0x40026BB1C0000000
  %88 = fcmp olt float %87, 0x3810000000000000
  br i1 %88, label %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit13, label %89

; <label>:89                                      ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i12
  %90 = fdiv float 1.000000e+00, %87
  br label %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit13

_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit13:       ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i12, %89
  %91 = phi float [ %90, %89 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i12 ]
  %92 = fmul float %54, %91
  %93 = fmul float %55, %91
  %94 = getelementptr inbounds i8* %a_, i64 8
  %95 = bitcast i8* %94 to float*
  %96 = getelementptr inbounds i8* %a_, i64 20
  %97 = bitcast i8* %96 to float*
  %98 = getelementptr inbounds i8* %a_, i64 32
  %99 = bitcast i8* %98 to float*
  %100 = load float* %95, align 4, !tbaa !1
  %101 = load float* %97, align 4, !tbaa !1
  %102 = load float* %99, align 4, !tbaa !1
  %103 = fcmp olt float %100, 0x3810000000000000
  br i1 %103, label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i, label %104

; <label>:104                                     ; preds = %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit13
  %105 = fcmp ogt float %100, 0x47EFFFFFE0000000
  %106 = bitcast float %100 to i32
  %phitmp.i.i.i = select i1 %105, i32 2139095039, i32 %106
  br label %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i: ; preds = %104, %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit13
  %107 = phi i32 [ %phitmp.i.i.i, %104 ], [ 8388608, %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit13 ]
  %108 = lshr i32 %107, 23
  %109 = add nsw i32 %108, -127
  %110 = and i32 %107, 8388607
  %111 = or i32 %110, 1065353216
  %112 = bitcast i32 %111 to float
  %113 = fadd float %112, -1.000000e+00
  %114 = fmul float %113, %113
  %115 = fmul float %114, %114
  %116 = fmul float %113, 0xBF831161A0000000
  %117 = fadd float %116, 0x3FAAA83920000000
  %118 = fmul float %113, 0x3FDEA2C5A0000000
  %119 = fadd float %118, 0xBFE713CA80000000
  %120 = fmul float %113, %117
  %121 = fadd float %120, 0xBFC19A9FA0000000
  %122 = fmul float %113, %121
  %123 = fadd float %122, 0x3FCEF5B7A0000000
  %124 = fmul float %113, %123
  %125 = fadd float %124, 0xBFD63A40C0000000
  %126 = fmul float %113, %119
  %127 = fadd float %126, 0x3FF7154200000000
  %128 = fmul float %115, %125
  %129 = fmul float %113, %127
  %130 = fadd float %129, %128
  %131 = sitofp i32 %109 to float
  %132 = fadd float %131, %130
  %133 = fmul float %132, 0x3FD3441360000000
  %134 = fmul float %100, 0x40026BB1C0000000
  %135 = fcmp olt float %134, 0x3810000000000000
  br i1 %135, label %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit, label %136

; <label>:136                                     ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i
  %137 = fdiv float 1.000000e+00, %134
  br label %_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit

_ZN3OSL10fast_log10ERKNS_5Dual2IfEE.exit:         ; preds = %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i, %136
  %138 = phi float [ %137, %136 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_710fast_log10IfEET_RKS2_.exit.i ]
  %139 = fmul float %101, %138
  %140 = fmul float %102, %138
  %141 = bitcast i8* %r_ to float*
  store float %39, float* %141, align 4, !tbaa !5
  %142 = getelementptr inbounds i8* %r_, i64 4
  %143 = bitcast i8* %142 to float*
  store float %86, float* %143, align 4, !tbaa !7
  %144 = getelementptr inbounds i8* %r_, i64 8
  %145 = bitcast i8* %144 to float*
  store float %133, float* %145, align 4, !tbaa !8
  %146 = getelementptr inbounds i8* %r_, i64 12
  %147 = bitcast i8* %146 to float*
  store float %45, float* %147, align 4, !tbaa !5
  %148 = getelementptr inbounds i8* %r_, i64 16
  %149 = bitcast i8* %148 to float*
  store float %92, float* %149, align 4, !tbaa !7
  %150 = getelementptr inbounds i8* %r_, i64 20
  %151 = bitcast i8* %150 to float*
  store float %139, float* %151, align 4, !tbaa !8
  %152 = getelementptr inbounds i8* %r_, i64 24
  %153 = bitcast i8* %152 to float*
  store float %46, float* %153, align 4, !tbaa !5
  %154 = getelementptr inbounds i8* %r_, i64 28
  %155 = bitcast i8* %154 to float*
  store float %93, float* %155, align 4, !tbaa !7
  %156 = getelementptr inbounds i8* %r_, i64 32
  %157 = bitcast i8* %156 to float*
  store float %140, float* %157, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_exp_ff(float %a) #3 {
  %1 = fmul float %a, 0x3FF7154760000000
  %2 = fcmp olt float %1, -1.260000e+02
  br i1 %2, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit, label %3

; <label>:3                                       ; preds = %0
  %4 = fcmp ogt float %1, 1.260000e+02
  %5 = select i1 %4, float 1.260000e+02, float %1
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit:   ; preds = %0, %3
  %6 = phi float [ %5, %3 ], [ -1.260000e+02, %0 ]
  %7 = fptosi float %6 to i32
  %8 = sitofp i32 %7 to float
  %9 = fsub float %6, %8
  %10 = fsub float 1.000000e+00, %9
  %11 = fsub float 1.000000e+00, %10
  %12 = fmul float %11, 0x3F55D889C0000000
  %13 = fadd float %12, 0x3F84177340000000
  %14 = fmul float %11, %13
  %15 = fadd float %14, 0x3FAC6CE660000000
  %16 = fmul float %11, %15
  %17 = fadd float %16, 0x3FCEBE3240000000
  %18 = fmul float %11, %17
  %19 = fadd float %18, 0x3FE62E3E20000000
  %20 = fmul float %11, %19
  %21 = fadd float %20, 1.000000e+00
  %22 = bitcast float %21 to i32
  %23 = shl i32 %7, 23
  %24 = add i32 %22, %23
  %25 = bitcast i32 %24 to float
  ret float %25
}

; Function Attrs: nounwind uwtable
define void @osl_exp_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FF7154760000000
  %4 = fcmp olt float %3, -1.260000e+02
  br i1 %4, label %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit, label %5

; <label>:5                                       ; preds = %0
  %6 = fcmp ogt float %3, 1.260000e+02
  %7 = select i1 %6, float 1.260000e+02, float %3
  br label %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit

_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit:            ; preds = %0, %5
  %8 = phi float [ %7, %5 ], [ -1.260000e+02, %0 ]
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fsub float %8, %10
  %12 = fsub float 1.000000e+00, %11
  %13 = fsub float 1.000000e+00, %12
  %14 = fmul float %13, 0x3F55D889C0000000
  %15 = fadd float %14, 0x3F84177340000000
  %16 = fmul float %13, %15
  %17 = fadd float %16, 0x3FAC6CE660000000
  %18 = fmul float %13, %17
  %19 = fadd float %18, 0x3FCEBE3240000000
  %20 = fmul float %13, %19
  %21 = fadd float %20, 0x3FE62E3E20000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 1.000000e+00
  %24 = bitcast float %23 to i32
  %25 = shl i32 %9, 23
  %26 = add i32 %24, %25
  %27 = bitcast i32 %26 to float
  %28 = getelementptr inbounds i8* %a, i64 4
  %29 = bitcast i8* %28 to float*
  %30 = load float* %29, align 4, !tbaa !1
  %31 = fmul float %30, %27
  %32 = getelementptr inbounds i8* %a, i64 8
  %33 = bitcast i8* %32 to float*
  %34 = load float* %33, align 4, !tbaa !1
  %35 = fmul float %34, %27
  %36 = insertelement <2 x float> undef, float %27, i32 0
  %37 = insertelement <2 x float> %36, float %31, i32 1
  %38 = bitcast i8* %r to <2 x float>*
  store <2 x float> %37, <2 x float>* %38, align 4
  %39 = getelementptr inbounds i8* %r, i64 8
  %40 = bitcast i8* %39 to float*
  store float %35, float* %40, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_exp_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fmul float %2, 0x3FF7154760000000
  %4 = fcmp olt float %3, -1.260000e+02
  br i1 %4, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit2, label %5

; <label>:5                                       ; preds = %0
  %6 = fcmp ogt float %3, 1.260000e+02
  %7 = select i1 %6, float 1.260000e+02, float %3
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit2

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit2:  ; preds = %0, %5
  %8 = phi float [ %7, %5 ], [ -1.260000e+02, %0 ]
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fsub float %8, %10
  %12 = fsub float 1.000000e+00, %11
  %13 = fsub float 1.000000e+00, %12
  %14 = fmul float %13, 0x3F55D889C0000000
  %15 = fadd float %14, 0x3F84177340000000
  %16 = fmul float %13, %15
  %17 = fadd float %16, 0x3FAC6CE660000000
  %18 = fmul float %13, %17
  %19 = fadd float %18, 0x3FCEBE3240000000
  %20 = fmul float %13, %19
  %21 = fadd float %20, 0x3FE62E3E20000000
  %22 = fmul float %13, %21
  %23 = fadd float %22, 1.000000e+00
  %24 = bitcast float %23 to i32
  %25 = shl i32 %9, 23
  %26 = add i32 %24, %25
  %27 = bitcast i32 %26 to float
  %28 = bitcast i8* %r_ to float*
  store float %27, float* %28, align 4, !tbaa !1
  %29 = getelementptr inbounds i8* %a_, i64 4
  %30 = bitcast i8* %29 to float*
  %31 = load float* %30, align 4, !tbaa !1
  %32 = fmul float %31, 0x3FF7154760000000
  %33 = fcmp olt float %32, -1.260000e+02
  br i1 %33, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit1, label %34

; <label>:34                                      ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit2
  %35 = fcmp ogt float %32, 1.260000e+02
  %36 = select i1 %35, float 1.260000e+02, float %32
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit1

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit1:  ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit2, %34
  %37 = phi float [ %36, %34 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit2 ]
  %38 = fptosi float %37 to i32
  %39 = sitofp i32 %38 to float
  %40 = fsub float %37, %39
  %41 = fsub float 1.000000e+00, %40
  %42 = fsub float 1.000000e+00, %41
  %43 = fmul float %42, 0x3F55D889C0000000
  %44 = fadd float %43, 0x3F84177340000000
  %45 = fmul float %42, %44
  %46 = fadd float %45, 0x3FAC6CE660000000
  %47 = fmul float %42, %46
  %48 = fadd float %47, 0x3FCEBE3240000000
  %49 = fmul float %42, %48
  %50 = fadd float %49, 0x3FE62E3E20000000
  %51 = fmul float %42, %50
  %52 = fadd float %51, 1.000000e+00
  %53 = bitcast float %52 to i32
  %54 = shl i32 %38, 23
  %55 = add i32 %53, %54
  %56 = bitcast i32 %55 to float
  %57 = getelementptr inbounds i8* %r_, i64 4
  %58 = bitcast i8* %57 to float*
  store float %56, float* %58, align 4, !tbaa !1
  %59 = getelementptr inbounds i8* %a_, i64 8
  %60 = bitcast i8* %59 to float*
  %61 = load float* %60, align 4, !tbaa !1
  %62 = fmul float %61, 0x3FF7154760000000
  %63 = fcmp olt float %62, -1.260000e+02
  br i1 %63, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit, label %64

; <label>:64                                      ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit1
  %65 = fcmp ogt float %62, 1.260000e+02
  %66 = select i1 %65, float 1.260000e+02, float %62
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit:   ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit1, %64
  %67 = phi float [ %66, %64 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit1 ]
  %68 = fptosi float %67 to i32
  %69 = sitofp i32 %68 to float
  %70 = fsub float %67, %69
  %71 = fsub float 1.000000e+00, %70
  %72 = fsub float 1.000000e+00, %71
  %73 = fmul float %72, 0x3F55D889C0000000
  %74 = fadd float %73, 0x3F84177340000000
  %75 = fmul float %72, %74
  %76 = fadd float %75, 0x3FAC6CE660000000
  %77 = fmul float %72, %76
  %78 = fadd float %77, 0x3FCEBE3240000000
  %79 = fmul float %72, %78
  %80 = fadd float %79, 0x3FE62E3E20000000
  %81 = fmul float %72, %80
  %82 = fadd float %81, 1.000000e+00
  %83 = bitcast float %82 to i32
  %84 = shl i32 %68, 23
  %85 = add i32 %83, %84
  %86 = bitcast i32 %85 to float
  %87 = getelementptr inbounds i8* %r_, i64 8
  %88 = bitcast i8* %87 to float*
  store float %86, float* %88, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_exp_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = fmul float %6, 0x3FF7154760000000
  %10 = fcmp olt float %9, -1.260000e+02
  br i1 %10, label %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit12, label %11

; <label>:11                                      ; preds = %0
  %12 = fcmp ogt float %9, 1.260000e+02
  %13 = select i1 %12, float 1.260000e+02, float %9
  br label %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit12

_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit12:          ; preds = %0, %11
  %14 = phi float [ %13, %11 ], [ -1.260000e+02, %0 ]
  %15 = fptosi float %14 to i32
  %16 = sitofp i32 %15 to float
  %17 = fsub float %14, %16
  %18 = fsub float 1.000000e+00, %17
  %19 = fsub float 1.000000e+00, %18
  %20 = fmul float %19, 0x3F55D889C0000000
  %21 = fadd float %20, 0x3F84177340000000
  %22 = fmul float %19, %21
  %23 = fadd float %22, 0x3FAC6CE660000000
  %24 = fmul float %19, %23
  %25 = fadd float %24, 0x3FCEBE3240000000
  %26 = fmul float %19, %25
  %27 = fadd float %26, 0x3FE62E3E20000000
  %28 = fmul float %19, %27
  %29 = fadd float %28, 1.000000e+00
  %30 = bitcast float %29 to i32
  %31 = shl i32 %15, 23
  %32 = add i32 %30, %31
  %33 = bitcast i32 %32 to float
  %34 = fmul float %7, %33
  %35 = fmul float %8, %33
  %36 = getelementptr inbounds i8* %a_, i64 4
  %37 = bitcast i8* %36 to float*
  %38 = getelementptr inbounds i8* %a_, i64 16
  %39 = bitcast i8* %38 to float*
  %40 = getelementptr inbounds i8* %a_, i64 28
  %41 = bitcast i8* %40 to float*
  %42 = load float* %37, align 4, !tbaa !1
  %43 = load float* %39, align 4, !tbaa !1
  %44 = load float* %41, align 4, !tbaa !1
  %45 = fmul float %42, 0x3FF7154760000000
  %46 = fcmp olt float %45, -1.260000e+02
  br i1 %46, label %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit11, label %47

; <label>:47                                      ; preds = %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit12
  %48 = fcmp ogt float %45, 1.260000e+02
  %49 = select i1 %48, float 1.260000e+02, float %45
  br label %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit11

_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit11:          ; preds = %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit12, %47
  %50 = phi float [ %49, %47 ], [ -1.260000e+02, %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit12 ]
  %51 = fptosi float %50 to i32
  %52 = sitofp i32 %51 to float
  %53 = fsub float %50, %52
  %54 = fsub float 1.000000e+00, %53
  %55 = fsub float 1.000000e+00, %54
  %56 = fmul float %55, 0x3F55D889C0000000
  %57 = fadd float %56, 0x3F84177340000000
  %58 = fmul float %55, %57
  %59 = fadd float %58, 0x3FAC6CE660000000
  %60 = fmul float %55, %59
  %61 = fadd float %60, 0x3FCEBE3240000000
  %62 = fmul float %55, %61
  %63 = fadd float %62, 0x3FE62E3E20000000
  %64 = fmul float %55, %63
  %65 = fadd float %64, 1.000000e+00
  %66 = bitcast float %65 to i32
  %67 = shl i32 %51, 23
  %68 = add i32 %66, %67
  %69 = bitcast i32 %68 to float
  %70 = fmul float %43, %69
  %71 = fmul float %44, %69
  %72 = getelementptr inbounds i8* %a_, i64 8
  %73 = bitcast i8* %72 to float*
  %74 = getelementptr inbounds i8* %a_, i64 20
  %75 = bitcast i8* %74 to float*
  %76 = getelementptr inbounds i8* %a_, i64 32
  %77 = bitcast i8* %76 to float*
  %78 = load float* %73, align 4, !tbaa !1
  %79 = load float* %75, align 4, !tbaa !1
  %80 = load float* %77, align 4, !tbaa !1
  %81 = fmul float %78, 0x3FF7154760000000
  %82 = fcmp olt float %81, -1.260000e+02
  br i1 %82, label %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit, label %83

; <label>:83                                      ; preds = %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit11
  %84 = fcmp ogt float %81, 1.260000e+02
  %85 = select i1 %84, float 1.260000e+02, float %81
  br label %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit

_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit:            ; preds = %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit11, %83
  %86 = phi float [ %85, %83 ], [ -1.260000e+02, %_ZN3OSL8fast_expERKNS_5Dual2IfEE.exit11 ]
  %87 = fptosi float %86 to i32
  %88 = sitofp i32 %87 to float
  %89 = fsub float %86, %88
  %90 = fsub float 1.000000e+00, %89
  %91 = fsub float 1.000000e+00, %90
  %92 = fmul float %91, 0x3F55D889C0000000
  %93 = fadd float %92, 0x3F84177340000000
  %94 = fmul float %91, %93
  %95 = fadd float %94, 0x3FAC6CE660000000
  %96 = fmul float %91, %95
  %97 = fadd float %96, 0x3FCEBE3240000000
  %98 = fmul float %91, %97
  %99 = fadd float %98, 0x3FE62E3E20000000
  %100 = fmul float %91, %99
  %101 = fadd float %100, 1.000000e+00
  %102 = bitcast float %101 to i32
  %103 = shl i32 %87, 23
  %104 = add i32 %102, %103
  %105 = bitcast i32 %104 to float
  %106 = fmul float %79, %105
  %107 = fmul float %80, %105
  %108 = bitcast i8* %r_ to float*
  store float %33, float* %108, align 4, !tbaa !5
  %109 = getelementptr inbounds i8* %r_, i64 4
  %110 = bitcast i8* %109 to float*
  store float %69, float* %110, align 4, !tbaa !7
  %111 = getelementptr inbounds i8* %r_, i64 8
  %112 = bitcast i8* %111 to float*
  store float %105, float* %112, align 4, !tbaa !8
  %113 = getelementptr inbounds i8* %r_, i64 12
  %114 = bitcast i8* %113 to float*
  store float %34, float* %114, align 4, !tbaa !5
  %115 = getelementptr inbounds i8* %r_, i64 16
  %116 = bitcast i8* %115 to float*
  store float %70, float* %116, align 4, !tbaa !7
  %117 = getelementptr inbounds i8* %r_, i64 20
  %118 = bitcast i8* %117 to float*
  store float %106, float* %118, align 4, !tbaa !8
  %119 = getelementptr inbounds i8* %r_, i64 24
  %120 = bitcast i8* %119 to float*
  store float %35, float* %120, align 4, !tbaa !5
  %121 = getelementptr inbounds i8* %r_, i64 28
  %122 = bitcast i8* %121 to float*
  store float %71, float* %122, align 4, !tbaa !7
  %123 = getelementptr inbounds i8* %r_, i64 32
  %124 = bitcast i8* %123 to float*
  store float %107, float* %124, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_exp2_ff(float %a) #3 {
  %1 = fcmp olt float %a, -1.260000e+02
  br i1 %1, label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit, label %2

; <label>:2                                       ; preds = %0
  %3 = fcmp ogt float %a, 1.260000e+02
  %4 = select i1 %3, float 1.260000e+02, float %a
  br label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit

_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit:  ; preds = %0, %2
  %5 = phi float [ %4, %2 ], [ -1.260000e+02, %0 ]
  %6 = fptosi float %5 to i32
  %7 = sitofp i32 %6 to float
  %8 = fsub float %5, %7
  %9 = fsub float 1.000000e+00, %8
  %10 = fsub float 1.000000e+00, %9
  %11 = fmul float %10, 0x3F55D889C0000000
  %12 = fadd float %11, 0x3F84177340000000
  %13 = fmul float %10, %12
  %14 = fadd float %13, 0x3FAC6CE660000000
  %15 = fmul float %10, %14
  %16 = fadd float %15, 0x3FCEBE3240000000
  %17 = fmul float %10, %16
  %18 = fadd float %17, 0x3FE62E3E20000000
  %19 = fmul float %10, %18
  %20 = fadd float %19, 1.000000e+00
  %21 = bitcast float %20 to i32
  %22 = shl i32 %6, 23
  %23 = add i32 %21, %22
  %24 = bitcast i32 %23 to float
  ret float %24
}

; Function Attrs: nounwind uwtable
define void @osl_exp2_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, -1.260000e+02
  br i1 %3, label %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 1.260000e+02
  %6 = select i1 %5, float 1.260000e+02, float %2
  br label %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit:           ; preds = %0, %4
  %7 = phi float [ %6, %4 ], [ -1.260000e+02, %0 ]
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fsub float %7, %9
  %11 = fsub float 1.000000e+00, %10
  %12 = fsub float 1.000000e+00, %11
  %13 = fmul float %12, 0x3F55D889C0000000
  %14 = fadd float %13, 0x3F84177340000000
  %15 = fmul float %12, %14
  %16 = fadd float %15, 0x3FAC6CE660000000
  %17 = fmul float %12, %16
  %18 = fadd float %17, 0x3FCEBE3240000000
  %19 = fmul float %12, %18
  %20 = fadd float %19, 0x3FE62E3E20000000
  %21 = fmul float %12, %20
  %22 = fadd float %21, 1.000000e+00
  %23 = bitcast float %22 to i32
  %24 = shl i32 %8, 23
  %25 = add i32 %23, %24
  %26 = bitcast i32 %25 to float
  %27 = fmul float %26, 0x3FE62E4300000000
  %28 = getelementptr inbounds i8* %a, i64 4
  %29 = bitcast i8* %28 to float*
  %30 = load float* %29, align 4, !tbaa !1
  %31 = fmul float %30, %27
  %32 = getelementptr inbounds i8* %a, i64 8
  %33 = bitcast i8* %32 to float*
  %34 = load float* %33, align 4, !tbaa !1
  %35 = fmul float %34, %27
  %36 = insertelement <2 x float> undef, float %26, i32 0
  %37 = insertelement <2 x float> %36, float %31, i32 1
  %38 = bitcast i8* %r to <2 x float>*
  store <2 x float> %37, <2 x float>* %38, align 4
  %39 = getelementptr inbounds i8* %r, i64 8
  %40 = bitcast i8* %39 to float*
  store float %35, float* %40, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_exp2_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, -1.260000e+02
  br i1 %3, label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit2, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp ogt float %2, 1.260000e+02
  %6 = select i1 %5, float 1.260000e+02, float %2
  br label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit2

_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit2: ; preds = %0, %4
  %7 = phi float [ %6, %4 ], [ -1.260000e+02, %0 ]
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fsub float %7, %9
  %11 = fsub float 1.000000e+00, %10
  %12 = fsub float 1.000000e+00, %11
  %13 = fmul float %12, 0x3F55D889C0000000
  %14 = fadd float %13, 0x3F84177340000000
  %15 = fmul float %12, %14
  %16 = fadd float %15, 0x3FAC6CE660000000
  %17 = fmul float %12, %16
  %18 = fadd float %17, 0x3FCEBE3240000000
  %19 = fmul float %12, %18
  %20 = fadd float %19, 0x3FE62E3E20000000
  %21 = fmul float %12, %20
  %22 = fadd float %21, 1.000000e+00
  %23 = bitcast float %22 to i32
  %24 = shl i32 %8, 23
  %25 = add i32 %23, %24
  %26 = bitcast i32 %25 to float
  %27 = bitcast i8* %r_ to float*
  store float %26, float* %27, align 4, !tbaa !1
  %28 = getelementptr inbounds i8* %a_, i64 4
  %29 = bitcast i8* %28 to float*
  %30 = load float* %29, align 4, !tbaa !1
  %31 = fcmp olt float %30, -1.260000e+02
  br i1 %31, label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit1, label %32

; <label>:32                                      ; preds = %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit2
  %33 = fcmp ogt float %30, 1.260000e+02
  %34 = select i1 %33, float 1.260000e+02, float %30
  br label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit1

_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit1: ; preds = %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit2, %32
  %35 = phi float [ %34, %32 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit2 ]
  %36 = fptosi float %35 to i32
  %37 = sitofp i32 %36 to float
  %38 = fsub float %35, %37
  %39 = fsub float 1.000000e+00, %38
  %40 = fsub float 1.000000e+00, %39
  %41 = fmul float %40, 0x3F55D889C0000000
  %42 = fadd float %41, 0x3F84177340000000
  %43 = fmul float %40, %42
  %44 = fadd float %43, 0x3FAC6CE660000000
  %45 = fmul float %40, %44
  %46 = fadd float %45, 0x3FCEBE3240000000
  %47 = fmul float %40, %46
  %48 = fadd float %47, 0x3FE62E3E20000000
  %49 = fmul float %40, %48
  %50 = fadd float %49, 1.000000e+00
  %51 = bitcast float %50 to i32
  %52 = shl i32 %36, 23
  %53 = add i32 %51, %52
  %54 = bitcast i32 %53 to float
  %55 = getelementptr inbounds i8* %r_, i64 4
  %56 = bitcast i8* %55 to float*
  store float %54, float* %56, align 4, !tbaa !1
  %57 = getelementptr inbounds i8* %a_, i64 8
  %58 = bitcast i8* %57 to float*
  %59 = load float* %58, align 4, !tbaa !1
  %60 = fcmp olt float %59, -1.260000e+02
  br i1 %60, label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit, label %61

; <label>:61                                      ; preds = %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit1
  %62 = fcmp ogt float %59, 1.260000e+02
  %63 = select i1 %62, float 1.260000e+02, float %59
  br label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit

_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit:  ; preds = %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit1, %61
  %64 = phi float [ %63, %61 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit1 ]
  %65 = fptosi float %64 to i32
  %66 = sitofp i32 %65 to float
  %67 = fsub float %64, %66
  %68 = fsub float 1.000000e+00, %67
  %69 = fsub float 1.000000e+00, %68
  %70 = fmul float %69, 0x3F55D889C0000000
  %71 = fadd float %70, 0x3F84177340000000
  %72 = fmul float %69, %71
  %73 = fadd float %72, 0x3FAC6CE660000000
  %74 = fmul float %69, %73
  %75 = fadd float %74, 0x3FCEBE3240000000
  %76 = fmul float %69, %75
  %77 = fadd float %76, 0x3FE62E3E20000000
  %78 = fmul float %69, %77
  %79 = fadd float %78, 1.000000e+00
  %80 = bitcast float %79 to i32
  %81 = shl i32 %65, 23
  %82 = add i32 %80, %81
  %83 = bitcast i32 %82 to float
  %84 = getelementptr inbounds i8* %r_, i64 8
  %85 = bitcast i8* %84 to float*
  store float %83, float* %85, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_exp2_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = fcmp olt float %6, -1.260000e+02
  br i1 %9, label %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit12, label %10

; <label>:10                                      ; preds = %0
  %11 = fcmp ogt float %6, 1.260000e+02
  %12 = select i1 %11, float 1.260000e+02, float %6
  br label %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit12

_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit12:         ; preds = %0, %10
  %13 = phi float [ %12, %10 ], [ -1.260000e+02, %0 ]
  %14 = fptosi float %13 to i32
  %15 = sitofp i32 %14 to float
  %16 = fsub float %13, %15
  %17 = fsub float 1.000000e+00, %16
  %18 = fsub float 1.000000e+00, %17
  %19 = fmul float %18, 0x3F55D889C0000000
  %20 = fadd float %19, 0x3F84177340000000
  %21 = fmul float %18, %20
  %22 = fadd float %21, 0x3FAC6CE660000000
  %23 = fmul float %18, %22
  %24 = fadd float %23, 0x3FCEBE3240000000
  %25 = fmul float %18, %24
  %26 = fadd float %25, 0x3FE62E3E20000000
  %27 = fmul float %18, %26
  %28 = fadd float %27, 1.000000e+00
  %29 = bitcast float %28 to i32
  %30 = shl i32 %14, 23
  %31 = add i32 %29, %30
  %32 = bitcast i32 %31 to float
  %33 = fmul float %32, 0x3FE62E4300000000
  %34 = fmul float %7, %33
  %35 = fmul float %8, %33
  %36 = getelementptr inbounds i8* %a_, i64 4
  %37 = bitcast i8* %36 to float*
  %38 = getelementptr inbounds i8* %a_, i64 16
  %39 = bitcast i8* %38 to float*
  %40 = getelementptr inbounds i8* %a_, i64 28
  %41 = bitcast i8* %40 to float*
  %42 = load float* %37, align 4, !tbaa !1
  %43 = load float* %39, align 4, !tbaa !1
  %44 = load float* %41, align 4, !tbaa !1
  %45 = fcmp olt float %42, -1.260000e+02
  br i1 %45, label %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit11, label %46

; <label>:46                                      ; preds = %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit12
  %47 = fcmp ogt float %42, 1.260000e+02
  %48 = select i1 %47, float 1.260000e+02, float %42
  br label %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit11

_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit11:         ; preds = %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit12, %46
  %49 = phi float [ %48, %46 ], [ -1.260000e+02, %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit12 ]
  %50 = fptosi float %49 to i32
  %51 = sitofp i32 %50 to float
  %52 = fsub float %49, %51
  %53 = fsub float 1.000000e+00, %52
  %54 = fsub float 1.000000e+00, %53
  %55 = fmul float %54, 0x3F55D889C0000000
  %56 = fadd float %55, 0x3F84177340000000
  %57 = fmul float %54, %56
  %58 = fadd float %57, 0x3FAC6CE660000000
  %59 = fmul float %54, %58
  %60 = fadd float %59, 0x3FCEBE3240000000
  %61 = fmul float %54, %60
  %62 = fadd float %61, 0x3FE62E3E20000000
  %63 = fmul float %54, %62
  %64 = fadd float %63, 1.000000e+00
  %65 = bitcast float %64 to i32
  %66 = shl i32 %50, 23
  %67 = add i32 %65, %66
  %68 = bitcast i32 %67 to float
  %69 = fmul float %68, 0x3FE62E4300000000
  %70 = fmul float %43, %69
  %71 = fmul float %44, %69
  %72 = getelementptr inbounds i8* %a_, i64 8
  %73 = bitcast i8* %72 to float*
  %74 = getelementptr inbounds i8* %a_, i64 20
  %75 = bitcast i8* %74 to float*
  %76 = getelementptr inbounds i8* %a_, i64 32
  %77 = bitcast i8* %76 to float*
  %78 = load float* %73, align 4, !tbaa !1
  %79 = load float* %75, align 4, !tbaa !1
  %80 = load float* %77, align 4, !tbaa !1
  %81 = fcmp olt float %78, -1.260000e+02
  br i1 %81, label %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit, label %82

; <label>:82                                      ; preds = %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit11
  %83 = fcmp ogt float %78, 1.260000e+02
  %84 = select i1 %83, float 1.260000e+02, float %78
  br label %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit

_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit:           ; preds = %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit11, %82
  %85 = phi float [ %84, %82 ], [ -1.260000e+02, %_ZN3OSL9fast_exp2ERKNS_5Dual2IfEE.exit11 ]
  %86 = fptosi float %85 to i32
  %87 = sitofp i32 %86 to float
  %88 = fsub float %85, %87
  %89 = fsub float 1.000000e+00, %88
  %90 = fsub float 1.000000e+00, %89
  %91 = fmul float %90, 0x3F55D889C0000000
  %92 = fadd float %91, 0x3F84177340000000
  %93 = fmul float %90, %92
  %94 = fadd float %93, 0x3FAC6CE660000000
  %95 = fmul float %90, %94
  %96 = fadd float %95, 0x3FCEBE3240000000
  %97 = fmul float %90, %96
  %98 = fadd float %97, 0x3FE62E3E20000000
  %99 = fmul float %90, %98
  %100 = fadd float %99, 1.000000e+00
  %101 = bitcast float %100 to i32
  %102 = shl i32 %86, 23
  %103 = add i32 %101, %102
  %104 = bitcast i32 %103 to float
  %105 = fmul float %104, 0x3FE62E4300000000
  %106 = fmul float %79, %105
  %107 = fmul float %80, %105
  %108 = bitcast i8* %r_ to float*
  store float %32, float* %108, align 4, !tbaa !5
  %109 = getelementptr inbounds i8* %r_, i64 4
  %110 = bitcast i8* %109 to float*
  store float %68, float* %110, align 4, !tbaa !7
  %111 = getelementptr inbounds i8* %r_, i64 8
  %112 = bitcast i8* %111 to float*
  store float %104, float* %112, align 4, !tbaa !8
  %113 = getelementptr inbounds i8* %r_, i64 12
  %114 = bitcast i8* %113 to float*
  store float %34, float* %114, align 4, !tbaa !5
  %115 = getelementptr inbounds i8* %r_, i64 16
  %116 = bitcast i8* %115 to float*
  store float %70, float* %116, align 4, !tbaa !7
  %117 = getelementptr inbounds i8* %r_, i64 20
  %118 = bitcast i8* %117 to float*
  store float %106, float* %118, align 4, !tbaa !8
  %119 = getelementptr inbounds i8* %r_, i64 24
  %120 = bitcast i8* %119 to float*
  store float %35, float* %120, align 4, !tbaa !5
  %121 = getelementptr inbounds i8* %r_, i64 28
  %122 = bitcast i8* %121 to float*
  store float %71, float* %122, align 4, !tbaa !7
  %123 = getelementptr inbounds i8* %r_, i64 32
  %124 = bitcast i8* %123 to float*
  store float %107, float* %124, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_expm1_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fcmp olt float %1, 0x3F9EB851E0000000
  br i1 %2, label %3, label %10

; <label>:3                                       ; preds = %0
  %4 = fsub float 1.000000e+00, %a
  %5 = fsub float 1.000000e+00, %4
  %6 = fmul float %5, %5
  %7 = fmul float %6, 5.000000e-01
  %8 = fadd float %5, %7
  %9 = tail call float @copysignf(float %8, float %a) #12
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit

; <label>:10                                      ; preds = %0
  %11 = fmul float %a, 0x3FF7154760000000
  %12 = fcmp olt float %11, -1.260000e+02
  br i1 %12, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, label %13

; <label>:13                                      ; preds = %10
  %14 = fcmp ogt float %11, 1.260000e+02
  %15 = select i1 %14, float 1.260000e+02, float %11
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i: ; preds = %13, %10
  %16 = phi float [ %15, %13 ], [ -1.260000e+02, %10 ]
  %17 = fptosi float %16 to i32
  %18 = sitofp i32 %17 to float
  %19 = fsub float %16, %18
  %20 = fsub float 1.000000e+00, %19
  %21 = fsub float 1.000000e+00, %20
  %22 = fmul float %21, 0x3F55D889C0000000
  %23 = fadd float %22, 0x3F84177340000000
  %24 = fmul float %21, %23
  %25 = fadd float %24, 0x3FAC6CE660000000
  %26 = fmul float %21, %25
  %27 = fadd float %26, 0x3FCEBE3240000000
  %28 = fmul float %21, %27
  %29 = fadd float %28, 0x3FE62E3E20000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 1.000000e+00
  %32 = bitcast float %31 to i32
  %33 = shl i32 %17, 23
  %34 = add i32 %32, %33
  %35 = bitcast i32 %34 to float
  %36 = fadd float %35, -1.000000e+00
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit

_ZN11OpenImageIO4v1_710fast_expm1Ef.exit:         ; preds = %3, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i
  %.0.i = phi float [ %9, %3 ], [ %36, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i ]
  ret float %.0.i
}

; Function Attrs: nounwind uwtable
define void @osl_expm1_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp olt float %3, 0x3F9EB851E0000000
  br i1 %4, label %5, label %12

; <label>:5                                       ; preds = %0
  %6 = fsub float 1.000000e+00, %2
  %7 = fsub float 1.000000e+00, %6
  %8 = fmul float %7, %7
  %9 = fmul float %8, 5.000000e-01
  %10 = fadd float %7, %9
  %11 = tail call float @copysignf(float %10, float %2) #12
  %.pre.i = fmul float %2, 0x3FF7154760000000
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i

; <label>:12                                      ; preds = %0
  %13 = fmul float %2, 0x3FF7154760000000
  %14 = fcmp olt float %13, -1.260000e+02
  br i1 %14, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i, label %15

; <label>:15                                      ; preds = %12
  %16 = fcmp ogt float %13, 1.260000e+02
  %17 = select i1 %16, float 1.260000e+02, float %13
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i: ; preds = %15, %12
  %18 = phi float [ %17, %15 ], [ -1.260000e+02, %12 ]
  %19 = fptosi float %18 to i32
  %20 = sitofp i32 %19 to float
  %21 = fsub float %18, %20
  %22 = fsub float 1.000000e+00, %21
  %23 = fsub float 1.000000e+00, %22
  %24 = fmul float %23, 0x3F55D889C0000000
  %25 = fadd float %24, 0x3F84177340000000
  %26 = fmul float %23, %25
  %27 = fadd float %26, 0x3FAC6CE660000000
  %28 = fmul float %23, %27
  %29 = fadd float %28, 0x3FCEBE3240000000
  %30 = fmul float %23, %29
  %31 = fadd float %30, 0x3FE62E3E20000000
  %32 = fmul float %23, %31
  %33 = fadd float %32, 1.000000e+00
  %34 = bitcast float %33 to i32
  %35 = shl i32 %19, 23
  %36 = add i32 %34, %35
  %37 = bitcast i32 %36 to float
  %38 = fadd float %37, -1.000000e+00
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i

_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i:       ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i, %5
  %.pre-phi.i = phi float [ %.pre.i, %5 ], [ %13, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i ]
  %.0.i.i = phi float [ %11, %5 ], [ %38, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i ]
  %39 = fcmp olt float %.pre-phi.i, -1.260000e+02
  br i1 %39, label %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit, label %40

; <label>:40                                      ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i
  %41 = fcmp ogt float %.pre-phi.i, 1.260000e+02
  %42 = select i1 %41, float 1.260000e+02, float %.pre-phi.i
  br label %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit

_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit:         ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i, %40
  %43 = phi float [ %42, %40 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i ]
  %44 = fptosi float %43 to i32
  %45 = sitofp i32 %44 to float
  %46 = fsub float %43, %45
  %47 = fsub float 1.000000e+00, %46
  %48 = fsub float 1.000000e+00, %47
  %49 = fmul float %48, 0x3F55D889C0000000
  %50 = fadd float %49, 0x3F84177340000000
  %51 = fmul float %48, %50
  %52 = fadd float %51, 0x3FAC6CE660000000
  %53 = fmul float %48, %52
  %54 = fadd float %53, 0x3FCEBE3240000000
  %55 = fmul float %48, %54
  %56 = fadd float %55, 0x3FE62E3E20000000
  %57 = fmul float %48, %56
  %58 = fadd float %57, 1.000000e+00
  %59 = bitcast float %58 to i32
  %60 = shl i32 %44, 23
  %61 = add i32 %59, %60
  %62 = bitcast i32 %61 to float
  %63 = getelementptr inbounds i8* %a, i64 4
  %64 = bitcast i8* %63 to float*
  %65 = load float* %64, align 4, !tbaa !1
  %66 = fmul float %65, %62
  %67 = getelementptr inbounds i8* %a, i64 8
  %68 = bitcast i8* %67 to float*
  %69 = load float* %68, align 4, !tbaa !1
  %70 = fmul float %69, %62
  %71 = insertelement <2 x float> undef, float %.0.i.i, i32 0
  %72 = insertelement <2 x float> %71, float %66, i32 1
  %73 = bitcast i8* %r to <2 x float>*
  store <2 x float> %72, <2 x float>* %73, align 4
  %74 = getelementptr inbounds i8* %r, i64 8
  %75 = bitcast i8* %74 to float*
  store float %70, float* %75, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_expm1_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp olt float %3, 0x3F9EB851E0000000
  br i1 %4, label %5, label %12

; <label>:5                                       ; preds = %0
  %6 = fsub float 1.000000e+00, %2
  %7 = fsub float 1.000000e+00, %6
  %8 = fmul float %7, %7
  %9 = fmul float %8, 5.000000e-01
  %10 = fadd float %7, %9
  %11 = tail call float @copysignf(float %10, float %2) #12
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit3

; <label>:12                                      ; preds = %0
  %13 = fmul float %2, 0x3FF7154760000000
  %14 = fcmp olt float %13, -1.260000e+02
  br i1 %14, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1, label %15

; <label>:15                                      ; preds = %12
  %16 = fcmp ogt float %13, 1.260000e+02
  %17 = select i1 %16, float 1.260000e+02, float %13
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1: ; preds = %15, %12
  %18 = phi float [ %17, %15 ], [ -1.260000e+02, %12 ]
  %19 = fptosi float %18 to i32
  %20 = sitofp i32 %19 to float
  %21 = fsub float %18, %20
  %22 = fsub float 1.000000e+00, %21
  %23 = fsub float 1.000000e+00, %22
  %24 = fmul float %23, 0x3F55D889C0000000
  %25 = fadd float %24, 0x3F84177340000000
  %26 = fmul float %23, %25
  %27 = fadd float %26, 0x3FAC6CE660000000
  %28 = fmul float %23, %27
  %29 = fadd float %28, 0x3FCEBE3240000000
  %30 = fmul float %23, %29
  %31 = fadd float %30, 0x3FE62E3E20000000
  %32 = fmul float %23, %31
  %33 = fadd float %32, 1.000000e+00
  %34 = bitcast float %33 to i32
  %35 = shl i32 %19, 23
  %36 = add i32 %34, %35
  %37 = bitcast i32 %36 to float
  %38 = fadd float %37, -1.000000e+00
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit3

_ZN11OpenImageIO4v1_710fast_expm1Ef.exit3:        ; preds = %5, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1
  %.0.i2 = phi float [ %11, %5 ], [ %38, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i1 ]
  %39 = bitcast i8* %r_ to float*
  store float %.0.i2, float* %39, align 4, !tbaa !1
  %40 = getelementptr inbounds i8* %a_, i64 4
  %41 = bitcast i8* %40 to float*
  %42 = load float* %41, align 4, !tbaa !1
  %43 = tail call float @fabsf(float %42) #12
  %44 = fcmp olt float %43, 0x3F9EB851E0000000
  br i1 %44, label %45, label %52

; <label>:45                                      ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit3
  %46 = fsub float 1.000000e+00, %42
  %47 = fsub float 1.000000e+00, %46
  %48 = fmul float %47, %47
  %49 = fmul float %48, 5.000000e-01
  %50 = fadd float %47, %49
  %51 = tail call float @copysignf(float %50, float %42) #12
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit6

; <label>:52                                      ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit3
  %53 = fmul float %42, 0x3FF7154760000000
  %54 = fcmp olt float %53, -1.260000e+02
  br i1 %54, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4, label %55

; <label>:55                                      ; preds = %52
  %56 = fcmp ogt float %53, 1.260000e+02
  %57 = select i1 %56, float 1.260000e+02, float %53
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4: ; preds = %55, %52
  %58 = phi float [ %57, %55 ], [ -1.260000e+02, %52 ]
  %59 = fptosi float %58 to i32
  %60 = sitofp i32 %59 to float
  %61 = fsub float %58, %60
  %62 = fsub float 1.000000e+00, %61
  %63 = fsub float 1.000000e+00, %62
  %64 = fmul float %63, 0x3F55D889C0000000
  %65 = fadd float %64, 0x3F84177340000000
  %66 = fmul float %63, %65
  %67 = fadd float %66, 0x3FAC6CE660000000
  %68 = fmul float %63, %67
  %69 = fadd float %68, 0x3FCEBE3240000000
  %70 = fmul float %63, %69
  %71 = fadd float %70, 0x3FE62E3E20000000
  %72 = fmul float %63, %71
  %73 = fadd float %72, 1.000000e+00
  %74 = bitcast float %73 to i32
  %75 = shl i32 %59, 23
  %76 = add i32 %74, %75
  %77 = bitcast i32 %76 to float
  %78 = fadd float %77, -1.000000e+00
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit6

_ZN11OpenImageIO4v1_710fast_expm1Ef.exit6:        ; preds = %45, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4
  %.0.i5 = phi float [ %51, %45 ], [ %78, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i4 ]
  %79 = getelementptr inbounds i8* %r_, i64 4
  %80 = bitcast i8* %79 to float*
  store float %.0.i5, float* %80, align 4, !tbaa !1
  %81 = getelementptr inbounds i8* %a_, i64 8
  %82 = bitcast i8* %81 to float*
  %83 = load float* %82, align 4, !tbaa !1
  %84 = tail call float @fabsf(float %83) #12
  %85 = fcmp olt float %84, 0x3F9EB851E0000000
  br i1 %85, label %86, label %93

; <label>:86                                      ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit6
  %87 = fsub float 1.000000e+00, %83
  %88 = fsub float 1.000000e+00, %87
  %89 = fmul float %88, %88
  %90 = fmul float %89, 5.000000e-01
  %91 = fadd float %88, %90
  %92 = tail call float @copysignf(float %91, float %83) #12
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit

; <label>:93                                      ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit6
  %94 = fmul float %83, 0x3FF7154760000000
  %95 = fcmp olt float %94, -1.260000e+02
  br i1 %95, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i, label %96

; <label>:96                                      ; preds = %93
  %97 = fcmp ogt float %94, 1.260000e+02
  %98 = select i1 %97, float 1.260000e+02, float %94
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i: ; preds = %96, %93
  %99 = phi float [ %98, %96 ], [ -1.260000e+02, %93 ]
  %100 = fptosi float %99 to i32
  %101 = sitofp i32 %100 to float
  %102 = fsub float %99, %101
  %103 = fsub float 1.000000e+00, %102
  %104 = fsub float 1.000000e+00, %103
  %105 = fmul float %104, 0x3F55D889C0000000
  %106 = fadd float %105, 0x3F84177340000000
  %107 = fmul float %104, %106
  %108 = fadd float %107, 0x3FAC6CE660000000
  %109 = fmul float %104, %108
  %110 = fadd float %109, 0x3FCEBE3240000000
  %111 = fmul float %104, %110
  %112 = fadd float %111, 0x3FE62E3E20000000
  %113 = fmul float %104, %112
  %114 = fadd float %113, 1.000000e+00
  %115 = bitcast float %114 to i32
  %116 = shl i32 %100, 23
  %117 = add i32 %115, %116
  %118 = bitcast i32 %117 to float
  %119 = fadd float %118, -1.000000e+00
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit

_ZN11OpenImageIO4v1_710fast_expm1Ef.exit:         ; preds = %86, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i
  %.0.i = phi float [ %92, %86 ], [ %119, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i ]
  %120 = getelementptr inbounds i8* %r_, i64 8
  %121 = bitcast i8* %120 to float*
  store float %.0.i, float* %121, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_expm1_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = tail call float @fabsf(float %6) #12
  %10 = fcmp olt float %9, 0x3F9EB851E0000000
  br i1 %10, label %11, label %18

; <label>:11                                      ; preds = %0
  %12 = fsub float 1.000000e+00, %6
  %13 = fsub float 1.000000e+00, %12
  %14 = fmul float %13, %13
  %15 = fmul float %14, 5.000000e-01
  %16 = fadd float %13, %15
  %17 = tail call float @copysignf(float %16, float %6) #12
  %.pre.i17 = fmul float %6, 0x3FF7154760000000
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i21

; <label>:18                                      ; preds = %0
  %19 = fmul float %6, 0x3FF7154760000000
  %20 = fcmp olt float %19, -1.260000e+02
  br i1 %20, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i18, label %21

; <label>:21                                      ; preds = %18
  %22 = fcmp ogt float %19, 1.260000e+02
  %23 = select i1 %22, float 1.260000e+02, float %19
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i18

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i18: ; preds = %21, %18
  %24 = phi float [ %23, %21 ], [ -1.260000e+02, %18 ]
  %25 = fptosi float %24 to i32
  %26 = sitofp i32 %25 to float
  %27 = fsub float %24, %26
  %28 = fsub float 1.000000e+00, %27
  %29 = fsub float 1.000000e+00, %28
  %30 = fmul float %29, 0x3F55D889C0000000
  %31 = fadd float %30, 0x3F84177340000000
  %32 = fmul float %29, %31
  %33 = fadd float %32, 0x3FAC6CE660000000
  %34 = fmul float %29, %33
  %35 = fadd float %34, 0x3FCEBE3240000000
  %36 = fmul float %29, %35
  %37 = fadd float %36, 0x3FE62E3E20000000
  %38 = fmul float %29, %37
  %39 = fadd float %38, 1.000000e+00
  %40 = bitcast float %39 to i32
  %41 = shl i32 %25, 23
  %42 = add i32 %40, %41
  %43 = bitcast i32 %42 to float
  %44 = fadd float %43, -1.000000e+00
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i21

_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i21:     ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i18, %11
  %.pre-phi.i19 = phi float [ %.pre.i17, %11 ], [ %19, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i18 ]
  %.0.i.i20 = phi float [ %17, %11 ], [ %44, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i18 ]
  %45 = fcmp olt float %.pre-phi.i19, -1.260000e+02
  br i1 %45, label %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit22, label %46

; <label>:46                                      ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i21
  %47 = fcmp ogt float %.pre-phi.i19, 1.260000e+02
  %48 = select i1 %47, float 1.260000e+02, float %.pre-phi.i19
  br label %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit22

_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit22:       ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i21, %46
  %49 = phi float [ %48, %46 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i21 ]
  %50 = fptosi float %49 to i32
  %51 = sitofp i32 %50 to float
  %52 = fsub float %49, %51
  %53 = fsub float 1.000000e+00, %52
  %54 = fsub float 1.000000e+00, %53
  %55 = fmul float %54, 0x3F55D889C0000000
  %56 = fadd float %55, 0x3F84177340000000
  %57 = fmul float %54, %56
  %58 = fadd float %57, 0x3FAC6CE660000000
  %59 = fmul float %54, %58
  %60 = fadd float %59, 0x3FCEBE3240000000
  %61 = fmul float %54, %60
  %62 = fadd float %61, 0x3FE62E3E20000000
  %63 = fmul float %54, %62
  %64 = fadd float %63, 1.000000e+00
  %65 = bitcast float %64 to i32
  %66 = shl i32 %50, 23
  %67 = add i32 %65, %66
  %68 = bitcast i32 %67 to float
  %69 = fmul float %7, %68
  %70 = fmul float %8, %68
  %71 = getelementptr inbounds i8* %a_, i64 4
  %72 = bitcast i8* %71 to float*
  %73 = getelementptr inbounds i8* %a_, i64 16
  %74 = bitcast i8* %73 to float*
  %75 = getelementptr inbounds i8* %a_, i64 28
  %76 = bitcast i8* %75 to float*
  %77 = load float* %72, align 4, !tbaa !1
  %78 = load float* %74, align 4, !tbaa !1
  %79 = load float* %76, align 4, !tbaa !1
  %80 = tail call float @fabsf(float %77) #12
  %81 = fcmp olt float %80, 0x3F9EB851E0000000
  br i1 %81, label %82, label %89

; <label>:82                                      ; preds = %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit22
  %83 = fsub float 1.000000e+00, %77
  %84 = fsub float 1.000000e+00, %83
  %85 = fmul float %84, %84
  %86 = fmul float %85, 5.000000e-01
  %87 = fadd float %84, %86
  %88 = tail call float @copysignf(float %87, float %77) #12
  %.pre.i11 = fmul float %77, 0x3FF7154760000000
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i15

; <label>:89                                      ; preds = %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit22
  %90 = fmul float %77, 0x3FF7154760000000
  %91 = fcmp olt float %90, -1.260000e+02
  br i1 %91, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i12, label %92

; <label>:92                                      ; preds = %89
  %93 = fcmp ogt float %90, 1.260000e+02
  %94 = select i1 %93, float 1.260000e+02, float %90
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i12

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i12: ; preds = %92, %89
  %95 = phi float [ %94, %92 ], [ -1.260000e+02, %89 ]
  %96 = fptosi float %95 to i32
  %97 = sitofp i32 %96 to float
  %98 = fsub float %95, %97
  %99 = fsub float 1.000000e+00, %98
  %100 = fsub float 1.000000e+00, %99
  %101 = fmul float %100, 0x3F55D889C0000000
  %102 = fadd float %101, 0x3F84177340000000
  %103 = fmul float %100, %102
  %104 = fadd float %103, 0x3FAC6CE660000000
  %105 = fmul float %100, %104
  %106 = fadd float %105, 0x3FCEBE3240000000
  %107 = fmul float %100, %106
  %108 = fadd float %107, 0x3FE62E3E20000000
  %109 = fmul float %100, %108
  %110 = fadd float %109, 1.000000e+00
  %111 = bitcast float %110 to i32
  %112 = shl i32 %96, 23
  %113 = add i32 %111, %112
  %114 = bitcast i32 %113 to float
  %115 = fadd float %114, -1.000000e+00
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i15

_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i15:     ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i12, %82
  %.pre-phi.i13 = phi float [ %.pre.i11, %82 ], [ %90, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i12 ]
  %.0.i.i14 = phi float [ %88, %82 ], [ %115, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i12 ]
  %116 = fcmp olt float %.pre-phi.i13, -1.260000e+02
  br i1 %116, label %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit16, label %117

; <label>:117                                     ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i15
  %118 = fcmp ogt float %.pre-phi.i13, 1.260000e+02
  %119 = select i1 %118, float 1.260000e+02, float %.pre-phi.i13
  br label %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit16

_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit16:       ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i15, %117
  %120 = phi float [ %119, %117 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i15 ]
  %121 = fptosi float %120 to i32
  %122 = sitofp i32 %121 to float
  %123 = fsub float %120, %122
  %124 = fsub float 1.000000e+00, %123
  %125 = fsub float 1.000000e+00, %124
  %126 = fmul float %125, 0x3F55D889C0000000
  %127 = fadd float %126, 0x3F84177340000000
  %128 = fmul float %125, %127
  %129 = fadd float %128, 0x3FAC6CE660000000
  %130 = fmul float %125, %129
  %131 = fadd float %130, 0x3FCEBE3240000000
  %132 = fmul float %125, %131
  %133 = fadd float %132, 0x3FE62E3E20000000
  %134 = fmul float %125, %133
  %135 = fadd float %134, 1.000000e+00
  %136 = bitcast float %135 to i32
  %137 = shl i32 %121, 23
  %138 = add i32 %136, %137
  %139 = bitcast i32 %138 to float
  %140 = fmul float %78, %139
  %141 = fmul float %79, %139
  %142 = getelementptr inbounds i8* %a_, i64 8
  %143 = bitcast i8* %142 to float*
  %144 = getelementptr inbounds i8* %a_, i64 20
  %145 = bitcast i8* %144 to float*
  %146 = getelementptr inbounds i8* %a_, i64 32
  %147 = bitcast i8* %146 to float*
  %148 = load float* %143, align 4, !tbaa !1
  %149 = load float* %145, align 4, !tbaa !1
  %150 = load float* %147, align 4, !tbaa !1
  %151 = tail call float @fabsf(float %148) #12
  %152 = fcmp olt float %151, 0x3F9EB851E0000000
  br i1 %152, label %153, label %160

; <label>:153                                     ; preds = %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit16
  %154 = fsub float 1.000000e+00, %148
  %155 = fsub float 1.000000e+00, %154
  %156 = fmul float %155, %155
  %157 = fmul float %156, 5.000000e-01
  %158 = fadd float %155, %157
  %159 = tail call float @copysignf(float %158, float %148) #12
  %.pre.i = fmul float %148, 0x3FF7154760000000
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i

; <label>:160                                     ; preds = %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit16
  %161 = fmul float %148, 0x3FF7154760000000
  %162 = fcmp olt float %161, -1.260000e+02
  br i1 %162, label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i, label %163

; <label>:163                                     ; preds = %160
  %164 = fcmp ogt float %161, 1.260000e+02
  %165 = select i1 %164, float 1.260000e+02, float %161
  br label %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i

_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i: ; preds = %163, %160
  %166 = phi float [ %165, %163 ], [ -1.260000e+02, %160 ]
  %167 = fptosi float %166 to i32
  %168 = sitofp i32 %167 to float
  %169 = fsub float %166, %168
  %170 = fsub float 1.000000e+00, %169
  %171 = fsub float 1.000000e+00, %170
  %172 = fmul float %171, 0x3F55D889C0000000
  %173 = fadd float %172, 0x3F84177340000000
  %174 = fmul float %171, %173
  %175 = fadd float %174, 0x3FAC6CE660000000
  %176 = fmul float %171, %175
  %177 = fadd float %176, 0x3FCEBE3240000000
  %178 = fmul float %171, %177
  %179 = fadd float %178, 0x3FE62E3E20000000
  %180 = fmul float %171, %179
  %181 = fadd float %180, 1.000000e+00
  %182 = bitcast float %181 to i32
  %183 = shl i32 %167, 23
  %184 = add i32 %182, %183
  %185 = bitcast i32 %184 to float
  %186 = fadd float %185, -1.000000e+00
  br label %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i

_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i:       ; preds = %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i, %153
  %.pre-phi.i = phi float [ %.pre.i, %153 ], [ %161, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i ]
  %.0.i.i = phi float [ %159, %153 ], [ %186, %_ZN11OpenImageIO4v1_78fast_expIfEET_RKS2_.exit.i.i ]
  %187 = fcmp olt float %.pre-phi.i, -1.260000e+02
  br i1 %187, label %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit, label %188

; <label>:188                                     ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i
  %189 = fcmp ogt float %.pre-phi.i, 1.260000e+02
  %190 = select i1 %189, float 1.260000e+02, float %.pre-phi.i
  br label %_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit

_ZN3OSL10fast_expm1ERKNS_5Dual2IfEE.exit:         ; preds = %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i, %188
  %191 = phi float [ %190, %188 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_710fast_expm1Ef.exit.i ]
  %192 = fptosi float %191 to i32
  %193 = sitofp i32 %192 to float
  %194 = fsub float %191, %193
  %195 = fsub float 1.000000e+00, %194
  %196 = fsub float 1.000000e+00, %195
  %197 = fmul float %196, 0x3F55D889C0000000
  %198 = fadd float %197, 0x3F84177340000000
  %199 = fmul float %196, %198
  %200 = fadd float %199, 0x3FAC6CE660000000
  %201 = fmul float %196, %200
  %202 = fadd float %201, 0x3FCEBE3240000000
  %203 = fmul float %196, %202
  %204 = fadd float %203, 0x3FE62E3E20000000
  %205 = fmul float %196, %204
  %206 = fadd float %205, 1.000000e+00
  %207 = bitcast float %206 to i32
  %208 = shl i32 %192, 23
  %209 = add i32 %207, %208
  %210 = bitcast i32 %209 to float
  %211 = fmul float %149, %210
  %212 = fmul float %150, %210
  %213 = bitcast i8* %r_ to float*
  store float %.0.i.i20, float* %213, align 4, !tbaa !5
  %214 = getelementptr inbounds i8* %r_, i64 4
  %215 = bitcast i8* %214 to float*
  store float %.0.i.i14, float* %215, align 4, !tbaa !7
  %216 = getelementptr inbounds i8* %r_, i64 8
  %217 = bitcast i8* %216 to float*
  store float %.0.i.i, float* %217, align 4, !tbaa !8
  %218 = getelementptr inbounds i8* %r_, i64 12
  %219 = bitcast i8* %218 to float*
  store float %69, float* %219, align 4, !tbaa !5
  %220 = getelementptr inbounds i8* %r_, i64 16
  %221 = bitcast i8* %220 to float*
  store float %140, float* %221, align 4, !tbaa !7
  %222 = getelementptr inbounds i8* %r_, i64 20
  %223 = bitcast i8* %222 to float*
  store float %211, float* %223, align 4, !tbaa !8
  %224 = getelementptr inbounds i8* %r_, i64 24
  %225 = bitcast i8* %224 to float*
  store float %70, float* %225, align 4, !tbaa !5
  %226 = getelementptr inbounds i8* %r_, i64 28
  %227 = bitcast i8* %226 to float*
  store float %141, float* %227, align 4, !tbaa !7
  %228 = getelementptr inbounds i8* %r_, i64 32
  %229 = bitcast i8* %228 to float*
  store float %212, float* %229, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_pow_fff(float %a, float %b) #3 {
  %1 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %a, float %b)
  ret float %1
}

; Function Attrs: inlinehint nounwind readnone uwtable
define linkonce_odr float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %x, float %y) #6 {
  %1 = fcmp oeq float %y, 0.000000e+00
  br i1 %1, label %88, label %2

; <label>:2                                       ; preds = %0
  %3 = fcmp oeq float %x, 0.000000e+00
  br i1 %3, label %88, label %4

; <label>:4                                       ; preds = %2
  %5 = fcmp oeq float %y, 1.000000e+00
  br i1 %5, label %88, label %6

; <label>:6                                       ; preds = %4
  %7 = fcmp oeq float %y, 2.000000e+00
  br i1 %7, label %8, label %12

; <label>:8                                       ; preds = %6
  %9 = fmul float %x, %x
  %10 = fcmp ogt float %9, 0x47EFFFFFE0000000
  %11 = select i1 %10, float 0x47EFFFFFE0000000, float %9
  br label %88

; <label>:12                                      ; preds = %6
  %13 = fcmp olt float %x, 0.000000e+00
  br i1 %13, label %14, label %30

; <label>:14                                      ; preds = %12
  %15 = bitcast float %y to i32
  %16 = and i32 %15, 2147483647
  %17 = icmp ugt i32 %16, 1266679807
  br i1 %17, label %30, label %18

; <label>:18                                      ; preds = %14
  %19 = icmp ugt i32 %16, 1065353215
  br i1 %19, label %20, label %88

; <label>:20                                      ; preds = %18
  %21 = lshr i32 %16, 23
  %22 = sub i32 150, %21
  %23 = lshr i32 %16, %22
  %24 = shl i32 %23, %22
  %25 = icmp eq i32 %24, %16
  br i1 %25, label %26, label %88

; <label>:26                                      ; preds = %20
  %27 = shl i32 %23, 31
  %28 = or i32 %27, 1065353216
  %29 = bitcast i32 %28 to float
  br label %30

; <label>:30                                      ; preds = %26, %14, %12
  %sign.0 = phi float [ 1.000000e+00, %14 ], [ %29, %26 ], [ 1.000000e+00, %12 ]
  %31 = tail call float @fabsf(float %x) #12
  %32 = fcmp olt float %31, 0x3810000000000000
  br i1 %32, label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit, label %33

; <label>:33                                      ; preds = %30
  %34 = fcmp ogt float %31, 0x47EFFFFFE0000000
  %35 = bitcast float %31 to i32
  %phitmp.i = select i1 %34, i32 2139095039, i32 %35
  br label %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit

_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit:  ; preds = %30, %33
  %36 = phi i32 [ %phitmp.i, %33 ], [ 8388608, %30 ]
  %37 = lshr i32 %36, 23
  %38 = add nsw i32 %37, -127
  %39 = and i32 %36, 8388607
  %40 = or i32 %39, 1065353216
  %41 = bitcast i32 %40 to float
  %42 = fadd float %41, -1.000000e+00
  %43 = fmul float %42, %42
  %44 = fmul float %43, %43
  %45 = fmul float %42, 0xBF831161A0000000
  %46 = fadd float %45, 0x3FAAA83920000000
  %47 = fmul float %42, 0x3FDEA2C5A0000000
  %48 = fadd float %47, 0xBFE713CA80000000
  %49 = fmul float %42, %46
  %50 = fadd float %49, 0xBFC19A9FA0000000
  %51 = fmul float %42, %50
  %52 = fadd float %51, 0x3FCEF5B7A0000000
  %53 = fmul float %42, %52
  %54 = fadd float %53, 0xBFD63A40C0000000
  %55 = fmul float %42, %48
  %56 = fadd float %55, 0x3FF7154200000000
  %57 = fmul float %44, %54
  %58 = fmul float %42, %56
  %59 = fadd float %58, %57
  %60 = sitofp i32 %38 to float
  %61 = fadd float %60, %59
  %62 = fmul float %61, %y
  %63 = fcmp olt float %62, -1.260000e+02
  br i1 %63, label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit, label %64

; <label>:64                                      ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit
  %65 = fcmp ogt float %62, 1.260000e+02
  %66 = select i1 %65, float 1.260000e+02, float %62
  br label %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit

_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit:  ; preds = %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit, %64
  %67 = phi float [ %66, %64 ], [ -1.260000e+02, %_ZN11OpenImageIO4v1_79fast_log2IfEET_RKS2_.exit ]
  %68 = fptosi float %67 to i32
  %69 = sitofp i32 %68 to float
  %70 = fsub float %67, %69
  %71 = fsub float 1.000000e+00, %70
  %72 = fsub float 1.000000e+00, %71
  %73 = fmul float %72, 0x3F55D889C0000000
  %74 = fadd float %73, 0x3F84177340000000
  %75 = fmul float %72, %74
  %76 = fadd float %75, 0x3FAC6CE660000000
  %77 = fmul float %72, %76
  %78 = fadd float %77, 0x3FCEBE3240000000
  %79 = fmul float %72, %78
  %80 = fadd float %79, 0x3FE62E3E20000000
  %81 = fmul float %72, %80
  %82 = fadd float %81, 1.000000e+00
  %83 = bitcast float %82 to i32
  %84 = shl i32 %68, 23
  %85 = add i32 %83, %84
  %86 = bitcast i32 %85 to float
  %87 = fmul float %sign.0, %86
  br label %88

; <label>:88                                      ; preds = %18, %20, %4, %2, %0, %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit, %8
  %.0 = phi float [ %11, %8 ], [ %87, %_ZN11OpenImageIO4v1_79fast_exp2IfEET_RKS2_.exit ], [ 1.000000e+00, %0 ], [ 0.000000e+00, %2 ], [ %x, %4 ], [ 0.000000e+00, %20 ], [ 0.000000e+00, %18 ]
  ret float %.0
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dfdfdf(i8* nocapture %r, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = bitcast i8* %b to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = fadd float %4, -1.000000e+00
  %6 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %2, float %5) #2
  %7 = fmul float %2, %6
  %8 = fcmp ogt float %2, 0.000000e+00
  br i1 %8, label %9, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

; <label>:9                                       ; preds = %0
  %10 = fcmp olt float %2, 0x3810000000000000
  br i1 %10, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, label %11

; <label>:11                                      ; preds = %9
  %12 = fcmp ogt float %2, 0x47EFFFFFE0000000
  %13 = bitcast float %2 to i32
  %phitmp.i.i.i = select i1 %12, i32 2139095039, i32 %13
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i: ; preds = %11, %9
  %14 = phi i32 [ %phitmp.i.i.i, %11 ], [ 8388608, %9 ]
  %15 = lshr i32 %14, 23
  %16 = add nsw i32 %15, -127
  %17 = and i32 %14, 8388607
  %18 = or i32 %17, 1065353216
  %19 = bitcast i32 %18 to float
  %20 = fadd float %19, -1.000000e+00
  %21 = fmul float %20, %20
  %22 = fmul float %21, %21
  %23 = fmul float %20, 0xBF831161A0000000
  %24 = fadd float %23, 0x3FAAA83920000000
  %25 = fmul float %20, 0x3FDEA2C5A0000000
  %26 = fadd float %25, 0xBFE713CA80000000
  %27 = fmul float %20, %24
  %28 = fadd float %27, 0xBFC19A9FA0000000
  %29 = fmul float %20, %28
  %30 = fadd float %29, 0x3FCEF5B7A0000000
  %31 = fmul float %20, %30
  %32 = fadd float %31, 0xBFD63A40C0000000
  %33 = fmul float %20, %26
  %34 = fadd float %33, 0x3FF7154200000000
  %35 = fmul float %22, %32
  %36 = fmul float %20, %34
  %37 = fadd float %36, %35
  %38 = sitofp i32 %16 to float
  %39 = fadd float %38, %37
  %40 = fmul float %39, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit:   ; preds = %0, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i
  %41 = phi float [ %40, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i ], [ 0.000000e+00, %0 ]
  %42 = fmul float %4, %6
  %43 = getelementptr inbounds i8* %a, i64 4
  %44 = bitcast i8* %43 to float*
  %45 = load float* %44, align 4, !tbaa !1
  %46 = fmul float %42, %45
  %47 = fmul float %7, %41
  %48 = getelementptr inbounds i8* %b, i64 4
  %49 = bitcast i8* %48 to float*
  %50 = load float* %49, align 4, !tbaa !1
  %51 = fmul float %47, %50
  %52 = fadd float %46, %51
  %53 = getelementptr inbounds i8* %a, i64 8
  %54 = bitcast i8* %53 to float*
  %55 = load float* %54, align 4, !tbaa !1
  %56 = fmul float %42, %55
  %57 = getelementptr inbounds i8* %b, i64 8
  %58 = bitcast i8* %57 to float*
  %59 = load float* %58, align 4, !tbaa !1
  %60 = fmul float %47, %59
  %61 = fadd float %56, %60
  %62 = insertelement <2 x float> undef, float %7, i32 0
  %63 = insertelement <2 x float> %62, float %52, i32 1
  %64 = bitcast i8* %r to <2 x float>*
  store <2 x float> %63, <2 x float>* %64, align 4
  %65 = getelementptr inbounds i8* %r, i64 8
  %66 = bitcast i8* %65 to float*
  store float %61, float* %66, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dffdf(i8* nocapture %r, float %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %b to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fadd float %2, -1.000000e+00
  %4 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %a, float %3) #2
  %5 = fmul float %4, %a
  %6 = fcmp ogt float %a, 0.000000e+00
  br i1 %6, label %7, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

; <label>:7                                       ; preds = %0
  %8 = fcmp olt float %a, 0x3810000000000000
  br i1 %8, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, label %9

; <label>:9                                       ; preds = %7
  %10 = fcmp ogt float %a, 0x47EFFFFFE0000000
  %11 = bitcast float %a to i32
  %phitmp.i.i.i = select i1 %10, i32 2139095039, i32 %11
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i: ; preds = %9, %7
  %12 = phi i32 [ %phitmp.i.i.i, %9 ], [ 8388608, %7 ]
  %13 = lshr i32 %12, 23
  %14 = add nsw i32 %13, -127
  %15 = and i32 %12, 8388607
  %16 = or i32 %15, 1065353216
  %17 = bitcast i32 %16 to float
  %18 = fadd float %17, -1.000000e+00
  %19 = fmul float %18, %18
  %20 = fmul float %19, %19
  %21 = fmul float %18, 0xBF831161A0000000
  %22 = fadd float %21, 0x3FAAA83920000000
  %23 = fmul float %18, 0x3FDEA2C5A0000000
  %24 = fadd float %23, 0xBFE713CA80000000
  %25 = fmul float %18, %22
  %26 = fadd float %25, 0xBFC19A9FA0000000
  %27 = fmul float %18, %26
  %28 = fadd float %27, 0x3FCEF5B7A0000000
  %29 = fmul float %18, %28
  %30 = fadd float %29, 0xBFD63A40C0000000
  %31 = fmul float %18, %24
  %32 = fadd float %31, 0x3FF7154200000000
  %33 = fmul float %20, %30
  %34 = fmul float %18, %32
  %35 = fadd float %34, %33
  %36 = sitofp i32 %14 to float
  %37 = fadd float %36, %35
  %38 = fmul float %37, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit:   ; preds = %0, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i
  %39 = phi float [ %38, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i ], [ 0.000000e+00, %0 ]
  %40 = fmul float %2, %4
  %41 = fmul float %40, 0.000000e+00
  %42 = fmul float %5, %39
  %43 = getelementptr inbounds i8* %b, i64 4
  %44 = bitcast i8* %43 to float*
  %45 = load float* %44, align 4, !tbaa !1
  %46 = fmul float %42, %45
  %47 = fadd float %41, %46
  %48 = getelementptr inbounds i8* %b, i64 8
  %49 = bitcast i8* %48 to float*
  %50 = load float* %49, align 4, !tbaa !1
  %51 = fmul float %42, %50
  %52 = fadd float %41, %51
  %53 = insertelement <2 x float> undef, float %5, i32 0
  %54 = insertelement <2 x float> %53, float %47, i32 1
  %55 = bitcast i8* %r to <2 x float>*
  store <2 x float> %54, <2 x float>* %55, align 4
  %56 = getelementptr inbounds i8* %r, i64 8
  %57 = bitcast i8* %56 to float*
  store float %52, float* %57, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dfdff(i8* nocapture %r, i8* nocapture readonly %a, float %b) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fadd float %b, -1.000000e+00
  %4 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %2, float %3) #2
  %5 = fmul float %2, %4
  %6 = fcmp ogt float %2, 0.000000e+00
  br i1 %6, label %7, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

; <label>:7                                       ; preds = %0
  %8 = fcmp olt float %2, 0x3810000000000000
  br i1 %8, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, label %9

; <label>:9                                       ; preds = %7
  %10 = fcmp ogt float %2, 0x47EFFFFFE0000000
  %11 = bitcast float %2 to i32
  %phitmp.i.i.i = select i1 %10, i32 2139095039, i32 %11
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i: ; preds = %9, %7
  %12 = phi i32 [ %phitmp.i.i.i, %9 ], [ 8388608, %7 ]
  %13 = lshr i32 %12, 23
  %14 = add nsw i32 %13, -127
  %15 = and i32 %12, 8388607
  %16 = or i32 %15, 1065353216
  %17 = bitcast i32 %16 to float
  %18 = fadd float %17, -1.000000e+00
  %19 = fmul float %18, %18
  %20 = fmul float %19, %19
  %21 = fmul float %18, 0xBF831161A0000000
  %22 = fadd float %21, 0x3FAAA83920000000
  %23 = fmul float %18, 0x3FDEA2C5A0000000
  %24 = fadd float %23, 0xBFE713CA80000000
  %25 = fmul float %18, %22
  %26 = fadd float %25, 0xBFC19A9FA0000000
  %27 = fmul float %18, %26
  %28 = fadd float %27, 0x3FCEF5B7A0000000
  %29 = fmul float %18, %28
  %30 = fadd float %29, 0xBFD63A40C0000000
  %31 = fmul float %18, %24
  %32 = fadd float %31, 0x3FF7154200000000
  %33 = fmul float %20, %30
  %34 = fmul float %18, %32
  %35 = fadd float %34, %33
  %36 = sitofp i32 %14 to float
  %37 = fadd float %36, %35
  %38 = fmul float %37, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit:   ; preds = %0, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i
  %39 = phi float [ %38, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i ], [ 0.000000e+00, %0 ]
  %40 = fmul float %4, %b
  %41 = getelementptr inbounds i8* %a, i64 4
  %42 = bitcast i8* %41 to float*
  %43 = load float* %42, align 4, !tbaa !1
  %44 = fmul float %40, %43
  %45 = fmul float %5, %39
  %46 = fmul float %45, 0.000000e+00
  %47 = fadd float %44, %46
  %48 = getelementptr inbounds i8* %a, i64 8
  %49 = bitcast i8* %48 to float*
  %50 = load float* %49, align 4, !tbaa !1
  %51 = fmul float %40, %50
  %52 = fadd float %46, %51
  %53 = insertelement <2 x float> undef, float %5, i32 0
  %54 = insertelement <2 x float> %53, float %47, i32 1
  %55 = bitcast i8* %r to <2 x float>*
  store <2 x float> %54, <2 x float>* %55, align 4
  %56 = getelementptr inbounds i8* %r, i64 8
  %57 = bitcast i8* %56 to float*
  store float %52, float* %57, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_vvv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = bitcast i8* %b_ to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %2, float %4)
  %6 = bitcast i8* %r_ to float*
  store float %5, float* %6, align 4, !tbaa !1
  %7 = getelementptr inbounds i8* %a_, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float* %8, align 4, !tbaa !1
  %10 = getelementptr inbounds i8* %b_, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float* %11, align 4, !tbaa !1
  %13 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %9, float %12)
  %14 = getelementptr inbounds i8* %r_, i64 4
  %15 = bitcast i8* %14 to float*
  store float %13, float* %15, align 4, !tbaa !1
  %16 = getelementptr inbounds i8* %a_, i64 8
  %17 = bitcast i8* %16 to float*
  %18 = load float* %17, align 4, !tbaa !1
  %19 = getelementptr inbounds i8* %b_, i64 8
  %20 = bitcast i8* %19 to float*
  %21 = load float* %20, align 4, !tbaa !1
  %22 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %18, float %21)
  %23 = getelementptr inbounds i8* %r_, i64 8
  %24 = bitcast i8* %23 to float*
  store float %22, float* %24, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvdvdv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = bitcast i8* %b_ to float*
  %10 = getelementptr inbounds i8* %b_, i64 12
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8* %b_, i64 24
  %13 = bitcast i8* %12 to float*
  %14 = load float* %9, align 4, !tbaa !1
  %15 = load float* %11, align 4, !tbaa !1
  %16 = load float* %13, align 4, !tbaa !1
  %17 = fadd float %14, -1.000000e+00
  %18 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %6, float %17) #2
  %19 = fmul float %6, %18
  %20 = fcmp ogt float %6, 0.000000e+00
  br i1 %20, label %21, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16

; <label>:21                                      ; preds = %0
  %22 = fcmp olt float %6, 0x3810000000000000
  br i1 %22, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15, label %23

; <label>:23                                      ; preds = %21
  %24 = fcmp ogt float %6, 0x47EFFFFFE0000000
  %25 = bitcast float %6 to i32
  %phitmp.i.i.i14 = select i1 %24, i32 2139095039, i32 %25
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15: ; preds = %23, %21
  %26 = phi i32 [ %phitmp.i.i.i14, %23 ], [ 8388608, %21 ]
  %27 = lshr i32 %26, 23
  %28 = add nsw i32 %27, -127
  %29 = and i32 %26, 8388607
  %30 = or i32 %29, 1065353216
  %31 = bitcast i32 %30 to float
  %32 = fadd float %31, -1.000000e+00
  %33 = fmul float %32, %32
  %34 = fmul float %33, %33
  %35 = fmul float %32, 0xBF831161A0000000
  %36 = fadd float %35, 0x3FAAA83920000000
  %37 = fmul float %32, 0x3FDEA2C5A0000000
  %38 = fadd float %37, 0xBFE713CA80000000
  %39 = fmul float %32, %36
  %40 = fadd float %39, 0xBFC19A9FA0000000
  %41 = fmul float %32, %40
  %42 = fadd float %41, 0x3FCEF5B7A0000000
  %43 = fmul float %32, %42
  %44 = fadd float %43, 0xBFD63A40C0000000
  %45 = fmul float %32, %38
  %46 = fadd float %45, 0x3FF7154200000000
  %47 = fmul float %34, %44
  %48 = fmul float %32, %46
  %49 = fadd float %48, %47
  %50 = sitofp i32 %28 to float
  %51 = fadd float %50, %49
  %52 = fmul float %51, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16: ; preds = %0, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15
  %53 = phi float [ %52, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15 ], [ 0.000000e+00, %0 ]
  %54 = fmul float %14, %18
  %55 = fmul float %7, %54
  %56 = fmul float %19, %53
  %57 = fmul float %15, %56
  %58 = fadd float %55, %57
  %59 = fmul float %8, %54
  %60 = fmul float %16, %56
  %61 = fadd float %59, %60
  %62 = getelementptr inbounds i8* %a_, i64 4
  %63 = bitcast i8* %62 to float*
  %64 = getelementptr inbounds i8* %a_, i64 16
  %65 = bitcast i8* %64 to float*
  %66 = getelementptr inbounds i8* %a_, i64 28
  %67 = bitcast i8* %66 to float*
  %68 = load float* %63, align 4, !tbaa !1
  %69 = load float* %65, align 4, !tbaa !1
  %70 = load float* %67, align 4, !tbaa !1
  %71 = getelementptr inbounds i8* %b_, i64 4
  %72 = bitcast i8* %71 to float*
  %73 = getelementptr inbounds i8* %b_, i64 16
  %74 = bitcast i8* %73 to float*
  %75 = getelementptr inbounds i8* %b_, i64 28
  %76 = bitcast i8* %75 to float*
  %77 = load float* %72, align 4, !tbaa !1
  %78 = load float* %74, align 4, !tbaa !1
  %79 = load float* %76, align 4, !tbaa !1
  %80 = fadd float %77, -1.000000e+00
  %81 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %68, float %80) #2
  %82 = fmul float %68, %81
  %83 = fcmp ogt float %68, 0.000000e+00
  br i1 %83, label %84, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13

; <label>:84                                      ; preds = %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16
  %85 = fcmp olt float %68, 0x3810000000000000
  br i1 %85, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12, label %86

; <label>:86                                      ; preds = %84
  %87 = fcmp ogt float %68, 0x47EFFFFFE0000000
  %88 = bitcast float %68 to i32
  %phitmp.i.i.i11 = select i1 %87, i32 2139095039, i32 %88
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12: ; preds = %86, %84
  %89 = phi i32 [ %phitmp.i.i.i11, %86 ], [ 8388608, %84 ]
  %90 = lshr i32 %89, 23
  %91 = add nsw i32 %90, -127
  %92 = and i32 %89, 8388607
  %93 = or i32 %92, 1065353216
  %94 = bitcast i32 %93 to float
  %95 = fadd float %94, -1.000000e+00
  %96 = fmul float %95, %95
  %97 = fmul float %96, %96
  %98 = fmul float %95, 0xBF831161A0000000
  %99 = fadd float %98, 0x3FAAA83920000000
  %100 = fmul float %95, 0x3FDEA2C5A0000000
  %101 = fadd float %100, 0xBFE713CA80000000
  %102 = fmul float %95, %99
  %103 = fadd float %102, 0xBFC19A9FA0000000
  %104 = fmul float %95, %103
  %105 = fadd float %104, 0x3FCEF5B7A0000000
  %106 = fmul float %95, %105
  %107 = fadd float %106, 0xBFD63A40C0000000
  %108 = fmul float %95, %101
  %109 = fadd float %108, 0x3FF7154200000000
  %110 = fmul float %97, %107
  %111 = fmul float %95, %109
  %112 = fadd float %111, %110
  %113 = sitofp i32 %91 to float
  %114 = fadd float %113, %112
  %115 = fmul float %114, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13: ; preds = %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12
  %116 = phi float [ %115, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12 ], [ 0.000000e+00, %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16 ]
  %117 = fmul float %77, %81
  %118 = fmul float %69, %117
  %119 = fmul float %82, %116
  %120 = fmul float %78, %119
  %121 = fadd float %118, %120
  %122 = fmul float %70, %117
  %123 = fmul float %79, %119
  %124 = fadd float %122, %123
  %125 = getelementptr inbounds i8* %a_, i64 8
  %126 = bitcast i8* %125 to float*
  %127 = getelementptr inbounds i8* %a_, i64 20
  %128 = bitcast i8* %127 to float*
  %129 = getelementptr inbounds i8* %a_, i64 32
  %130 = bitcast i8* %129 to float*
  %131 = load float* %126, align 4, !tbaa !1
  %132 = load float* %128, align 4, !tbaa !1
  %133 = load float* %130, align 4, !tbaa !1
  %134 = getelementptr inbounds i8* %b_, i64 8
  %135 = bitcast i8* %134 to float*
  %136 = getelementptr inbounds i8* %b_, i64 20
  %137 = bitcast i8* %136 to float*
  %138 = getelementptr inbounds i8* %b_, i64 32
  %139 = bitcast i8* %138 to float*
  %140 = load float* %135, align 4, !tbaa !1
  %141 = load float* %137, align 4, !tbaa !1
  %142 = load float* %139, align 4, !tbaa !1
  %143 = fadd float %140, -1.000000e+00
  %144 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %131, float %143) #2
  %145 = fmul float %131, %144
  %146 = fcmp ogt float %131, 0.000000e+00
  br i1 %146, label %147, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

; <label>:147                                     ; preds = %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13
  %148 = fcmp olt float %131, 0x3810000000000000
  br i1 %148, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, label %149

; <label>:149                                     ; preds = %147
  %150 = fcmp ogt float %131, 0x47EFFFFFE0000000
  %151 = bitcast float %131 to i32
  %phitmp.i.i.i = select i1 %150, i32 2139095039, i32 %151
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i: ; preds = %149, %147
  %152 = phi i32 [ %phitmp.i.i.i, %149 ], [ 8388608, %147 ]
  %153 = lshr i32 %152, 23
  %154 = add nsw i32 %153, -127
  %155 = and i32 %152, 8388607
  %156 = or i32 %155, 1065353216
  %157 = bitcast i32 %156 to float
  %158 = fadd float %157, -1.000000e+00
  %159 = fmul float %158, %158
  %160 = fmul float %159, %159
  %161 = fmul float %158, 0xBF831161A0000000
  %162 = fadd float %161, 0x3FAAA83920000000
  %163 = fmul float %158, 0x3FDEA2C5A0000000
  %164 = fadd float %163, 0xBFE713CA80000000
  %165 = fmul float %158, %162
  %166 = fadd float %165, 0xBFC19A9FA0000000
  %167 = fmul float %158, %166
  %168 = fadd float %167, 0x3FCEF5B7A0000000
  %169 = fmul float %158, %168
  %170 = fadd float %169, 0xBFD63A40C0000000
  %171 = fmul float %158, %164
  %172 = fadd float %171, 0x3FF7154200000000
  %173 = fmul float %160, %170
  %174 = fmul float %158, %172
  %175 = fadd float %174, %173
  %176 = sitofp i32 %154 to float
  %177 = fadd float %176, %175
  %178 = fmul float %177, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit:   ; preds = %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i
  %179 = phi float [ %178, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i ], [ 0.000000e+00, %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13 ]
  %180 = fmul float %140, %144
  %181 = fmul float %132, %180
  %182 = fmul float %145, %179
  %183 = fmul float %141, %182
  %184 = fadd float %181, %183
  %185 = fmul float %133, %180
  %186 = fmul float %142, %182
  %187 = fadd float %185, %186
  %188 = bitcast i8* %r_ to float*
  store float %19, float* %188, align 4, !tbaa !5
  %189 = getelementptr inbounds i8* %r_, i64 4
  %190 = bitcast i8* %189 to float*
  store float %82, float* %190, align 4, !tbaa !7
  %191 = getelementptr inbounds i8* %r_, i64 8
  %192 = bitcast i8* %191 to float*
  store float %145, float* %192, align 4, !tbaa !8
  %193 = getelementptr inbounds i8* %r_, i64 12
  %194 = bitcast i8* %193 to float*
  store float %58, float* %194, align 4, !tbaa !5
  %195 = getelementptr inbounds i8* %r_, i64 16
  %196 = bitcast i8* %195 to float*
  store float %121, float* %196, align 4, !tbaa !7
  %197 = getelementptr inbounds i8* %r_, i64 20
  %198 = bitcast i8* %197 to float*
  store float %184, float* %198, align 4, !tbaa !8
  %199 = getelementptr inbounds i8* %r_, i64 24
  %200 = bitcast i8* %199 to float*
  store float %61, float* %200, align 4, !tbaa !5
  %201 = getelementptr inbounds i8* %r_, i64 28
  %202 = bitcast i8* %201 to float*
  store float %124, float* %202, align 4, !tbaa !7
  %203 = getelementptr inbounds i8* %r_, i64 32
  %204 = bitcast i8* %203 to float*
  store float %187, float* %204, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvvdv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %a = alloca %"class.OSL::Dual2.0", align 4
  %1 = bitcast %"class.OSL::Dual2.0"* %a to i8*
  call void @llvm.lifetime.start(i64 36, i8* %1) #2
  %2 = bitcast i8* %a_ to float*
  %3 = load float* %2, align 4, !tbaa !5
  %4 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 0
  store float %3, float* %4, align 4, !tbaa !5
  %5 = getelementptr inbounds i8* %a_, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4, !tbaa !7
  %8 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 1
  store float %7, float* %8, align 4, !tbaa !7
  %9 = getelementptr inbounds i8* %a_, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !8
  %12 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 2
  store float %11, float* %12, align 4, !tbaa !8
  %13 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 1, i32 0
  %14 = bitcast float* %13 to i8*
  call void @llvm.memset.p0i8.i64(i8* %14, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_pow_dvdvdv(i8* %r_, i8* %1, i8* %b_)
  call void @llvm.lifetime.end(i64 36, i8* %1) #2
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvdvv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %b = alloca %"class.OSL::Dual2.0", align 4
  %1 = bitcast %"class.OSL::Dual2.0"* %b to i8*
  call void @llvm.lifetime.start(i64 36, i8* %1) #2
  %2 = bitcast i8* %b_ to float*
  %3 = load float* %2, align 4, !tbaa !5
  %4 = getelementptr inbounds %"class.OSL::Dual2.0"* %b, i64 0, i32 0, i32 0
  store float %3, float* %4, align 4, !tbaa !5
  %5 = getelementptr inbounds i8* %b_, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4, !tbaa !7
  %8 = getelementptr inbounds %"class.OSL::Dual2.0"* %b, i64 0, i32 0, i32 1
  store float %7, float* %8, align 4, !tbaa !7
  %9 = getelementptr inbounds i8* %b_, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !8
  %12 = getelementptr inbounds %"class.OSL::Dual2.0"* %b, i64 0, i32 0, i32 2
  store float %11, float* %12, align 4, !tbaa !8
  %13 = getelementptr inbounds %"class.OSL::Dual2.0"* %b, i64 0, i32 1, i32 0
  %14 = bitcast float* %13 to i8*
  call void @llvm.memset.p0i8.i64(i8* %14, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_pow_dvdvdv(i8* %r_, i8* %a_, i8* %1)
  call void @llvm.lifetime.end(i64 36, i8* %1) #2
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_vvf(i8* nocapture %r_, i8* nocapture readonly %a_, float %b) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %2, float %b)
  %4 = bitcast i8* %r_ to float*
  store float %3, float* %4, align 4, !tbaa !1
  %5 = getelementptr inbounds i8* %a_, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4, !tbaa !1
  %8 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %7, float %b)
  %9 = getelementptr inbounds i8* %r_, i64 4
  %10 = bitcast i8* %9 to float*
  store float %8, float* %10, align 4, !tbaa !1
  %11 = getelementptr inbounds i8* %a_, i64 8
  %12 = bitcast i8* %11 to float*
  %13 = load float* %12, align 4, !tbaa !1
  %14 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %13, float %b)
  %15 = getelementptr inbounds i8* %r_, i64 8
  %16 = bitcast i8* %15 to float*
  store float %14, float* %16, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvdvdf(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = bitcast i8* %b_ to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = fadd float %10, -1.000000e+00
  %12 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %6, float %11) #2
  %13 = fmul float %6, %12
  %14 = fcmp ogt float %6, 0.000000e+00
  br i1 %14, label %15, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16

; <label>:15                                      ; preds = %0
  %16 = fcmp olt float %6, 0x3810000000000000
  br i1 %16, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15, label %17

; <label>:17                                      ; preds = %15
  %18 = fcmp ogt float %6, 0x47EFFFFFE0000000
  %19 = bitcast float %6 to i32
  %phitmp.i.i.i14 = select i1 %18, i32 2139095039, i32 %19
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15: ; preds = %17, %15
  %20 = phi i32 [ %phitmp.i.i.i14, %17 ], [ 8388608, %15 ]
  %21 = lshr i32 %20, 23
  %22 = add nsw i32 %21, -127
  %23 = and i32 %20, 8388607
  %24 = or i32 %23, 1065353216
  %25 = bitcast i32 %24 to float
  %26 = fadd float %25, -1.000000e+00
  %27 = fmul float %26, %26
  %28 = fmul float %27, %27
  %29 = fmul float %26, 0xBF831161A0000000
  %30 = fadd float %29, 0x3FAAA83920000000
  %31 = fmul float %26, 0x3FDEA2C5A0000000
  %32 = fadd float %31, 0xBFE713CA80000000
  %33 = fmul float %26, %30
  %34 = fadd float %33, 0xBFC19A9FA0000000
  %35 = fmul float %26, %34
  %36 = fadd float %35, 0x3FCEF5B7A0000000
  %37 = fmul float %26, %36
  %38 = fadd float %37, 0xBFD63A40C0000000
  %39 = fmul float %26, %32
  %40 = fadd float %39, 0x3FF7154200000000
  %41 = fmul float %28, %38
  %42 = fmul float %26, %40
  %43 = fadd float %42, %41
  %44 = sitofp i32 %22 to float
  %45 = fadd float %44, %43
  %46 = fmul float %45, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16: ; preds = %0, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15
  %47 = phi float [ %46, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i15 ], [ 0.000000e+00, %0 ]
  %48 = fmul float %10, %12
  %49 = fmul float %7, %48
  %50 = fmul float %13, %47
  %51 = getelementptr inbounds i8* %b_, i64 4
  %52 = bitcast i8* %51 to float*
  %53 = load float* %52, align 4, !tbaa !1
  %54 = fmul float %50, %53
  %55 = fadd float %49, %54
  %56 = fmul float %8, %48
  %57 = getelementptr inbounds i8* %b_, i64 8
  %58 = bitcast i8* %57 to float*
  %59 = load float* %58, align 4, !tbaa !1
  %60 = fmul float %50, %59
  %61 = fadd float %56, %60
  %62 = getelementptr inbounds i8* %a_, i64 4
  %63 = bitcast i8* %62 to float*
  %64 = getelementptr inbounds i8* %a_, i64 16
  %65 = bitcast i8* %64 to float*
  %66 = getelementptr inbounds i8* %a_, i64 28
  %67 = bitcast i8* %66 to float*
  %68 = load float* %63, align 4, !tbaa !1
  %69 = load float* %65, align 4, !tbaa !1
  %70 = load float* %67, align 4, !tbaa !1
  %71 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %68, float %11) #2
  %72 = fmul float %68, %71
  %73 = fcmp ogt float %68, 0.000000e+00
  br i1 %73, label %74, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13

; <label>:74                                      ; preds = %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16
  %75 = fcmp olt float %68, 0x3810000000000000
  br i1 %75, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12, label %76

; <label>:76                                      ; preds = %74
  %77 = fcmp ogt float %68, 0x47EFFFFFE0000000
  %78 = bitcast float %68 to i32
  %phitmp.i.i.i11 = select i1 %77, i32 2139095039, i32 %78
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12: ; preds = %76, %74
  %79 = phi i32 [ %phitmp.i.i.i11, %76 ], [ 8388608, %74 ]
  %80 = lshr i32 %79, 23
  %81 = add nsw i32 %80, -127
  %82 = and i32 %79, 8388607
  %83 = or i32 %82, 1065353216
  %84 = bitcast i32 %83 to float
  %85 = fadd float %84, -1.000000e+00
  %86 = fmul float %85, %85
  %87 = fmul float %86, %86
  %88 = fmul float %85, 0xBF831161A0000000
  %89 = fadd float %88, 0x3FAAA83920000000
  %90 = fmul float %85, 0x3FDEA2C5A0000000
  %91 = fadd float %90, 0xBFE713CA80000000
  %92 = fmul float %85, %89
  %93 = fadd float %92, 0xBFC19A9FA0000000
  %94 = fmul float %85, %93
  %95 = fadd float %94, 0x3FCEF5B7A0000000
  %96 = fmul float %85, %95
  %97 = fadd float %96, 0xBFD63A40C0000000
  %98 = fmul float %85, %91
  %99 = fadd float %98, 0x3FF7154200000000
  %100 = fmul float %87, %97
  %101 = fmul float %85, %99
  %102 = fadd float %101, %100
  %103 = sitofp i32 %81 to float
  %104 = fadd float %103, %102
  %105 = fmul float %104, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13: ; preds = %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12
  %106 = phi float [ %105, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i12 ], [ 0.000000e+00, %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit16 ]
  %107 = fmul float %10, %71
  %108 = fmul float %69, %107
  %109 = fmul float %72, %106
  %110 = fmul float %109, %53
  %111 = fadd float %108, %110
  %112 = fmul float %70, %107
  %113 = fmul float %109, %59
  %114 = fadd float %112, %113
  %115 = getelementptr inbounds i8* %a_, i64 8
  %116 = bitcast i8* %115 to float*
  %117 = getelementptr inbounds i8* %a_, i64 20
  %118 = bitcast i8* %117 to float*
  %119 = getelementptr inbounds i8* %a_, i64 32
  %120 = bitcast i8* %119 to float*
  %121 = load float* %116, align 4, !tbaa !1
  %122 = load float* %118, align 4, !tbaa !1
  %123 = load float* %120, align 4, !tbaa !1
  %124 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %121, float %11) #2
  %125 = fmul float %121, %124
  %126 = fcmp ogt float %121, 0.000000e+00
  br i1 %126, label %127, label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

; <label>:127                                     ; preds = %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13
  %128 = fcmp olt float %121, 0x3810000000000000
  br i1 %128, label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i, label %129

; <label>:129                                     ; preds = %127
  %130 = fcmp ogt float %121, 0x47EFFFFFE0000000
  %131 = bitcast float %121 to i32
  %phitmp.i.i.i = select i1 %130, i32 2139095039, i32 %131
  br label %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i

_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i: ; preds = %129, %127
  %132 = phi i32 [ %phitmp.i.i.i, %129 ], [ 8388608, %127 ]
  %133 = lshr i32 %132, 23
  %134 = add nsw i32 %133, -127
  %135 = and i32 %132, 8388607
  %136 = or i32 %135, 1065353216
  %137 = bitcast i32 %136 to float
  %138 = fadd float %137, -1.000000e+00
  %139 = fmul float %138, %138
  %140 = fmul float %139, %139
  %141 = fmul float %138, 0xBF831161A0000000
  %142 = fadd float %141, 0x3FAAA83920000000
  %143 = fmul float %138, 0x3FDEA2C5A0000000
  %144 = fadd float %143, 0xBFE713CA80000000
  %145 = fmul float %138, %142
  %146 = fadd float %145, 0xBFC19A9FA0000000
  %147 = fmul float %138, %146
  %148 = fadd float %147, 0x3FCEF5B7A0000000
  %149 = fmul float %138, %148
  %150 = fadd float %149, 0xBFD63A40C0000000
  %151 = fmul float %138, %144
  %152 = fadd float %151, 0x3FF7154200000000
  %153 = fmul float %140, %150
  %154 = fmul float %138, %152
  %155 = fadd float %154, %153
  %156 = sitofp i32 %134 to float
  %157 = fadd float %156, %155
  %158 = fmul float %157, 0x3FE62E4300000000
  br label %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit

_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit:   ; preds = %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i
  %159 = phi float [ %158, %_ZN11OpenImageIO4v1_78fast_logIfEET_RKS2_.exit.i ], [ 0.000000e+00, %_ZN3OSL13fast_safe_powERKNS_5Dual2IfEES3_.exit13 ]
  %160 = fmul float %10, %124
  %161 = fmul float %122, %160
  %162 = fmul float %125, %159
  %163 = fmul float %162, %53
  %164 = fadd float %161, %163
  %165 = fmul float %123, %160
  %166 = fmul float %162, %59
  %167 = fadd float %165, %166
  %168 = bitcast i8* %r_ to float*
  store float %13, float* %168, align 4, !tbaa !5
  %169 = getelementptr inbounds i8* %r_, i64 4
  %170 = bitcast i8* %169 to float*
  store float %72, float* %170, align 4, !tbaa !7
  %171 = getelementptr inbounds i8* %r_, i64 8
  %172 = bitcast i8* %171 to float*
  store float %125, float* %172, align 4, !tbaa !8
  %173 = getelementptr inbounds i8* %r_, i64 12
  %174 = bitcast i8* %173 to float*
  store float %55, float* %174, align 4, !tbaa !5
  %175 = getelementptr inbounds i8* %r_, i64 16
  %176 = bitcast i8* %175 to float*
  store float %111, float* %176, align 4, !tbaa !7
  %177 = getelementptr inbounds i8* %r_, i64 20
  %178 = bitcast i8* %177 to float*
  store float %164, float* %178, align 4, !tbaa !8
  %179 = getelementptr inbounds i8* %r_, i64 24
  %180 = bitcast i8* %179 to float*
  store float %61, float* %180, align 4, !tbaa !5
  %181 = getelementptr inbounds i8* %r_, i64 28
  %182 = bitcast i8* %181 to float*
  store float %114, float* %182, align 4, !tbaa !7
  %183 = getelementptr inbounds i8* %r_, i64 32
  %184 = bitcast i8* %183 to float*
  store float %167, float* %184, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvvdf(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %a = alloca %"class.OSL::Dual2.0", align 4
  %1 = bitcast %"class.OSL::Dual2.0"* %a to i8*
  call void @llvm.lifetime.start(i64 36, i8* %1) #2
  %2 = bitcast i8* %a_ to float*
  %3 = load float* %2, align 4, !tbaa !5
  %4 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 0
  store float %3, float* %4, align 4, !tbaa !5
  %5 = getelementptr inbounds i8* %a_, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4, !tbaa !7
  %8 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 1
  store float %7, float* %8, align 4, !tbaa !7
  %9 = getelementptr inbounds i8* %a_, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !8
  %12 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 0, i32 2
  store float %11, float* %12, align 4, !tbaa !8
  %13 = getelementptr inbounds %"class.OSL::Dual2.0"* %a, i64 0, i32 1, i32 0
  %14 = bitcast float* %13 to i8*
  call void @llvm.memset.p0i8.i64(i8* %14, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_pow_dvdvdf(i8* %r_, i8* %1, i8* %b_)
  call void @llvm.lifetime.end(i64 36, i8* %1) #2
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvdvf(i8* nocapture %r_, i8* nocapture readonly %a_, float %b_) #4 {
  %b = alloca %"class.OSL::Dual2", align 4
  %1 = getelementptr inbounds %"class.OSL::Dual2"* %b, i64 0, i32 0
  store float %b_, float* %1, align 4, !tbaa !9
  %2 = getelementptr inbounds %"class.OSL::Dual2"* %b, i64 0, i32 1
  store float 0.000000e+00, float* %2, align 4, !tbaa !11
  %3 = getelementptr inbounds %"class.OSL::Dual2"* %b, i64 0, i32 2
  store float 0.000000e+00, float* %3, align 4, !tbaa !12
  %4 = bitcast %"class.OSL::Dual2"* %b to i8*
  call void @osl_pow_dvdvdf(i8* %r_, i8* %a_, i8* %4)
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_erf_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fsub float 1.000000e+00, %1
  %3 = fsub float 1.000000e+00, %2
  %4 = fmul float %3, 0x3F0693ECE0000000
  %5 = fadd float %4, 0x3F32200720000000
  %6 = fmul float %3, %5
  %7 = fadd float %6, 0x3F23ECC0E0000000
  %8 = fmul float %3, %7
  %9 = fadd float %8, 0x3F82FC6D20000000
  %10 = fmul float %3, %9
  %11 = fadd float %10, 0x3FA5A5FCE0000000
  %12 = fmul float %3, %11
  %13 = fadd float %12, 0x3FB20DCCE0000000
  %14 = fmul float %3, %13
  %15 = fadd float %14, 1.000000e+00
  %16 = fmul float %15, %15
  %17 = fmul float %16, %16
  %18 = fmul float %17, %17
  %19 = fmul float %18, %18
  %20 = fdiv float 1.000000e+00, %19
  %21 = fsub float 1.000000e+00, %20
  %22 = tail call float @copysignf(float %21, float %a) #12
  ret float %22
}

; Function Attrs: nounwind uwtable
define void @osl_erf_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @erff(float %2) #12
  %4 = fmul float %2, %2
  %5 = fsub float -0.000000e+00, %4
  %6 = tail call float @expf(float %5) #12
  %7 = fmul float %6, 0x3FF20DD760000000
  %8 = getelementptr inbounds i8* %a, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = fmul float %7, %10
  %12 = getelementptr inbounds i8* %a, i64 8
  %13 = bitcast i8* %12 to float*
  %14 = load float* %13, align 4, !tbaa !1
  %15 = fmul float %7, %14
  %16 = insertelement <2 x float> undef, float %3, i32 0
  %17 = insertelement <2 x float> %16, float %11, i32 1
  %18 = bitcast i8* %r to <2 x float>*
  store <2 x float> %17, <2 x float>* %18, align 4
  %19 = getelementptr inbounds i8* %r, i64 8
  %20 = bitcast i8* %19 to float*
  store float %15, float* %20, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_erf_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fsub float 1.000000e+00, %3
  %5 = fsub float 1.000000e+00, %4
  %6 = fmul float %5, 0x3F0693ECE0000000
  %7 = fadd float %6, 0x3F32200720000000
  %8 = fmul float %5, %7
  %9 = fadd float %8, 0x3F23ECC0E0000000
  %10 = fmul float %5, %9
  %11 = fadd float %10, 0x3F82FC6D20000000
  %12 = fmul float %5, %11
  %13 = fadd float %12, 0x3FA5A5FCE0000000
  %14 = fmul float %5, %13
  %15 = fadd float %14, 0x3FB20DCCE0000000
  %16 = fmul float %5, %15
  %17 = fadd float %16, 1.000000e+00
  %18 = fmul float %17, %17
  %19 = fmul float %18, %18
  %20 = fmul float %19, %19
  %21 = fmul float %20, %20
  %22 = fdiv float 1.000000e+00, %21
  %23 = fsub float 1.000000e+00, %22
  %24 = tail call float @copysignf(float %23, float %2) #12
  %25 = bitcast i8* %r_ to float*
  store float %24, float* %25, align 4, !tbaa !1
  %26 = getelementptr inbounds i8* %a_, i64 4
  %27 = bitcast i8* %26 to float*
  %28 = load float* %27, align 4, !tbaa !1
  %29 = tail call float @fabsf(float %28) #12
  %30 = fsub float 1.000000e+00, %29
  %31 = fsub float 1.000000e+00, %30
  %32 = fmul float %31, 0x3F0693ECE0000000
  %33 = fadd float %32, 0x3F32200720000000
  %34 = fmul float %31, %33
  %35 = fadd float %34, 0x3F23ECC0E0000000
  %36 = fmul float %31, %35
  %37 = fadd float %36, 0x3F82FC6D20000000
  %38 = fmul float %31, %37
  %39 = fadd float %38, 0x3FA5A5FCE0000000
  %40 = fmul float %31, %39
  %41 = fadd float %40, 0x3FB20DCCE0000000
  %42 = fmul float %31, %41
  %43 = fadd float %42, 1.000000e+00
  %44 = fmul float %43, %43
  %45 = fmul float %44, %44
  %46 = fmul float %45, %45
  %47 = fmul float %46, %46
  %48 = fdiv float 1.000000e+00, %47
  %49 = fsub float 1.000000e+00, %48
  %50 = tail call float @copysignf(float %49, float %28) #12
  %51 = getelementptr inbounds i8* %r_, i64 4
  %52 = bitcast i8* %51 to float*
  store float %50, float* %52, align 4, !tbaa !1
  %53 = getelementptr inbounds i8* %a_, i64 8
  %54 = bitcast i8* %53 to float*
  %55 = load float* %54, align 4, !tbaa !1
  %56 = tail call float @fabsf(float %55) #12
  %57 = fsub float 1.000000e+00, %56
  %58 = fsub float 1.000000e+00, %57
  %59 = fmul float %58, 0x3F0693ECE0000000
  %60 = fadd float %59, 0x3F32200720000000
  %61 = fmul float %58, %60
  %62 = fadd float %61, 0x3F23ECC0E0000000
  %63 = fmul float %58, %62
  %64 = fadd float %63, 0x3F82FC6D20000000
  %65 = fmul float %58, %64
  %66 = fadd float %65, 0x3FA5A5FCE0000000
  %67 = fmul float %58, %66
  %68 = fadd float %67, 0x3FB20DCCE0000000
  %69 = fmul float %58, %68
  %70 = fadd float %69, 1.000000e+00
  %71 = fmul float %70, %70
  %72 = fmul float %71, %71
  %73 = fmul float %72, %72
  %74 = fmul float %73, %73
  %75 = fdiv float 1.000000e+00, %74
  %76 = fsub float 1.000000e+00, %75
  %77 = tail call float @copysignf(float %76, float %55) #12
  %78 = getelementptr inbounds i8* %r_, i64 8
  %79 = bitcast i8* %78 to float*
  store float %77, float* %79, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_erf_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = load float* %1, align 4, !tbaa !1
  %4 = tail call float @erff(float %3) #12
  %5 = fmul float %3, %3
  %6 = fsub float -0.000000e+00, %5
  %7 = tail call float @expf(float %6) #12
  %8 = fmul float %7, 0x3FF20DD760000000
  %9 = getelementptr inbounds i8* %a_, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = getelementptr inbounds i8* %a_, i64 28
  %12 = bitcast i8* %11 to float*
  %13 = load float* %10, align 4, !tbaa !1
  %14 = load float* %12, align 4, !tbaa !1
  %15 = tail call float @erff(float %13) #12
  %16 = fmul float %13, %13
  %17 = fsub float -0.000000e+00, %16
  %18 = tail call float @expf(float %17) #12
  %19 = fmul float %18, 0x3FF20DD760000000
  %20 = fmul float %14, %19
  %21 = getelementptr inbounds i8* %a_, i64 8
  %22 = bitcast i8* %21 to float*
  %23 = getelementptr inbounds i8* %a_, i64 32
  %24 = bitcast i8* %23 to float*
  %25 = load float* %22, align 4, !tbaa !1
  %26 = bitcast i8* %2 to <4 x float>*
  %27 = load <4 x float>* %26, align 4, !tbaa !1
  %28 = load float* %24, align 4, !tbaa !1
  %29 = tail call float @erff(float %25) #12
  %30 = fmul float %25, %25
  %31 = fsub float -0.000000e+00, %30
  %32 = tail call float @expf(float %31) #12
  %33 = fmul float %32, 0x3FF20DD760000000
  %34 = insertelement <4 x float> undef, float %8, i32 0
  %35 = insertelement <4 x float> %34, float %19, i32 1
  %36 = insertelement <4 x float> %35, float %33, i32 2
  %37 = insertelement <4 x float> %36, float %8, i32 3
  %38 = fmul <4 x float> %27, %37
  %39 = fmul float %28, %33
  %40 = bitcast i8* %r_ to float*
  store float %4, float* %40, align 4, !tbaa !5
  %41 = getelementptr inbounds i8* %r_, i64 4
  %42 = bitcast i8* %41 to float*
  store float %15, float* %42, align 4, !tbaa !7
  %43 = getelementptr inbounds i8* %r_, i64 8
  %44 = bitcast i8* %43 to float*
  store float %29, float* %44, align 4, !tbaa !8
  %45 = getelementptr inbounds i8* %r_, i64 12
  %46 = bitcast i8* %45 to <4 x float>*
  store <4 x float> %38, <4 x float>* %46, align 4, !tbaa !1
  %47 = getelementptr inbounds i8* %r_, i64 28
  %48 = bitcast i8* %47 to float*
  store float %20, float* %48, align 4, !tbaa !7
  %49 = getelementptr inbounds i8* %r_, i64 32
  %50 = bitcast i8* %49 to float*
  store float %39, float* %50, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_erfc_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  %2 = fsub float 1.000000e+00, %1
  %3 = fsub float 1.000000e+00, %2
  %4 = fmul float %3, 0x3F0693ECE0000000
  %5 = fadd float %4, 0x3F32200720000000
  %6 = fmul float %3, %5
  %7 = fadd float %6, 0x3F23ECC0E0000000
  %8 = fmul float %3, %7
  %9 = fadd float %8, 0x3F82FC6D20000000
  %10 = fmul float %3, %9
  %11 = fadd float %10, 0x3FA5A5FCE0000000
  %12 = fmul float %3, %11
  %13 = fadd float %12, 0x3FB20DCCE0000000
  %14 = fmul float %3, %13
  %15 = fadd float %14, 1.000000e+00
  %16 = fmul float %15, %15
  %17 = fmul float %16, %16
  %18 = fmul float %17, %17
  %19 = fmul float %18, %18
  %20 = fdiv float 1.000000e+00, %19
  %21 = fsub float 1.000000e+00, %20
  %22 = tail call float @copysignf(float %21, float %a) #12
  %23 = fsub float 1.000000e+00, %22
  ret float %23
}

; Function Attrs: nounwind uwtable
define void @osl_erfc_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @erfcf(float %2) #12
  %4 = fmul float %2, %2
  %5 = fsub float -0.000000e+00, %4
  %6 = tail call float @expf(float %5) #12
  %7 = fmul float %6, 0xBFF20DD760000000
  %8 = getelementptr inbounds i8* %a, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = fmul float %7, %10
  %12 = getelementptr inbounds i8* %a, i64 8
  %13 = bitcast i8* %12 to float*
  %14 = load float* %13, align 4, !tbaa !1
  %15 = fmul float %7, %14
  %16 = insertelement <2 x float> undef, float %3, i32 0
  %17 = insertelement <2 x float> %16, float %11, i32 1
  %18 = bitcast i8* %r to <2 x float>*
  store <2 x float> %17, <2 x float>* %18, align 4
  %19 = getelementptr inbounds i8* %r, i64 8
  %20 = bitcast i8* %19 to float*
  store float %15, float* %20, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_erfc_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fsub float 1.000000e+00, %3
  %5 = fsub float 1.000000e+00, %4
  %6 = fmul float %5, 0x3F0693ECE0000000
  %7 = fadd float %6, 0x3F32200720000000
  %8 = fmul float %5, %7
  %9 = fadd float %8, 0x3F23ECC0E0000000
  %10 = fmul float %5, %9
  %11 = fadd float %10, 0x3F82FC6D20000000
  %12 = fmul float %5, %11
  %13 = fadd float %12, 0x3FA5A5FCE0000000
  %14 = fmul float %5, %13
  %15 = fadd float %14, 0x3FB20DCCE0000000
  %16 = fmul float %5, %15
  %17 = fadd float %16, 1.000000e+00
  %18 = fmul float %17, %17
  %19 = fmul float %18, %18
  %20 = fmul float %19, %19
  %21 = fmul float %20, %20
  %22 = fdiv float 1.000000e+00, %21
  %23 = fsub float 1.000000e+00, %22
  %24 = tail call float @copysignf(float %23, float %2) #12
  %25 = fsub float 1.000000e+00, %24
  %26 = bitcast i8* %r_ to float*
  store float %25, float* %26, align 4, !tbaa !1
  %27 = getelementptr inbounds i8* %a_, i64 4
  %28 = bitcast i8* %27 to float*
  %29 = load float* %28, align 4, !tbaa !1
  %30 = tail call float @fabsf(float %29) #12
  %31 = fsub float 1.000000e+00, %30
  %32 = fsub float 1.000000e+00, %31
  %33 = fmul float %32, 0x3F0693ECE0000000
  %34 = fadd float %33, 0x3F32200720000000
  %35 = fmul float %32, %34
  %36 = fadd float %35, 0x3F23ECC0E0000000
  %37 = fmul float %32, %36
  %38 = fadd float %37, 0x3F82FC6D20000000
  %39 = fmul float %32, %38
  %40 = fadd float %39, 0x3FA5A5FCE0000000
  %41 = fmul float %32, %40
  %42 = fadd float %41, 0x3FB20DCCE0000000
  %43 = fmul float %32, %42
  %44 = fadd float %43, 1.000000e+00
  %45 = fmul float %44, %44
  %46 = fmul float %45, %45
  %47 = fmul float %46, %46
  %48 = fmul float %47, %47
  %49 = fdiv float 1.000000e+00, %48
  %50 = fsub float 1.000000e+00, %49
  %51 = tail call float @copysignf(float %50, float %29) #12
  %52 = fsub float 1.000000e+00, %51
  %53 = getelementptr inbounds i8* %r_, i64 4
  %54 = bitcast i8* %53 to float*
  store float %52, float* %54, align 4, !tbaa !1
  %55 = getelementptr inbounds i8* %a_, i64 8
  %56 = bitcast i8* %55 to float*
  %57 = load float* %56, align 4, !tbaa !1
  %58 = tail call float @fabsf(float %57) #12
  %59 = fsub float 1.000000e+00, %58
  %60 = fsub float 1.000000e+00, %59
  %61 = fmul float %60, 0x3F0693ECE0000000
  %62 = fadd float %61, 0x3F32200720000000
  %63 = fmul float %60, %62
  %64 = fadd float %63, 0x3F23ECC0E0000000
  %65 = fmul float %60, %64
  %66 = fadd float %65, 0x3F82FC6D20000000
  %67 = fmul float %60, %66
  %68 = fadd float %67, 0x3FA5A5FCE0000000
  %69 = fmul float %60, %68
  %70 = fadd float %69, 0x3FB20DCCE0000000
  %71 = fmul float %60, %70
  %72 = fadd float %71, 1.000000e+00
  %73 = fmul float %72, %72
  %74 = fmul float %73, %73
  %75 = fmul float %74, %74
  %76 = fmul float %75, %75
  %77 = fdiv float 1.000000e+00, %76
  %78 = fsub float 1.000000e+00, %77
  %79 = tail call float @copysignf(float %78, float %57) #12
  %80 = fsub float 1.000000e+00, %79
  %81 = getelementptr inbounds i8* %r_, i64 8
  %82 = bitcast i8* %81 to float*
  store float %80, float* %82, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_erfc_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = load float* %1, align 4, !tbaa !1
  %4 = tail call float @erfcf(float %3) #12
  %5 = fmul float %3, %3
  %6 = fsub float -0.000000e+00, %5
  %7 = tail call float @expf(float %6) #12
  %8 = fmul float %7, 0xBFF20DD760000000
  %9 = getelementptr inbounds i8* %a_, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = getelementptr inbounds i8* %a_, i64 28
  %12 = bitcast i8* %11 to float*
  %13 = load float* %10, align 4, !tbaa !1
  %14 = load float* %12, align 4, !tbaa !1
  %15 = tail call float @erfcf(float %13) #12
  %16 = fmul float %13, %13
  %17 = fsub float -0.000000e+00, %16
  %18 = tail call float @expf(float %17) #12
  %19 = fmul float %18, 0xBFF20DD760000000
  %20 = fmul float %14, %19
  %21 = getelementptr inbounds i8* %a_, i64 8
  %22 = bitcast i8* %21 to float*
  %23 = getelementptr inbounds i8* %a_, i64 32
  %24 = bitcast i8* %23 to float*
  %25 = load float* %22, align 4, !tbaa !1
  %26 = bitcast i8* %2 to <4 x float>*
  %27 = load <4 x float>* %26, align 4, !tbaa !1
  %28 = load float* %24, align 4, !tbaa !1
  %29 = tail call float @erfcf(float %25) #12
  %30 = fmul float %25, %25
  %31 = fsub float -0.000000e+00, %30
  %32 = tail call float @expf(float %31) #12
  %33 = fmul float %32, 0xBFF20DD760000000
  %34 = insertelement <4 x float> undef, float %8, i32 0
  %35 = insertelement <4 x float> %34, float %19, i32 1
  %36 = insertelement <4 x float> %35, float %33, i32 2
  %37 = insertelement <4 x float> %36, float %8, i32 3
  %38 = fmul <4 x float> %27, %37
  %39 = fmul float %28, %33
  %40 = bitcast i8* %r_ to float*
  store float %4, float* %40, align 4, !tbaa !5
  %41 = getelementptr inbounds i8* %r_, i64 4
  %42 = bitcast i8* %41 to float*
  store float %15, float* %42, align 4, !tbaa !7
  %43 = getelementptr inbounds i8* %r_, i64 8
  %44 = bitcast i8* %43 to float*
  store float %29, float* %44, align 4, !tbaa !8
  %45 = getelementptr inbounds i8* %r_, i64 12
  %46 = bitcast i8* %45 to <4 x float>*
  store <4 x float> %38, <4 x float>* %46, align 4, !tbaa !1
  %47 = getelementptr inbounds i8* %r_, i64 28
  %48 = bitcast i8* %47 to float*
  store float %20, float* %48, align 4, !tbaa !7
  %49 = getelementptr inbounds i8* %r_, i64 32
  %50 = bitcast i8* %49 to float*
  store float %39, float* %50, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_sqrt_ff(float %a) #3 {
  %1 = fcmp ult float %a, 0.000000e+00
  br i1 %1, label %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit, label %2

; <label>:2                                       ; preds = %0
  %3 = tail call float @sqrtf(float %a) #12
  br label %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit

_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit:    ; preds = %0, %2
  %4 = phi float [ %3, %2 ], [ 0.000000e+00, %0 ]
  ret float %4
}

; Function Attrs: nounwind uwtable
define void @osl_sqrt_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp ugt float %2, 0.000000e+00
  br i1 %3, label %4, label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit

; <label>:4                                       ; preds = %0
  %5 = tail call float @sqrtf(float %2) #12
  %6 = fmul float %5, 2.000000e+00
  %7 = fdiv float 1.000000e+00, %6
  %8 = getelementptr inbounds i8* %a, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = fmul float %7, %10
  %12 = getelementptr inbounds i8* %a, i64 8
  %13 = bitcast i8* %12 to float*
  %14 = load float* %13, align 4, !tbaa !1
  %15 = fmul float %7, %14
  %16 = insertelement <2 x float> undef, float %5, i32 0
  %17 = insertelement <2 x float> %16, float %11, i32 1
  br label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit

_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit:         ; preds = %0, %4
  %18 = phi float [ %15, %4 ], [ 0.000000e+00, %0 ]
  %19 = phi <2 x float> [ %17, %4 ], [ zeroinitializer, %0 ]
  %20 = bitcast i8* %r to <2 x float>*
  store <2 x float> %19, <2 x float>* %20, align 4
  %21 = getelementptr inbounds i8* %r, i64 8
  %22 = bitcast i8* %21 to float*
  store float %18, float* %22, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sqrt_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp ult float %2, 0.000000e+00
  br i1 %3, label %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit1, label %4

; <label>:4                                       ; preds = %0
  %5 = tail call float @sqrtf(float %2) #12
  br label %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit1

_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit1:   ; preds = %0, %4
  %6 = phi float [ %5, %4 ], [ 0.000000e+00, %0 ]
  %7 = bitcast i8* %r_ to float*
  store float %6, float* %7, align 4, !tbaa !1
  %8 = getelementptr inbounds i8* %a_, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = fcmp ult float %10, 0.000000e+00
  br i1 %11, label %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit2, label %12

; <label>:12                                      ; preds = %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit1
  %13 = tail call float @sqrtf(float %10) #12
  br label %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit2

_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit2:   ; preds = %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit1, %12
  %14 = phi float [ %13, %12 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit1 ]
  %15 = getelementptr inbounds i8* %r_, i64 4
  %16 = bitcast i8* %15 to float*
  store float %14, float* %16, align 4, !tbaa !1
  %17 = getelementptr inbounds i8* %a_, i64 8
  %18 = bitcast i8* %17 to float*
  %19 = load float* %18, align 4, !tbaa !1
  %20 = fcmp ult float %19, 0.000000e+00
  br i1 %20, label %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit, label %21

; <label>:21                                      ; preds = %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit2
  %22 = tail call float @sqrtf(float %19) #12
  br label %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit

_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit:    ; preds = %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit2, %21
  %23 = phi float [ %22, %21 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_79safe_sqrtIfEET_S2_.exit2 ]
  %24 = getelementptr inbounds i8* %r_, i64 8
  %25 = bitcast i8* %24 to float*
  store float %23, float* %25, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sqrt_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp ugt float %2, 0.000000e+00
  br i1 %3, label %4, label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit12

; <label>:4                                       ; preds = %0
  %5 = getelementptr inbounds i8* %a_, i64 24
  %6 = getelementptr inbounds i8* %a_, i64 12
  %7 = bitcast i8* %5 to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = bitcast i8* %6 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = tail call float @sqrtf(float %2) #12
  %12 = fmul float %11, 2.000000e+00
  %13 = fdiv float 1.000000e+00, %12
  %14 = fmul float %10, %13
  %15 = fmul float %8, %13
  %16 = insertelement <2 x float> undef, float %11, i32 0
  %17 = insertelement <2 x float> %16, float %14, i32 1
  br label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit12

_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit12:       ; preds = %0, %4
  %18 = phi float [ %15, %4 ], [ 0.000000e+00, %0 ]
  %19 = phi <2 x float> [ %17, %4 ], [ zeroinitializer, %0 ]
  %20 = getelementptr inbounds i8* %a_, i64 4
  %21 = bitcast i8* %20 to float*
  %22 = load float* %21, align 4, !tbaa !1
  %23 = fcmp ugt float %22, 0.000000e+00
  br i1 %23, label %24, label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit11

; <label>:24                                      ; preds = %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit12
  %25 = getelementptr inbounds i8* %a_, i64 28
  %26 = bitcast i8* %25 to float*
  %27 = load float* %26, align 4, !tbaa !1
  %28 = getelementptr inbounds i8* %a_, i64 16
  %29 = bitcast i8* %28 to float*
  %30 = load float* %29, align 4, !tbaa !1
  %31 = tail call float @sqrtf(float %22) #12
  %32 = fmul float %31, 2.000000e+00
  %33 = fdiv float 1.000000e+00, %32
  %34 = fmul float %30, %33
  %35 = fmul float %27, %33
  %36 = insertelement <2 x float> undef, float %31, i32 0
  %37 = insertelement <2 x float> %36, float %34, i32 1
  br label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit11

_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit11:       ; preds = %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit12, %24
  %38 = phi float [ %35, %24 ], [ 0.000000e+00, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit12 ]
  %39 = phi <2 x float> [ %37, %24 ], [ zeroinitializer, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit12 ]
  %40 = getelementptr inbounds i8* %a_, i64 8
  %41 = bitcast i8* %40 to float*
  %42 = load float* %41, align 4, !tbaa !1
  %43 = fcmp ugt float %42, 0.000000e+00
  br i1 %43, label %44, label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit

; <label>:44                                      ; preds = %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit11
  %45 = getelementptr inbounds i8* %a_, i64 32
  %46 = bitcast i8* %45 to float*
  %47 = load float* %46, align 4, !tbaa !1
  %48 = getelementptr inbounds i8* %a_, i64 20
  %49 = bitcast i8* %48 to float*
  %50 = load float* %49, align 4, !tbaa !1
  %51 = tail call float @sqrtf(float %42) #12
  %52 = fmul float %51, 2.000000e+00
  %53 = fdiv float 1.000000e+00, %52
  %54 = fmul float %50, %53
  %55 = fmul float %47, %53
  %56 = insertelement <2 x float> undef, float %51, i32 0
  %57 = insertelement <2 x float> %56, float %54, i32 1
  br label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit

_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit:         ; preds = %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit11, %44
  %58 = phi float [ %55, %44 ], [ 0.000000e+00, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit11 ]
  %59 = phi <2 x float> [ %57, %44 ], [ zeroinitializer, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit11 ]
  %60 = extractelement <2 x float> %19, i32 0
  %61 = extractelement <2 x float> %39, i32 0
  %62 = extractelement <2 x float> %59, i32 0
  %63 = extractelement <2 x float> %19, i32 1
  %64 = extractelement <2 x float> %39, i32 1
  %65 = extractelement <2 x float> %59, i32 1
  %66 = bitcast i8* %r_ to float*
  store float %60, float* %66, align 4, !tbaa !5
  %67 = getelementptr inbounds i8* %r_, i64 4
  %68 = bitcast i8* %67 to float*
  store float %61, float* %68, align 4, !tbaa !7
  %69 = getelementptr inbounds i8* %r_, i64 8
  %70 = bitcast i8* %69 to float*
  store float %62, float* %70, align 4, !tbaa !8
  %71 = getelementptr inbounds i8* %r_, i64 12
  %72 = bitcast i8* %71 to float*
  store float %63, float* %72, align 4, !tbaa !5
  %73 = getelementptr inbounds i8* %r_, i64 16
  %74 = bitcast i8* %73 to float*
  store float %64, float* %74, align 4, !tbaa !7
  %75 = getelementptr inbounds i8* %r_, i64 20
  %76 = bitcast i8* %75 to float*
  store float %65, float* %76, align 4, !tbaa !8
  %77 = getelementptr inbounds i8* %r_, i64 24
  %78 = bitcast i8* %77 to float*
  store float %18, float* %78, align 4, !tbaa !5
  %79 = getelementptr inbounds i8* %r_, i64 28
  %80 = bitcast i8* %79 to float*
  store float %38, float* %80, align 4, !tbaa !7
  %81 = getelementptr inbounds i8* %r_, i64 32
  %82 = bitcast i8* %81 to float*
  store float %58, float* %82, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_inversesqrt_ff(float %a) #3 {
  %1 = fcmp ogt float %a, 0.000000e+00
  br i1 %1, label %2, label %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit

; <label>:2                                       ; preds = %0
  %3 = tail call float @sqrtf(float %a) #12
  %4 = fdiv float 1.000000e+00, %3
  br label %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit

_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit: ; preds = %0, %2
  %5 = phi float [ %4, %2 ], [ 0.000000e+00, %0 ]
  ret float %5
}

; Function Attrs: nounwind uwtable
define void @osl_inversesqrt_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp ugt float %2, 0.000000e+00
  br i1 %3, label %4, label %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit

; <label>:4                                       ; preds = %0
  %5 = tail call float @sqrtf(float %2) #12
  %6 = fmul float %2, 2.000000e+00
  %7 = fmul float %6, %5
  %8 = fdiv float -1.000000e+00, %7
  %9 = fdiv float 1.000000e+00, %5
  %10 = getelementptr inbounds i8* %a, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float* %11, align 4, !tbaa !1
  %13 = fmul float %8, %12
  %14 = getelementptr inbounds i8* %a, i64 8
  %15 = bitcast i8* %14 to float*
  %16 = load float* %15, align 4, !tbaa !1
  %17 = fmul float %8, %16
  %18 = insertelement <2 x float> undef, float %9, i32 0
  %19 = insertelement <2 x float> %18, float %13, i32 1
  br label %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit

_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit: ; preds = %0, %4
  %20 = phi float [ %17, %4 ], [ 0.000000e+00, %0 ]
  %21 = phi <2 x float> [ %19, %4 ], [ zeroinitializer, %0 ]
  %22 = bitcast i8* %r to <2 x float>*
  store <2 x float> %21, <2 x float>* %22, align 4
  %23 = getelementptr inbounds i8* %r, i64 8
  %24 = bitcast i8* %23 to float*
  store float %20, float* %24, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_inversesqrt_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp ogt float %2, 0.000000e+00
  br i1 %3, label %4, label %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit1

; <label>:4                                       ; preds = %0
  %5 = tail call float @sqrtf(float %2) #12
  %6 = fdiv float 1.000000e+00, %5
  br label %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit1

_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit1: ; preds = %0, %4
  %7 = phi float [ %6, %4 ], [ 0.000000e+00, %0 ]
  %8 = bitcast i8* %r_ to float*
  store float %7, float* %8, align 4, !tbaa !1
  %9 = getelementptr inbounds i8* %a_, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !1
  %12 = fcmp ogt float %11, 0.000000e+00
  br i1 %12, label %13, label %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit2

; <label>:13                                      ; preds = %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit1
  %14 = tail call float @sqrtf(float %11) #12
  %15 = fdiv float 1.000000e+00, %14
  br label %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit2

_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit2: ; preds = %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit1, %13
  %16 = phi float [ %15, %13 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit1 ]
  %17 = getelementptr inbounds i8* %r_, i64 4
  %18 = bitcast i8* %17 to float*
  store float %16, float* %18, align 4, !tbaa !1
  %19 = getelementptr inbounds i8* %a_, i64 8
  %20 = bitcast i8* %19 to float*
  %21 = load float* %20, align 4, !tbaa !1
  %22 = fcmp ogt float %21, 0.000000e+00
  br i1 %22, label %23, label %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit

; <label>:23                                      ; preds = %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit2
  %24 = tail call float @sqrtf(float %21) #12
  %25 = fdiv float 1.000000e+00, %24
  br label %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit

_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit: ; preds = %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit2, %23
  %26 = phi float [ %25, %23 ], [ 0.000000e+00, %_ZN11OpenImageIO4v1_716safe_inversesqrtIfEET_S2_.exit2 ]
  %27 = getelementptr inbounds i8* %r_, i64 8
  %28 = bitcast i8* %27 to float*
  store float %26, float* %28, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_inversesqrt_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp ugt float %2, 0.000000e+00
  br i1 %3, label %4, label %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit12

; <label>:4                                       ; preds = %0
  %5 = getelementptr inbounds i8* %a_, i64 24
  %6 = getelementptr inbounds i8* %a_, i64 12
  %7 = bitcast i8* %5 to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = bitcast i8* %6 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = tail call float @sqrtf(float %2) #12
  %12 = fmul float %2, 2.000000e+00
  %13 = fmul float %12, %11
  %14 = fdiv float -1.000000e+00, %13
  %15 = fdiv float 1.000000e+00, %11
  %16 = fmul float %10, %14
  %17 = fmul float %8, %14
  %18 = insertelement <2 x float> undef, float %15, i32 0
  %19 = insertelement <2 x float> %18, float %16, i32 1
  br label %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit12

_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit12: ; preds = %0, %4
  %20 = phi float [ %17, %4 ], [ 0.000000e+00, %0 ]
  %21 = phi <2 x float> [ %19, %4 ], [ zeroinitializer, %0 ]
  %22 = getelementptr inbounds i8* %a_, i64 4
  %23 = bitcast i8* %22 to float*
  %24 = load float* %23, align 4, !tbaa !1
  %25 = fcmp ugt float %24, 0.000000e+00
  br i1 %25, label %26, label %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit11

; <label>:26                                      ; preds = %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit12
  %27 = getelementptr inbounds i8* %a_, i64 28
  %28 = bitcast i8* %27 to float*
  %29 = load float* %28, align 4, !tbaa !1
  %30 = getelementptr inbounds i8* %a_, i64 16
  %31 = bitcast i8* %30 to float*
  %32 = load float* %31, align 4, !tbaa !1
  %33 = tail call float @sqrtf(float %24) #12
  %34 = fmul float %24, 2.000000e+00
  %35 = fmul float %34, %33
  %36 = fdiv float -1.000000e+00, %35
  %37 = fdiv float 1.000000e+00, %33
  %38 = fmul float %32, %36
  %39 = fmul float %29, %36
  %40 = insertelement <2 x float> undef, float %37, i32 0
  %41 = insertelement <2 x float> %40, float %38, i32 1
  br label %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit11

_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit11: ; preds = %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit12, %26
  %42 = phi float [ %39, %26 ], [ 0.000000e+00, %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit12 ]
  %43 = phi <2 x float> [ %41, %26 ], [ zeroinitializer, %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit12 ]
  %44 = getelementptr inbounds i8* %a_, i64 8
  %45 = bitcast i8* %44 to float*
  %46 = load float* %45, align 4, !tbaa !1
  %47 = fcmp ugt float %46, 0.000000e+00
  br i1 %47, label %48, label %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit

; <label>:48                                      ; preds = %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit11
  %49 = getelementptr inbounds i8* %a_, i64 32
  %50 = bitcast i8* %49 to float*
  %51 = load float* %50, align 4, !tbaa !1
  %52 = getelementptr inbounds i8* %a_, i64 20
  %53 = bitcast i8* %52 to float*
  %54 = load float* %53, align 4, !tbaa !1
  %55 = tail call float @sqrtf(float %46) #12
  %56 = fmul float %46, 2.000000e+00
  %57 = fmul float %56, %55
  %58 = fdiv float -1.000000e+00, %57
  %59 = fdiv float 1.000000e+00, %55
  %60 = fmul float %54, %58
  %61 = fmul float %51, %58
  %62 = insertelement <2 x float> undef, float %59, i32 0
  %63 = insertelement <2 x float> %62, float %60, i32 1
  br label %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit

_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit: ; preds = %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit11, %48
  %64 = phi float [ %61, %48 ], [ 0.000000e+00, %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit11 ]
  %65 = phi <2 x float> [ %63, %48 ], [ zeroinitializer, %_ZN3OSL11inversesqrtIfEENS_5Dual2IT_EERKS3_.exit11 ]
  %66 = extractelement <2 x float> %21, i32 0
  %67 = extractelement <2 x float> %43, i32 0
  %68 = extractelement <2 x float> %65, i32 0
  %69 = extractelement <2 x float> %21, i32 1
  %70 = extractelement <2 x float> %43, i32 1
  %71 = extractelement <2 x float> %65, i32 1
  %72 = bitcast i8* %r_ to float*
  store float %66, float* %72, align 4, !tbaa !5
  %73 = getelementptr inbounds i8* %r_, i64 4
  %74 = bitcast i8* %73 to float*
  store float %67, float* %74, align 4, !tbaa !7
  %75 = getelementptr inbounds i8* %r_, i64 8
  %76 = bitcast i8* %75 to float*
  store float %68, float* %76, align 4, !tbaa !8
  %77 = getelementptr inbounds i8* %r_, i64 12
  %78 = bitcast i8* %77 to float*
  store float %69, float* %78, align 4, !tbaa !5
  %79 = getelementptr inbounds i8* %r_, i64 16
  %80 = bitcast i8* %79 to float*
  store float %70, float* %80, align 4, !tbaa !7
  %81 = getelementptr inbounds i8* %r_, i64 20
  %82 = bitcast i8* %81 to float*
  store float %71, float* %82, align 4, !tbaa !8
  %83 = getelementptr inbounds i8* %r_, i64 24
  %84 = bitcast i8* %83 to float*
  store float %20, float* %84, align 4, !tbaa !5
  %85 = getelementptr inbounds i8* %r_, i64 28
  %86 = bitcast i8* %85 to float*
  store float %42, float* %86, align 4, !tbaa !7
  %87 = getelementptr inbounds i8* %r_, i64 32
  %88 = bitcast i8* %87 to float*
  store float %64, float* %88, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_logb_ff(float %x) #3 {
  %1 = tail call float @fabsf(float %x) #12
  %2 = fcmp olt float %1, 0x3810000000000000
  br i1 %2, label %_ZN11OpenImageIO4v1_79fast_logbEf.exit, label %3

; <label>:3                                       ; preds = %0
  %4 = fcmp ogt float %1, 0x47EFFFFFE0000000
  br i1 %4, label %5, label %_ZN11OpenImageIO4v1_79fast_logbEf.exit

; <label>:5                                       ; preds = %3
  br label %_ZN11OpenImageIO4v1_79fast_logbEf.exit

_ZN11OpenImageIO4v1_79fast_logbEf.exit:           ; preds = %0, %3, %5
  %.1.i = phi float [ 0x47EFFFFFE0000000, %5 ], [ %1, %3 ], [ 0x3810000000000000, %0 ]
  %6 = bitcast float %.1.i to i32
  %7 = lshr i32 %6, 23
  %8 = add nsw i32 %7, -127
  %9 = sitofp i32 %8 to float
  ret float %9
}

; Function Attrs: nounwind uwtable
define void @osl_logb_vv(i8* nocapture %r, i8* nocapture readonly %x_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = fcmp olt float %3, 0x3810000000000000
  br i1 %4, label %_ZN11OpenImageIO4v1_79fast_logbEf.exit2, label %5

; <label>:5                                       ; preds = %0
  %6 = fcmp ogt float %3, 0x47EFFFFFE0000000
  br i1 %6, label %7, label %_ZN11OpenImageIO4v1_79fast_logbEf.exit2

; <label>:7                                       ; preds = %5
  br label %_ZN11OpenImageIO4v1_79fast_logbEf.exit2

_ZN11OpenImageIO4v1_79fast_logbEf.exit2:          ; preds = %0, %5, %7
  %.1.i1 = phi float [ 0x47EFFFFFE0000000, %7 ], [ %3, %5 ], [ 0x3810000000000000, %0 ]
  %8 = bitcast float %.1.i1 to i32
  %9 = lshr i32 %8, 23
  %10 = add nsw i32 %9, -127
  %11 = sitofp i32 %10 to float
  %12 = getelementptr inbounds i8* %x_, i64 4
  %13 = bitcast i8* %12 to float*
  %14 = load float* %13, align 4, !tbaa !1
  %15 = tail call float @fabsf(float %14) #12
  %16 = fcmp olt float %15, 0x3810000000000000
  br i1 %16, label %_ZN11OpenImageIO4v1_79fast_logbEf.exit4, label %17

; <label>:17                                      ; preds = %_ZN11OpenImageIO4v1_79fast_logbEf.exit2
  %18 = fcmp ogt float %15, 0x47EFFFFFE0000000
  br i1 %18, label %19, label %_ZN11OpenImageIO4v1_79fast_logbEf.exit4

; <label>:19                                      ; preds = %17
  br label %_ZN11OpenImageIO4v1_79fast_logbEf.exit4

_ZN11OpenImageIO4v1_79fast_logbEf.exit4:          ; preds = %_ZN11OpenImageIO4v1_79fast_logbEf.exit2, %17, %19
  %.1.i3 = phi float [ 0x47EFFFFFE0000000, %19 ], [ %15, %17 ], [ 0x3810000000000000, %_ZN11OpenImageIO4v1_79fast_logbEf.exit2 ]
  %20 = bitcast float %.1.i3 to i32
  %21 = lshr i32 %20, 23
  %22 = add nsw i32 %21, -127
  %23 = sitofp i32 %22 to float
  %24 = getelementptr inbounds i8* %x_, i64 8
  %25 = bitcast i8* %24 to float*
  %26 = load float* %25, align 4, !tbaa !1
  %27 = tail call float @fabsf(float %26) #12
  %28 = fcmp olt float %27, 0x3810000000000000
  br i1 %28, label %_ZN11OpenImageIO4v1_79fast_logbEf.exit, label %29

; <label>:29                                      ; preds = %_ZN11OpenImageIO4v1_79fast_logbEf.exit4
  %30 = fcmp ogt float %27, 0x47EFFFFFE0000000
  br i1 %30, label %31, label %_ZN11OpenImageIO4v1_79fast_logbEf.exit

; <label>:31                                      ; preds = %29
  br label %_ZN11OpenImageIO4v1_79fast_logbEf.exit

_ZN11OpenImageIO4v1_79fast_logbEf.exit:           ; preds = %_ZN11OpenImageIO4v1_79fast_logbEf.exit4, %29, %31
  %.1.i = phi float [ 0x47EFFFFFE0000000, %31 ], [ %27, %29 ], [ 0x3810000000000000, %_ZN11OpenImageIO4v1_79fast_logbEf.exit4 ]
  %32 = bitcast float %.1.i to i32
  %33 = lshr i32 %32, 23
  %34 = add nsw i32 %33, -127
  %35 = sitofp i32 %34 to float
  %36 = bitcast i8* %r to float*
  store float %11, float* %36, align 4, !tbaa !5
  %37 = getelementptr inbounds i8* %r, i64 4
  %38 = bitcast i8* %37 to float*
  store float %23, float* %38, align 4, !tbaa !7
  %39 = getelementptr inbounds i8* %r, i64 8
  %40 = bitcast i8* %39 to float*
  store float %35, float* %40, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_floor_ff(float %x) #3 {
  %1 = tail call float @floorf(float %x) #12
  ret float %1
}

; Function Attrs: nounwind readnone
declare float @floorf(float) #7

; Function Attrs: nounwind uwtable
define void @osl_floor_vv(i8* nocapture %r, i8* nocapture readonly %x_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @floorf(float %2) #12
  %4 = getelementptr inbounds i8* %x_, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !1
  %7 = tail call float @floorf(float %6) #12
  %8 = getelementptr inbounds i8* %x_, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = tail call float @floorf(float %10) #12
  %12 = bitcast i8* %r to float*
  store float %3, float* %12, align 4, !tbaa !5
  %13 = getelementptr inbounds i8* %r, i64 4
  %14 = bitcast i8* %13 to float*
  store float %7, float* %14, align 4, !tbaa !7
  %15 = getelementptr inbounds i8* %r, i64 8
  %16 = bitcast i8* %15 to float*
  store float %11, float* %16, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_ceil_ff(float %x) #3 {
  %1 = tail call float @ceilf(float %x) #12
  ret float %1
}

; Function Attrs: nounwind readnone
declare float @ceilf(float) #7

; Function Attrs: nounwind uwtable
define void @osl_ceil_vv(i8* nocapture %r, i8* nocapture readonly %x_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @ceilf(float %2) #12
  %4 = getelementptr inbounds i8* %x_, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !1
  %7 = tail call float @ceilf(float %6) #12
  %8 = getelementptr inbounds i8* %x_, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = tail call float @ceilf(float %10) #12
  %12 = bitcast i8* %r to float*
  store float %3, float* %12, align 4, !tbaa !5
  %13 = getelementptr inbounds i8* %r, i64 4
  %14 = bitcast i8* %13 to float*
  store float %7, float* %14, align 4, !tbaa !7
  %15 = getelementptr inbounds i8* %r, i64 8
  %16 = bitcast i8* %15 to float*
  store float %11, float* %16, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_round_ff(float %x) #3 {
  %1 = tail call float @roundf(float %x) #12
  ret float %1
}

; Function Attrs: nounwind readnone
declare float @roundf(float) #7

; Function Attrs: nounwind uwtable
define void @osl_round_vv(i8* nocapture %r, i8* nocapture readonly %x_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @roundf(float %2) #12
  %4 = getelementptr inbounds i8* %x_, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !1
  %7 = tail call float @roundf(float %6) #12
  %8 = getelementptr inbounds i8* %x_, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = tail call float @roundf(float %10) #12
  %12 = bitcast i8* %r to float*
  store float %3, float* %12, align 4, !tbaa !5
  %13 = getelementptr inbounds i8* %r, i64 4
  %14 = bitcast i8* %13 to float*
  store float %7, float* %14, align 4, !tbaa !7
  %15 = getelementptr inbounds i8* %r, i64 8
  %16 = bitcast i8* %15 to float*
  store float %11, float* %16, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_trunc_ff(float %x) #3 {
  %1 = tail call float @truncf(float %x) #12
  ret float %1
}

; Function Attrs: nounwind readnone
declare float @truncf(float) #7

; Function Attrs: nounwind uwtable
define void @osl_trunc_vv(i8* nocapture %r, i8* nocapture readonly %x_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @truncf(float %2) #12
  %4 = getelementptr inbounds i8* %x_, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !1
  %7 = tail call float @truncf(float %6) #12
  %8 = getelementptr inbounds i8* %x_, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = tail call float @truncf(float %10) #12
  %12 = bitcast i8* %r to float*
  store float %3, float* %12, align 4, !tbaa !5
  %13 = getelementptr inbounds i8* %r, i64 4
  %14 = bitcast i8* %13 to float*
  store float %7, float* %14, align 4, !tbaa !7
  %15 = getelementptr inbounds i8* %r, i64 8
  %16 = bitcast i8* %15 to float*
  store float %11, float* %16, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_sign_ff(float %x) #3 {
  %1 = fcmp olt float %x, 0.000000e+00
  br i1 %1, label %5, label %2

; <label>:2                                       ; preds = %0
  %3 = fcmp oeq float %x, 0.000000e+00
  %4 = select i1 %3, float 0.000000e+00, float 1.000000e+00
  br label %5

; <label>:5                                       ; preds = %0, %2
  %6 = phi float [ %4, %2 ], [ -1.000000e+00, %0 ]
  ret float %6
}

; Function Attrs: nounwind uwtable
define void @osl_sign_vv(i8* nocapture %r, i8* nocapture readonly %x_) #4 {
  %1 = bitcast i8* %x_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp olt float %2, 0.000000e+00
  br i1 %3, label %osl_sign_ff.exit2, label %4

; <label>:4                                       ; preds = %0
  %5 = fcmp oeq float %2, 0.000000e+00
  %6 = select i1 %5, float 0.000000e+00, float 1.000000e+00
  br label %osl_sign_ff.exit2

osl_sign_ff.exit2:                                ; preds = %0, %4
  %7 = phi float [ %6, %4 ], [ -1.000000e+00, %0 ]
  %8 = getelementptr inbounds i8* %x_, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !1
  %11 = fcmp olt float %10, 0.000000e+00
  br i1 %11, label %osl_sign_ff.exit1, label %12

; <label>:12                                      ; preds = %osl_sign_ff.exit2
  %13 = fcmp oeq float %10, 0.000000e+00
  %14 = select i1 %13, float 0.000000e+00, float 1.000000e+00
  br label %osl_sign_ff.exit1

osl_sign_ff.exit1:                                ; preds = %osl_sign_ff.exit2, %12
  %15 = phi float [ %14, %12 ], [ -1.000000e+00, %osl_sign_ff.exit2 ]
  %16 = getelementptr inbounds i8* %x_, i64 8
  %17 = bitcast i8* %16 to float*
  %18 = load float* %17, align 4, !tbaa !1
  %19 = fcmp olt float %18, 0.000000e+00
  br i1 %19, label %osl_sign_ff.exit, label %20

; <label>:20                                      ; preds = %osl_sign_ff.exit1
  %21 = fcmp oeq float %18, 0.000000e+00
  %22 = select i1 %21, float 0.000000e+00, float 1.000000e+00
  br label %osl_sign_ff.exit

osl_sign_ff.exit:                                 ; preds = %osl_sign_ff.exit1, %20
  %23 = phi float [ %22, %20 ], [ -1.000000e+00, %osl_sign_ff.exit1 ]
  %24 = bitcast i8* %r to float*
  store float %7, float* %24, align 4, !tbaa !5
  %25 = getelementptr inbounds i8* %r, i64 4
  %26 = bitcast i8* %25 to float*
  store float %15, float* %26, align 4, !tbaa !7
  %27 = getelementptr inbounds i8* %r, i64 8
  %28 = bitcast i8* %27 to float*
  store float %23, float* %28, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_step_fff(float %edge, float %x) #3 {
  %1 = fcmp olt float %x, %edge
  %2 = select i1 %1, float 0.000000e+00, float 1.000000e+00
  ret float %2
}

; Function Attrs: nounwind uwtable
define void @osl_step_vvv(i8* nocapture %result, i8* nocapture readonly %edge, i8* nocapture readonly %x) #4 {
  %1 = bitcast i8* %x to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = bitcast i8* %edge to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %2, %4
  %6 = select i1 %5, float 0.000000e+00, float 1.000000e+00
  %7 = getelementptr inbounds i8* %x, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float* %8, align 4, !tbaa !1
  %10 = getelementptr inbounds i8* %edge, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float* %11, align 4, !tbaa !1
  %13 = fcmp olt float %9, %12
  %14 = select i1 %13, float 0.000000e+00, float 1.000000e+00
  %15 = getelementptr inbounds i8* %x, i64 8
  %16 = bitcast i8* %15 to float*
  %17 = load float* %16, align 4, !tbaa !1
  %18 = getelementptr inbounds i8* %edge, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = load float* %19, align 4, !tbaa !1
  %21 = fcmp olt float %17, %20
  %22 = select i1 %21, float 0.000000e+00, float 1.000000e+00
  %23 = bitcast i8* %result to float*
  store float %6, float* %23, align 4, !tbaa !5
  %24 = getelementptr inbounds i8* %result, i64 4
  %25 = bitcast i8* %24 to float*
  store float %14, float* %25, align 4, !tbaa !7
  %26 = getelementptr inbounds i8* %result, i64 8
  %27 = bitcast i8* %26 to float*
  store float %22, float* %27, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define i32 @osl_isnan_if(float %f) #3 {
  %1 = fcmp uno float %f, 0.000000e+00
  %2 = zext i1 %1 to i32
  ret i32 %2
}

; Function Attrs: nounwind readnone uwtable
define i32 @osl_isinf_if(float %f) #3 {
  %1 = tail call float @fabsf(float %f) #2
  %2 = fcmp oeq float %1, 0x7FF0000000000000
  %3 = zext i1 %2 to i32
  ret i32 %3
}

; Function Attrs: nounwind readnone uwtable
define i32 @osl_isfinite_if(float %f) #3 {
  %1 = fcmp ord float %f, 0.000000e+00
  %2 = tail call float @fabsf(float %f) #2
  %3 = fcmp une float %2, 0x7FF0000000000000
  %4 = and i1 %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
}

; Function Attrs: nounwind readnone uwtable
define i32 @osl_abs_ii(i32 %x) #3 {
  %ispos = icmp sgt i32 %x, -1
  %neg = sub i32 0, %x
  %1 = select i1 %ispos, i32 %x, i32 %neg
  ret i32 %1
}

; Function Attrs: nounwind readnone uwtable
define i32 @osl_fabs_ii(i32 %x) #3 {
  %ispos = icmp sgt i32 %x, -1
  %neg = sub i32 0, %x
  %1 = select i1 %ispos, i32 %x, i32 %neg
  ret i32 %1
}

; Function Attrs: nounwind readnone uwtable
define float @osl_abs_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  ret float %1
}

; Function Attrs: nounwind readnone
declare float @fabsf(float) #7

; Function Attrs: nounwind uwtable
define void @osl_abs_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp ult float %2, 0.000000e+00
  %4 = bitcast i8* %a to <2 x float>*
  %5 = load <2 x float>* %4, align 4, !tbaa !1
  %6 = getelementptr inbounds i8* %a, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4
  br i1 %3, label %9, label %_Z5fabsfRKN3OSL5Dual2IfEE.exit

; <label>:9                                       ; preds = %0
  %10 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %5
  %11 = fsub float -0.000000e+00, %8
  br label %_Z5fabsfRKN3OSL5Dual2IfEE.exit

_Z5fabsfRKN3OSL5Dual2IfEE.exit:                   ; preds = %0, %9
  %.sroa.03.0.i = phi <2 x float> [ %10, %9 ], [ %5, %0 ]
  %.sroa.3.0.i = phi float [ %11, %9 ], [ %8, %0 ]
  %12 = bitcast i8* %r to <2 x float>*
  store <2 x float> %.sroa.03.0.i, <2 x float>* %12, align 4
  %13 = getelementptr inbounds i8* %r, i64 8
  %14 = bitcast i8* %13 to float*
  store float %.sroa.3.0.i, float* %14, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_abs_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = bitcast i8* %r_ to float*
  store float %3, float* %4, align 4, !tbaa !1
  %5 = getelementptr inbounds i8* %a_, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4, !tbaa !1
  %8 = tail call float @fabsf(float %7) #12
  %9 = getelementptr inbounds i8* %r_, i64 4
  %10 = bitcast i8* %9 to float*
  store float %8, float* %10, align 4, !tbaa !1
  %11 = getelementptr inbounds i8* %a_, i64 8
  %12 = bitcast i8* %11 to float*
  %13 = load float* %12, align 4, !tbaa !1
  %14 = tail call float @fabsf(float %13) #12
  %15 = getelementptr inbounds i8* %r_, i64 8
  %16 = bitcast i8* %15 to float*
  store float %14, float* %16, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_abs_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = insertelement <2 x float> undef, float %6, i32 0
  %8 = load float* %3, align 4, !tbaa !1
  %9 = insertelement <2 x float> %7, float %8, i32 1
  %10 = load float* %5, align 4, !tbaa !1
  %11 = fcmp ult float %6, 0.000000e+00
  br i1 %11, label %12, label %_Z5fabsfRKN3OSL5Dual2IfEE.exit16

; <label>:12                                      ; preds = %0
  %13 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %9
  %14 = fsub float -0.000000e+00, %10
  br label %_Z5fabsfRKN3OSL5Dual2IfEE.exit16

_Z5fabsfRKN3OSL5Dual2IfEE.exit16:                 ; preds = %0, %12
  %.sroa.03.0.i14 = phi <2 x float> [ %13, %12 ], [ %9, %0 ]
  %.sroa.3.0.i15 = phi float [ %14, %12 ], [ %10, %0 ]
  %15 = getelementptr inbounds i8* %a_, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = getelementptr inbounds i8* %a_, i64 16
  %18 = bitcast i8* %17 to float*
  %19 = getelementptr inbounds i8* %a_, i64 28
  %20 = bitcast i8* %19 to float*
  %21 = load float* %16, align 4, !tbaa !1
  %22 = insertelement <2 x float> undef, float %21, i32 0
  %23 = load float* %18, align 4, !tbaa !1
  %24 = insertelement <2 x float> %22, float %23, i32 1
  %25 = load float* %20, align 4, !tbaa !1
  %26 = fcmp ult float %21, 0.000000e+00
  br i1 %26, label %27, label %_Z5fabsfRKN3OSL5Dual2IfEE.exit13

; <label>:27                                      ; preds = %_Z5fabsfRKN3OSL5Dual2IfEE.exit16
  %28 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %24
  %29 = fsub float -0.000000e+00, %25
  br label %_Z5fabsfRKN3OSL5Dual2IfEE.exit13

_Z5fabsfRKN3OSL5Dual2IfEE.exit13:                 ; preds = %_Z5fabsfRKN3OSL5Dual2IfEE.exit16, %27
  %.sroa.03.0.i11 = phi <2 x float> [ %28, %27 ], [ %24, %_Z5fabsfRKN3OSL5Dual2IfEE.exit16 ]
  %.sroa.3.0.i12 = phi float [ %29, %27 ], [ %25, %_Z5fabsfRKN3OSL5Dual2IfEE.exit16 ]
  %30 = getelementptr inbounds i8* %a_, i64 8
  %31 = bitcast i8* %30 to float*
  %32 = getelementptr inbounds i8* %a_, i64 20
  %33 = bitcast i8* %32 to float*
  %34 = getelementptr inbounds i8* %a_, i64 32
  %35 = bitcast i8* %34 to float*
  %36 = load float* %31, align 4, !tbaa !1
  %37 = insertelement <2 x float> undef, float %36, i32 0
  %38 = load float* %33, align 4, !tbaa !1
  %39 = insertelement <2 x float> %37, float %38, i32 1
  %40 = load float* %35, align 4, !tbaa !1
  %41 = fcmp ult float %36, 0.000000e+00
  br i1 %41, label %42, label %_Z5fabsfRKN3OSL5Dual2IfEE.exit

; <label>:42                                      ; preds = %_Z5fabsfRKN3OSL5Dual2IfEE.exit13
  %43 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %39
  %44 = fsub float -0.000000e+00, %40
  br label %_Z5fabsfRKN3OSL5Dual2IfEE.exit

_Z5fabsfRKN3OSL5Dual2IfEE.exit:                   ; preds = %_Z5fabsfRKN3OSL5Dual2IfEE.exit13, %42
  %.sroa.03.0.i = phi <2 x float> [ %43, %42 ], [ %39, %_Z5fabsfRKN3OSL5Dual2IfEE.exit13 ]
  %.sroa.3.0.i = phi float [ %44, %42 ], [ %40, %_Z5fabsfRKN3OSL5Dual2IfEE.exit13 ]
  %45 = extractelement <2 x float> %.sroa.03.0.i14, i32 0
  %46 = extractelement <2 x float> %.sroa.03.0.i11, i32 0
  %47 = extractelement <2 x float> %.sroa.03.0.i, i32 0
  %48 = extractelement <2 x float> %.sroa.03.0.i14, i32 1
  %49 = extractelement <2 x float> %.sroa.03.0.i11, i32 1
  %50 = extractelement <2 x float> %.sroa.03.0.i, i32 1
  %51 = bitcast i8* %r_ to float*
  store float %45, float* %51, align 4, !tbaa !5
  %52 = getelementptr inbounds i8* %r_, i64 4
  %53 = bitcast i8* %52 to float*
  store float %46, float* %53, align 4, !tbaa !7
  %54 = getelementptr inbounds i8* %r_, i64 8
  %55 = bitcast i8* %54 to float*
  store float %47, float* %55, align 4, !tbaa !8
  %56 = getelementptr inbounds i8* %r_, i64 12
  %57 = bitcast i8* %56 to float*
  store float %48, float* %57, align 4, !tbaa !5
  %58 = getelementptr inbounds i8* %r_, i64 16
  %59 = bitcast i8* %58 to float*
  store float %49, float* %59, align 4, !tbaa !7
  %60 = getelementptr inbounds i8* %r_, i64 20
  %61 = bitcast i8* %60 to float*
  store float %50, float* %61, align 4, !tbaa !8
  %62 = getelementptr inbounds i8* %r_, i64 24
  %63 = bitcast i8* %62 to float*
  store float %.sroa.3.0.i15, float* %63, align 4, !tbaa !5
  %64 = getelementptr inbounds i8* %r_, i64 28
  %65 = bitcast i8* %64 to float*
  store float %.sroa.3.0.i12, float* %65, align 4, !tbaa !7
  %66 = getelementptr inbounds i8* %r_, i64 32
  %67 = bitcast i8* %66 to float*
  store float %.sroa.3.0.i, float* %67, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_fabs_ff(float %a) #3 {
  %1 = tail call float @fabsf(float %a) #12
  ret float %1
}

; Function Attrs: nounwind uwtable
define void @osl_fabs_dfdf(i8* nocapture %r, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp ult float %2, 0.000000e+00
  %4 = bitcast i8* %a to <2 x float>*
  %5 = load <2 x float>* %4, align 4, !tbaa !1
  %6 = getelementptr inbounds i8* %a, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4
  br i1 %3, label %9, label %_Z5fabsfRKN3OSL5Dual2IfEE.exit

; <label>:9                                       ; preds = %0
  %10 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %5
  %11 = fsub float -0.000000e+00, %8
  br label %_Z5fabsfRKN3OSL5Dual2IfEE.exit

_Z5fabsfRKN3OSL5Dual2IfEE.exit:                   ; preds = %0, %9
  %.sroa.03.0.i = phi <2 x float> [ %10, %9 ], [ %5, %0 ]
  %.sroa.3.0.i = phi float [ %11, %9 ], [ %8, %0 ]
  %12 = bitcast i8* %r to <2 x float>*
  store <2 x float> %.sroa.03.0.i, <2 x float>* %12, align 4
  %13 = getelementptr inbounds i8* %r, i64 8
  %14 = bitcast i8* %13 to float*
  store float %.sroa.3.0.i, float* %14, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fabs_vv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = tail call float @fabsf(float %2) #12
  %4 = bitcast i8* %r_ to float*
  store float %3, float* %4, align 4, !tbaa !1
  %5 = getelementptr inbounds i8* %a_, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4, !tbaa !1
  %8 = tail call float @fabsf(float %7) #12
  %9 = getelementptr inbounds i8* %r_, i64 4
  %10 = bitcast i8* %9 to float*
  store float %8, float* %10, align 4, !tbaa !1
  %11 = getelementptr inbounds i8* %a_, i64 8
  %12 = bitcast i8* %11 to float*
  %13 = load float* %12, align 4, !tbaa !1
  %14 = tail call float @fabsf(float %13) #12
  %15 = getelementptr inbounds i8* %r_, i64 8
  %16 = bitcast i8* %15 to float*
  store float %14, float* %16, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fabs_dvdv(i8* nocapture %r_, i8* nocapture readonly %a_) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = getelementptr inbounds i8* %a_, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = insertelement <2 x float> undef, float %6, i32 0
  %8 = load float* %3, align 4, !tbaa !1
  %9 = insertelement <2 x float> %7, float %8, i32 1
  %10 = load float* %5, align 4, !tbaa !1
  %11 = fcmp ult float %6, 0.000000e+00
  br i1 %11, label %12, label %_Z5fabsfRKN3OSL5Dual2IfEE.exit16

; <label>:12                                      ; preds = %0
  %13 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %9
  %14 = fsub float -0.000000e+00, %10
  br label %_Z5fabsfRKN3OSL5Dual2IfEE.exit16

_Z5fabsfRKN3OSL5Dual2IfEE.exit16:                 ; preds = %0, %12
  %.sroa.03.0.i14 = phi <2 x float> [ %13, %12 ], [ %9, %0 ]
  %.sroa.3.0.i15 = phi float [ %14, %12 ], [ %10, %0 ]
  %15 = getelementptr inbounds i8* %a_, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = getelementptr inbounds i8* %a_, i64 16
  %18 = bitcast i8* %17 to float*
  %19 = getelementptr inbounds i8* %a_, i64 28
  %20 = bitcast i8* %19 to float*
  %21 = load float* %16, align 4, !tbaa !1
  %22 = insertelement <2 x float> undef, float %21, i32 0
  %23 = load float* %18, align 4, !tbaa !1
  %24 = insertelement <2 x float> %22, float %23, i32 1
  %25 = load float* %20, align 4, !tbaa !1
  %26 = fcmp ult float %21, 0.000000e+00
  br i1 %26, label %27, label %_Z5fabsfRKN3OSL5Dual2IfEE.exit13

; <label>:27                                      ; preds = %_Z5fabsfRKN3OSL5Dual2IfEE.exit16
  %28 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %24
  %29 = fsub float -0.000000e+00, %25
  br label %_Z5fabsfRKN3OSL5Dual2IfEE.exit13

_Z5fabsfRKN3OSL5Dual2IfEE.exit13:                 ; preds = %_Z5fabsfRKN3OSL5Dual2IfEE.exit16, %27
  %.sroa.03.0.i11 = phi <2 x float> [ %28, %27 ], [ %24, %_Z5fabsfRKN3OSL5Dual2IfEE.exit16 ]
  %.sroa.3.0.i12 = phi float [ %29, %27 ], [ %25, %_Z5fabsfRKN3OSL5Dual2IfEE.exit16 ]
  %30 = getelementptr inbounds i8* %a_, i64 8
  %31 = bitcast i8* %30 to float*
  %32 = getelementptr inbounds i8* %a_, i64 20
  %33 = bitcast i8* %32 to float*
  %34 = getelementptr inbounds i8* %a_, i64 32
  %35 = bitcast i8* %34 to float*
  %36 = load float* %31, align 4, !tbaa !1
  %37 = insertelement <2 x float> undef, float %36, i32 0
  %38 = load float* %33, align 4, !tbaa !1
  %39 = insertelement <2 x float> %37, float %38, i32 1
  %40 = load float* %35, align 4, !tbaa !1
  %41 = fcmp ult float %36, 0.000000e+00
  br i1 %41, label %42, label %_Z5fabsfRKN3OSL5Dual2IfEE.exit

; <label>:42                                      ; preds = %_Z5fabsfRKN3OSL5Dual2IfEE.exit13
  %43 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %39
  %44 = fsub float -0.000000e+00, %40
  br label %_Z5fabsfRKN3OSL5Dual2IfEE.exit

_Z5fabsfRKN3OSL5Dual2IfEE.exit:                   ; preds = %_Z5fabsfRKN3OSL5Dual2IfEE.exit13, %42
  %.sroa.03.0.i = phi <2 x float> [ %43, %42 ], [ %39, %_Z5fabsfRKN3OSL5Dual2IfEE.exit13 ]
  %.sroa.3.0.i = phi float [ %44, %42 ], [ %40, %_Z5fabsfRKN3OSL5Dual2IfEE.exit13 ]
  %45 = extractelement <2 x float> %.sroa.03.0.i14, i32 0
  %46 = extractelement <2 x float> %.sroa.03.0.i11, i32 0
  %47 = extractelement <2 x float> %.sroa.03.0.i, i32 0
  %48 = extractelement <2 x float> %.sroa.03.0.i14, i32 1
  %49 = extractelement <2 x float> %.sroa.03.0.i11, i32 1
  %50 = extractelement <2 x float> %.sroa.03.0.i, i32 1
  %51 = bitcast i8* %r_ to float*
  store float %45, float* %51, align 4, !tbaa !5
  %52 = getelementptr inbounds i8* %r_, i64 4
  %53 = bitcast i8* %52 to float*
  store float %46, float* %53, align 4, !tbaa !7
  %54 = getelementptr inbounds i8* %r_, i64 8
  %55 = bitcast i8* %54 to float*
  store float %47, float* %55, align 4, !tbaa !8
  %56 = getelementptr inbounds i8* %r_, i64 12
  %57 = bitcast i8* %56 to float*
  store float %48, float* %57, align 4, !tbaa !5
  %58 = getelementptr inbounds i8* %r_, i64 16
  %59 = bitcast i8* %58 to float*
  store float %49, float* %59, align 4, !tbaa !7
  %60 = getelementptr inbounds i8* %r_, i64 20
  %61 = bitcast i8* %60 to float*
  store float %50, float* %61, align 4, !tbaa !8
  %62 = getelementptr inbounds i8* %r_, i64 24
  %63 = bitcast i8* %62 to float*
  store float %.sroa.3.0.i15, float* %63, align 4, !tbaa !5
  %64 = getelementptr inbounds i8* %r_, i64 28
  %65 = bitcast i8* %64 to float*
  store float %.sroa.3.0.i12, float* %65, align 4, !tbaa !7
  %66 = getelementptr inbounds i8* %r_, i64 32
  %67 = bitcast i8* %66 to float*
  store float %.sroa.3.0.i, float* %67, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
;define <4 x float> @osl_abs_w4fw4f(<4 x float> %a) #3 {
;  %1 = extractelement <4 x float> %a, i32 0
;  %2 = fcmp ult float %1, 0.000000e+00
;  br i1 %2, label %3, label %5

; <label>:3                                       ; preds = %0
;  %4 = fsub float -0.000000e+00, %1
;  br label %5

; <label>:5                                       ; preds = %0, %3
;  %6 = phi float [ %4, %3 ], [ %1, %0 ]
;  %7 = insertelement <4 x float> undef, float %6, i32 0
;  %8 = extractelement <4 x float> %a, i32 1
;  %9 = fcmp ult float %8, 0.000000e+00
;  br i1 %9, label %10, label %12

; <label>:10                                      ; preds = %5
;  %11 = fsub float -0.000000e+00, %8
;  br label %12

; <label>:12                                      ; preds = %5, %10
;  %13 = phi float [ %11, %10 ], [ %8, %5 ]
;  %14 = insertelement <4 x float> %7, float %13, i32 1
;  %15 = extractelement <4 x float> %a, i32 2
;  %16 = fcmp ult float %15, 0.000000e+00
;  br i1 %16, label %17, label %19

; <label>:17                                      ; preds = %12
;  %18 = fsub float -0.000000e+00, %15
;  br label %19

; <label>:19                                      ; preds = %12, %17
;  %20 = phi float [ %18, %17 ], [ %15, %12 ]
;  %21 = insertelement <4 x float> %14, float %20, i32 2
;  %22 = extractelement <4 x float> %a, i32 3
;  %23 = fcmp ult float %22, 0.000000e+00
;  br i1 %23, label %24, label %26

; <label>:24                                      ; preds = %19
;  %25 = fsub float -0.000000e+00, %22
;  br label %26

; <label>:26                                      ; preds = %19, %24
;  %27 = phi float [ %25, %24 ], [ %22, %19 ]
;  %28 = insertelement <4 x float> %21, float %27, i32 3
;  ret <4 x float> %28
;}

declare <4 x float> @llvm.fabs.v4f32(<4 x float>)

define <4 x float> @osl_abs_w4fw4f(<4 x float> %a) #3 {
   %r = call <4 x float> @llvm.fabs.v4f32(<4 x float> %a)
   ret <4 x float> %r
  }

declare <8 x float> @llvm.fabs.v8f32(<8 x float>)

define <8 x float> @osl_abs_w8fw8f(<8 x float> %a) #3 {
   %r = call <8 x float> @llvm.fabs.v8f32(<8 x float> %a)
   ret <8 x float> %r
  }

declare <16 x float> @llvm.fabs.v16f32(<16 x float>)

define <16 x float> @osl_abs_w16fw16f(<16 x float> %a) #3 {
   %r = call <16 x float> @llvm.fabs.v16f32(<16 x float> %a)
   ret <16 x float> %r
  }

; Function Attrs: nounwind readnone uwtable
define float @osl_fmod_fff(float %a, float %b) #3 {
  %1 = fcmp une float %b, 0.000000e+00
  br i1 %1, label %2, label %_Z9safe_fmodff.exit

; <label>:2                                       ; preds = %0
  %3 = tail call float @fmodf(float %a, float %b) #12
  br label %_Z9safe_fmodff.exit

_Z9safe_fmodff.exit:                              ; preds = %0, %2
  %4 = phi float [ %3, %2 ], [ 0.000000e+00, %0 ]
  ret float %4
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dfdfdf(i8* nocapture %r, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %b to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp une float %2, 0.000000e+00
  br i1 %3, label %4, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

; <label>:4                                       ; preds = %0
  %5 = bitcast i8* %a to float*
  %6 = load float* %5, align 4, !tbaa !1
  %7 = tail call float @fmodf(float %6, float %2) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit:            ; preds = %0, %4
  %8 = phi float [ %7, %4 ], [ 0.000000e+00, %0 ]
  %9 = getelementptr inbounds i8* %a, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = getelementptr inbounds i8* %a, i64 8
  %12 = bitcast i8* %11 to float*
  %13 = insertelement <2 x float> undef, float %8, i32 0
  %14 = load float* %10, align 4, !tbaa !1
  %15 = insertelement <2 x float> %13, float %14, i32 1
  %16 = load float* %12, align 4, !tbaa !1
  %17 = bitcast i8* %r to <2 x float>*
  store <2 x float> %15, <2 x float>* %17, align 4
  %18 = getelementptr inbounds i8* %r, i64 8
  %19 = bitcast i8* %18 to float*
  store float %16, float* %19, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dffdf(i8* nocapture %r, float %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %b to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp une float %2, 0.000000e+00
  br i1 %3, label %4, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

; <label>:4                                       ; preds = %0
  %5 = tail call float @fmodf(float %a, float %2) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit:            ; preds = %0, %4
  %6 = phi float [ %5, %4 ], [ 0.000000e+00, %0 ]
  %7 = insertelement <2 x float> undef, float %6, i32 0
  %8 = insertelement <2 x float> %7, float 0.000000e+00, i32 1
  %9 = bitcast i8* %r to <2 x float>*
  store <2 x float> %8, <2 x float>* %9, align 4
  %10 = getelementptr inbounds i8* %r, i64 8
  %11 = bitcast i8* %10 to float*
  store float 0.000000e+00, float* %11, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dfdff(i8* nocapture %r, i8* nocapture readonly %a, float %b) #4 {
  %1 = fcmp une float %b, 0.000000e+00
  br i1 %1, label %2, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

; <label>:2                                       ; preds = %0
  %3 = bitcast i8* %a to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = tail call float @fmodf(float %4, float %b) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit:            ; preds = %0, %2
  %6 = phi float [ %5, %2 ], [ 0.000000e+00, %0 ]
  %7 = getelementptr inbounds i8* %a, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = getelementptr inbounds i8* %a, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = insertelement <2 x float> undef, float %6, i32 0
  %12 = load float* %8, align 4, !tbaa !1
  %13 = insertelement <2 x float> %11, float %12, i32 1
  %14 = load float* %10, align 4, !tbaa !1
  %15 = bitcast i8* %r to <2 x float>*
  store <2 x float> %13, <2 x float>* %15, align 4
  %16 = getelementptr inbounds i8* %r, i64 8
  %17 = bitcast i8* %16 to float*
  store float %14, float* %17, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_vvv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %b_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp une float %2, 0.000000e+00
  br i1 %3, label %4, label %_Z9safe_fmodff.exit2

; <label>:4                                       ; preds = %0
  %5 = bitcast i8* %a_ to float*
  %6 = load float* %5, align 4, !tbaa !1
  %7 = tail call float @fmodf(float %6, float %2) #12
  br label %_Z9safe_fmodff.exit2

_Z9safe_fmodff.exit2:                             ; preds = %0, %4
  %8 = phi float [ %7, %4 ], [ 0.000000e+00, %0 ]
  %9 = bitcast i8* %r_ to float*
  store float %8, float* %9, align 4, !tbaa !1
  %10 = getelementptr inbounds i8* %b_, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float* %11, align 4, !tbaa !1
  %13 = fcmp une float %12, 0.000000e+00
  br i1 %13, label %14, label %_Z9safe_fmodff.exit1

; <label>:14                                      ; preds = %_Z9safe_fmodff.exit2
  %15 = getelementptr inbounds i8* %a_, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float* %16, align 4, !tbaa !1
  %18 = tail call float @fmodf(float %17, float %12) #12
  br label %_Z9safe_fmodff.exit1

_Z9safe_fmodff.exit1:                             ; preds = %_Z9safe_fmodff.exit2, %14
  %19 = phi float [ %18, %14 ], [ 0.000000e+00, %_Z9safe_fmodff.exit2 ]
  %20 = getelementptr inbounds i8* %r_, i64 4
  %21 = bitcast i8* %20 to float*
  store float %19, float* %21, align 4, !tbaa !1
  %22 = getelementptr inbounds i8* %b_, i64 8
  %23 = bitcast i8* %22 to float*
  %24 = load float* %23, align 4, !tbaa !1
  %25 = fcmp une float %24, 0.000000e+00
  br i1 %25, label %26, label %_Z9safe_fmodff.exit

; <label>:26                                      ; preds = %_Z9safe_fmodff.exit1
  %27 = getelementptr inbounds i8* %a_, i64 8
  %28 = bitcast i8* %27 to float*
  %29 = load float* %28, align 4, !tbaa !1
  %30 = tail call float @fmodf(float %29, float %24) #12
  br label %_Z9safe_fmodff.exit

_Z9safe_fmodff.exit:                              ; preds = %_Z9safe_fmodff.exit1, %26
  %31 = phi float [ %30, %26 ], [ 0.000000e+00, %_Z9safe_fmodff.exit1 ]
  %32 = getelementptr inbounds i8* %r_, i64 8
  %33 = bitcast i8* %32 to float*
  store float %31, float* %33, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvdvdv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = getelementptr inbounds i8* %a_, i64 12
  %2 = bitcast i8* %1 to float*
  %3 = getelementptr inbounds i8* %a_, i64 24
  %4 = bitcast i8* %3 to float*
  %5 = load float* %2, align 4, !tbaa !1
  %6 = load float* %4, align 4, !tbaa !1
  %7 = bitcast i8* %b_ to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = fcmp une float %8, 0.000000e+00
  br i1 %9, label %10, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12

; <label>:10                                      ; preds = %0
  %11 = bitcast i8* %a_ to float*
  %12 = load float* %11, align 4, !tbaa !1
  %13 = tail call float @fmodf(float %12, float %8) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12:          ; preds = %0, %10
  %14 = phi float [ %13, %10 ], [ 0.000000e+00, %0 ]
  %15 = getelementptr inbounds i8* %a_, i64 16
  %16 = bitcast i8* %15 to float*
  %17 = getelementptr inbounds i8* %a_, i64 28
  %18 = bitcast i8* %17 to float*
  %19 = load float* %16, align 4, !tbaa !1
  %20 = load float* %18, align 4, !tbaa !1
  %21 = getelementptr inbounds i8* %b_, i64 4
  %22 = bitcast i8* %21 to float*
  %23 = load float* %22, align 4, !tbaa !1
  %24 = fcmp une float %23, 0.000000e+00
  br i1 %24, label %25, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11

; <label>:25                                      ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12
  %26 = getelementptr inbounds i8* %a_, i64 4
  %27 = bitcast i8* %26 to float*
  %28 = load float* %27, align 4, !tbaa !1
  %29 = tail call float @fmodf(float %28, float %23) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11:          ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12, %25
  %30 = phi float [ %29, %25 ], [ 0.000000e+00, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12 ]
  %31 = getelementptr inbounds i8* %a_, i64 20
  %32 = bitcast i8* %31 to float*
  %33 = getelementptr inbounds i8* %a_, i64 32
  %34 = bitcast i8* %33 to float*
  %35 = load float* %32, align 4, !tbaa !1
  %36 = load float* %34, align 4, !tbaa !1
  %37 = getelementptr inbounds i8* %b_, i64 8
  %38 = bitcast i8* %37 to float*
  %39 = load float* %38, align 4, !tbaa !1
  %40 = fcmp une float %39, 0.000000e+00
  br i1 %40, label %41, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

; <label>:41                                      ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11
  %42 = getelementptr inbounds i8* %a_, i64 8
  %43 = bitcast i8* %42 to float*
  %44 = load float* %43, align 4, !tbaa !1
  %45 = tail call float @fmodf(float %44, float %39) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit:            ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11, %41
  %46 = phi float [ %45, %41 ], [ 0.000000e+00, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11 ]
  %47 = bitcast i8* %r_ to float*
  store float %14, float* %47, align 4, !tbaa !5
  %48 = getelementptr inbounds i8* %r_, i64 4
  %49 = bitcast i8* %48 to float*
  store float %30, float* %49, align 4, !tbaa !7
  %50 = getelementptr inbounds i8* %r_, i64 8
  %51 = bitcast i8* %50 to float*
  store float %46, float* %51, align 4, !tbaa !8
  %52 = getelementptr inbounds i8* %r_, i64 12
  %53 = bitcast i8* %52 to float*
  store float %5, float* %53, align 4, !tbaa !5
  %54 = getelementptr inbounds i8* %r_, i64 16
  %55 = bitcast i8* %54 to float*
  store float %19, float* %55, align 4, !tbaa !7
  %56 = getelementptr inbounds i8* %r_, i64 20
  %57 = bitcast i8* %56 to float*
  store float %35, float* %57, align 4, !tbaa !8
  %58 = getelementptr inbounds i8* %r_, i64 24
  %59 = bitcast i8* %58 to float*
  store float %6, float* %59, align 4, !tbaa !5
  %60 = getelementptr inbounds i8* %r_, i64 28
  %61 = bitcast i8* %60 to float*
  store float %20, float* %61, align 4, !tbaa !7
  %62 = getelementptr inbounds i8* %r_, i64 32
  %63 = bitcast i8* %62 to float*
  store float %36, float* %63, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvvdv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = getelementptr inbounds i8* %a_, i64 4
  %2 = bitcast i8* %1 to float*
  %3 = load float* %2, align 4, !tbaa !7
  %4 = getelementptr inbounds i8* %a_, i64 8
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !8
  %7 = bitcast i8* %b_ to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = fcmp une float %8, 0.000000e+00
  br i1 %9, label %10, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i

; <label>:10                                      ; preds = %0
  %11 = bitcast i8* %a_ to float*
  %12 = load float* %11, align 4, !tbaa !5
  %13 = tail call float @fmodf(float %12, float %8) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i:        ; preds = %10, %0
  %14 = phi float [ %13, %10 ], [ 0.000000e+00, %0 ]
  %15 = getelementptr inbounds i8* %b_, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float* %16, align 4, !tbaa !1
  %18 = fcmp une float %17, 0.000000e+00
  br i1 %18, label %19, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i

; <label>:19                                      ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i
  %20 = tail call float @fmodf(float %3, float %17) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i:        ; preds = %19, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i
  %21 = phi float [ %20, %19 ], [ 0.000000e+00, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i ]
  %22 = getelementptr inbounds i8* %b_, i64 8
  %23 = bitcast i8* %22 to float*
  %24 = load float* %23, align 4, !tbaa !1
  %25 = fcmp une float %24, 0.000000e+00
  br i1 %25, label %26, label %osl_fmod_dvdvdv.exit

; <label>:26                                      ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i
  %27 = tail call float @fmodf(float %6, float %24) #12
  br label %osl_fmod_dvdvdv.exit

osl_fmod_dvdvdv.exit:                             ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i, %26
  %28 = phi float [ %27, %26 ], [ 0.000000e+00, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i ]
  %29 = bitcast i8* %r_ to float*
  store float %14, float* %29, align 4, !tbaa !5
  %30 = getelementptr inbounds i8* %r_, i64 4
  %31 = bitcast i8* %30 to float*
  store float %21, float* %31, align 4, !tbaa !7
  %32 = getelementptr inbounds i8* %r_, i64 8
  %33 = bitcast i8* %32 to float*
  store float %28, float* %33, align 4, !tbaa !8
  %34 = getelementptr inbounds i8* %r_, i64 12
  call void @llvm.memset.p0i8.i64(i8* %34, i8 0, i64 24, i32 4, i1 false)
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvdvv(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %b_ to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = getelementptr inbounds i8* %b_, i64 4
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4, !tbaa !7
  %6 = getelementptr inbounds i8* %b_, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !8
  %9 = getelementptr inbounds i8* %a_, i64 12
  %10 = bitcast i8* %9 to float*
  %11 = getelementptr inbounds i8* %a_, i64 24
  %12 = bitcast i8* %11 to float*
  %13 = load float* %10, align 4, !tbaa !1
  %14 = load float* %12, align 4, !tbaa !1
  %15 = fcmp une float %2, 0.000000e+00
  br i1 %15, label %16, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i

; <label>:16                                      ; preds = %0
  %17 = bitcast i8* %a_ to float*
  %18 = load float* %17, align 4, !tbaa !1
  %19 = tail call float @fmodf(float %18, float %2) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i:        ; preds = %16, %0
  %20 = phi float [ %19, %16 ], [ 0.000000e+00, %0 ]
  %21 = getelementptr inbounds i8* %a_, i64 16
  %22 = bitcast i8* %21 to float*
  %23 = getelementptr inbounds i8* %a_, i64 28
  %24 = bitcast i8* %23 to float*
  %25 = load float* %22, align 4, !tbaa !1
  %26 = load float* %24, align 4, !tbaa !1
  %27 = fcmp une float %5, 0.000000e+00
  br i1 %27, label %28, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i

; <label>:28                                      ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i
  %29 = getelementptr inbounds i8* %a_, i64 4
  %30 = bitcast i8* %29 to float*
  %31 = load float* %30, align 4, !tbaa !1
  %32 = tail call float @fmodf(float %31, float %5) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i:        ; preds = %28, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i
  %33 = phi float [ %32, %28 ], [ 0.000000e+00, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit12.i ]
  %34 = getelementptr inbounds i8* %a_, i64 20
  %35 = bitcast i8* %34 to float*
  %36 = getelementptr inbounds i8* %a_, i64 32
  %37 = bitcast i8* %36 to float*
  %38 = load float* %35, align 4, !tbaa !1
  %39 = load float* %37, align 4, !tbaa !1
  %40 = fcmp une float %8, 0.000000e+00
  br i1 %40, label %41, label %osl_fmod_dvdvdv.exit

; <label>:41                                      ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i
  %42 = getelementptr inbounds i8* %a_, i64 8
  %43 = bitcast i8* %42 to float*
  %44 = load float* %43, align 4, !tbaa !1
  %45 = tail call float @fmodf(float %44, float %8) #12
  br label %osl_fmod_dvdvdv.exit

osl_fmod_dvdvdv.exit:                             ; preds = %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i, %41
  %46 = phi float [ %45, %41 ], [ 0.000000e+00, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.i ]
  %47 = bitcast i8* %r_ to float*
  store float %20, float* %47, align 4, !tbaa !5
  %48 = getelementptr inbounds i8* %r_, i64 4
  %49 = bitcast i8* %48 to float*
  store float %33, float* %49, align 4, !tbaa !7
  %50 = getelementptr inbounds i8* %r_, i64 8
  %51 = bitcast i8* %50 to float*
  store float %46, float* %51, align 4, !tbaa !8
  %52 = getelementptr inbounds i8* %r_, i64 12
  %53 = bitcast i8* %52 to float*
  store float %13, float* %53, align 4, !tbaa !5
  %54 = getelementptr inbounds i8* %r_, i64 16
  %55 = bitcast i8* %54 to float*
  store float %25, float* %55, align 4, !tbaa !7
  %56 = getelementptr inbounds i8* %r_, i64 20
  %57 = bitcast i8* %56 to float*
  store float %38, float* %57, align 4, !tbaa !8
  %58 = getelementptr inbounds i8* %r_, i64 24
  %59 = bitcast i8* %58 to float*
  store float %14, float* %59, align 4, !tbaa !5
  %60 = getelementptr inbounds i8* %r_, i64 28
  %61 = bitcast i8* %60 to float*
  store float %26, float* %61, align 4, !tbaa !7
  %62 = getelementptr inbounds i8* %r_, i64 32
  %63 = bitcast i8* %62 to float*
  store float %39, float* %63, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_vvf(i8* nocapture %r_, i8* nocapture readonly %a_, float %b) #4 {
  %1 = fcmp une float %b, 0.000000e+00
  br i1 %1, label %5, label %_Z9safe_fmodff.exit2.thread4

_Z9safe_fmodff.exit2.thread4:                     ; preds = %0
  %2 = bitcast i8* %r_ to float*
  store float 0.000000e+00, float* %2, align 4, !tbaa !1
  %3 = getelementptr inbounds i8* %r_, i64 4
  %4 = bitcast i8* %3 to float*
  store float 0.000000e+00, float* %4, align 4, !tbaa !1
  br label %_Z9safe_fmodff.exit

; <label>:5                                       ; preds = %0
  %6 = bitcast i8* %a_ to float*
  %7 = load float* %6, align 4, !tbaa !1
  %8 = tail call float @fmodf(float %7, float %b) #12
  %9 = bitcast i8* %r_ to float*
  store float %8, float* %9, align 4, !tbaa !1
  %10 = getelementptr inbounds i8* %a_, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float* %11, align 4, !tbaa !1
  %13 = tail call float @fmodf(float %12, float %b) #12
  %14 = getelementptr inbounds i8* %r_, i64 4
  %15 = bitcast i8* %14 to float*
  store float %13, float* %15, align 4, !tbaa !1
  %16 = getelementptr inbounds i8* %a_, i64 8
  %17 = bitcast i8* %16 to float*
  %18 = load float* %17, align 4, !tbaa !1
  %19 = tail call float @fmodf(float %18, float %b) #12
  br label %_Z9safe_fmodff.exit

_Z9safe_fmodff.exit:                              ; preds = %_Z9safe_fmodff.exit2.thread4, %5
  %20 = phi float* [ %9, %5 ], [ %2, %_Z9safe_fmodff.exit2.thread4 ]
  %21 = phi float [ %19, %5 ], [ 0.000000e+00, %_Z9safe_fmodff.exit2.thread4 ]
  %22 = getelementptr inbounds float* %20, i64 2
  store float %21, float* %22, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvdvdf(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = getelementptr inbounds i8* %a_, i64 12
  %2 = bitcast i8* %1 to float*
  %3 = getelementptr inbounds i8* %a_, i64 24
  %4 = bitcast i8* %3 to float*
  %5 = load float* %2, align 4, !tbaa !1
  %6 = load float* %4, align 4, !tbaa !1
  %7 = bitcast i8* %b_ to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = fcmp une float %8, 0.000000e+00
  br i1 %9, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread:   ; preds = %0
  %10 = bitcast i8* %a_ to float*
  %11 = load float* %10, align 4, !tbaa !1
  %12 = tail call float @fmodf(float %11, float %8) #12
  %13 = getelementptr inbounds i8* %a_, i64 4
  %14 = bitcast i8* %13 to float*
  %15 = load float* %14, align 4, !tbaa !1
  %16 = tail call float @fmodf(float %15, float %8) #12
  %17 = getelementptr inbounds i8* %a_, i64 8
  %18 = bitcast i8* %17 to float*
  %19 = load float* %18, align 4, !tbaa !1
  %20 = tail call float @fmodf(float %19, float %8) #12
  br label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit:            ; preds = %0, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread
  %21 = phi float [ %16, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread ], [ 0.000000e+00, %0 ]
  %22 = phi float [ %12, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread ], [ 0.000000e+00, %0 ]
  %23 = phi float [ %20, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread ], [ 0.000000e+00, %0 ]
  %.in31.in = getelementptr inbounds i8* %a_, i64 28
  %.in30.in = getelementptr inbounds i8* %a_, i64 16
  %.in29.in = getelementptr inbounds i8* %a_, i64 20
  %.in.in = getelementptr inbounds i8* %a_, i64 32
  %.in31 = bitcast i8* %.in31.in to float*
  %.in30 = bitcast i8* %.in30.in to float*
  %.in29 = bitcast i8* %.in29.in to float*
  %.in = bitcast i8* %.in.in to float*
  %24 = load float* %.in31, align 4
  %25 = load float* %.in30, align 4
  %26 = load float* %.in29, align 4
  %27 = load float* %.in, align 4
  %28 = bitcast i8* %r_ to float*
  store float %22, float* %28, align 4, !tbaa !5
  %29 = getelementptr inbounds i8* %r_, i64 4
  %30 = bitcast i8* %29 to float*
  store float %21, float* %30, align 4, !tbaa !7
  %31 = getelementptr inbounds i8* %r_, i64 8
  %32 = bitcast i8* %31 to float*
  store float %23, float* %32, align 4, !tbaa !8
  %33 = getelementptr inbounds i8* %r_, i64 12
  %34 = bitcast i8* %33 to float*
  store float %5, float* %34, align 4, !tbaa !5
  %35 = getelementptr inbounds i8* %r_, i64 16
  %36 = bitcast i8* %35 to float*
  store float %25, float* %36, align 4, !tbaa !7
  %37 = getelementptr inbounds i8* %r_, i64 20
  %38 = bitcast i8* %37 to float*
  store float %26, float* %38, align 4, !tbaa !8
  %39 = getelementptr inbounds i8* %r_, i64 24
  %40 = bitcast i8* %39 to float*
  store float %6, float* %40, align 4, !tbaa !5
  %41 = getelementptr inbounds i8* %r_, i64 28
  %42 = bitcast i8* %41 to float*
  store float %24, float* %42, align 4, !tbaa !7
  %43 = getelementptr inbounds i8* %r_, i64 32
  %44 = bitcast i8* %43 to float*
  store float %27, float* %44, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvvdf(i8* nocapture %r_, i8* nocapture readonly %a_, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %b_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp une float %2, 0.000000e+00
  br i1 %3, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i, label %osl_fmod_dvdvdf.exit

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i: ; preds = %0
  %4 = getelementptr inbounds i8* %a_, i64 8
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !8
  %7 = getelementptr inbounds i8* %a_, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float* %8, align 4, !tbaa !7
  %10 = bitcast i8* %a_ to float*
  %11 = load float* %10, align 4, !tbaa !5
  %12 = tail call float @fmodf(float %11, float %2) #12
  %13 = tail call float @fmodf(float %9, float %2) #12
  %14 = tail call float @fmodf(float %6, float %2) #12
  br label %osl_fmod_dvdvdf.exit

osl_fmod_dvdvdf.exit:                             ; preds = %0, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i
  %15 = phi float [ %13, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i ], [ 0.000000e+00, %0 ]
  %16 = phi float [ %12, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i ], [ 0.000000e+00, %0 ]
  %17 = phi float [ %14, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i ], [ 0.000000e+00, %0 ]
  %18 = bitcast i8* %r_ to float*
  store float %16, float* %18, align 4, !tbaa !5
  %19 = getelementptr inbounds i8* %r_, i64 4
  %20 = bitcast i8* %19 to float*
  store float %15, float* %20, align 4, !tbaa !7
  %21 = getelementptr inbounds i8* %r_, i64 8
  %22 = bitcast i8* %21 to float*
  store float %17, float* %22, align 4, !tbaa !8
  %23 = getelementptr inbounds i8* %r_, i64 12
  call void @llvm.memset.p0i8.i64(i8* %23, i8 0, i64 24, i32 4, i1 false)
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvdvf(i8* nocapture %r_, i8* nocapture readonly %a_, float %b_) #4 {
  %1 = getelementptr inbounds i8* %a_, i64 12
  %2 = bitcast i8* %1 to float*
  %3 = getelementptr inbounds i8* %a_, i64 24
  %4 = bitcast i8* %3 to float*
  %5 = load float* %2, align 4, !tbaa !1
  %6 = load float* %4, align 4, !tbaa !1
  %7 = fcmp une float %b_, 0.000000e+00
  br i1 %7, label %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i, label %osl_fmod_dvdvdf.exit

_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i: ; preds = %0
  %8 = bitcast i8* %a_ to float*
  %9 = load float* %8, align 4, !tbaa !1
  %10 = tail call float @fmodf(float %9, float %b_) #12
  %11 = getelementptr inbounds i8* %a_, i64 4
  %12 = bitcast i8* %11 to float*
  %13 = load float* %12, align 4, !tbaa !1
  %14 = tail call float @fmodf(float %13, float %b_) #12
  %15 = getelementptr inbounds i8* %a_, i64 8
  %16 = bitcast i8* %15 to float*
  %17 = load float* %16, align 4, !tbaa !1
  %18 = tail call float @fmodf(float %17, float %b_) #12
  br label %osl_fmod_dvdvdf.exit

osl_fmod_dvdvdf.exit:                             ; preds = %0, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i
  %19 = phi float [ %14, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i ], [ 0.000000e+00, %0 ]
  %20 = phi float [ %10, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i ], [ 0.000000e+00, %0 ]
  %21 = phi float [ %18, %_Z9safe_fmodRKN3OSL5Dual2IfEES3_.exit11.thread.i ], [ 0.000000e+00, %0 ]
  %.in31.in.i = getelementptr inbounds i8* %a_, i64 28
  %.in30.in.i = getelementptr inbounds i8* %a_, i64 16
  %.in29.in.i = getelementptr inbounds i8* %a_, i64 20
  %.in.in.i = getelementptr inbounds i8* %a_, i64 32
  %.in31.i = bitcast i8* %.in31.in.i to float*
  %.in30.i = bitcast i8* %.in30.in.i to float*
  %.in29.i = bitcast i8* %.in29.in.i to float*
  %.in.i = bitcast i8* %.in.in.i to float*
  %22 = load float* %.in31.i, align 4
  %23 = load float* %.in30.i, align 4
  %24 = load float* %.in29.i, align 4
  %25 = load float* %.in.i, align 4
  %26 = bitcast i8* %r_ to float*
  store float %20, float* %26, align 4, !tbaa !5
  %27 = getelementptr inbounds i8* %r_, i64 4
  %28 = bitcast i8* %27 to float*
  store float %19, float* %28, align 4, !tbaa !7
  %29 = getelementptr inbounds i8* %r_, i64 8
  %30 = bitcast i8* %29 to float*
  store float %21, float* %30, align 4, !tbaa !8
  %31 = getelementptr inbounds i8* %r_, i64 12
  %32 = bitcast i8* %31 to float*
  store float %5, float* %32, align 4, !tbaa !5
  %33 = getelementptr inbounds i8* %r_, i64 16
  %34 = bitcast i8* %33 to float*
  store float %23, float* %34, align 4, !tbaa !7
  %35 = getelementptr inbounds i8* %r_, i64 20
  %36 = bitcast i8* %35 to float*
  store float %24, float* %36, align 4, !tbaa !8
  %37 = getelementptr inbounds i8* %r_, i64 24
  %38 = bitcast i8* %37 to float*
  store float %6, float* %38, align 4, !tbaa !5
  %39 = getelementptr inbounds i8* %r_, i64 28
  %40 = bitcast i8* %39 to float*
  store float %22, float* %40, align 4, !tbaa !7
  %41 = getelementptr inbounds i8* %r_, i64 32
  %42 = bitcast i8* %41 to float*
  store float %25, float* %42, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_safe_div_fff(float %a, float %b) #3 {
  %1 = fcmp une float %b, 0.000000e+00
  br i1 %1, label %2, label %4

; <label>:2                                       ; preds = %0
  %3 = fdiv float %a, %b
  br label %4

; <label>:4                                       ; preds = %0, %2
  %5 = phi float [ %3, %2 ], [ 0.000000e+00, %0 ]
  ret float %5
}

; Function Attrs: nounwind readnone uwtable
define i32 @osl_safe_div_iii(i32 %a, i32 %b) #3 {
  %1 = icmp eq i32 %b, 0
  br i1 %1, label %4, label %2

; <label>:2                                       ; preds = %0
  %3 = sdiv i32 %a, %b
  br label %4

; <label>:4                                       ; preds = %0, %2
  %5 = phi i32 [ %3, %2 ], [ 0, %0 ]
  ret i32 %5
}

; Function Attrs: nounwind readnone uwtable
define float @osl_smoothstep_ffff(float %e0, float %e1, float %x) #3 {
  %1 = fcmp olt float %x, %e0
  br i1 %1, label %_ZN3OSL10smoothstepEfff.exit, label %2

; <label>:2                                       ; preds = %0
  %3 = fcmp ult float %x, %e1
  br i1 %3, label %4, label %_ZN3OSL10smoothstepEfff.exit

; <label>:4                                       ; preds = %2
  %5 = fsub float %x, %e0
  %6 = fsub float %e1, %e0
  %7 = fdiv float %5, %6
  %8 = fmul float %7, 2.000000e+00
  %9 = fsub float 3.000000e+00, %8
  %10 = fmul float %7, %7
  %11 = fmul float %10, %9
  br label %_ZN3OSL10smoothstepEfff.exit

_ZN3OSL10smoothstepEfff.exit:                     ; preds = %0, %2, %4
  %.0.i = phi float [ %11, %4 ], [ 0.000000e+00, %0 ], [ 1.000000e+00, %2 ]
  ret float %.0.i
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfffdf(i8* nocapture %result, float %e0_, float %e1_, i8* nocapture readonly %x_) #4 {
  %1 = insertelement <2 x float> undef, float %e0_, i32 0
  %2 = insertelement <2 x float> %1, float 0.000000e+00, i32 1
  %3 = insertelement <2 x float> undef, float %e1_, i32 0
  %4 = insertelement <2 x float> %3, float 0.000000e+00, i32 1
  %5 = bitcast i8* %x_ to <2 x float>*
  %6 = load <2 x float>* %5, align 4
  %7 = getelementptr inbounds i8* %x_, i64 8
  %8 = bitcast i8* %7 to float*
  %9 = load float* %8, align 4
  %10 = extractelement <2 x float> %6, i32 0
  %11 = fcmp olt float %10, %e0_
  br i1 %11, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit, label %12

; <label>:12                                      ; preds = %0
  %13 = fcmp ult float %10, %e1_
  br i1 %13, label %14, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

; <label>:14                                      ; preds = %12
  %15 = fsub <2 x float> %6, %2
  %16 = fsub <2 x float> %4, %2
  %17 = extractelement <2 x float> %16, i32 0
  %18 = fdiv float 1.000000e+00, %17
  %19 = extractelement <2 x float> %15, i32 0
  %20 = fmul float %19, %18
  %21 = extractelement <2 x float> %15, i32 1
  %22 = extractelement <2 x float> %16, i32 1
  %23 = fmul float %22, %20
  %24 = fsub float %21, %23
  %25 = fmul float %18, %24
  %26 = fmul float %20, 0.000000e+00
  %27 = fsub float %9, %26
  %28 = fmul float %18, %27
  %29 = insertelement <2 x float> undef, float %20, i32 0
  %30 = insertelement <2 x float> %29, float %25, i32 1
  %31 = fmul <2 x float> %30, <float 2.000000e+00, float 2.000000e+00>
  %32 = fmul float %28, 2.000000e+00
  %33 = fsub <2 x float> <float 3.000000e+00, float -0.000000e+00>, %31
  %34 = extractelement <2 x float> %33, i32 0
  %35 = fmul float %20, %34
  %36 = fmul float %25, %34
  %37 = extractelement <2 x float> %33, i32 1
  %38 = fmul float %20, %37
  %39 = fadd float %36, %38
  %40 = fmul float %28, %34
  %41 = fmul float %20, %32
  %42 = fsub float %40, %41
  %43 = fmul float %20, %35
  %44 = fmul float %25, %35
  %45 = fmul float %20, %39
  %46 = fadd float %44, %45
  %47 = fmul float %28, %35
  %48 = fmul float %20, %42
  %49 = fadd float %47, %48
  %50 = insertelement <2 x float> undef, float %43, i32 0
  %51 = insertelement <2 x float> %50, float %46, i32 1
  br label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit: ; preds = %0, %12, %14
  %52 = phi float [ %49, %14 ], [ 0.000000e+00, %0 ], [ 0.000000e+00, %12 ]
  %53 = phi <2 x float> [ %51, %14 ], [ zeroinitializer, %0 ], [ <float 1.000000e+00, float 0.000000e+00>, %12 ]
  %54 = bitcast i8* %result to <2 x float>*
  store <2 x float> %53, <2 x float>* %54, align 4
  %55 = getelementptr inbounds i8* %result, i64 8
  %56 = bitcast i8* %55 to float*
  store float %52, float* %56, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dffdff(i8* nocapture %result, float %e0_, i8* nocapture readonly %e1_, float %x_) #4 {
  %1 = insertelement <2 x float> undef, float %e0_, i32 0
  %2 = insertelement <2 x float> %1, float 0.000000e+00, i32 1
  %3 = bitcast i8* %e1_ to <2 x float>*
  %4 = load <2 x float>* %3, align 4
  %5 = getelementptr inbounds i8* %e1_, i64 8
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4
  %8 = insertelement <2 x float> undef, float %x_, i32 0
  %9 = insertelement <2 x float> %8, float 0.000000e+00, i32 1
  %10 = fcmp olt float %x_, %e0_
  br i1 %10, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit, label %11

; <label>:11                                      ; preds = %0
  %12 = extractelement <2 x float> %4, i32 0
  %13 = fcmp ugt float %12, %x_
  br i1 %13, label %14, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

; <label>:14                                      ; preds = %11
  %15 = fsub <2 x float> %9, %2
  %16 = fsub <2 x float> %4, %2
  %17 = extractelement <2 x float> %16, i32 0
  %18 = fdiv float 1.000000e+00, %17
  %19 = extractelement <2 x float> %15, i32 0
  %20 = fmul float %19, %18
  %21 = extractelement <2 x float> %15, i32 1
  %22 = extractelement <2 x float> %16, i32 1
  %23 = fmul float %22, %20
  %24 = fsub float %21, %23
  %25 = fmul float %18, %24
  %26 = fmul float %7, %20
  %27 = fsub float 0.000000e+00, %26
  %28 = fmul float %18, %27
  %29 = insertelement <2 x float> undef, float %20, i32 0
  %30 = insertelement <2 x float> %29, float %25, i32 1
  %31 = fmul <2 x float> %30, <float 2.000000e+00, float 2.000000e+00>
  %32 = fmul float %28, 2.000000e+00
  %33 = fsub <2 x float> <float 3.000000e+00, float -0.000000e+00>, %31
  %34 = extractelement <2 x float> %33, i32 0
  %35 = fmul float %20, %34
  %36 = fmul float %25, %34
  %37 = extractelement <2 x float> %33, i32 1
  %38 = fmul float %20, %37
  %39 = fadd float %36, %38
  %40 = fmul float %28, %34
  %41 = fmul float %20, %32
  %42 = fsub float %40, %41
  %43 = fmul float %20, %35
  %44 = fmul float %25, %35
  %45 = fmul float %20, %39
  %46 = fadd float %44, %45
  %47 = fmul float %28, %35
  %48 = fmul float %20, %42
  %49 = fadd float %47, %48
  %50 = insertelement <2 x float> undef, float %43, i32 0
  %51 = insertelement <2 x float> %50, float %46, i32 1
  br label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit: ; preds = %0, %11, %14
  %52 = phi float [ %49, %14 ], [ 0.000000e+00, %0 ], [ 0.000000e+00, %11 ]
  %53 = phi <2 x float> [ %51, %14 ], [ zeroinitializer, %0 ], [ <float 1.000000e+00, float 0.000000e+00>, %11 ]
  %54 = bitcast i8* %result to <2 x float>*
  store <2 x float> %53, <2 x float>* %54, align 4
  %55 = getelementptr inbounds i8* %result, i64 8
  %56 = bitcast i8* %55 to float*
  store float %52, float* %56, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dffdfdf(i8* nocapture %result, float %e0_, i8* nocapture readonly %e1_, i8* nocapture readonly %x_) #4 {
  %1 = insertelement <2 x float> undef, float %e0_, i32 0
  %2 = insertelement <2 x float> %1, float 0.000000e+00, i32 1
  %3 = bitcast i8* %e1_ to <2 x float>*
  %4 = load <2 x float>* %3, align 4
  %5 = getelementptr inbounds i8* %e1_, i64 8
  %6 = bitcast i8* %5 to float*
  %7 = load float* %6, align 4
  %8 = bitcast i8* %x_ to <2 x float>*
  %9 = load <2 x float>* %8, align 4
  %10 = getelementptr inbounds i8* %x_, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float* %11, align 4
  %13 = extractelement <2 x float> %9, i32 0
  %14 = fcmp olt float %13, %e0_
  br i1 %14, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit, label %15

; <label>:15                                      ; preds = %0
  %16 = extractelement <2 x float> %4, i32 0
  %17 = fcmp ult float %13, %16
  br i1 %17, label %18, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

; <label>:18                                      ; preds = %15
  %19 = fsub <2 x float> %9, %2
  %20 = fsub <2 x float> %4, %2
  %21 = extractelement <2 x float> %20, i32 0
  %22 = fdiv float 1.000000e+00, %21
  %23 = extractelement <2 x float> %19, i32 0
  %24 = fmul float %23, %22
  %25 = extractelement <2 x float> %19, i32 1
  %26 = extractelement <2 x float> %20, i32 1
  %27 = fmul float %26, %24
  %28 = fsub float %25, %27
  %29 = fmul float %22, %28
  %30 = fmul float %7, %24
  %31 = fsub float %12, %30
  %32 = fmul float %22, %31
  %33 = insertelement <2 x float> undef, float %24, i32 0
  %34 = insertelement <2 x float> %33, float %29, i32 1
  %35 = fmul <2 x float> %34, <float 2.000000e+00, float 2.000000e+00>
  %36 = fmul float %32, 2.000000e+00
  %37 = fsub <2 x float> <float 3.000000e+00, float -0.000000e+00>, %35
  %38 = extractelement <2 x float> %37, i32 0
  %39 = fmul float %24, %38
  %40 = fmul float %29, %38
  %41 = extractelement <2 x float> %37, i32 1
  %42 = fmul float %24, %41
  %43 = fadd float %40, %42
  %44 = fmul float %32, %38
  %45 = fmul float %24, %36
  %46 = fsub float %44, %45
  %47 = fmul float %24, %39
  %48 = fmul float %29, %39
  %49 = fmul float %24, %43
  %50 = fadd float %48, %49
  %51 = fmul float %32, %39
  %52 = fmul float %24, %46
  %53 = fadd float %51, %52
  %54 = insertelement <2 x float> undef, float %47, i32 0
  %55 = insertelement <2 x float> %54, float %50, i32 1
  br label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit: ; preds = %0, %15, %18
  %56 = phi float [ %53, %18 ], [ 0.000000e+00, %0 ], [ 0.000000e+00, %15 ]
  %57 = phi <2 x float> [ %55, %18 ], [ zeroinitializer, %0 ], [ <float 1.000000e+00, float 0.000000e+00>, %15 ]
  %58 = bitcast i8* %result to <2 x float>*
  store <2 x float> %57, <2 x float>* %58, align 4
  %59 = getelementptr inbounds i8* %result, i64 8
  %60 = bitcast i8* %59 to float*
  store float %56, float* %60, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfdfff(i8* nocapture %result, i8* nocapture readonly %e0_, float %e1_, float %x_) #4 {
  %1 = bitcast i8* %e0_ to <2 x float>*
  %2 = load <2 x float>* %1, align 4
  %3 = getelementptr inbounds i8* %e0_, i64 8
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4
  %6 = insertelement <2 x float> undef, float %e1_, i32 0
  %7 = insertelement <2 x float> %6, float 0.000000e+00, i32 1
  %8 = insertelement <2 x float> undef, float %x_, i32 0
  %9 = insertelement <2 x float> %8, float 0.000000e+00, i32 1
  %10 = extractelement <2 x float> %2, i32 0
  %11 = fcmp ogt float %10, %x_
  br i1 %11, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit, label %12

; <label>:12                                      ; preds = %0
  %13 = fcmp ult float %x_, %e1_
  br i1 %13, label %14, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

; <label>:14                                      ; preds = %12
  %15 = fsub <2 x float> %9, %2
  %16 = fsub float 0.000000e+00, %5
  %17 = fsub <2 x float> %7, %2
  %18 = extractelement <2 x float> %17, i32 0
  %19 = fdiv float 1.000000e+00, %18
  %20 = extractelement <2 x float> %15, i32 0
  %21 = fmul float %20, %19
  %22 = extractelement <2 x float> %15, i32 1
  %23 = extractelement <2 x float> %17, i32 1
  %24 = fmul float %23, %21
  %25 = fsub float %22, %24
  %26 = fmul float %19, %25
  %27 = fmul float %16, %21
  %28 = fsub float %16, %27
  %29 = fmul float %19, %28
  %30 = insertelement <2 x float> undef, float %21, i32 0
  %31 = insertelement <2 x float> %30, float %26, i32 1
  %32 = fmul <2 x float> %31, <float 2.000000e+00, float 2.000000e+00>
  %33 = fmul float %29, 2.000000e+00
  %34 = fsub <2 x float> <float 3.000000e+00, float -0.000000e+00>, %32
  %35 = extractelement <2 x float> %34, i32 0
  %36 = fmul float %21, %35
  %37 = fmul float %26, %35
  %38 = extractelement <2 x float> %34, i32 1
  %39 = fmul float %21, %38
  %40 = fadd float %37, %39
  %41 = fmul float %29, %35
  %42 = fmul float %21, %33
  %43 = fsub float %41, %42
  %44 = fmul float %21, %36
  %45 = fmul float %26, %36
  %46 = fmul float %21, %40
  %47 = fadd float %45, %46
  %48 = fmul float %29, %36
  %49 = fmul float %21, %43
  %50 = fadd float %48, %49
  %51 = insertelement <2 x float> undef, float %44, i32 0
  %52 = insertelement <2 x float> %51, float %47, i32 1
  br label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit: ; preds = %0, %12, %14
  %53 = phi float [ %50, %14 ], [ 0.000000e+00, %0 ], [ 0.000000e+00, %12 ]
  %54 = phi <2 x float> [ %52, %14 ], [ zeroinitializer, %0 ], [ <float 1.000000e+00, float 0.000000e+00>, %12 ]
  %55 = bitcast i8* %result to <2 x float>*
  store <2 x float> %54, <2 x float>* %55, align 4
  %56 = getelementptr inbounds i8* %result, i64 8
  %57 = bitcast i8* %56 to float*
  store float %53, float* %57, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfdffdf(i8* nocapture %result, i8* nocapture readonly %e0_, float %e1_, i8* nocapture readonly %x_) #4 {
  %1 = bitcast i8* %e0_ to <2 x float>*
  %2 = load <2 x float>* %1, align 4
  %3 = getelementptr inbounds i8* %e0_, i64 8
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4
  %6 = insertelement <2 x float> undef, float %e1_, i32 0
  %7 = insertelement <2 x float> %6, float 0.000000e+00, i32 1
  %8 = bitcast i8* %x_ to <2 x float>*
  %9 = load <2 x float>* %8, align 4
  %10 = getelementptr inbounds i8* %x_, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float* %11, align 4
  %13 = extractelement <2 x float> %9, i32 0
  %14 = extractelement <2 x float> %2, i32 0
  %15 = fcmp olt float %13, %14
  br i1 %15, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit, label %16

; <label>:16                                      ; preds = %0
  %17 = fcmp ult float %13, %e1_
  br i1 %17, label %18, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

; <label>:18                                      ; preds = %16
  %19 = fsub <2 x float> %9, %2
  %20 = fsub float %12, %5
  %21 = fsub <2 x float> %7, %2
  %22 = fsub float 0.000000e+00, %5
  %23 = extractelement <2 x float> %21, i32 0
  %24 = fdiv float 1.000000e+00, %23
  %25 = extractelement <2 x float> %19, i32 0
  %26 = fmul float %25, %24
  %27 = extractelement <2 x float> %19, i32 1
  %28 = extractelement <2 x float> %21, i32 1
  %29 = fmul float %28, %26
  %30 = fsub float %27, %29
  %31 = fmul float %24, %30
  %32 = fmul float %22, %26
  %33 = fsub float %20, %32
  %34 = fmul float %24, %33
  %35 = insertelement <2 x float> undef, float %26, i32 0
  %36 = insertelement <2 x float> %35, float %31, i32 1
  %37 = fmul <2 x float> %36, <float 2.000000e+00, float 2.000000e+00>
  %38 = fmul float %34, 2.000000e+00
  %39 = fsub <2 x float> <float 3.000000e+00, float -0.000000e+00>, %37
  %40 = extractelement <2 x float> %39, i32 0
  %41 = fmul float %26, %40
  %42 = fmul float %31, %40
  %43 = extractelement <2 x float> %39, i32 1
  %44 = fmul float %26, %43
  %45 = fadd float %42, %44
  %46 = fmul float %34, %40
  %47 = fmul float %26, %38
  %48 = fsub float %46, %47
  %49 = fmul float %26, %41
  %50 = fmul float %31, %41
  %51 = fmul float %26, %45
  %52 = fadd float %50, %51
  %53 = fmul float %34, %41
  %54 = fmul float %26, %48
  %55 = fadd float %53, %54
  %56 = insertelement <2 x float> undef, float %49, i32 0
  %57 = insertelement <2 x float> %56, float %52, i32 1
  br label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit: ; preds = %0, %16, %18
  %58 = phi float [ %55, %18 ], [ 0.000000e+00, %0 ], [ 0.000000e+00, %16 ]
  %59 = phi <2 x float> [ %57, %18 ], [ zeroinitializer, %0 ], [ <float 1.000000e+00, float 0.000000e+00>, %16 ]
  %60 = bitcast i8* %result to <2 x float>*
  store <2 x float> %59, <2 x float>* %60, align 4
  %61 = getelementptr inbounds i8* %result, i64 8
  %62 = bitcast i8* %61 to float*
  store float %58, float* %62, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfdfdff(i8* nocapture %result, i8* nocapture readonly %e0_, i8* nocapture readonly %e1_, float %x_) #4 {
  %1 = bitcast i8* %e0_ to <2 x float>*
  %2 = load <2 x float>* %1, align 4
  %3 = getelementptr inbounds i8* %e0_, i64 8
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4
  %6 = bitcast i8* %e1_ to <2 x float>*
  %7 = load <2 x float>* %6, align 4
  %8 = getelementptr inbounds i8* %e1_, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4
  %11 = insertelement <2 x float> undef, float %x_, i32 0
  %12 = insertelement <2 x float> %11, float 0.000000e+00, i32 1
  %13 = extractelement <2 x float> %2, i32 0
  %14 = fcmp ogt float %13, %x_
  br i1 %14, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit, label %15

; <label>:15                                      ; preds = %0
  %16 = extractelement <2 x float> %7, i32 0
  %17 = fcmp ugt float %16, %x_
  br i1 %17, label %18, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

; <label>:18                                      ; preds = %15
  %19 = fsub <2 x float> %12, %2
  %20 = fsub float 0.000000e+00, %5
  %21 = fsub <2 x float> %7, %2
  %22 = fsub float %10, %5
  %23 = extractelement <2 x float> %21, i32 0
  %24 = fdiv float 1.000000e+00, %23
  %25 = extractelement <2 x float> %19, i32 0
  %26 = fmul float %25, %24
  %27 = extractelement <2 x float> %19, i32 1
  %28 = extractelement <2 x float> %21, i32 1
  %29 = fmul float %28, %26
  %30 = fsub float %27, %29
  %31 = fmul float %24, %30
  %32 = fmul float %22, %26
  %33 = fsub float %20, %32
  %34 = fmul float %24, %33
  %35 = insertelement <2 x float> undef, float %26, i32 0
  %36 = insertelement <2 x float> %35, float %31, i32 1
  %37 = fmul <2 x float> %36, <float 2.000000e+00, float 2.000000e+00>
  %38 = fmul float %34, 2.000000e+00
  %39 = fsub <2 x float> <float 3.000000e+00, float -0.000000e+00>, %37
  %40 = extractelement <2 x float> %39, i32 0
  %41 = fmul float %26, %40
  %42 = fmul float %31, %40
  %43 = extractelement <2 x float> %39, i32 1
  %44 = fmul float %26, %43
  %45 = fadd float %42, %44
  %46 = fmul float %34, %40
  %47 = fmul float %26, %38
  %48 = fsub float %46, %47
  %49 = fmul float %26, %41
  %50 = fmul float %31, %41
  %51 = fmul float %26, %45
  %52 = fadd float %50, %51
  %53 = fmul float %34, %41
  %54 = fmul float %26, %48
  %55 = fadd float %53, %54
  %56 = insertelement <2 x float> undef, float %49, i32 0
  %57 = insertelement <2 x float> %56, float %52, i32 1
  br label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit: ; preds = %0, %15, %18
  %58 = phi float [ %55, %18 ], [ 0.000000e+00, %0 ], [ 0.000000e+00, %15 ]
  %59 = phi <2 x float> [ %57, %18 ], [ zeroinitializer, %0 ], [ <float 1.000000e+00, float 0.000000e+00>, %15 ]
  %60 = bitcast i8* %result to <2 x float>*
  store <2 x float> %59, <2 x float>* %60, align 4
  %61 = getelementptr inbounds i8* %result, i64 8
  %62 = bitcast i8* %61 to float*
  store float %58, float* %62, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfdfdfdf(i8* nocapture %result, i8* nocapture readonly %e0_, i8* nocapture readonly %e1_, i8* nocapture readonly %x_) #4 {
  %1 = bitcast i8* %e0_ to <2 x float>*
  %2 = load <2 x float>* %1, align 4
  %3 = getelementptr inbounds i8* %e0_, i64 8
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4
  %6 = bitcast i8* %e1_ to <2 x float>*
  %7 = load <2 x float>* %6, align 4
  %8 = getelementptr inbounds i8* %e1_, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4
  %11 = bitcast i8* %x_ to <2 x float>*
  %12 = load <2 x float>* %11, align 4
  %13 = getelementptr inbounds i8* %x_, i64 8
  %14 = bitcast i8* %13 to float*
  %15 = load float* %14, align 4
  %16 = extractelement <2 x float> %12, i32 0
  %17 = extractelement <2 x float> %2, i32 0
  %18 = fcmp olt float %16, %17
  br i1 %18, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit, label %19

; <label>:19                                      ; preds = %0
  %20 = extractelement <2 x float> %7, i32 0
  %21 = fcmp ult float %16, %20
  br i1 %21, label %22, label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

; <label>:22                                      ; preds = %19
  %23 = fsub <2 x float> %12, %2
  %24 = fsub float %15, %5
  %25 = fsub <2 x float> %7, %2
  %26 = fsub float %10, %5
  %27 = extractelement <2 x float> %25, i32 0
  %28 = fdiv float 1.000000e+00, %27
  %29 = extractelement <2 x float> %23, i32 0
  %30 = fmul float %29, %28
  %31 = extractelement <2 x float> %23, i32 1
  %32 = extractelement <2 x float> %25, i32 1
  %33 = fmul float %32, %30
  %34 = fsub float %31, %33
  %35 = fmul float %28, %34
  %36 = fmul float %26, %30
  %37 = fsub float %24, %36
  %38 = fmul float %28, %37
  %39 = insertelement <2 x float> undef, float %30, i32 0
  %40 = insertelement <2 x float> %39, float %35, i32 1
  %41 = fmul <2 x float> %40, <float 2.000000e+00, float 2.000000e+00>
  %42 = fmul float %38, 2.000000e+00
  %43 = fsub <2 x float> <float 3.000000e+00, float -0.000000e+00>, %41
  %44 = extractelement <2 x float> %43, i32 0
  %45 = fmul float %30, %44
  %46 = fmul float %35, %44
  %47 = extractelement <2 x float> %43, i32 1
  %48 = fmul float %30, %47
  %49 = fadd float %46, %48
  %50 = fmul float %38, %44
  %51 = fmul float %30, %42
  %52 = fsub float %50, %51
  %53 = fmul float %30, %45
  %54 = fmul float %35, %45
  %55 = fmul float %30, %49
  %56 = fadd float %54, %55
  %57 = fmul float %38, %45
  %58 = fmul float %30, %52
  %59 = fadd float %57, %58
  %60 = insertelement <2 x float> undef, float %53, i32 0
  %61 = insertelement <2 x float> %60, float %56, i32 1
  br label %_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit

_ZN3OSL10smoothstepIfEENS_5Dual2IT_EERKS3_S5_S5_.exit: ; preds = %0, %19, %22
  %62 = phi float [ %59, %22 ], [ 0.000000e+00, %0 ], [ 0.000000e+00, %19 ]
  %63 = phi <2 x float> [ %61, %22 ], [ zeroinitializer, %0 ], [ <float 1.000000e+00, float 0.000000e+00>, %19 ]
  %64 = bitcast i8* %result to <2 x float>*
  store <2 x float> %63, <2 x float>* %64, align 4
  %65 = getelementptr inbounds i8* %result, i64 8
  %66 = bitcast i8* %65 to float*
  store float %62, float* %66, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_transform_vmv(i8* nocapture %result, i8* nocapture readonly %M_, i8* nocapture readonly %v_) #4 {
  %1 = bitcast i8* %v_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = getelementptr inbounds i8* %v_, i64 4
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4, !tbaa !1
  %6 = getelementptr inbounds i8* %v_, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = getelementptr inbounds i8* %M_, i64 12
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !1
  %12 = fmul float %2, %11
  %13 = getelementptr inbounds i8* %M_, i64 28
  %14 = bitcast i8* %13 to float*
  %15 = load float* %14, align 4, !tbaa !1
  %16 = fmul float %5, %15
  %17 = fadd float %12, %16
  %18 = getelementptr inbounds i8* %M_, i64 44
  %19 = bitcast i8* %18 to float*
  %20 = load float* %19, align 4, !tbaa !1
  %21 = fmul float %8, %20
  %22 = fadd float %17, %21
  %23 = getelementptr inbounds i8* %M_, i64 60
  %24 = bitcast i8* %23 to float*
  %25 = load float* %24, align 4, !tbaa !1
  %26 = fadd float %25, %22
  %27 = fcmp une float %26, 0.000000e+00
  br i1 %27, label %28, label %_ZN3OSL20robust_multVecMatrixERKN9Imath_2_28Matrix44IfEERKNS0_4Vec3IfEERS6_.exit

; <label>:28                                      ; preds = %0
  %29 = getelementptr inbounds i8* %M_, i64 48
  %30 = bitcast i8* %29 to float*
  %31 = getelementptr inbounds i8* %M_, i64 32
  %32 = bitcast i8* %31 to float*
  %33 = getelementptr inbounds i8* %M_, i64 16
  %34 = bitcast i8* %33 to float*
  %35 = bitcast i8* %M_ to float*
  %36 = getelementptr inbounds i8* %M_, i64 56
  %37 = bitcast i8* %36 to float*
  %38 = load float* %37, align 4, !tbaa !1
  %39 = getelementptr inbounds i8* %M_, i64 40
  %40 = bitcast i8* %39 to float*
  %41 = load float* %40, align 4, !tbaa !1
  %42 = getelementptr inbounds i8* %M_, i64 24
  %43 = bitcast i8* %42 to float*
  %44 = load float* %43, align 4, !tbaa !1
  %45 = getelementptr inbounds i8* %M_, i64 8
  %46 = bitcast i8* %45 to float*
  %47 = load float* %46, align 4, !tbaa !1
  %48 = getelementptr inbounds i8* %M_, i64 52
  %49 = bitcast i8* %48 to float*
  %50 = load float* %49, align 4, !tbaa !1
  %51 = getelementptr inbounds i8* %M_, i64 36
  %52 = bitcast i8* %51 to float*
  %53 = load float* %52, align 4, !tbaa !1
  %54 = getelementptr inbounds i8* %M_, i64 20
  %55 = bitcast i8* %54 to float*
  %56 = load float* %55, align 4, !tbaa !1
  %57 = getelementptr inbounds i8* %M_, i64 4
  %58 = bitcast i8* %57 to float*
  %59 = load float* %58, align 4, !tbaa !1
  %60 = load float* %30, align 4, !tbaa !1
  %61 = load float* %32, align 4, !tbaa !1
  %62 = load float* %34, align 4, !tbaa !1
  %63 = load float* %35, align 4, !tbaa !1
  %64 = fmul float %2, %47
  %65 = fmul float %5, %44
  %66 = fadd float %65, %64
  %67 = fmul float %8, %41
  %68 = fadd float %67, %66
  %69 = fadd float %38, %68
  %70 = fmul float %2, %59
  %71 = fmul float %5, %56
  %72 = fadd float %71, %70
  %73 = fmul float %8, %53
  %74 = fadd float %73, %72
  %75 = fadd float %50, %74
  %76 = fmul float %2, %63
  %77 = fmul float %5, %62
  %78 = fadd float %77, %76
  %79 = fmul float %8, %61
  %80 = fadd float %79, %78
  %81 = fadd float %60, %80
  %82 = fdiv float %81, %26
  %83 = fdiv float %75, %26
  %84 = fdiv float %69, %26
  br label %_ZN3OSL20robust_multVecMatrixERKN9Imath_2_28Matrix44IfEERKNS0_4Vec3IfEERS6_.exit

_ZN3OSL20robust_multVecMatrixERKN9Imath_2_28Matrix44IfEERKNS0_4Vec3IfEERS6_.exit: ; preds = %0, %28
  %.sink2.i = phi float [ %82, %28 ], [ 0.000000e+00, %0 ]
  %.sink1.i = phi float [ %83, %28 ], [ 0.000000e+00, %0 ]
  %.sink.i = phi float [ %84, %28 ], [ 0.000000e+00, %0 ]
  %85 = bitcast i8* %result to float*
  store float %.sink2.i, float* %85, align 4
  %86 = getelementptr inbounds i8* %result, i64 4
  %87 = bitcast i8* %86 to float*
  store float %.sink1.i, float* %87, align 4
  %88 = getelementptr inbounds i8* %result, i64 8
  %89 = bitcast i8* %88 to float*
  store float %.sink.i, float* %89, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_transform_dvmdv(i8* nocapture %result, i8* nocapture readonly %M_, i8* nocapture readonly %v_) #4 {
  %1 = bitcast i8* %v_ to %"class.OSL::Dual2.0"*
  %2 = bitcast i8* %M_ to %"class.Imath_2_2::Matrix44"*
  %3 = bitcast i8* %result to %"class.OSL::Dual2.0"*
  tail call void @_ZN3OSL20robust_multVecMatrixERKN9Imath_2_28Matrix44IfEERKNS_5Dual2INS0_4Vec3IfEEEERS8_(%"class.Imath_2_2::Matrix44"* dereferenceable(64) %2, %"class.OSL::Dual2.0"* dereferenceable(36) %1, %"class.OSL::Dual2.0"* dereferenceable(36) %3)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN3OSL20robust_multVecMatrixERKN9Imath_2_28Matrix44IfEERKNS_5Dual2INS0_4Vec3IfEEEERS8_(%"class.Imath_2_2::Matrix44"* nocapture readonly dereferenceable(64) %M, %"class.OSL::Dual2.0"* nocapture readonly dereferenceable(36) %in, %"class.OSL::Dual2.0"* nocapture dereferenceable(36) %out) #8 {
  %din = alloca %"class.Imath_2_2::Vec3.1", align 8
  %1 = bitcast %"class.Imath_2_2::Vec3.1"* %din to i8*
  call void @llvm.lifetime.start(i64 36, i8* %1) #2
  %2 = getelementptr inbounds %"class.Imath_2_2::Vec3.1"* %din, i64 0, i32 0
  %3 = getelementptr inbounds %"class.OSL::Dual2.0"* %in, i64 0, i32 0, i32 0
  %4 = getelementptr inbounds %"class.OSL::Dual2.0"* %in, i64 0, i32 1, i32 0
  %5 = getelementptr inbounds %"class.OSL::Dual2.0"* %in, i64 0, i32 2, i32 0
  %6 = load float* %3, align 4, !tbaa !1
  %7 = getelementptr inbounds %"class.Imath_2_2::Vec3.1"* %din, i64 0, i32 0, i32 0
  store float %6, float* %7, align 8, !tbaa !9
  %8 = load float* %4, align 4, !tbaa !1
  %9 = getelementptr inbounds %"class.Imath_2_2::Vec3.1"* %din, i64 0, i32 0, i32 1
  store float %8, float* %9, align 4, !tbaa !11
  %10 = load float* %5, align 4, !tbaa !1
  %11 = getelementptr inbounds %"class.Imath_2_2::Vec3.1"* %din, i64 0, i32 0, i32 2
  store float %10, float* %11, align 8, !tbaa !12
  %12 = getelementptr inbounds float* %3, i64 1
  %13 = getelementptr inbounds float* %4, i64 1
  %14 = getelementptr inbounds float* %5, i64 1
  %15 = load float* %12, align 4, !tbaa !1
  %16 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 1, i32 0
  store float %15, float* %16, align 4, !tbaa !9
  %17 = load float* %13, align 4, !tbaa !1
  %18 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 1, i32 1
  store float %17, float* %18, align 4, !tbaa !11
  %19 = load float* %14, align 4, !tbaa !1
  %20 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 1, i32 2
  store float %19, float* %20, align 4, !tbaa !12
  %21 = getelementptr inbounds float* %3, i64 2
  %22 = getelementptr inbounds float* %4, i64 2
  %23 = getelementptr inbounds float* %5, i64 2
  %24 = load float* %21, align 4, !tbaa !1
  %25 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 2, i32 0
  store float %24, float* %25, align 8, !tbaa !9
  %26 = load float* %22, align 4, !tbaa !1
  %27 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 2, i32 1
  store float %26, float* %27, align 4, !tbaa !11
  %28 = load float* %23, align 4, !tbaa !1
  %29 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 2, i32 2
  store float %28, float* %29, align 8, !tbaa !12
  %30 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 0, i64 0
  %31 = load float* %30, align 4, !tbaa !1
  %32 = bitcast %"class.Imath_2_2::Vec3.1"* %din to <2 x float>*
  %33 = load <2 x float>* %32, align 8, !tbaa !1
  %34 = insertelement <2 x float> undef, float %31, i32 0
  %35 = insertelement <2 x float> %34, float %31, i32 1
  %36 = fmul <2 x float> %33, %35
  %37 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 1
  %38 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 1, i64 0
  %39 = load float* %38, align 4, !tbaa !1
  %40 = bitcast %"class.OSL::Dual2"* %37 to <2 x float>*
  %41 = load <2 x float>* %40, align 4, !tbaa !1
  %42 = insertelement <2 x float> undef, float %39, i32 0
  %43 = insertelement <2 x float> %42, float %39, i32 1
  %44 = fmul <2 x float> %41, %43
  %45 = fadd <2 x float> %36, %44
  %46 = getelementptr inbounds %"class.OSL::Dual2"* %2, i64 2
  %47 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 2, i64 0
  %48 = load float* %47, align 4, !tbaa !1
  %49 = bitcast %"class.OSL::Dual2"* %46 to <2 x float>*
  %50 = load <2 x float>* %49, align 8, !tbaa !1
  %51 = insertelement <2 x float> undef, float %48, i32 0
  %52 = insertelement <2 x float> %51, float %48, i32 1
  %53 = fmul <2 x float> %50, %52
  %54 = fadd <2 x float> %45, %53
  %55 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 0, i64 1
  %56 = load float* %55, align 4, !tbaa !1
  %57 = insertelement <2 x float> undef, float %56, i32 0
  %58 = insertelement <2 x float> %57, float %56, i32 1
  %59 = fmul <2 x float> %33, %58
  %60 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 1, i64 1
  %61 = load float* %60, align 4, !tbaa !1
  %62 = insertelement <2 x float> undef, float %61, i32 0
  %63 = insertelement <2 x float> %62, float %61, i32 1
  %64 = fmul <2 x float> %41, %63
  %65 = fadd <2 x float> %59, %64
  %66 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 2, i64 1
  %67 = load float* %66, align 4, !tbaa !1
  %68 = insertelement <2 x float> undef, float %67, i32 0
  %69 = insertelement <2 x float> %68, float %67, i32 1
  %70 = fmul <2 x float> %50, %69
  %71 = fadd <2 x float> %65, %70
  %72 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 0, i64 2
  %73 = load float* %72, align 4, !tbaa !1
  %74 = insertelement <2 x float> undef, float %73, i32 0
  %75 = insertelement <2 x float> %74, float %73, i32 1
  %76 = fmul <2 x float> %33, %75
  %77 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 1, i64 2
  %78 = load float* %77, align 4, !tbaa !1
  %79 = insertelement <2 x float> undef, float %78, i32 0
  %80 = insertelement <2 x float> %79, float %78, i32 1
  %81 = fmul <2 x float> %41, %80
  %82 = fadd <2 x float> %76, %81
  %83 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 2, i64 2
  %84 = load float* %83, align 4, !tbaa !1
  %85 = insertelement <2 x float> undef, float %84, i32 0
  %86 = insertelement <2 x float> %85, float %84, i32 1
  %87 = fmul <2 x float> %50, %86
  %88 = fadd <2 x float> %82, %87
  %89 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 0, i64 3
  %90 = load float* %89, align 4, !tbaa !1
  %91 = insertelement <2 x float> undef, float %90, i32 0
  %92 = insertelement <2 x float> %91, float %90, i32 1
  %93 = fmul <2 x float> %33, %92
  %94 = fmul float %10, %90
  %95 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 1, i64 3
  %96 = load float* %95, align 4, !tbaa !1
  %97 = insertelement <2 x float> undef, float %96, i32 0
  %98 = insertelement <2 x float> %97, float %96, i32 1
  %99 = fmul <2 x float> %41, %98
  %100 = fmul float %19, %96
  %101 = fadd <2 x float> %93, %99
  %102 = fadd float %94, %100
  %103 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 2, i64 3
  %104 = load float* %103, align 4, !tbaa !1
  %105 = insertelement <2 x float> undef, float %104, i32 0
  %106 = insertelement <2 x float> %105, float %104, i32 1
  %107 = fmul <2 x float> %50, %106
  %108 = fmul float %28, %104
  %109 = fadd <2 x float> %101, %107
  %110 = fadd float %102, %108
  %111 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 3, i64 3
  %112 = extractelement <2 x float> %109, i32 0
  %113 = load float* %111, align 4, !tbaa !1
  %114 = fadd float %113, %112
  %115 = fcmp une float %114, 0.000000e+00
  br i1 %115, label %116, label %177

; <label>:116                                     ; preds = %0
  %117 = extractelement <2 x float> %88, i32 0
  %118 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 3, i64 2
  %119 = load float* %118, align 4, !tbaa !1
  %120 = fadd float %117, %119
  %121 = extractelement <2 x float> %71, i32 0
  %122 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 3, i64 1
  %123 = load float* %122, align 4, !tbaa !1
  %124 = fadd float %121, %123
  %125 = extractelement <2 x float> %54, i32 0
  %126 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %M, i64 0, i32 0, i64 3, i64 0
  %127 = load float* %126, align 4, !tbaa !1
  %128 = fadd float %125, %127
  %129 = fmul float %10, %73
  %130 = fmul float %19, %78
  %131 = fadd float %129, %130
  %132 = fmul float %28, %84
  %133 = fadd float %131, %132
  %134 = fmul float %10, %56
  %135 = fmul float %19, %61
  %136 = fadd float %134, %135
  %137 = fmul float %28, %67
  %138 = fadd float %136, %137
  %139 = fmul float %31, %10
  %140 = fmul float %39, %19
  %141 = fadd float %139, %140
  %142 = fmul float %48, %28
  %143 = fadd float %141, %142
  %144 = fdiv float 1.000000e+00, %114
  %145 = fmul float %144, %128
  %146 = extractelement <2 x float> %54, i32 1
  %147 = extractelement <2 x float> %109, i32 1
  %148 = fmul float %124, %144
  %149 = extractelement <2 x float> %71, i32 1
  %150 = fmul float %110, %148
  %151 = fsub float %138, %150
  %152 = fmul float %144, %151
  %153 = fmul float %120, %144
  %154 = extractelement <2 x float> %88, i32 1
  %155 = insertelement <4 x float> undef, float %147, i32 0
  %156 = insertelement <4 x float> %155, float %147, i32 1
  %157 = insertelement <4 x float> %156, float %147, i32 2
  %158 = insertelement <4 x float> %157, float %110, i32 3
  %159 = insertelement <4 x float> undef, float %145, i32 0
  %160 = insertelement <4 x float> %159, float %148, i32 1
  %161 = insertelement <4 x float> %160, float %153, i32 2
  %162 = insertelement <4 x float> %161, float %145, i32 3
  %163 = fmul <4 x float> %158, %162
  %164 = insertelement <4 x float> undef, float %146, i32 0
  %165 = insertelement <4 x float> %164, float %149, i32 1
  %166 = insertelement <4 x float> %165, float %154, i32 2
  %167 = insertelement <4 x float> %166, float %143, i32 3
  %168 = fsub <4 x float> %167, %163
  %169 = insertelement <4 x float> undef, float %144, i32 0
  %170 = insertelement <4 x float> %169, float %144, i32 1
  %171 = insertelement <4 x float> %170, float %144, i32 2
  %172 = insertelement <4 x float> %171, float %144, i32 3
  %173 = fmul <4 x float> %168, %172
  %174 = fmul float %110, %153
  %175 = fsub float %133, %174
  %176 = fmul float %144, %175
  br label %177

; <label>:177                                     ; preds = %0, %116
  %178 = phi float [ %153, %116 ], [ 0.000000e+00, %0 ]
  %179 = phi float [ %148, %116 ], [ 0.000000e+00, %0 ]
  %180 = phi float [ %145, %116 ], [ 0.000000e+00, %0 ]
  %181 = phi float [ %176, %116 ], [ 0.000000e+00, %0 ]
  %182 = phi float [ %152, %116 ], [ 0.000000e+00, %0 ]
  %183 = phi <4 x float> [ %173, %116 ], [ zeroinitializer, %0 ]
  %184 = getelementptr inbounds %"class.OSL::Dual2.0"* %out, i64 0, i32 0, i32 0
  store float %180, float* %184, align 4, !tbaa !5
  %185 = getelementptr inbounds %"class.OSL::Dual2.0"* %out, i64 0, i32 0, i32 1
  store float %179, float* %185, align 4, !tbaa !7
  %186 = getelementptr inbounds %"class.OSL::Dual2.0"* %out, i64 0, i32 0, i32 2
  store float %178, float* %186, align 4, !tbaa !8
  %187 = getelementptr inbounds %"class.OSL::Dual2.0"* %out, i64 0, i32 1, i32 0
  %188 = bitcast float* %187 to <4 x float>*
  store <4 x float> %183, <4 x float>* %188, align 4, !tbaa !1
  %189 = getelementptr inbounds %"class.OSL::Dual2.0"* %out, i64 0, i32 2, i32 1
  store float %182, float* %189, align 4, !tbaa !7
  %190 = getelementptr inbounds %"class.OSL::Dual2.0"* %out, i64 0, i32 2, i32 2
  store float %181, float* %190, align 4, !tbaa !8
  call void @llvm.lifetime.end(i64 36, i8* %1) #2
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_transformv_vmv(i8* nocapture %result, i8* nocapture readonly %M_, i8* nocapture readonly %v_) #4 {
  %1 = bitcast i8* %v_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = bitcast i8* %M_ to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = fmul float %2, %4
  %6 = getelementptr inbounds i8* %v_, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = getelementptr inbounds i8* %M_, i64 16
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !1
  %12 = fmul float %8, %11
  %13 = fadd float %5, %12
  %14 = getelementptr inbounds i8* %v_, i64 8
  %15 = bitcast i8* %14 to float*
  %16 = load float* %15, align 4, !tbaa !1
  %17 = getelementptr inbounds i8* %M_, i64 32
  %18 = bitcast i8* %17 to float*
  %19 = load float* %18, align 4, !tbaa !1
  %20 = fmul float %16, %19
  %21 = fadd float %13, %20
  %22 = getelementptr inbounds i8* %M_, i64 4
  %23 = bitcast i8* %22 to float*
  %24 = load float* %23, align 4, !tbaa !1
  %25 = fmul float %2, %24
  %26 = getelementptr inbounds i8* %M_, i64 20
  %27 = bitcast i8* %26 to float*
  %28 = load float* %27, align 4, !tbaa !1
  %29 = fmul float %8, %28
  %30 = fadd float %25, %29
  %31 = getelementptr inbounds i8* %M_, i64 36
  %32 = bitcast i8* %31 to float*
  %33 = load float* %32, align 4, !tbaa !1
  %34 = fmul float %16, %33
  %35 = fadd float %30, %34
  %36 = getelementptr inbounds i8* %M_, i64 8
  %37 = bitcast i8* %36 to float*
  %38 = load float* %37, align 4, !tbaa !1
  %39 = fmul float %2, %38
  %40 = getelementptr inbounds i8* %M_, i64 24
  %41 = bitcast i8* %40 to float*
  %42 = load float* %41, align 4, !tbaa !1
  %43 = fmul float %8, %42
  %44 = fadd float %39, %43
  %45 = getelementptr inbounds i8* %M_, i64 40
  %46 = bitcast i8* %45 to float*
  %47 = load float* %46, align 4, !tbaa !1
  %48 = fmul float %16, %47
  %49 = fadd float %44, %48
  %50 = bitcast i8* %result to float*
  store float %21, float* %50, align 4, !tbaa !5
  %51 = getelementptr inbounds i8* %result, i64 4
  %52 = bitcast i8* %51 to float*
  store float %35, float* %52, align 4, !tbaa !7
  %53 = getelementptr inbounds i8* %result, i64 8
  %54 = bitcast i8* %53 to float*
  store float %49, float* %54, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_transformv_dvmdv(i8* nocapture %result, i8* nocapture readonly %M_, i8* nocapture readonly %v_) #4 {
  %1 = bitcast i8* %v_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = bitcast i8* %M_ to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = fmul float %2, %4
  %6 = getelementptr inbounds i8* %v_, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = getelementptr inbounds i8* %M_, i64 16
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !1
  %12 = fmul float %8, %11
  %13 = fadd float %5, %12
  %14 = getelementptr inbounds i8* %v_, i64 8
  %15 = bitcast i8* %14 to float*
  %16 = load float* %15, align 4, !tbaa !1
  %17 = getelementptr inbounds i8* %M_, i64 32
  %18 = bitcast i8* %17 to float*
  %19 = load float* %18, align 4, !tbaa !1
  %20 = fmul float %16, %19
  %21 = fadd float %13, %20
  %22 = getelementptr inbounds i8* %M_, i64 4
  %23 = bitcast i8* %22 to float*
  %24 = load float* %23, align 4, !tbaa !1
  %25 = fmul float %2, %24
  %26 = getelementptr inbounds i8* %M_, i64 20
  %27 = bitcast i8* %26 to float*
  %28 = load float* %27, align 4, !tbaa !1
  %29 = fmul float %8, %28
  %30 = fadd float %25, %29
  %31 = getelementptr inbounds i8* %M_, i64 36
  %32 = bitcast i8* %31 to float*
  %33 = load float* %32, align 4, !tbaa !1
  %34 = fmul float %16, %33
  %35 = fadd float %30, %34
  %36 = getelementptr inbounds i8* %M_, i64 8
  %37 = bitcast i8* %36 to float*
  %38 = load float* %37, align 4, !tbaa !1
  %39 = fmul float %2, %38
  %40 = getelementptr inbounds i8* %M_, i64 24
  %41 = bitcast i8* %40 to float*
  %42 = load float* %41, align 4, !tbaa !1
  %43 = fmul float %8, %42
  %44 = fadd float %39, %43
  %45 = getelementptr inbounds i8* %M_, i64 40
  %46 = bitcast i8* %45 to float*
  %47 = load float* %46, align 4, !tbaa !1
  %48 = fmul float %16, %47
  %49 = fadd float %44, %48
  %50 = bitcast i8* %result to float*
  store float %21, float* %50, align 4, !tbaa !5
  %51 = getelementptr inbounds i8* %result, i64 4
  %52 = bitcast i8* %51 to float*
  store float %35, float* %52, align 4, !tbaa !7
  %53 = getelementptr inbounds i8* %result, i64 8
  %54 = bitcast i8* %53 to float*
  store float %49, float* %54, align 4, !tbaa !8
  %55 = getelementptr inbounds i8* %v_, i64 12
  %56 = bitcast i8* %55 to float*
  %57 = load float* %56, align 4, !tbaa !1
  %58 = load float* %3, align 4, !tbaa !1
  %59 = fmul float %57, %58
  %60 = getelementptr inbounds i8* %v_, i64 16
  %61 = bitcast i8* %60 to float*
  %62 = load float* %61, align 4, !tbaa !1
  %63 = load float* %10, align 4, !tbaa !1
  %64 = fmul float %62, %63
  %65 = fadd float %59, %64
  %66 = getelementptr inbounds i8* %v_, i64 20
  %67 = bitcast i8* %66 to float*
  %68 = load float* %67, align 4, !tbaa !1
  %69 = load float* %18, align 4, !tbaa !1
  %70 = fmul float %68, %69
  %71 = fadd float %65, %70
  %72 = load float* %23, align 4, !tbaa !1
  %73 = fmul float %57, %72
  %74 = load float* %27, align 4, !tbaa !1
  %75 = fmul float %62, %74
  %76 = fadd float %73, %75
  %77 = load float* %32, align 4, !tbaa !1
  %78 = fmul float %68, %77
  %79 = fadd float %76, %78
  %80 = load float* %37, align 4, !tbaa !1
  %81 = fmul float %57, %80
  %82 = load float* %41, align 4, !tbaa !1
  %83 = fmul float %62, %82
  %84 = fadd float %81, %83
  %85 = load float* %46, align 4, !tbaa !1
  %86 = fmul float %68, %85
  %87 = fadd float %84, %86
  %88 = getelementptr inbounds i8* %result, i64 12
  %89 = bitcast i8* %88 to float*
  store float %71, float* %89, align 4, !tbaa !5
  %90 = getelementptr inbounds i8* %result, i64 16
  %91 = bitcast i8* %90 to float*
  store float %79, float* %91, align 4, !tbaa !7
  %92 = getelementptr inbounds i8* %result, i64 20
  %93 = bitcast i8* %92 to float*
  store float %87, float* %93, align 4, !tbaa !8
  %94 = getelementptr inbounds i8* %v_, i64 24
  %95 = bitcast i8* %94 to float*
  %96 = load float* %95, align 4, !tbaa !1
  %97 = load float* %3, align 4, !tbaa !1
  %98 = fmul float %96, %97
  %99 = getelementptr inbounds i8* %v_, i64 28
  %100 = bitcast i8* %99 to float*
  %101 = load float* %100, align 4, !tbaa !1
  %102 = load float* %10, align 4, !tbaa !1
  %103 = fmul float %101, %102
  %104 = fadd float %98, %103
  %105 = getelementptr inbounds i8* %v_, i64 32
  %106 = bitcast i8* %105 to float*
  %107 = load float* %106, align 4, !tbaa !1
  %108 = load float* %18, align 4, !tbaa !1
  %109 = fmul float %107, %108
  %110 = fadd float %104, %109
  %111 = load float* %23, align 4, !tbaa !1
  %112 = fmul float %96, %111
  %113 = load float* %27, align 4, !tbaa !1
  %114 = fmul float %101, %113
  %115 = fadd float %112, %114
  %116 = load float* %32, align 4, !tbaa !1
  %117 = fmul float %107, %116
  %118 = fadd float %115, %117
  %119 = load float* %37, align 4, !tbaa !1
  %120 = fmul float %96, %119
  %121 = load float* %41, align 4, !tbaa !1
  %122 = fmul float %101, %121
  %123 = fadd float %120, %122
  %124 = load float* %46, align 4, !tbaa !1
  %125 = fmul float %107, %124
  %126 = fadd float %123, %125
  %127 = getelementptr inbounds i8* %result, i64 24
  %128 = bitcast i8* %127 to float*
  store float %110, float* %128, align 4, !tbaa !5
  %129 = getelementptr inbounds i8* %result, i64 28
  %130 = bitcast i8* %129 to float*
  store float %118, float* %130, align 4, !tbaa !7
  %131 = getelementptr inbounds i8* %result, i64 32
  %132 = bitcast i8* %131 to float*
  store float %126, float* %132, align 4, !tbaa !8
  ret void
}

; Function Attrs: uwtable
define void @osl_transformn_vmv(i8* nocapture %result, i8* nocapture readonly %M_, i8* nocapture readonly %v_) #9 {
  %1 = alloca %"class.Imath_2_2::Matrix44", align 4
  %2 = bitcast i8* %M_ to %"class.Imath_2_2::Matrix44"*
  call void @_ZNK9Imath_2_28Matrix44IfE7inverseEb(%"class.Imath_2_2::Matrix44"* sret %1, %"class.Imath_2_2::Matrix44"* %2, i1 zeroext false)
  %3 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 0
  %4 = load float* %3, align 4, !tbaa !1
  %5 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 0
  %6 = load float* %5, align 4, !tbaa !1
  %7 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 0
  %8 = load float* %7, align 4, !tbaa !1
  %9 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 1
  %10 = load float* %9, align 4, !tbaa !1
  %11 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 1
  %12 = load float* %11, align 4, !tbaa !1
  %13 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 1
  %14 = load float* %13, align 4, !tbaa !1
  %15 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 2
  %16 = load float* %15, align 4, !tbaa !1
  %17 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 2
  %18 = load float* %17, align 4, !tbaa !1
  %19 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 2
  %20 = load float* %19, align 4, !tbaa !1
  %21 = bitcast i8* %v_ to float*
  %22 = load float* %21, align 4, !tbaa !1
  %23 = fmul float %4, %22
  %24 = getelementptr inbounds i8* %v_, i64 4
  %25 = bitcast i8* %24 to float*
  %26 = load float* %25, align 4, !tbaa !1
  %27 = fmul float %10, %26
  %28 = fadd float %23, %27
  %29 = getelementptr inbounds i8* %v_, i64 8
  %30 = bitcast i8* %29 to float*
  %31 = load float* %30, align 4, !tbaa !1
  %32 = fmul float %16, %31
  %33 = fadd float %28, %32
  %34 = fmul float %6, %22
  %35 = fmul float %12, %26
  %36 = fadd float %34, %35
  %37 = fmul float %18, %31
  %38 = fadd float %36, %37
  %39 = fmul float %8, %22
  %40 = fmul float %14, %26
  %41 = fadd float %39, %40
  %42 = fmul float %20, %31
  %43 = fadd float %41, %42
  %44 = bitcast i8* %result to float*
  store float %33, float* %44, align 4, !tbaa !5
  %45 = getelementptr inbounds i8* %result, i64 4
  %46 = bitcast i8* %45 to float*
  store float %38, float* %46, align 4, !tbaa !7
  %47 = getelementptr inbounds i8* %result, i64 8
  %48 = bitcast i8* %47 to float*
  store float %43, float* %48, align 4, !tbaa !8
  ret void
}

; Function Attrs: uwtable
define linkonce_odr void @_ZNK9Imath_2_28Matrix44IfE7inverseEb(%"class.Imath_2_2::Matrix44"* noalias sret %agg.result, %"class.Imath_2_2::Matrix44"* nocapture readonly %this, i1 zeroext %singExc) #9 align 2 {
  %s = alloca %"class.Imath_2_2::Matrix44", align 16
  %1 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 0, i64 3
  %2 = load float* %1, align 4, !tbaa !1
  %3 = fcmp une float %2, 0.000000e+00
  br i1 %3, label %16, label %4

; <label>:4                                       ; preds = %0
  %5 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 1, i64 3
  %6 = load float* %5, align 4, !tbaa !1
  %7 = fcmp une float %6, 0.000000e+00
  br i1 %7, label %16, label %8

; <label>:8                                       ; preds = %4
  %9 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 2, i64 3
  %10 = load float* %9, align 4, !tbaa !1
  %11 = fcmp une float %10, 0.000000e+00
  br i1 %11, label %16, label %12

; <label>:12                                      ; preds = %8
  %13 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 3, i64 3
  %14 = load float* %13, align 4, !tbaa !1
  %15 = fcmp une float %14, 1.000000e+00
  br i1 %15, label %16, label %23

; <label>:16                                      ; preds = %12, %8, %4, %0
  invoke void @_ZNK9Imath_2_28Matrix44IfE9gjInverseEb(%"class.Imath_2_2::Matrix44"* sret %agg.result, %"class.Imath_2_2::Matrix44"* %this, i1 zeroext %singExc)
          to label %202 unwind label %17

; <label>:17                                      ; preds = %136, %16
  %18 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          filter [1 x i8*] [i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN7Iex_2_27MathExcE to i8*)]
  %19 = extractvalue { i8*, i32 } %18, 1
  %20 = icmp slt i32 %19, 0
  br i1 %20, label %21, label %203

; <label>:21                                      ; preds = %17
  %22 = extractvalue { i8*, i32 } %18, 0
  tail call void @__cxa_call_unexpected(i8* %22) #13
  unreachable

; <label>:23                                      ; preds = %12
  %24 = bitcast %"class.Imath_2_2::Matrix44"* %s to i8*
  call void @llvm.lifetime.start(i64 64, i8* %24) #2
  %25 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 1, i64 1
  %26 = load float* %25, align 4, !tbaa !1
  %27 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 2, i64 2
  %28 = load float* %27, align 4, !tbaa !1
  %29 = fmul float %26, %28
  %30 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 2, i64 1
  %31 = load float* %30, align 4, !tbaa !1
  %32 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 1, i64 2
  %33 = load float* %32, align 4, !tbaa !1
  %34 = fmul float %31, %33
  %35 = fsub float %29, %34
  %36 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 0, i64 2
  %37 = load float* %36, align 4, !tbaa !1
  %38 = fmul float %31, %37
  %39 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 0, i64 1
  %40 = load float* %39, align 4, !tbaa !1
  %41 = fmul float %28, %40
  %42 = fsub float %38, %41
  %43 = fmul float %33, %40
  %44 = fmul float %26, %37
  %45 = fsub float %43, %44
  %46 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 2, i64 0
  %47 = load float* %46, align 4, !tbaa !1
  %48 = fmul float %33, %47
  %49 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 1, i64 0
  %50 = load float* %49, align 4, !tbaa !1
  %51 = fmul float %28, %50
  %52 = fsub float %48, %51
  %53 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 0, i64 0
  %54 = load float* %53, align 4, !tbaa !1
  %55 = fmul float %28, %54
  %56 = fmul float %37, %47
  %57 = fsub float %55, %56
  %58 = fmul float %37, %50
  %59 = fmul float %33, %54
  %60 = fsub float %58, %59
  %61 = fmul float %31, %50
  %62 = fmul float %26, %47
  %63 = fsub float %61, %62
  %64 = fmul float %40, %47
  %65 = fmul float %31, %54
  %66 = fsub float %64, %65
  %67 = fmul float %26, %54
  %68 = fmul float %40, %50
  %69 = fsub float %67, %68
  %70 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 0, i64 0
  store float %35, float* %70, align 16, !tbaa !1
  %71 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 0, i64 1
  store float %42, float* %71, align 4, !tbaa !1
  %72 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 0, i64 2
  store float %45, float* %72, align 8, !tbaa !1
  %73 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 0, i64 3
  store float 0.000000e+00, float* %73, align 4, !tbaa !1
  %74 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 1, i64 0
  store float %52, float* %74, align 16, !tbaa !1
  %75 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 1, i64 1
  store float %57, float* %75, align 4, !tbaa !1
  %76 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 1, i64 2
  store float %60, float* %76, align 8, !tbaa !1
  %77 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 1, i64 3
  store float 0.000000e+00, float* %77, align 4, !tbaa !1
  %78 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 2, i64 0
  store float %63, float* %78, align 16, !tbaa !1
  %79 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 2, i64 1
  store float %66, float* %79, align 4, !tbaa !1
  %80 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 2, i64 2
  store float %69, float* %80, align 8, !tbaa !1
  %81 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 2, i64 3
  %82 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 3, i64 0
  %83 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 3, i64 1
  %84 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 3, i64 2
  %85 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 3, i64 3
  %86 = bitcast float* %81 to i8*
  call void @llvm.memset.p0i8.i64(i8* %86, i8 0, i64 16, i32 4, i1 false)
  store float 1.000000e+00, float* %85, align 4, !tbaa !1
  %87 = fmul float %54, %35
  %88 = fmul float %40, %52
  %89 = fadd float %87, %88
  %90 = fmul float %37, %63
  %91 = fadd float %89, %90
  %92 = fcmp ogt float %91, 0.000000e+00
  br i1 %92, label %_ZN9Imath_2_23absIfEET_S1_.exit4, label %93

; <label>:93                                      ; preds = %23
  %94 = fsub float -0.000000e+00, %91
  br label %_ZN9Imath_2_23absIfEET_S1_.exit4

_ZN9Imath_2_23absIfEET_S1_.exit4:                 ; preds = %23, %93
  %95 = phi float [ %94, %93 ], [ %91, %23 ]
  %96 = fcmp ult float %95, 1.000000e+00
  br i1 %96, label %118, label %.preheader6

.preheader6:                                      ; preds = %_ZN9Imath_2_23absIfEET_S1_.exit4
  %97 = fdiv float %35, %91
  store float %97, float* %70, align 16, !tbaa !1
  %98 = fdiv float %42, %91
  store float %98, float* %71, align 4, !tbaa !1
  %99 = fdiv float %45, %91
  store float %99, float* %72, align 8, !tbaa !1
  %100 = fdiv float %52, %91
  store float %100, float* %74, align 16, !tbaa !1
  %101 = fdiv float %57, %91
  store float %101, float* %75, align 4, !tbaa !1
  %102 = fdiv float %60, %91
  store float %102, float* %76, align 8, !tbaa !1
  %103 = fdiv float %63, %91
  store float %103, float* %78, align 16, !tbaa !1
  %104 = fdiv float %66, %91
  store float %104, float* %79, align 4, !tbaa !1
  %105 = fdiv float %69, %91
  store float %105, float* %80, align 8, !tbaa !1
  %106 = insertelement <4 x float> undef, float %97, i32 0
  %107 = insertelement <4 x float> %106, float %98, i32 1
  %108 = insertelement <4 x float> %107, float %99, i32 2
  %109 = insertelement <4 x float> %108, float 0.000000e+00, i32 3
  %110 = insertelement <4 x float> undef, float %100, i32 0
  %111 = insertelement <4 x float> %110, float %101, i32 1
  %112 = insertelement <4 x float> %111, float %102, i32 2
  %113 = insertelement <4 x float> %112, float 0.000000e+00, i32 3
  %114 = insertelement <4 x float> undef, float %103, i32 0
  %115 = insertelement <4 x float> %114, float %104, i32 1
  %116 = insertelement <4 x float> %115, float %105, i32 2
  %117 = insertelement <4 x float> %116, float 0.000000e+00, i32 3
  br label %155

; <label>:118                                     ; preds = %_ZN9Imath_2_23absIfEET_S1_.exit4
  br i1 %92, label %_ZN9Imath_2_23absIfEET_S1_.exit3, label %119

; <label>:119                                     ; preds = %118
  %120 = fsub float -0.000000e+00, %91
  br label %_ZN9Imath_2_23absIfEET_S1_.exit3

_ZN9Imath_2_23absIfEET_S1_.exit3:                 ; preds = %118, %119
  %121 = phi float [ %120, %119 ], [ %91, %118 ]
  %122 = fmul float %121, 0x47D0000000000000
  br label %.preheader

.preheader:                                       ; preds = %_ZN9Imath_2_23absIfEET_S1_.exit3, %146
  %indvars.iv12 = phi i64 [ 0, %_ZN9Imath_2_23absIfEET_S1_.exit3 ], [ %indvars.iv.next13, %146 ]
  br label %123

; <label>:123                                     ; preds = %.preheader, %131
  %indvars.iv = phi i64 [ 0, %.preheader ], [ %indvars.iv.next, %131 ]
  %124 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv12, i64 %indvars.iv
  %125 = load float* %124, align 4, !tbaa !1
  %126 = fcmp ogt float %125, 0.000000e+00
  br i1 %126, label %_ZN9Imath_2_23absIfEET_S1_.exit, label %127

; <label>:127                                     ; preds = %123
  %128 = fsub float -0.000000e+00, %125
  br label %_ZN9Imath_2_23absIfEET_S1_.exit

_ZN9Imath_2_23absIfEET_S1_.exit:                  ; preds = %123, %127
  %129 = phi float [ %128, %127 ], [ %125, %123 ]
  %130 = fcmp ogt float %122, %129
  br i1 %130, label %131, label %135

; <label>:131                                     ; preds = %_ZN9Imath_2_23absIfEET_S1_.exit
  %132 = fdiv float %125, %91
  store float %132, float* %124, align 4, !tbaa !1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %133 = trunc i64 %indvars.iv.next to i32
  %134 = icmp slt i32 %133, 3
  br i1 %134, label %123, label %146

; <label>:135                                     ; preds = %_ZN9Imath_2_23absIfEET_S1_.exit
  br i1 %singExc, label %136, label %140

; <label>:136                                     ; preds = %135
  %137 = tail call i8* @__cxa_allocate_exception(i64 24) #2
  %138 = bitcast i8* %137 to %"class.Iex_2_2::BaseExc"*
  tail call void @_ZN7Iex_2_27BaseExcC2EPKc(%"class.Iex_2_2::BaseExc"* %138, i8* getelementptr inbounds ([31 x i8]* @.str, i64 0, i64 0)) #2
  %139 = bitcast i8* %137 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ([5 x i8*]* @_ZTVN9Imath_2_213SingMatrixExcE, i64 0, i64 2) to i32 (...)**), i32 (...)*** %139, align 8, !tbaa !13
  invoke void @__cxa_throw(i8* %137, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN9Imath_2_213SingMatrixExcE to i8*), i8* bitcast (void (%"class.Iex_2_2::BaseExc"*)* @_ZN7Iex_2_27BaseExcD2Ev to i8*)) #13
          to label %204 unwind label %17

; <label>:140                                     ; preds = %135
  %141 = bitcast %"class.Imath_2_2::Matrix44"* %agg.result to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %141, i8 0, i64 60, i32 4, i1 false) #2
  %142 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 0, i64 0
  store float 1.000000e+00, float* %142, align 4, !tbaa !1
  %143 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 1, i64 1
  store float 1.000000e+00, float* %143, align 4, !tbaa !1
  %144 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 2, i64 2
  store float 1.000000e+00, float* %144, align 4, !tbaa !1
  %145 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 3, i64 3
  store float 1.000000e+00, float* %145, align 4, !tbaa !1
  br label %202

; <label>:146                                     ; preds = %131
  %indvars.iv.next13 = add nuw nsw i64 %indvars.iv12, 1
  %147 = trunc i64 %indvars.iv.next13 to i32
  %148 = icmp slt i32 %147, 3
  br i1 %148, label %.preheader, label %.loopexit

.loopexit:                                        ; preds = %146
  %149 = bitcast %"class.Imath_2_2::Matrix44"* %s to <4 x float>*
  %150 = load <4 x float>* %149, align 16, !tbaa !1
  %151 = bitcast float* %74 to <4 x float>*
  %152 = load <4 x float>* %151, align 16, !tbaa !1
  %153 = bitcast float* %78 to <4 x float>*
  %154 = load <4 x float>* %153, align 16, !tbaa !1
  %.pre32 = load float* %85, align 4, !tbaa !1
  br label %155

; <label>:155                                     ; preds = %.preheader6, %.loopexit
  %156 = phi float [ 1.000000e+00, %.preheader6 ], [ %.pre32, %.loopexit ]
  %157 = phi <4 x float> [ %109, %.preheader6 ], [ %150, %.loopexit ]
  %158 = phi <4 x float> [ %113, %.preheader6 ], [ %152, %.loopexit ]
  %159 = phi <4 x float> [ %117, %.preheader6 ], [ %154, %.loopexit ]
  %160 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 3, i64 0
  %161 = load float* %160, align 4, !tbaa !1
  %162 = extractelement <4 x float> %157, i32 0
  %163 = fmul float %161, %162
  %164 = fsub float -0.000000e+00, %163
  %165 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 3, i64 1
  %166 = load float* %165, align 4, !tbaa !1
  %167 = extractelement <4 x float> %158, i32 0
  %168 = fmul float %166, %167
  %169 = fsub float %164, %168
  %170 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 3, i64 2
  %171 = load float* %170, align 4, !tbaa !1
  %172 = extractelement <4 x float> %159, i32 0
  %173 = fmul float %171, %172
  %174 = fsub float %169, %173
  store float %174, float* %82, align 16, !tbaa !1
  %175 = extractelement <4 x float> %157, i32 1
  %176 = fmul float %161, %175
  %177 = fsub float -0.000000e+00, %176
  %178 = extractelement <4 x float> %158, i32 1
  %179 = fmul float %166, %178
  %180 = fsub float %177, %179
  %181 = extractelement <4 x float> %159, i32 1
  %182 = fmul float %171, %181
  %183 = fsub float %180, %182
  store float %183, float* %83, align 4, !tbaa !1
  %184 = extractelement <4 x float> %157, i32 2
  %185 = fmul float %161, %184
  %186 = fsub float -0.000000e+00, %185
  %187 = extractelement <4 x float> %158, i32 2
  %188 = fmul float %166, %187
  %189 = fsub float %186, %188
  %190 = extractelement <4 x float> %159, i32 2
  %191 = fmul float %171, %190
  %192 = fsub float %189, %191
  store float %192, float* %84, align 8, !tbaa !1
  %193 = bitcast %"class.Imath_2_2::Matrix44"* %agg.result to <4 x float>*
  store <4 x float> %157, <4 x float>* %193, align 4, !tbaa !1
  %194 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 1, i64 0
  %195 = bitcast float* %194 to <4 x float>*
  store <4 x float> %158, <4 x float>* %195, align 4, !tbaa !1
  %196 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 2, i64 0
  %197 = bitcast float* %196 to <4 x float>*
  store <4 x float> %159, <4 x float>* %197, align 4, !tbaa !1
  %198 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 3, i64 0
  store float %174, float* %198, align 4, !tbaa !1
  %199 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 3, i64 1
  store float %183, float* %199, align 4, !tbaa !1
  %200 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 3, i64 2
  store float %192, float* %200, align 4, !tbaa !1
  %201 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 3, i64 3
  store float %156, float* %201, align 4, !tbaa !1
  br label %202

; <label>:202                                     ; preds = %140, %155, %16
  ret void

; <label>:203                                     ; preds = %17
  resume { i8*, i32 } %18

; <label>:204                                     ; preds = %136
  unreachable
}

; Function Attrs: uwtable
define void @osl_transformn_dvmdv(i8* nocapture %result, i8* nocapture readonly %M_, i8* nocapture readonly %v_) #9 {
  %1 = alloca %"class.Imath_2_2::Matrix44", align 4
  %2 = bitcast i8* %M_ to %"class.Imath_2_2::Matrix44"*
  call void @_ZNK9Imath_2_28Matrix44IfE7inverseEb(%"class.Imath_2_2::Matrix44"* sret %1, %"class.Imath_2_2::Matrix44"* %2, i1 zeroext false)
  %3 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 0
  %4 = load float* %3, align 4, !tbaa !1
  %5 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 0
  %6 = load float* %5, align 4, !tbaa !1
  %7 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 0
  %8 = load float* %7, align 4, !tbaa !1
  %9 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 1
  %10 = load float* %9, align 4, !tbaa !1
  %11 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 1
  %12 = load float* %11, align 4, !tbaa !1
  %13 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 1
  %14 = load float* %13, align 4, !tbaa !1
  %15 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 2
  %16 = load float* %15, align 4, !tbaa !1
  %17 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 2
  %18 = load float* %17, align 4, !tbaa !1
  %19 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 2
  %20 = load float* %19, align 4, !tbaa !1
  %21 = bitcast i8* %v_ to float*
  %22 = load float* %21, align 4, !tbaa !1
  %23 = fmul float %4, %22
  %24 = getelementptr inbounds i8* %v_, i64 4
  %25 = bitcast i8* %24 to float*
  %26 = load float* %25, align 4, !tbaa !1
  %27 = fmul float %10, %26
  %28 = fadd float %23, %27
  %29 = getelementptr inbounds i8* %v_, i64 8
  %30 = bitcast i8* %29 to float*
  %31 = load float* %30, align 4, !tbaa !1
  %32 = fmul float %16, %31
  %33 = fadd float %28, %32
  %34 = fmul float %6, %22
  %35 = fmul float %12, %26
  %36 = fadd float %34, %35
  %37 = fmul float %18, %31
  %38 = fadd float %36, %37
  %39 = fmul float %8, %22
  %40 = fmul float %14, %26
  %41 = fadd float %39, %40
  %42 = fmul float %20, %31
  %43 = fadd float %41, %42
  %44 = bitcast i8* %result to float*
  store float %33, float* %44, align 4, !tbaa !5
  %45 = getelementptr inbounds i8* %result, i64 4
  %46 = bitcast i8* %45 to float*
  store float %38, float* %46, align 4, !tbaa !7
  %47 = getelementptr inbounds i8* %result, i64 8
  %48 = bitcast i8* %47 to float*
  store float %43, float* %48, align 4, !tbaa !8
  %49 = getelementptr inbounds i8* %v_, i64 12
  %50 = bitcast i8* %49 to float*
  %51 = load float* %50, align 4, !tbaa !1
  %52 = fmul float %4, %51
  %53 = getelementptr inbounds i8* %v_, i64 16
  %54 = bitcast i8* %53 to float*
  %55 = load float* %54, align 4, !tbaa !1
  %56 = fmul float %10, %55
  %57 = fadd float %52, %56
  %58 = getelementptr inbounds i8* %v_, i64 20
  %59 = bitcast i8* %58 to float*
  %60 = load float* %59, align 4, !tbaa !1
  %61 = fmul float %16, %60
  %62 = fadd float %57, %61
  %63 = fmul float %6, %51
  %64 = fmul float %12, %55
  %65 = fadd float %63, %64
  %66 = fmul float %18, %60
  %67 = fadd float %65, %66
  %68 = fmul float %8, %51
  %69 = fmul float %14, %55
  %70 = fadd float %68, %69
  %71 = fmul float %20, %60
  %72 = fadd float %70, %71
  %73 = getelementptr inbounds i8* %result, i64 12
  %74 = bitcast i8* %73 to float*
  store float %62, float* %74, align 4, !tbaa !5
  %75 = getelementptr inbounds i8* %result, i64 16
  %76 = bitcast i8* %75 to float*
  store float %67, float* %76, align 4, !tbaa !7
  %77 = getelementptr inbounds i8* %result, i64 20
  %78 = bitcast i8* %77 to float*
  store float %72, float* %78, align 4, !tbaa !8
  %79 = getelementptr inbounds i8* %v_, i64 24
  %80 = bitcast i8* %79 to float*
  %81 = load float* %80, align 4, !tbaa !1
  %82 = fmul float %4, %81
  %83 = getelementptr inbounds i8* %v_, i64 28
  %84 = bitcast i8* %83 to float*
  %85 = load float* %84, align 4, !tbaa !1
  %86 = fmul float %10, %85
  %87 = fadd float %82, %86
  %88 = getelementptr inbounds i8* %v_, i64 32
  %89 = bitcast i8* %88 to float*
  %90 = load float* %89, align 4, !tbaa !1
  %91 = fmul float %16, %90
  %92 = fadd float %87, %91
  %93 = fmul float %6, %81
  %94 = fmul float %12, %85
  %95 = fadd float %93, %94
  %96 = fmul float %18, %90
  %97 = fadd float %95, %96
  %98 = fmul float %8, %81
  %99 = fmul float %14, %85
  %100 = fadd float %98, %99
  %101 = fmul float %20, %90
  %102 = fadd float %100, %101
  %103 = getelementptr inbounds i8* %result, i64 24
  %104 = bitcast i8* %103 to float*
  store float %92, float* %104, align 4, !tbaa !5
  %105 = getelementptr inbounds i8* %result, i64 28
  %106 = bitcast i8* %105 to float*
  store float %97, float* %106, align 4, !tbaa !7
  %107 = getelementptr inbounds i8* %result, i64 32
  %108 = bitcast i8* %107 to float*
  store float %102, float* %108, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readonly uwtable
define float @osl_dot_fvv(i8* nocapture readonly %a, i8* nocapture readonly %b) #10 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = bitcast i8* %b to float*
  %4 = load float* %3, align 4, !tbaa !5
  %5 = fmul float %2, %4
  %6 = getelementptr inbounds i8* %a, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8* %b, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !7
  %12 = fmul float %8, %11
  %13 = fadd float %5, %12
  %14 = getelementptr inbounds i8* %a, i64 8
  %15 = bitcast i8* %14 to float*
  %16 = load float* %15, align 4, !tbaa !8
  %17 = getelementptr inbounds i8* %b, i64 8
  %18 = bitcast i8* %17 to float*
  %19 = load float* %18, align 4, !tbaa !8
  %20 = fmul float %16, %19
  %21 = fadd float %13, %20
  ret float %21
}

; Function Attrs: nounwind uwtable
define void @osl_dot_dfdvdv(i8* nocapture %result, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %a to float*
  %2 = getelementptr inbounds i8* %a, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = getelementptr inbounds i8* %a, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = getelementptr inbounds i8* %a, i64 16
  %12 = bitcast i8* %11 to float*
  %13 = getelementptr inbounds i8* %a, i64 28
  %14 = bitcast i8* %13 to float*
  %15 = load float* %10, align 4, !tbaa !1
  %16 = load float* %12, align 4, !tbaa !1
  %17 = load float* %14, align 4, !tbaa !1
  %18 = getelementptr inbounds i8* %a, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = getelementptr inbounds i8* %a, i64 20
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8* %a, i64 32
  %23 = bitcast i8* %22 to float*
  %24 = load float* %19, align 4, !tbaa !1
  %25 = load float* %21, align 4, !tbaa !1
  %26 = load float* %23, align 4, !tbaa !1
  %27 = bitcast i8* %b to float*
  %28 = getelementptr inbounds i8* %b, i64 12
  %29 = bitcast i8* %28 to float*
  %30 = getelementptr inbounds i8* %b, i64 24
  %31 = bitcast i8* %30 to float*
  %32 = load float* %27, align 4, !tbaa !1
  %33 = load float* %29, align 4, !tbaa !1
  %34 = load float* %31, align 4, !tbaa !1
  %35 = getelementptr inbounds i8* %b, i64 4
  %36 = bitcast i8* %35 to float*
  %37 = getelementptr inbounds i8* %b, i64 16
  %38 = bitcast i8* %37 to float*
  %39 = getelementptr inbounds i8* %b, i64 28
  %40 = bitcast i8* %39 to float*
  %41 = load float* %36, align 4, !tbaa !1
  %42 = load float* %38, align 4, !tbaa !1
  %43 = load float* %40, align 4, !tbaa !1
  %44 = getelementptr inbounds i8* %b, i64 8
  %45 = bitcast i8* %44 to float*
  %46 = getelementptr inbounds i8* %b, i64 20
  %47 = bitcast i8* %46 to float*
  %48 = getelementptr inbounds i8* %b, i64 32
  %49 = bitcast i8* %48 to float*
  %50 = load float* %45, align 4, !tbaa !1
  %51 = load float* %47, align 4, !tbaa !1
  %52 = load float* %49, align 4, !tbaa !1
  %53 = fmul float %6, %32
  %54 = fmul float %6, %33
  %55 = fmul float %7, %32
  %56 = fadd float %55, %54
  %57 = fmul float %6, %34
  %58 = fmul float %8, %32
  %59 = fadd float %58, %57
  %60 = insertelement <2 x float> undef, float %53, i32 0
  %61 = insertelement <2 x float> %60, float %56, i32 1
  %62 = fmul float %15, %41
  %63 = fmul float %15, %42
  %64 = fmul float %16, %41
  %65 = fadd float %64, %63
  %66 = fmul float %15, %43
  %67 = fmul float %17, %41
  %68 = fadd float %67, %66
  %69 = insertelement <2 x float> undef, float %62, i32 0
  %70 = insertelement <2 x float> %69, float %65, i32 1
  %71 = fadd <2 x float> %61, %70
  %72 = fadd float %59, %68
  %73 = fmul float %24, %50
  %74 = fmul float %24, %51
  %75 = fmul float %25, %50
  %76 = fadd float %75, %74
  %77 = fmul float %24, %52
  %78 = fmul float %26, %50
  %79 = fadd float %78, %77
  %80 = insertelement <2 x float> undef, float %73, i32 0
  %81 = insertelement <2 x float> %80, float %76, i32 1
  %82 = fadd <2 x float> %71, %81
  %83 = fadd float %72, %79
  %84 = bitcast i8* %result to <2 x float>*
  store <2 x float> %82, <2 x float>* %84, align 4
  %85 = getelementptr inbounds i8* %result, i64 8
  %86 = bitcast i8* %85 to float*
  store float %83, float* %86, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_dot_dfdvv(i8* nocapture %result, i8* nocapture readonly %a, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %b_ to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = getelementptr inbounds i8* %b_, i64 4
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4, !tbaa !7
  %6 = getelementptr inbounds i8* %b_, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !8
  %9 = bitcast i8* %a to float*
  %10 = getelementptr inbounds i8* %a, i64 12
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8* %a, i64 24
  %13 = bitcast i8* %12 to float*
  %14 = load float* %9, align 4, !tbaa !1
  %15 = load float* %11, align 4, !tbaa !1
  %16 = load float* %13, align 4, !tbaa !1
  %17 = getelementptr inbounds i8* %a, i64 4
  %18 = bitcast i8* %17 to float*
  %19 = getelementptr inbounds i8* %a, i64 16
  %20 = bitcast i8* %19 to float*
  %21 = getelementptr inbounds i8* %a, i64 28
  %22 = bitcast i8* %21 to float*
  %23 = load float* %18, align 4, !tbaa !1
  %24 = load float* %20, align 4, !tbaa !1
  %25 = load float* %22, align 4, !tbaa !1
  %26 = getelementptr inbounds i8* %a, i64 8
  %27 = bitcast i8* %26 to float*
  %28 = getelementptr inbounds i8* %a, i64 20
  %29 = bitcast i8* %28 to float*
  %30 = getelementptr inbounds i8* %a, i64 32
  %31 = bitcast i8* %30 to float*
  %32 = load float* %27, align 4, !tbaa !1
  %33 = load float* %29, align 4, !tbaa !1
  %34 = load float* %31, align 4, !tbaa !1
  %35 = fmul float %2, %14
  %36 = fmul float %14, 0.000000e+00
  %37 = fmul float %2, %15
  %38 = fadd float %36, %37
  %39 = fmul float %2, %16
  %40 = fadd float %36, %39
  %41 = insertelement <2 x float> undef, float %35, i32 0
  %42 = insertelement <2 x float> %41, float %38, i32 1
  %43 = fmul float %5, %23
  %44 = fmul float %23, 0.000000e+00
  %45 = fmul float %5, %24
  %46 = fadd float %44, %45
  %47 = fmul float %5, %25
  %48 = fadd float %44, %47
  %49 = insertelement <2 x float> undef, float %43, i32 0
  %50 = insertelement <2 x float> %49, float %46, i32 1
  %51 = fadd <2 x float> %42, %50
  %52 = fadd float %40, %48
  %53 = fmul float %8, %32
  %54 = fmul float %32, 0.000000e+00
  %55 = fmul float %8, %33
  %56 = fadd float %54, %55
  %57 = fmul float %8, %34
  %58 = fadd float %54, %57
  %59 = insertelement <2 x float> undef, float %53, i32 0
  %60 = insertelement <2 x float> %59, float %56, i32 1
  %61 = fadd <2 x float> %51, %60
  %62 = fadd float %52, %58
  %63 = bitcast i8* %result to <2 x float>*
  store <2 x float> %61, <2 x float>* %63, align 4
  %64 = getelementptr inbounds i8* %result, i64 8
  %65 = bitcast i8* %64 to float*
  store float %62, float* %65, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_dot_dfvdv(i8* nocapture %result, i8* nocapture readonly %a_, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = getelementptr inbounds i8* %a_, i64 4
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4, !tbaa !7
  %6 = getelementptr inbounds i8* %a_, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !8
  %9 = bitcast i8* %b to float*
  %10 = getelementptr inbounds i8* %b, i64 12
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8* %b, i64 24
  %13 = bitcast i8* %12 to float*
  %14 = load float* %9, align 4, !tbaa !1
  %15 = load float* %11, align 4, !tbaa !1
  %16 = load float* %13, align 4, !tbaa !1
  %17 = getelementptr inbounds i8* %b, i64 4
  %18 = bitcast i8* %17 to float*
  %19 = getelementptr inbounds i8* %b, i64 16
  %20 = bitcast i8* %19 to float*
  %21 = getelementptr inbounds i8* %b, i64 28
  %22 = bitcast i8* %21 to float*
  %23 = load float* %18, align 4, !tbaa !1
  %24 = load float* %20, align 4, !tbaa !1
  %25 = load float* %22, align 4, !tbaa !1
  %26 = getelementptr inbounds i8* %b, i64 8
  %27 = bitcast i8* %26 to float*
  %28 = getelementptr inbounds i8* %b, i64 20
  %29 = bitcast i8* %28 to float*
  %30 = getelementptr inbounds i8* %b, i64 32
  %31 = bitcast i8* %30 to float*
  %32 = load float* %27, align 4, !tbaa !1
  %33 = load float* %29, align 4, !tbaa !1
  %34 = load float* %31, align 4, !tbaa !1
  %35 = fmul float %2, %14
  %36 = fmul float %2, %15
  %37 = fmul float %14, 0.000000e+00
  %38 = fadd float %37, %36
  %39 = fmul float %2, %16
  %40 = fadd float %37, %39
  %41 = insertelement <2 x float> undef, float %35, i32 0
  %42 = insertelement <2 x float> %41, float %38, i32 1
  %43 = fmul float %5, %23
  %44 = fmul float %5, %24
  %45 = fmul float %23, 0.000000e+00
  %46 = fadd float %45, %44
  %47 = fmul float %5, %25
  %48 = fadd float %45, %47
  %49 = insertelement <2 x float> undef, float %43, i32 0
  %50 = insertelement <2 x float> %49, float %46, i32 1
  %51 = fadd <2 x float> %42, %50
  %52 = fadd float %40, %48
  %53 = fmul float %8, %32
  %54 = fmul float %8, %33
  %55 = fmul float %32, 0.000000e+00
  %56 = fadd float %55, %54
  %57 = fmul float %8, %34
  %58 = fadd float %55, %57
  %59 = insertelement <2 x float> undef, float %53, i32 0
  %60 = insertelement <2 x float> %59, float %56, i32 1
  %61 = fadd <2 x float> %51, %60
  %62 = fadd float %52, %58
  %63 = bitcast i8* %result to <2 x float>*
  store <2 x float> %61, <2 x float>* %63, align 4
  %64 = getelementptr inbounds i8* %result, i64 8
  %65 = bitcast i8* %64 to float*
  store float %62, float* %65, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cross_vvv(i8* nocapture %result, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = getelementptr inbounds i8* %a, i64 4
  %2 = bitcast i8* %1 to float*
  %3 = load float* %2, align 4, !tbaa !7
  %4 = getelementptr inbounds i8* %b, i64 8
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !8
  %7 = fmul float %3, %6
  %8 = getelementptr inbounds i8* %a, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !8
  %11 = getelementptr inbounds i8* %b, i64 4
  %12 = bitcast i8* %11 to float*
  %13 = load float* %12, align 4, !tbaa !7
  %14 = fmul float %10, %13
  %15 = fsub float %7, %14
  %16 = bitcast i8* %b to float*
  %17 = load float* %16, align 4, !tbaa !5
  %18 = fmul float %10, %17
  %19 = bitcast i8* %a to float*
  %20 = load float* %19, align 4, !tbaa !5
  %21 = fmul float %6, %20
  %22 = fsub float %18, %21
  %23 = fmul float %13, %20
  %24 = fmul float %3, %17
  %25 = fsub float %23, %24
  %26 = bitcast i8* %result to float*
  store float %15, float* %26, align 4, !tbaa !5
  %27 = getelementptr inbounds i8* %result, i64 4
  %28 = bitcast i8* %27 to float*
  store float %22, float* %28, align 4, !tbaa !7
  %29 = getelementptr inbounds i8* %result, i64 8
  %30 = bitcast i8* %29 to float*
  store float %25, float* %30, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cross_dvdvdv(i8* nocapture %result, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %a to float*
  %2 = getelementptr inbounds i8* %a, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = getelementptr inbounds i8* %a, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = getelementptr inbounds i8* %a, i64 16
  %12 = bitcast i8* %11 to float*
  %13 = getelementptr inbounds i8* %a, i64 28
  %14 = bitcast i8* %13 to float*
  %15 = load float* %10, align 4, !tbaa !1
  %16 = load float* %12, align 4, !tbaa !1
  %17 = load float* %14, align 4, !tbaa !1
  %18 = getelementptr inbounds i8* %a, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = getelementptr inbounds i8* %a, i64 20
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8* %a, i64 32
  %23 = bitcast i8* %22 to float*
  %24 = load float* %19, align 4, !tbaa !1
  %25 = load float* %21, align 4, !tbaa !1
  %26 = load float* %23, align 4, !tbaa !1
  %27 = bitcast i8* %b to float*
  %28 = getelementptr inbounds i8* %b, i64 12
  %29 = bitcast i8* %28 to float*
  %30 = getelementptr inbounds i8* %b, i64 24
  %31 = bitcast i8* %30 to float*
  %32 = load float* %27, align 4, !tbaa !1
  %33 = load float* %29, align 4, !tbaa !1
  %34 = load float* %31, align 4, !tbaa !1
  %35 = getelementptr inbounds i8* %b, i64 4
  %36 = bitcast i8* %35 to float*
  %37 = getelementptr inbounds i8* %b, i64 16
  %38 = bitcast i8* %37 to float*
  %39 = getelementptr inbounds i8* %b, i64 28
  %40 = bitcast i8* %39 to float*
  %41 = load float* %36, align 4, !tbaa !1
  %42 = load float* %38, align 4, !tbaa !1
  %43 = load float* %40, align 4, !tbaa !1
  %44 = getelementptr inbounds i8* %b, i64 8
  %45 = bitcast i8* %44 to float*
  %46 = getelementptr inbounds i8* %b, i64 20
  %47 = bitcast i8* %46 to float*
  %48 = getelementptr inbounds i8* %b, i64 32
  %49 = bitcast i8* %48 to float*
  %50 = load float* %45, align 4, !tbaa !1
  %51 = load float* %47, align 4, !tbaa !1
  %52 = load float* %49, align 4, !tbaa !1
  %53 = fmul float %15, %50
  %54 = fmul float %15, %51
  %55 = fmul float %16, %50
  %56 = fadd float %55, %54
  %57 = fmul float %15, %52
  %58 = fmul float %17, %50
  %59 = fadd float %58, %57
  %60 = insertelement <2 x float> undef, float %53, i32 0
  %61 = insertelement <2 x float> %60, float %56, i32 1
  %62 = fmul float %24, %41
  %63 = fmul float %24, %42
  %64 = fmul float %25, %41
  %65 = fadd float %64, %63
  %66 = fmul float %24, %43
  %67 = fmul float %26, %41
  %68 = fadd float %67, %66
  %69 = insertelement <2 x float> undef, float %62, i32 0
  %70 = insertelement <2 x float> %69, float %65, i32 1
  %71 = fsub <2 x float> %61, %70
  %72 = fsub float %59, %68
  %73 = fmul float %24, %32
  %74 = fmul float %24, %33
  %75 = fmul float %25, %32
  %76 = fadd float %75, %74
  %77 = fmul float %24, %34
  %78 = fmul float %26, %32
  %79 = fadd float %78, %77
  %80 = insertelement <2 x float> undef, float %73, i32 0
  %81 = insertelement <2 x float> %80, float %76, i32 1
  %82 = fmul float %6, %50
  %83 = fmul float %6, %51
  %84 = fmul float %7, %50
  %85 = fadd float %84, %83
  %86 = fmul float %6, %52
  %87 = fmul float %8, %50
  %88 = fadd float %87, %86
  %89 = insertelement <2 x float> undef, float %82, i32 0
  %90 = insertelement <2 x float> %89, float %85, i32 1
  %91 = fsub <2 x float> %81, %90
  %92 = fsub float %79, %88
  %93 = fmul float %6, %41
  %94 = fmul float %6, %42
  %95 = fmul float %7, %41
  %96 = fadd float %95, %94
  %97 = fmul float %6, %43
  %98 = fmul float %8, %41
  %99 = fadd float %98, %97
  %100 = insertelement <2 x float> undef, float %93, i32 0
  %101 = insertelement <2 x float> %100, float %96, i32 1
  %102 = fmul float %15, %32
  %103 = fmul float %15, %33
  %104 = fmul float %16, %32
  %105 = fadd float %104, %103
  %106 = fmul float %15, %34
  %107 = fmul float %17, %32
  %108 = fadd float %107, %106
  %109 = insertelement <2 x float> undef, float %102, i32 0
  %110 = insertelement <2 x float> %109, float %105, i32 1
  %111 = fsub <2 x float> %101, %110
  %112 = fsub float %99, %108
  %113 = extractelement <2 x float> %71, i32 0
  %114 = extractelement <2 x float> %91, i32 0
  %115 = extractelement <2 x float> %111, i32 0
  %116 = extractelement <2 x float> %71, i32 1
  %117 = extractelement <2 x float> %91, i32 1
  %118 = extractelement <2 x float> %111, i32 1
  %119 = bitcast i8* %result to float*
  store float %113, float* %119, align 4, !tbaa !5
  %120 = getelementptr inbounds i8* %result, i64 4
  %121 = bitcast i8* %120 to float*
  store float %114, float* %121, align 4, !tbaa !7
  %122 = getelementptr inbounds i8* %result, i64 8
  %123 = bitcast i8* %122 to float*
  store float %115, float* %123, align 4, !tbaa !8
  %124 = getelementptr inbounds i8* %result, i64 12
  %125 = bitcast i8* %124 to float*
  store float %116, float* %125, align 4, !tbaa !5
  %126 = getelementptr inbounds i8* %result, i64 16
  %127 = bitcast i8* %126 to float*
  store float %117, float* %127, align 4, !tbaa !7
  %128 = getelementptr inbounds i8* %result, i64 20
  %129 = bitcast i8* %128 to float*
  store float %118, float* %129, align 4, !tbaa !8
  %130 = getelementptr inbounds i8* %result, i64 24
  %131 = bitcast i8* %130 to float*
  store float %72, float* %131, align 4, !tbaa !5
  %132 = getelementptr inbounds i8* %result, i64 28
  %133 = bitcast i8* %132 to float*
  store float %92, float* %133, align 4, !tbaa !7
  %134 = getelementptr inbounds i8* %result, i64 32
  %135 = bitcast i8* %134 to float*
  store float %112, float* %135, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cross_dvdvv(i8* nocapture %result, i8* nocapture readonly %a, i8* nocapture readonly %b_) #4 {
  %1 = bitcast i8* %b_ to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = getelementptr inbounds i8* %b_, i64 4
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4, !tbaa !7
  %6 = getelementptr inbounds i8* %b_, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !8
  %9 = bitcast i8* %a to float*
  %10 = getelementptr inbounds i8* %a, i64 12
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8* %a, i64 24
  %13 = bitcast i8* %12 to float*
  %14 = load float* %9, align 4, !tbaa !1
  %15 = load float* %11, align 4, !tbaa !1
  %16 = load float* %13, align 4, !tbaa !1
  %17 = getelementptr inbounds i8* %a, i64 4
  %18 = bitcast i8* %17 to float*
  %19 = getelementptr inbounds i8* %a, i64 16
  %20 = bitcast i8* %19 to float*
  %21 = getelementptr inbounds i8* %a, i64 28
  %22 = bitcast i8* %21 to float*
  %23 = load float* %18, align 4, !tbaa !1
  %24 = load float* %20, align 4, !tbaa !1
  %25 = load float* %22, align 4, !tbaa !1
  %26 = getelementptr inbounds i8* %a, i64 8
  %27 = bitcast i8* %26 to float*
  %28 = getelementptr inbounds i8* %a, i64 20
  %29 = bitcast i8* %28 to float*
  %30 = getelementptr inbounds i8* %a, i64 32
  %31 = bitcast i8* %30 to float*
  %32 = load float* %27, align 4, !tbaa !1
  %33 = load float* %29, align 4, !tbaa !1
  %34 = load float* %31, align 4, !tbaa !1
  %35 = fmul float %8, %23
  %36 = fmul float %23, 0.000000e+00
  %37 = fmul float %8, %24
  %38 = fadd float %36, %37
  %39 = fmul float %8, %25
  %40 = fadd float %36, %39
  %41 = insertelement <2 x float> undef, float %35, i32 0
  %42 = insertelement <2 x float> %41, float %38, i32 1
  %43 = fmul float %5, %32
  %44 = fmul float %32, 0.000000e+00
  %45 = fmul float %5, %33
  %46 = fadd float %44, %45
  %47 = fmul float %5, %34
  %48 = fadd float %44, %47
  %49 = insertelement <2 x float> undef, float %43, i32 0
  %50 = insertelement <2 x float> %49, float %46, i32 1
  %51 = fsub <2 x float> %42, %50
  %52 = fsub float %40, %48
  %53 = fmul float %2, %32
  %54 = fmul float %2, %33
  %55 = fadd float %44, %54
  %56 = fmul float %2, %34
  %57 = fadd float %44, %56
  %58 = insertelement <2 x float> undef, float %53, i32 0
  %59 = insertelement <2 x float> %58, float %55, i32 1
  %60 = fmul float %8, %14
  %61 = fmul float %14, 0.000000e+00
  %62 = fmul float %8, %15
  %63 = fadd float %61, %62
  %64 = fmul float %8, %16
  %65 = fadd float %61, %64
  %66 = insertelement <2 x float> undef, float %60, i32 0
  %67 = insertelement <2 x float> %66, float %63, i32 1
  %68 = fsub <2 x float> %59, %67
  %69 = fsub float %57, %65
  %70 = fmul float %5, %14
  %71 = fmul float %5, %15
  %72 = fadd float %61, %71
  %73 = fmul float %5, %16
  %74 = fadd float %61, %73
  %75 = insertelement <2 x float> undef, float %70, i32 0
  %76 = insertelement <2 x float> %75, float %72, i32 1
  %77 = fmul float %2, %23
  %78 = fmul float %2, %24
  %79 = fadd float %36, %78
  %80 = fmul float %2, %25
  %81 = fadd float %36, %80
  %82 = insertelement <2 x float> undef, float %77, i32 0
  %83 = insertelement <2 x float> %82, float %79, i32 1
  %84 = fsub <2 x float> %76, %83
  %85 = fsub float %74, %81
  %86 = extractelement <2 x float> %51, i32 0
  %87 = extractelement <2 x float> %68, i32 0
  %88 = extractelement <2 x float> %84, i32 0
  %89 = extractelement <2 x float> %51, i32 1
  %90 = extractelement <2 x float> %68, i32 1
  %91 = extractelement <2 x float> %84, i32 1
  %92 = bitcast i8* %result to float*
  store float %86, float* %92, align 4, !tbaa !5
  %93 = getelementptr inbounds i8* %result, i64 4
  %94 = bitcast i8* %93 to float*
  store float %87, float* %94, align 4, !tbaa !7
  %95 = getelementptr inbounds i8* %result, i64 8
  %96 = bitcast i8* %95 to float*
  store float %88, float* %96, align 4, !tbaa !8
  %97 = getelementptr inbounds i8* %result, i64 12
  %98 = bitcast i8* %97 to float*
  store float %89, float* %98, align 4, !tbaa !5
  %99 = getelementptr inbounds i8* %result, i64 16
  %100 = bitcast i8* %99 to float*
  store float %90, float* %100, align 4, !tbaa !7
  %101 = getelementptr inbounds i8* %result, i64 20
  %102 = bitcast i8* %101 to float*
  store float %91, float* %102, align 4, !tbaa !8
  %103 = getelementptr inbounds i8* %result, i64 24
  %104 = bitcast i8* %103 to float*
  store float %52, float* %104, align 4, !tbaa !5
  %105 = getelementptr inbounds i8* %result, i64 28
  %106 = bitcast i8* %105 to float*
  store float %69, float* %106, align 4, !tbaa !7
  %107 = getelementptr inbounds i8* %result, i64 32
  %108 = bitcast i8* %107 to float*
  store float %85, float* %108, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cross_dvvdv(i8* nocapture %result, i8* nocapture readonly %a_, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = getelementptr inbounds i8* %a_, i64 4
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4, !tbaa !7
  %6 = getelementptr inbounds i8* %a_, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !8
  %9 = bitcast i8* %b to float*
  %10 = getelementptr inbounds i8* %b, i64 12
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8* %b, i64 24
  %13 = bitcast i8* %12 to float*
  %14 = load float* %9, align 4, !tbaa !1
  %15 = load float* %11, align 4, !tbaa !1
  %16 = load float* %13, align 4, !tbaa !1
  %17 = getelementptr inbounds i8* %b, i64 4
  %18 = bitcast i8* %17 to float*
  %19 = getelementptr inbounds i8* %b, i64 16
  %20 = bitcast i8* %19 to float*
  %21 = getelementptr inbounds i8* %b, i64 28
  %22 = bitcast i8* %21 to float*
  %23 = load float* %18, align 4, !tbaa !1
  %24 = load float* %20, align 4, !tbaa !1
  %25 = load float* %22, align 4, !tbaa !1
  %26 = getelementptr inbounds i8* %b, i64 8
  %27 = bitcast i8* %26 to float*
  %28 = getelementptr inbounds i8* %b, i64 20
  %29 = bitcast i8* %28 to float*
  %30 = getelementptr inbounds i8* %b, i64 32
  %31 = bitcast i8* %30 to float*
  %32 = load float* %27, align 4, !tbaa !1
  %33 = load float* %29, align 4, !tbaa !1
  %34 = load float* %31, align 4, !tbaa !1
  %35 = fmul float %5, %32
  %36 = fmul float %5, %33
  %37 = fmul float %32, 0.000000e+00
  %38 = fadd float %37, %36
  %39 = fmul float %5, %34
  %40 = fadd float %37, %39
  %41 = insertelement <2 x float> undef, float %35, i32 0
  %42 = insertelement <2 x float> %41, float %38, i32 1
  %43 = fmul float %8, %23
  %44 = fmul float %8, %24
  %45 = fmul float %23, 0.000000e+00
  %46 = fadd float %45, %44
  %47 = fmul float %8, %25
  %48 = fadd float %45, %47
  %49 = insertelement <2 x float> undef, float %43, i32 0
  %50 = insertelement <2 x float> %49, float %46, i32 1
  %51 = fsub <2 x float> %42, %50
  %52 = fsub float %40, %48
  %53 = fmul float %8, %14
  %54 = fmul float %8, %15
  %55 = fmul float %14, 0.000000e+00
  %56 = fadd float %55, %54
  %57 = fmul float %8, %16
  %58 = fadd float %55, %57
  %59 = insertelement <2 x float> undef, float %53, i32 0
  %60 = insertelement <2 x float> %59, float %56, i32 1
  %61 = fmul float %2, %32
  %62 = fmul float %2, %33
  %63 = fadd float %37, %62
  %64 = fmul float %2, %34
  %65 = fadd float %37, %64
  %66 = insertelement <2 x float> undef, float %61, i32 0
  %67 = insertelement <2 x float> %66, float %63, i32 1
  %68 = fsub <2 x float> %60, %67
  %69 = fsub float %58, %65
  %70 = fmul float %2, %23
  %71 = fmul float %2, %24
  %72 = fadd float %45, %71
  %73 = fmul float %2, %25
  %74 = fadd float %45, %73
  %75 = insertelement <2 x float> undef, float %70, i32 0
  %76 = insertelement <2 x float> %75, float %72, i32 1
  %77 = fmul float %5, %14
  %78 = fmul float %5, %15
  %79 = fadd float %55, %78
  %80 = fmul float %5, %16
  %81 = fadd float %55, %80
  %82 = insertelement <2 x float> undef, float %77, i32 0
  %83 = insertelement <2 x float> %82, float %79, i32 1
  %84 = fsub <2 x float> %76, %83
  %85 = fsub float %74, %81
  %86 = extractelement <2 x float> %51, i32 0
  %87 = extractelement <2 x float> %68, i32 0
  %88 = extractelement <2 x float> %84, i32 0
  %89 = extractelement <2 x float> %51, i32 1
  %90 = extractelement <2 x float> %68, i32 1
  %91 = extractelement <2 x float> %84, i32 1
  %92 = bitcast i8* %result to float*
  store float %86, float* %92, align 4, !tbaa !5
  %93 = getelementptr inbounds i8* %result, i64 4
  %94 = bitcast i8* %93 to float*
  store float %87, float* %94, align 4, !tbaa !7
  %95 = getelementptr inbounds i8* %result, i64 8
  %96 = bitcast i8* %95 to float*
  store float %88, float* %96, align 4, !tbaa !8
  %97 = getelementptr inbounds i8* %result, i64 12
  %98 = bitcast i8* %97 to float*
  store float %89, float* %98, align 4, !tbaa !5
  %99 = getelementptr inbounds i8* %result, i64 16
  %100 = bitcast i8* %99 to float*
  store float %90, float* %100, align 4, !tbaa !7
  %101 = getelementptr inbounds i8* %result, i64 20
  %102 = bitcast i8* %101 to float*
  store float %91, float* %102, align 4, !tbaa !8
  %103 = getelementptr inbounds i8* %result, i64 24
  %104 = bitcast i8* %103 to float*
  store float %52, float* %104, align 4, !tbaa !5
  %105 = getelementptr inbounds i8* %result, i64 28
  %106 = bitcast i8* %105 to float*
  store float %69, float* %106, align 4, !tbaa !7
  %107 = getelementptr inbounds i8* %result, i64 32
  %108 = bitcast i8* %107 to float*
  store float %85, float* %108, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readonly uwtable
define float @osl_length_fv(i8* nocapture readonly %a) #10 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = fmul float %2, %2
  %4 = getelementptr inbounds i8* %a, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !7
  %7 = fmul float %6, %6
  %8 = fadd float %3, %7
  %9 = getelementptr inbounds i8* %a, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !8
  %12 = fmul float %11, %11
  %13 = fadd float %8, %12
  %14 = fcmp olt float %13, 0x3820000000000000
  br i1 %14, label %15, label %45

; <label>:15                                      ; preds = %0
  %16 = fcmp ult float %2, 0.000000e+00
  br i1 %16, label %17, label %19

; <label>:17                                      ; preds = %15
  %18 = fsub float -0.000000e+00, %2
  br label %19

; <label>:19                                      ; preds = %17, %15
  %20 = phi float [ %18, %17 ], [ %2, %15 ]
  %21 = fcmp ult float %6, 0.000000e+00
  br i1 %21, label %22, label %24

; <label>:22                                      ; preds = %19
  %23 = fsub float -0.000000e+00, %6
  br label %24

; <label>:24                                      ; preds = %22, %19
  %25 = phi float [ %23, %22 ], [ %6, %19 ]
  %26 = fcmp ult float %11, 0.000000e+00
  br i1 %26, label %27, label %29

; <label>:27                                      ; preds = %24
  %28 = fsub float -0.000000e+00, %11
  br label %29

; <label>:29                                      ; preds = %27, %24
  %30 = phi float [ %28, %27 ], [ %11, %24 ]
  %31 = fcmp olt float %20, %25
  %max.0.i.i = select i1 %31, float %25, float %20
  %32 = fcmp olt float %max.0.i.i, %30
  %max.1.i.i = select i1 %32, float %30, float %max.0.i.i
  %33 = fcmp oeq float %max.1.i.i, 0.000000e+00
  br i1 %33, label %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit, label %34

; <label>:34                                      ; preds = %29
  %35 = fdiv float %20, %max.1.i.i
  %36 = fdiv float %25, %max.1.i.i
  %37 = fdiv float %30, %max.1.i.i
  %38 = fmul float %35, %35
  %39 = fmul float %36, %36
  %40 = fadd float %38, %39
  %41 = fmul float %37, %37
  %42 = fadd float %40, %41
  %43 = tail call float @sqrtf(float %42) #12
  %44 = fmul float %max.1.i.i, %43
  br label %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit

; <label>:45                                      ; preds = %0
  %46 = tail call float @sqrtf(float %13) #12
  br label %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit

_ZNK9Imath_2_24Vec3IfE6lengthEv.exit:             ; preds = %29, %34, %45
  %.0.i = phi float [ %46, %45 ], [ %44, %34 ], [ 0.000000e+00, %29 ]
  ret float %.0.i
}

; Function Attrs: nounwind uwtable
define void @osl_length_dfdv(i8* nocapture %result, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = getelementptr inbounds i8* %a, i64 12
  %3 = bitcast i8* %2 to float*
  %4 = getelementptr inbounds i8* %a, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %1, align 4, !tbaa !1
  %7 = load float* %3, align 4, !tbaa !1
  %8 = load float* %5, align 4, !tbaa !1
  %9 = getelementptr inbounds i8* %a, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = getelementptr inbounds i8* %a, i64 16
  %12 = bitcast i8* %11 to float*
  %13 = getelementptr inbounds i8* %a, i64 28
  %14 = bitcast i8* %13 to float*
  %15 = load float* %10, align 4, !tbaa !1
  %16 = load float* %12, align 4, !tbaa !1
  %17 = load float* %14, align 4, !tbaa !1
  %18 = getelementptr inbounds i8* %a, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = getelementptr inbounds i8* %a, i64 20
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8* %a, i64 32
  %23 = bitcast i8* %22 to float*
  %24 = load float* %19, align 4, !tbaa !1
  %25 = load float* %21, align 4, !tbaa !1
  %26 = load float* %23, align 4, !tbaa !1
  %27 = fmul float %6, %6
  %28 = fmul float %6, %7
  %29 = fadd float %28, %28
  %30 = fmul float %6, %8
  %31 = insertelement <2 x float> undef, float %27, i32 0
  %32 = insertelement <2 x float> %31, float %29, i32 1
  %33 = fmul float %15, %15
  %34 = fmul float %15, %16
  %35 = fadd float %34, %34
  %36 = fmul float %15, %17
  %37 = insertelement <2 x float> undef, float %33, i32 0
  %38 = insertelement <2 x float> %37, float %35, i32 1
  %39 = fadd <2 x float> %32, %38
  %40 = fmul float %24, %24
  %41 = fmul float %24, %25
  %42 = fadd float %41, %41
  %43 = fmul float %24, %26
  %44 = insertelement <2 x float> undef, float %40, i32 0
  %45 = insertelement <2 x float> %44, float %42, i32 1
  %46 = fadd <2 x float> %39, %45
  %47 = extractelement <2 x float> %46, i32 0
  %48 = fcmp ugt float %47, 0.000000e+00
  br i1 %48, label %49, label %_ZN3OSL6lengthERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE.exit

; <label>:49                                      ; preds = %0
  %50 = fadd float %30, %30
  %51 = fadd float %36, %36
  %52 = fadd float %50, %51
  %53 = fadd float %43, %43
  %54 = fadd float %52, %53
  %55 = tail call float @sqrtf(float %47) #12
  %56 = fmul float %55, 2.000000e+00
  %57 = fdiv float 1.000000e+00, %56
  %58 = extractelement <2 x float> %46, i32 1
  %59 = fmul float %58, %57
  %60 = fmul float %54, %57
  %61 = insertelement <2 x float> undef, float %55, i32 0
  %62 = insertelement <2 x float> %61, float %59, i32 1
  br label %_ZN3OSL6lengthERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE.exit

_ZN3OSL6lengthERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE.exit: ; preds = %0, %49
  %63 = phi float [ %60, %49 ], [ 0.000000e+00, %0 ]
  %64 = phi <2 x float> [ %62, %49 ], [ zeroinitializer, %0 ]
  %65 = bitcast i8* %result to <2 x float>*
  store <2 x float> %64, <2 x float>* %65, align 4
  %66 = getelementptr inbounds i8* %result, i64 8
  %67 = bitcast i8* %66 to float*
  store float %63, float* %67, align 4
  ret void
}

; Function Attrs: nounwind readonly uwtable
define float @osl_distance_fvv(i8* nocapture readonly %a_, i8* nocapture readonly %b_) #10 {
  %1 = bitcast i8* %a_ to float*
  %2 = load float* %1, align 4, !tbaa !1
  %3 = bitcast i8* %b_ to float*
  %4 = load float* %3, align 4, !tbaa !1
  %5 = fsub float %2, %4
  %6 = getelementptr inbounds i8* %a_, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !1
  %9 = getelementptr inbounds i8* %b_, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !1
  %12 = fsub float %8, %11
  %13 = getelementptr inbounds i8* %a_, i64 8
  %14 = bitcast i8* %13 to float*
  %15 = load float* %14, align 4, !tbaa !1
  %16 = getelementptr inbounds i8* %b_, i64 8
  %17 = bitcast i8* %16 to float*
  %18 = load float* %17, align 4, !tbaa !1
  %19 = fsub float %15, %18
  %20 = fmul float %5, %5
  %21 = fmul float %12, %12
  %22 = fadd float %20, %21
  %23 = fmul float %19, %19
  %24 = fadd float %22, %23
  %25 = tail call float @sqrtf(float %24) #12
  ret float %25
}

; Function Attrs: nounwind readnone
declare float @sqrtf(float) #7

; Function Attrs: nounwind uwtable
define void @osl_distance_dfdvdv(i8* nocapture %result, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %a to <4 x float>*
  %2 = load <4 x float>* %1, align 4, !tbaa !1
  %3 = bitcast i8* %b to <4 x float>*
  %4 = load <4 x float>* %3, align 4, !tbaa !1
  %5 = fsub <4 x float> %2, %4
  %6 = getelementptr inbounds i8* %a, i64 16
  %7 = getelementptr inbounds i8* %b, i64 16
  %8 = bitcast i8* %6 to <4 x float>*
  %9 = load <4 x float>* %8, align 4, !tbaa !1
  %10 = bitcast i8* %7 to <4 x float>*
  %11 = load <4 x float>* %10, align 4, !tbaa !1
  %12 = fsub <4 x float> %9, %11
  %13 = getelementptr inbounds i8* %a, i64 32
  %14 = bitcast i8* %13 to float*
  %15 = load float* %14, align 4, !tbaa !8
  %16 = getelementptr inbounds i8* %b, i64 32
  %17 = bitcast i8* %16 to float*
  %18 = load float* %17, align 4, !tbaa !8
  %19 = fsub float %15, %18
  %20 = extractelement <4 x float> %5, i32 0
  %21 = extractelement <4 x float> %5, i32 3
  %22 = extractelement <4 x float> %12, i32 2
  %23 = extractelement <4 x float> %5, i32 1
  %24 = extractelement <4 x float> %12, i32 0
  %25 = extractelement <4 x float> %12, i32 3
  %26 = extractelement <4 x float> %5, i32 2
  %27 = extractelement <4 x float> %12, i32 1
  %28 = fmul float %20, %20
  %29 = fmul float %20, %21
  %30 = fadd float %29, %29
  %31 = fmul float %20, %22
  %32 = insertelement <2 x float> undef, float %28, i32 0
  %33 = insertelement <2 x float> %32, float %30, i32 1
  %34 = fmul float %23, %23
  %35 = fmul float %23, %24
  %36 = fadd float %35, %35
  %37 = fmul float %23, %25
  %38 = insertelement <2 x float> undef, float %34, i32 0
  %39 = insertelement <2 x float> %38, float %36, i32 1
  %40 = fadd <2 x float> %33, %39
  %41 = fmul float %26, %26
  %42 = fmul float %26, %27
  %43 = fadd float %42, %42
  %44 = fmul float %26, %19
  %45 = insertelement <2 x float> undef, float %41, i32 0
  %46 = insertelement <2 x float> %45, float %43, i32 1
  %47 = fadd <2 x float> %46, %40
  %48 = extractelement <2 x float> %47, i32 0
  %49 = fcmp ugt float %48, 0.000000e+00
  br i1 %49, label %50, label %_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit

; <label>:50                                      ; preds = %0
  %51 = fadd float %31, %31
  %52 = fadd float %37, %37
  %53 = fadd float %51, %52
  %54 = fadd float %44, %44
  %55 = fadd float %53, %54
  %56 = tail call float @sqrtf(float %48) #12
  %57 = fmul float %56, 2.000000e+00
  %58 = fdiv float 1.000000e+00, %57
  %59 = extractelement <2 x float> %47, i32 1
  %60 = fmul float %59, %58
  %61 = fmul float %55, %58
  %62 = insertelement <2 x float> undef, float %56, i32 0
  %63 = insertelement <2 x float> %62, float %60, i32 1
  br label %_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit

_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit: ; preds = %0, %50
  %64 = phi float [ %61, %50 ], [ 0.000000e+00, %0 ]
  %65 = phi <2 x float> [ %63, %50 ], [ zeroinitializer, %0 ]
  %66 = bitcast i8* %result to <2 x float>*
  store <2 x float> %65, <2 x float>* %66, align 4
  %67 = getelementptr inbounds i8* %result, i64 8
  %68 = bitcast i8* %67 to float*
  store float %64, float* %68, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_distance_dfdvv(i8* nocapture %result, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %b to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = getelementptr inbounds i8* %b, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !7
  %7 = insertelement <4 x float> %3, float %6, i32 1
  %8 = getelementptr inbounds i8* %b, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !8
  %11 = insertelement <4 x float> %7, float %10, i32 2
  %12 = insertelement <4 x float> %11, float 0.000000e+00, i32 3
  %13 = bitcast i8* %a to <4 x float>*
  %14 = load <4 x float>* %13, align 4, !tbaa !1
  %15 = fsub <4 x float> %14, %12
  %16 = getelementptr inbounds i8* %a, i64 16
  %17 = bitcast i8* %16 to <4 x float>*
  %18 = load <4 x float>* %17, align 4, !tbaa !1
  %19 = getelementptr inbounds i8* %a, i64 32
  %20 = bitcast i8* %19 to float*
  %21 = load float* %20, align 4, !tbaa !8
  %22 = extractelement <4 x float> %15, i32 0
  %23 = extractelement <4 x float> %15, i32 3
  %24 = extractelement <4 x float> %18, i32 2
  %25 = extractelement <4 x float> %15, i32 1
  %26 = extractelement <4 x float> %18, i32 0
  %27 = extractelement <4 x float> %18, i32 3
  %28 = extractelement <4 x float> %15, i32 2
  %29 = extractelement <4 x float> %18, i32 1
  %30 = fmul float %22, %22
  %31 = fmul float %22, %23
  %32 = fadd float %31, %31
  %33 = fmul float %24, %22
  %34 = insertelement <2 x float> undef, float %30, i32 0
  %35 = insertelement <2 x float> %34, float %32, i32 1
  %36 = fmul float %25, %25
  %37 = fmul float %26, %25
  %38 = fadd float %37, %37
  %39 = fmul float %27, %25
  %40 = insertelement <2 x float> undef, float %36, i32 0
  %41 = insertelement <2 x float> %40, float %38, i32 1
  %42 = fadd <2 x float> %35, %41
  %43 = fmul float %28, %28
  %44 = fmul float %29, %28
  %45 = fadd float %44, %44
  %46 = fmul float %21, %28
  %47 = insertelement <2 x float> undef, float %43, i32 0
  %48 = insertelement <2 x float> %47, float %45, i32 1
  %49 = fadd <2 x float> %48, %42
  %50 = extractelement <2 x float> %49, i32 0
  %51 = fcmp ugt float %50, 0.000000e+00
  br i1 %51, label %52, label %_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit

; <label>:52                                      ; preds = %0
  %53 = fadd float %33, %33
  %54 = fadd float %39, %39
  %55 = fadd float %53, %54
  %56 = fadd float %46, %46
  %57 = fadd float %56, %55
  %58 = tail call float @sqrtf(float %50) #12
  %59 = fmul float %58, 2.000000e+00
  %60 = fdiv float 1.000000e+00, %59
  %61 = extractelement <2 x float> %49, i32 1
  %62 = fmul float %61, %60
  %63 = fmul float %57, %60
  %64 = insertelement <2 x float> undef, float %58, i32 0
  %65 = insertelement <2 x float> %64, float %62, i32 1
  br label %_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit

_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit: ; preds = %0, %52
  %66 = phi float [ %63, %52 ], [ 0.000000e+00, %0 ]
  %67 = phi <2 x float> [ %65, %52 ], [ zeroinitializer, %0 ]
  %68 = bitcast i8* %result to <2 x float>*
  store <2 x float> %67, <2 x float>* %68, align 4
  %69 = getelementptr inbounds i8* %result, i64 8
  %70 = bitcast i8* %69 to float*
  store float %66, float* %70, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_distance_dfvdv(i8* nocapture %result, i8* nocapture readonly %a, i8* nocapture readonly %b) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = insertelement <4 x float> undef, float %2, i32 0
  %4 = getelementptr inbounds i8* %a, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !7
  %7 = insertelement <4 x float> %3, float %6, i32 1
  %8 = getelementptr inbounds i8* %a, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !8
  %11 = insertelement <4 x float> %7, float %10, i32 2
  %12 = insertelement <4 x float> %11, float 0.000000e+00, i32 3
  %13 = bitcast i8* %b to <4 x float>*
  %14 = load <4 x float>* %13, align 4, !tbaa !1
  %15 = fsub <4 x float> %12, %14
  %16 = getelementptr inbounds i8* %b, i64 16
  %17 = bitcast i8* %16 to <4 x float>*
  %18 = load <4 x float>* %17, align 4, !tbaa !1
  %19 = fsub <4 x float> zeroinitializer, %18
  %20 = getelementptr inbounds i8* %b, i64 32
  %21 = bitcast i8* %20 to float*
  %22 = load float* %21, align 4, !tbaa !8
  %23 = fsub float 0.000000e+00, %22
  %24 = extractelement <4 x float> %15, i32 0
  %25 = extractelement <4 x float> %15, i32 3
  %26 = extractelement <4 x float> %19, i32 2
  %27 = extractelement <4 x float> %15, i32 1
  %28 = extractelement <4 x float> %19, i32 0
  %29 = extractelement <4 x float> %19, i32 3
  %30 = extractelement <4 x float> %15, i32 2
  %31 = extractelement <4 x float> %19, i32 1
  %32 = fmul float %24, %24
  %33 = fmul float %24, %25
  %34 = fadd float %33, %33
  %35 = fmul float %24, %26
  %36 = insertelement <2 x float> undef, float %32, i32 0
  %37 = insertelement <2 x float> %36, float %34, i32 1
  %38 = fmul float %27, %27
  %39 = fmul float %27, %28
  %40 = fadd float %39, %39
  %41 = fmul float %27, %29
  %42 = insertelement <2 x float> undef, float %38, i32 0
  %43 = insertelement <2 x float> %42, float %40, i32 1
  %44 = fadd <2 x float> %37, %43
  %45 = fmul float %30, %30
  %46 = fmul float %30, %31
  %47 = fadd float %46, %46
  %48 = fmul float %30, %23
  %49 = insertelement <2 x float> undef, float %45, i32 0
  %50 = insertelement <2 x float> %49, float %47, i32 1
  %51 = fadd <2 x float> %50, %44
  %52 = extractelement <2 x float> %51, i32 0
  %53 = fcmp ugt float %52, 0.000000e+00
  br i1 %53, label %54, label %_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit

; <label>:54                                      ; preds = %0
  %55 = fadd float %35, %35
  %56 = fadd float %41, %41
  %57 = fadd float %55, %56
  %58 = fadd float %48, %48
  %59 = fadd float %58, %57
  %60 = tail call float @sqrtf(float %52) #12
  %61 = fmul float %60, 2.000000e+00
  %62 = fdiv float 1.000000e+00, %61
  %63 = extractelement <2 x float> %51, i32 1
  %64 = fmul float %63, %62
  %65 = fmul float %59, %62
  %66 = insertelement <2 x float> undef, float %60, i32 0
  %67 = insertelement <2 x float> %66, float %64, i32 1
  br label %_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit

_ZN3OSL8distanceERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_.exit: ; preds = %0, %54
  %68 = phi float [ %65, %54 ], [ 0.000000e+00, %0 ]
  %69 = phi <2 x float> [ %67, %54 ], [ zeroinitializer, %0 ]
  %70 = bitcast i8* %result to <2 x float>*
  store <2 x float> %69, <2 x float>* %70, align 4
  %71 = getelementptr inbounds i8* %result, i64 8
  %72 = bitcast i8* %71 to float*
  store float %68, float* %72, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_normalize_vv(i8* nocapture %result, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = fmul float %2, %2
  %4 = getelementptr inbounds i8* %a, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !7
  %7 = fmul float %6, %6
  %8 = fadd float %3, %7
  %9 = getelementptr inbounds i8* %a, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float* %10, align 4, !tbaa !8
  %12 = fmul float %11, %11
  %13 = fadd float %8, %12
  %14 = fcmp olt float %13, 0x3820000000000000
  br i1 %14, label %15, label %45

; <label>:15                                      ; preds = %0
  %16 = fcmp ult float %2, 0.000000e+00
  br i1 %16, label %17, label %19

; <label>:17                                      ; preds = %15
  %18 = fsub float -0.000000e+00, %2
  br label %19

; <label>:19                                      ; preds = %17, %15
  %20 = phi float [ %18, %17 ], [ %2, %15 ]
  %21 = fcmp ult float %6, 0.000000e+00
  br i1 %21, label %22, label %24

; <label>:22                                      ; preds = %19
  %23 = fsub float -0.000000e+00, %6
  br label %24

; <label>:24                                      ; preds = %22, %19
  %25 = phi float [ %23, %22 ], [ %6, %19 ]
  %26 = fcmp ult float %11, 0.000000e+00
  br i1 %26, label %27, label %29

; <label>:27                                      ; preds = %24
  %28 = fsub float -0.000000e+00, %11
  br label %29

; <label>:29                                      ; preds = %27, %24
  %30 = phi float [ %28, %27 ], [ %11, %24 ]
  %31 = fcmp olt float %20, %25
  %max.0.i.i.i = select i1 %31, float %25, float %20
  %32 = fcmp olt float %max.0.i.i.i, %30
  %max.1.i.i.i = select i1 %32, float %30, float %max.0.i.i.i
  %33 = fcmp oeq float %max.1.i.i.i, 0.000000e+00
  br i1 %33, label %_ZNK9Imath_2_24Vec3IfE10normalizedEv.exit, label %34

; <label>:34                                      ; preds = %29
  %35 = fdiv float %20, %max.1.i.i.i
  %36 = fdiv float %25, %max.1.i.i.i
  %37 = fdiv float %30, %max.1.i.i.i
  %38 = fmul float %35, %35
  %39 = fmul float %36, %36
  %40 = fadd float %38, %39
  %41 = fmul float %37, %37
  %42 = fadd float %40, %41
  %43 = tail call float @sqrtf(float %42) #12
  %44 = fmul float %max.1.i.i.i, %43
  br label %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit.i

; <label>:45                                      ; preds = %0
  %46 = tail call float @sqrtf(float %13) #12
  br label %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit.i

_ZNK9Imath_2_24Vec3IfE6lengthEv.exit.i:           ; preds = %45, %34
  %.0.i.i = phi float [ %46, %45 ], [ %44, %34 ]
  %47 = fcmp oeq float %.0.i.i, 0.000000e+00
  br i1 %47, label %_ZNK9Imath_2_24Vec3IfE10normalizedEv.exit, label %48

; <label>:48                                      ; preds = %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit.i
  %49 = fdiv float %2, %.0.i.i
  %50 = fdiv float %6, %.0.i.i
  %51 = fdiv float %11, %.0.i.i
  br label %_ZNK9Imath_2_24Vec3IfE10normalizedEv.exit

_ZNK9Imath_2_24Vec3IfE10normalizedEv.exit:        ; preds = %29, %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit.i, %48
  %52 = phi float [ %51, %48 ], [ 0.000000e+00, %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit.i ], [ 0.000000e+00, %29 ]
  %53 = phi float [ %50, %48 ], [ 0.000000e+00, %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit.i ], [ 0.000000e+00, %29 ]
  %54 = phi float [ %49, %48 ], [ 0.000000e+00, %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit.i ], [ 0.000000e+00, %29 ]
  %55 = bitcast i8* %result to float*
  store float %54, float* %55, align 4, !tbaa !5
  %56 = getelementptr inbounds i8* %result, i64 4
  %57 = bitcast i8* %56 to float*
  store float %53, float* %57, align 4, !tbaa !7
  %58 = getelementptr inbounds i8* %result, i64 8
  %59 = bitcast i8* %58 to float*
  store float %52, float* %59, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_normalize_dvdv(i8* nocapture %result, i8* nocapture readonly %a) #4 {
  %1 = bitcast i8* %a to float*
  %2 = load float* %1, align 4, !tbaa !5
  %3 = fcmp oeq float %2, 0.000000e+00
  %4 = getelementptr inbounds i8* %a, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !7
  %7 = fcmp oeq float %6, 0.000000e+00
  %or.cond.i = and i1 %3, %7
  %8 = getelementptr inbounds i8* %a, i64 8
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !8
  %11 = fcmp oeq float %10, 0.000000e+00
  %or.cond = and i1 %or.cond.i, %11
  br i1 %or.cond, label %_ZN3OSL9normalizeERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE.exit, label %._crit_edge.i

._crit_edge.i:                                    ; preds = %0
  %12 = getelementptr inbounds i8* %a, i64 12
  %13 = getelementptr inbounds i8* %a, i64 28
  %14 = bitcast i8* %13 to float*
  %15 = load float* %14, align 4, !tbaa !1
  %16 = getelementptr inbounds i8* %a, i64 32
  %17 = bitcast i8* %16 to float*
  %18 = bitcast i8* %12 to <4 x float>*
  %19 = load <4 x float>* %18, align 4, !tbaa !1
  %20 = load float* %17, align 4, !tbaa !1
  %21 = fmul float %2, %2
  %22 = extractelement <4 x float> %19, i32 0
  %23 = fmul float %2, %22
  %24 = fadd float %23, %23
  %25 = extractelement <4 x float> %19, i32 3
  %26 = fmul float %2, %25
  %27 = insertelement <2 x float> undef, float %21, i32 0
  %28 = insertelement <2 x float> %27, float %24, i32 1
  %29 = fmul float %6, %6
  %30 = extractelement <4 x float> %19, i32 1
  %31 = fmul float %6, %30
  %32 = fadd float %31, %31
  %33 = fmul float %6, %15
  %34 = insertelement <2 x float> undef, float %29, i32 0
  %35 = insertelement <2 x float> %34, float %32, i32 1
  %36 = fadd <2 x float> %28, %35
  %37 = fmul float %10, %10
  %38 = extractelement <4 x float> %19, i32 2
  %39 = fmul float %10, %38
  %40 = fadd float %39, %39
  %41 = fmul float %10, %20
  %42 = insertelement <2 x float> undef, float %37, i32 0
  %43 = insertelement <2 x float> %42, float %40, i32 1
  %44 = fadd <2 x float> %43, %36
  %45 = extractelement <2 x float> %44, i32 0
  %46 = fcmp ugt float %45, 0.000000e+00
  br i1 %46, label %47, label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i

; <label>:47                                      ; preds = %._crit_edge.i
  %48 = fadd float %26, %26
  %49 = fadd float %33, %33
  %50 = fadd float %49, %48
  %51 = fadd float %41, %41
  %52 = fadd float %51, %50
  %53 = tail call float @sqrtf(float %45) #12
  %54 = fmul float %53, 2.000000e+00
  %55 = fdiv float 1.000000e+00, %54
  %56 = extractelement <2 x float> %44, i32 1
  %57 = fmul float %56, %55
  %58 = fmul float %52, %55
  %59 = insertelement <2 x float> undef, float %53, i32 0
  %60 = insertelement <2 x float> %59, float %57, i32 1
  br label %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i

_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i:       ; preds = %47, %._crit_edge.i
  %61 = phi float [ %58, %47 ], [ 0.000000e+00, %._crit_edge.i ]
  %62 = phi <2 x float> [ %60, %47 ], [ zeroinitializer, %._crit_edge.i ]
  %63 = extractelement <2 x float> %62, i32 0
  %64 = fdiv float 1.000000e+00, %63
  %65 = extractelement <2 x float> %62, i32 1
  %66 = fmul float %64, %65
  %67 = fmul float %64, %66
  %68 = fsub float -0.000000e+00, %67
  %69 = fmul float %61, %64
  %70 = fmul float %64, %69
  %71 = fsub float -0.000000e+00, %70
  %72 = fmul float %2, %64
  %73 = fmul float %6, %64
  %74 = fmul float %6, %71
  %75 = fmul float %15, %64
  %76 = fadd float %75, %74
  %77 = fmul float %10, %64
  %78 = insertelement <4 x float> undef, float %2, i32 0
  %79 = insertelement <4 x float> %78, float %6, i32 1
  %80 = insertelement <4 x float> %79, float %10, i32 2
  %81 = insertelement <4 x float> %80, float %2, i32 3
  %82 = insertelement <4 x float> undef, float %68, i32 0
  %83 = insertelement <4 x float> %82, float %68, i32 1
  %84 = insertelement <4 x float> %83, float %68, i32 2
  %85 = insertelement <4 x float> %84, float %71, i32 3
  %86 = fmul <4 x float> %81, %85
  %87 = insertelement <4 x float> undef, float %64, i32 0
  %88 = insertelement <4 x float> %87, float %64, i32 1
  %89 = insertelement <4 x float> %88, float %64, i32 2
  %90 = insertelement <4 x float> %89, float %64, i32 3
  %91 = fmul <4 x float> %19, %90
  %92 = fadd <4 x float> %91, %86
  %93 = fmul float %10, %71
  %94 = fmul float %20, %64
  %95 = fadd float %94, %93
  br label %_ZN3OSL9normalizeERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE.exit

_ZN3OSL9normalizeERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE.exit: ; preds = %0, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i
  %96 = phi float [ %95, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i ], [ 0.000000e+00, %0 ]
  %97 = phi float [ %76, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i ], [ 0.000000e+00, %0 ]
  %98 = phi <4 x float> [ %92, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i ], [ zeroinitializer, %0 ]
  %99 = phi float [ %77, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i ], [ 0.000000e+00, %0 ]
  %100 = phi float [ %73, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i ], [ 0.000000e+00, %0 ]
  %101 = phi float [ %72, %_ZN3OSL4sqrtIfEENS_5Dual2IT_EERKS3_.exit.i ], [ 0.000000e+00, %0 ]
  %102 = bitcast i8* %result to float*
  store float %101, float* %102, align 4, !tbaa !5
  %103 = getelementptr inbounds i8* %result, i64 4
  %104 = bitcast i8* %103 to float*
  store float %100, float* %104, align 4, !tbaa !7
  %105 = getelementptr inbounds i8* %result, i64 8
  %106 = bitcast i8* %105 to float*
  store float %99, float* %106, align 4, !tbaa !8
  %107 = getelementptr inbounds i8* %result, i64 12
  %108 = bitcast i8* %107 to <4 x float>*
  store <4 x float> %98, <4 x float>* %108, align 4, !tbaa !1
  %109 = getelementptr inbounds i8* %result, i64 28
  %110 = bitcast i8* %109 to float*
  store float %97, float* %110, align 4, !tbaa !7
  %111 = getelementptr inbounds i8* %result, i64 32
  %112 = bitcast i8* %111 to float*
  store float %96, float* %112, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_calculatenormal(i8* nocapture %out, i8* nocapture readonly %sg_, i8* nocapture readonly %P_) #4 {
  %1 = getelementptr inbounds i8* %sg_, i64 288
  %2 = bitcast i8* %1 to i32*
  %3 = load i32* %2, align 4, !tbaa !15
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %33, label %5

; <label>:5                                       ; preds = %0
  %6 = getelementptr inbounds i8* %P_, i64 24
  %7 = getelementptr inbounds i8* %P_, i64 12
  %8 = getelementptr inbounds i8* %P_, i64 28
  %9 = bitcast i8* %8 to float*
  %10 = load float* %9, align 4, !tbaa !7
  %11 = getelementptr inbounds i8* %P_, i64 20
  %12 = bitcast i8* %11 to float*
  %13 = load float* %12, align 4, !tbaa !8
  %14 = fmul float %10, %13
  %15 = getelementptr inbounds i8* %P_, i64 32
  %16 = bitcast i8* %15 to float*
  %17 = load float* %16, align 4, !tbaa !8
  %18 = getelementptr inbounds i8* %P_, i64 16
  %19 = bitcast i8* %18 to float*
  %20 = load float* %19, align 4, !tbaa !7
  %21 = fmul float %17, %20
  %22 = fsub float %14, %21
  %23 = bitcast i8* %7 to float*
  %24 = load float* %23, align 4, !tbaa !5
  %25 = fmul float %17, %24
  %26 = bitcast i8* %6 to float*
  %27 = load float* %26, align 4, !tbaa !5
  %28 = fmul float %13, %27
  %29 = fsub float %25, %28
  %30 = fmul float %20, %27
  %31 = fmul float %10, %24
  %32 = fsub float %30, %31
  br label %_Z15calculatenormalPvb.exit

; <label>:33                                      ; preds = %0
  %34 = getelementptr inbounds i8* %P_, i64 12
  %35 = getelementptr inbounds i8* %P_, i64 24
  %36 = getelementptr inbounds i8* %P_, i64 16
  %37 = bitcast i8* %36 to float*
  %38 = load float* %37, align 4, !tbaa !7
  %39 = getelementptr inbounds i8* %P_, i64 32
  %40 = bitcast i8* %39 to float*
  %41 = load float* %40, align 4, !tbaa !8
  %42 = fmul float %38, %41
  %43 = getelementptr inbounds i8* %P_, i64 20
  %44 = bitcast i8* %43 to float*
  %45 = load float* %44, align 4, !tbaa !8
  %46 = getelementptr inbounds i8* %P_, i64 28
  %47 = bitcast i8* %46 to float*
  %48 = load float* %47, align 4, !tbaa !7
  %49 = fmul float %45, %48
  %50 = fsub float %42, %49
  %51 = bitcast i8* %35 to float*
  %52 = load float* %51, align 4, !tbaa !5
  %53 = fmul float %45, %52
  %54 = bitcast i8* %34 to float*
  %55 = load float* %54, align 4, !tbaa !5
  %56 = fmul float %41, %55
  %57 = fsub float %53, %56
  %58 = fmul float %48, %55
  %59 = fmul float %38, %52
  %60 = fsub float %58, %59
  br label %_Z15calculatenormalPvb.exit

_Z15calculatenormalPvb.exit:                      ; preds = %5, %33
  %.sink2.i = phi float [ %50, %33 ], [ %22, %5 ]
  %.sink1.i = phi float [ %57, %33 ], [ %29, %5 ]
  %.sink.i = phi float [ %60, %33 ], [ %32, %5 ]
  %61 = bitcast i8* %out to float*
  store float %.sink2.i, float* %61, align 4, !tbaa !5
  %62 = getelementptr inbounds i8* %out, i64 4
  %63 = bitcast i8* %62 to float*
  store float %.sink1.i, float* %63, align 4, !tbaa !7
  %64 = getelementptr inbounds i8* %out, i64 8
  %65 = bitcast i8* %64 to float*
  store float %.sink.i, float* %65, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readonly uwtable
define float @osl_area(i8* nocapture readonly %P_) #10 {
  %1 = getelementptr inbounds i8* %P_, i64 12
  %2 = getelementptr inbounds i8* %P_, i64 24
  %3 = getelementptr inbounds i8* %P_, i64 16
  %4 = bitcast i8* %3 to float*
  %5 = load float* %4, align 4, !tbaa !7
  %6 = getelementptr inbounds i8* %P_, i64 32
  %7 = bitcast i8* %6 to float*
  %8 = load float* %7, align 4, !tbaa !8
  %9 = fmul float %5, %8
  %10 = getelementptr inbounds i8* %P_, i64 20
  %11 = bitcast i8* %10 to float*
  %12 = load float* %11, align 4, !tbaa !8
  %13 = getelementptr inbounds i8* %P_, i64 28
  %14 = bitcast i8* %13 to float*
  %15 = load float* %14, align 4, !tbaa !7
  %16 = fmul float %12, %15
  %17 = fsub float %9, %16
  %18 = bitcast i8* %2 to float*
  %19 = load float* %18, align 4, !tbaa !5
  %20 = fmul float %12, %19
  %21 = bitcast i8* %1 to float*
  %22 = load float* %21, align 4, !tbaa !5
  %23 = fmul float %8, %22
  %24 = fsub float %20, %23
  %25 = fmul float %15, %22
  %26 = fmul float %5, %19
  %27 = fsub float %25, %26
  %28 = fmul float %17, %17
  %29 = fmul float %24, %24
  %30 = fadd float %28, %29
  %31 = fmul float %27, %27
  %32 = fadd float %31, %30
  %33 = fcmp olt float %32, 0x3820000000000000
  br i1 %33, label %34, label %64

; <label>:34                                      ; preds = %0
  %35 = fcmp ult float %17, 0.000000e+00
  br i1 %35, label %36, label %38

; <label>:36                                      ; preds = %34
  %37 = fsub float -0.000000e+00, %17
  br label %38

; <label>:38                                      ; preds = %36, %34
  %39 = phi float [ %37, %36 ], [ %17, %34 ]
  %40 = fcmp ult float %24, 0.000000e+00
  br i1 %40, label %41, label %43

; <label>:41                                      ; preds = %38
  %42 = fsub float -0.000000e+00, %24
  br label %43

; <label>:43                                      ; preds = %41, %38
  %44 = phi float [ %42, %41 ], [ %24, %38 ]
  %45 = fcmp ult float %27, 0.000000e+00
  br i1 %45, label %46, label %48

; <label>:46                                      ; preds = %43
  %47 = fsub float -0.000000e+00, %27
  br label %48

; <label>:48                                      ; preds = %46, %43
  %49 = phi float [ %47, %46 ], [ %27, %43 ]
  %50 = fcmp olt float %39, %44
  %max.0.i.i = select i1 %50, float %44, float %39
  %51 = fcmp olt float %max.0.i.i, %49
  %max.1.i.i = select i1 %51, float %49, float %max.0.i.i
  %52 = fcmp oeq float %max.1.i.i, 0.000000e+00
  br i1 %52, label %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit, label %53

; <label>:53                                      ; preds = %48
  %54 = fdiv float %39, %max.1.i.i
  %55 = fdiv float %44, %max.1.i.i
  %56 = fdiv float %49, %max.1.i.i
  %57 = fmul float %54, %54
  %58 = fmul float %55, %55
  %59 = fadd float %57, %58
  %60 = fmul float %56, %56
  %61 = fadd float %59, %60
  %62 = tail call float @sqrtf(float %61) #12
  %63 = fmul float %max.1.i.i, %62
  br label %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit

; <label>:64                                      ; preds = %0
  %65 = tail call float @sqrtf(float %32) #12
  br label %_ZNK9Imath_2_24Vec3IfE6lengthEv.exit

_ZNK9Imath_2_24Vec3IfE6lengthEv.exit:             ; preds = %48, %53, %64
  %.0.i = phi float [ %65, %64 ], [ %63, %53 ], [ 0.000000e+00, %48 ]
  ret float %.0.i
}

; Function Attrs: nounwind readonly uwtable
define float @osl_filterwidth_fdf(i8* nocapture readonly %x_) #10 {
  %1 = getelementptr inbounds i8* %x_, i64 4
  %2 = bitcast i8* %1 to float*
  %3 = load float* %2, align 4, !tbaa !1
  %4 = getelementptr inbounds i8* %x_, i64 8
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !1
  %7 = fmul float %3, %3
  %8 = fmul float %6, %6
  %9 = fadd float %7, %8
  %10 = tail call float @sqrtf(float %9) #12
  ret float %10
}

; Function Attrs: nounwind uwtable
define void @osl_filterwidth_vdv(i8* nocapture %out, i8* nocapture readonly %x_) #4 {
  %1 = getelementptr inbounds i8* %x_, i64 12
  %2 = bitcast i8* %1 to float*
  %3 = load float* %2, align 4, !tbaa !5
  %4 = getelementptr inbounds i8* %x_, i64 24
  %5 = bitcast i8* %4 to float*
  %6 = load float* %5, align 4, !tbaa !5
  %7 = fmul float %3, %3
  %8 = fmul float %6, %6
  %9 = fadd float %7, %8
  %10 = tail call float @sqrtf(float %9) #12
  %11 = bitcast i8* %out to float*
  store float %10, float* %11, align 4, !tbaa !5
  %12 = getelementptr inbounds i8* %x_, i64 16
  %13 = bitcast i8* %12 to float*
  %14 = load float* %13, align 4, !tbaa !7
  %15 = getelementptr inbounds i8* %x_, i64 28
  %16 = bitcast i8* %15 to float*
  %17 = load float* %16, align 4, !tbaa !7
  %18 = fmul float %14, %14
  %19 = fmul float %17, %17
  %20 = fadd float %18, %19
  %21 = tail call float @sqrtf(float %20) #12
  %22 = getelementptr inbounds i8* %out, i64 4
  %23 = bitcast i8* %22 to float*
  store float %21, float* %23, align 4, !tbaa !7
  %24 = getelementptr inbounds i8* %x_, i64 20
  %25 = bitcast i8* %24 to float*
  %26 = load float* %25, align 4, !tbaa !8
  %27 = getelementptr inbounds i8* %x_, i64 32
  %28 = bitcast i8* %27 to float*
  %29 = load float* %28, align 4, !tbaa !8
  %30 = fmul float %26, %26
  %31 = fmul float %29, %29
  %32 = fadd float %30, %31
  %33 = tail call float @sqrtf(float %32) #12
  %34 = getelementptr inbounds i8* %out, i64 8
  %35 = bitcast i8* %34 to float*
  store float %33, float* %35, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readonly uwtable
define i32 @osl_raytype_bit(i8* nocapture readonly %sg_, i32 %bit) #10 {
  %1 = getelementptr inbounds i8* %sg_, i64 284
  %2 = bitcast i8* %1 to i32*
  %3 = load i32* %2, align 4, !tbaa !19
  %4 = and i32 %3, %bit
  %5 = icmp ne i32 %4, 0
  %6 = zext i1 %5 to i32
  ret i32 %6
}

; Function Attrs: uwtable
define linkonce_odr void @_ZNK9Imath_2_28Matrix44IfE9gjInverseEb(%"class.Imath_2_2::Matrix44"* noalias sret %agg.result, %"class.Imath_2_2::Matrix44"* nocapture readonly %this, i1 zeroext %singExc) #9 align 2 {
  %s = alloca %"class.Imath_2_2::Matrix44", align 16
  %t = alloca %"class.Imath_2_2::Matrix44", align 16
  %1 = bitcast %"class.Imath_2_2::Matrix44"* %s to i8*
  call void @llvm.lifetime.start(i64 64, i8* %1) #2
  call void @llvm.memset.p0i8.i64(i8* %1, i8 0, i64 60, i32 16, i1 false) #2
  %2 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 0, i64 0
  store float 1.000000e+00, float* %2, align 16, !tbaa !1
  %3 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 1, i64 1
  store float 1.000000e+00, float* %3, align 4, !tbaa !1
  %4 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 2, i64 2
  store float 1.000000e+00, float* %4, align 8, !tbaa !1
  %5 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 3, i64 3
  store float 1.000000e+00, float* %5, align 4, !tbaa !1
  %6 = bitcast %"class.Imath_2_2::Matrix44"* %t to i8*
  call void @llvm.lifetime.start(i64 64, i8* %6) #2
  %7 = bitcast %"class.Imath_2_2::Matrix44"* %this to <4 x float>*
  %8 = load <4 x float>* %7, align 4, !tbaa !1
  %9 = bitcast %"class.Imath_2_2::Matrix44"* %t to <4 x float>*
  store <4 x float> %8, <4 x float>* %9, align 16, !tbaa !1
  %10 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 1, i64 0
  %11 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 1, i64 0
  %12 = bitcast float* %10 to <4 x float>*
  %13 = load <4 x float>* %12, align 4, !tbaa !1
  %14 = bitcast float* %11 to <4 x float>*
  store <4 x float> %13, <4 x float>* %14, align 16, !tbaa !1
  %15 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 2, i64 0
  %16 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 2, i64 0
  %17 = bitcast float* %15 to <4 x float>*
  %18 = load <4 x float>* %17, align 4, !tbaa !1
  %19 = bitcast float* %16 to <4 x float>*
  store <4 x float> %18, <4 x float>* %19, align 16, !tbaa !1
  %20 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %this, i64 0, i32 0, i64 3, i64 0
  %21 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 3, i64 0
  %22 = bitcast float* %20 to <4 x float>*
  %23 = load <4 x float>* %22, align 4, !tbaa !1
  %24 = bitcast float* %21 to <4 x float>*
  store <4 x float> %23, <4 x float>* %24, align 16, !tbaa !1
  %25 = extractelement <4 x float> %8, i32 0
  br label %28

.loopexit:                                        ; preds = %114
  %26 = trunc i64 %indvars.iv.next52 to i32
  %27 = icmp slt i32 %26, 3
  br i1 %27, label %.loopexit._crit_edge, label %.preheader4.preheader

.preheader4.preheader:                            ; preds = %.loopexit
  br label %.preheader4

.loopexit._crit_edge:                             ; preds = %.loopexit
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %.phi.trans.insert = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv.next52, i64 %indvars.iv.next52
  %.pre = load float* %.phi.trans.insert, align 4, !tbaa !1
  br label %28

; <label>:28                                      ; preds = %.loopexit._crit_edge, %0
  %29 = phi float [ %25, %0 ], [ %.pre, %.loopexit._crit_edge ]
  %indvars.iv51 = phi i64 [ 0, %0 ], [ %indvars.iv.next52, %.loopexit._crit_edge ]
  %indvars.iv45 = phi i64 [ 1, %0 ], [ %indvars.iv.next46, %.loopexit._crit_edge ]
  %30 = mul i64 %indvars.iv51, -1
  %31 = add i64 %30, 3
  %32 = trunc i64 %31 to i32
  %33 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv51, i64 %indvars.iv51
  %34 = fcmp olt float %29, 0.000000e+00
  br i1 %34, label %35, label %.lr.ph15

; <label>:35                                      ; preds = %28
  %36 = fsub float -0.000000e+00, %29
  br label %.lr.ph15

; <label>:37                                      ; preds = %160, %68
  %38 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          filter [1 x i8*] [i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN7Iex_2_27MathExcE to i8*)]
  %39 = extractvalue { i8*, i32 } %38, 1
  %40 = icmp slt i32 %39, 0
  br i1 %40, label %41, label %213

; <label>:41                                      ; preds = %37
  %42 = extractvalue { i8*, i32 } %38, 0
  tail call void @__cxa_call_unexpected(i8* %42) #13
  unreachable

.lr.ph15:                                         ; preds = %28, %35
  %pivotsize.0 = phi float [ %36, %35 ], [ %29, %28 ]
  %indvars.iv.next52 = add nuw nsw i64 %indvars.iv51, 1
  %43 = trunc i64 %indvars.iv51 to i32
  %xtraiter = and i32 %32, 1
  %lcmp.mod = icmp ne i32 %xtraiter, 0
  %lcmp.overflow = icmp eq i32 %32, 0
  %lcmp.or = or i1 %lcmp.overflow, %lcmp.mod
  br i1 %lcmp.or, label %44, label %.lr.ph15.split

; <label>:44                                      ; preds = %.lr.ph15
  %45 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv45, i64 %indvars.iv51
  %46 = load float* %45, align 4, !tbaa !1
  %47 = fcmp olt float %46, 0.000000e+00
  br i1 %47, label %48, label %50

; <label>:48                                      ; preds = %44
  %49 = fsub float -0.000000e+00, %46
  br label %50

; <label>:50                                      ; preds = %48, %44
  %tmp.0.unr = phi float [ %49, %48 ], [ %46, %44 ]
  %51 = fcmp ogt float %tmp.0.unr, %pivotsize.0
  %52 = trunc i64 %indvars.iv45 to i32
  %pivot.1.unr = select i1 %51, i32 %52, i32 %43
  %pivotsize.2.unr = select i1 %51, float %tmp.0.unr, float %pivotsize.0
  %indvars.iv.next31.unr = add nuw nsw i64 %indvars.iv45, 1
  %lftr.wideiv32.unr = trunc i64 %indvars.iv.next31.unr to i32
  %exitcond33.unr = icmp eq i32 %lftr.wideiv32.unr, 4
  br label %.lr.ph15.split

.lr.ph15.split:                                   ; preds = %50, %.lr.ph15
  %pivotsize.2.lcssa.unr = phi float [ 0.000000e+00, %.lr.ph15 ], [ %pivotsize.2.unr, %50 ]
  %pivot.1.lcssa.unr = phi i32 [ 0, %.lr.ph15 ], [ %pivot.1.unr, %50 ]
  %indvars.iv30.unr = phi i64 [ %indvars.iv45, %.lr.ph15 ], [ %indvars.iv.next31.unr, %50 ]
  %pivotsize.112.unr = phi float [ %pivotsize.0, %.lr.ph15 ], [ %pivotsize.2.unr, %50 ]
  %pivot.011.unr = phi i32 [ %43, %.lr.ph15 ], [ %pivot.1.unr, %50 ]
  %53 = icmp ult i32 %32, 2
  br i1 %53, label %66, label %.lr.ph15.split.split

.lr.ph15.split.split:                             ; preds = %.lr.ph15.split
  br label %54

; <label>:54                                      ; preds = %217, %.lr.ph15.split.split
  %indvars.iv30 = phi i64 [ %indvars.iv30.unr, %.lr.ph15.split.split ], [ %indvars.iv.next31.1, %217 ]
  %pivotsize.112 = phi float [ %pivotsize.112.unr, %.lr.ph15.split.split ], [ %pivotsize.2.1, %217 ]
  %pivot.011 = phi i32 [ %pivot.011.unr, %.lr.ph15.split.split ], [ %pivot.1.1, %217 ]
  %55 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv30, i64 %indvars.iv51
  %56 = load float* %55, align 4, !tbaa !1
  %57 = fcmp olt float %56, 0.000000e+00
  br i1 %57, label %58, label %60

; <label>:58                                      ; preds = %54
  %59 = fsub float -0.000000e+00, %56
  br label %60

; <label>:60                                      ; preds = %58, %54
  %tmp.0 = phi float [ %59, %58 ], [ %56, %54 ]
  %61 = fcmp ogt float %tmp.0, %pivotsize.112
  %62 = trunc i64 %indvars.iv30 to i32
  %pivot.1 = select i1 %61, i32 %62, i32 %pivot.011
  %pivotsize.2 = select i1 %61, float %tmp.0, float %pivotsize.112
  %indvars.iv.next31 = add nuw nsw i64 %indvars.iv30, 1
  %lftr.wideiv32 = trunc i64 %indvars.iv.next31 to i32
  %63 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv.next31, i64 %indvars.iv51
  %64 = load float* %63, align 4, !tbaa !1
  %65 = fcmp olt float %64, 0.000000e+00
  br i1 %65, label %215, label %217

.unr-lcssa:                                       ; preds = %217
  %pivotsize.2.lcssa.ph = phi float [ %pivotsize.2.1, %217 ]
  %pivot.1.lcssa.ph = phi i32 [ %pivot.1.1, %217 ]
  br label %66

; <label>:66                                      ; preds = %.lr.ph15.split, %.unr-lcssa
  %pivotsize.2.lcssa = phi float [ %pivotsize.2.lcssa.unr, %.lr.ph15.split ], [ %pivotsize.2.lcssa.ph, %.unr-lcssa ]
  %pivot.1.lcssa = phi i32 [ %pivot.1.lcssa.unr, %.lr.ph15.split ], [ %pivot.1.lcssa.ph, %.unr-lcssa ]
  %phitmp = fcmp oeq float %pivotsize.2.lcssa, 0.000000e+00
  br i1 %phitmp, label %67, label %78

; <label>:67                                      ; preds = %66
  br i1 %singExc, label %68, label %72

; <label>:68                                      ; preds = %67
  %69 = tail call i8* @__cxa_allocate_exception(i64 24) #2
  %70 = bitcast i8* %69 to %"class.Iex_2_2::BaseExc"*
  tail call void @_ZN7Iex_2_27BaseExcC2EPKc(%"class.Iex_2_2::BaseExc"* %70, i8* getelementptr inbounds ([31 x i8]* @.str, i64 0, i64 0)) #2
  %71 = bitcast i8* %69 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ([5 x i8*]* @_ZTVN9Imath_2_213SingMatrixExcE, i64 0, i64 2) to i32 (...)**), i32 (...)*** %71, align 8, !tbaa !13
  invoke void @__cxa_throw(i8* %69, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN9Imath_2_213SingMatrixExcE to i8*), i8* bitcast (void (%"class.Iex_2_2::BaseExc"*)* @_ZN7Iex_2_27BaseExcD2Ev to i8*)) #13
          to label %214 unwind label %37

; <label>:72                                      ; preds = %67
  %73 = bitcast %"class.Imath_2_2::Matrix44"* %agg.result to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %73, i8 0, i64 60, i32 4, i1 false) #2
  %74 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 0, i64 0
  store float 1.000000e+00, float* %74, align 4, !tbaa !1
  %75 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 1, i64 1
  store float 1.000000e+00, float* %75, align 4, !tbaa !1
  %76 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 2, i64 2
  store float 1.000000e+00, float* %76, align 4, !tbaa !1
  %77 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 3, i64 3
  store float 1.000000e+00, float* %77, align 4, !tbaa !1
  br label %212

; <label>:78                                      ; preds = %66
  %79 = icmp eq i32 %pivot.1.lcssa, %43
  br i1 %79, label %.lr.ph21, label %.preheader6

.preheader6:                                      ; preds = %78
  %80 = sext i32 %pivot.1.lcssa to i64
  %81 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv51, i64 0
  %82 = load float* %81, align 16, !tbaa !1
  %83 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %80, i64 0
  %84 = load float* %83, align 16, !tbaa !1
  store float %84, float* %81, align 16, !tbaa !1
  store float %82, float* %83, align 16, !tbaa !1
  %85 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv51, i64 0
  %86 = load float* %85, align 16, !tbaa !1
  %87 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %80, i64 0
  %88 = load float* %87, align 16, !tbaa !1
  store float %88, float* %85, align 16, !tbaa !1
  store float %86, float* %87, align 16, !tbaa !1
  %89 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv51, i64 1
  %90 = load float* %89, align 4, !tbaa !1
  %91 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %80, i64 1
  %92 = load float* %91, align 4, !tbaa !1
  store float %92, float* %89, align 4, !tbaa !1
  store float %90, float* %91, align 4, !tbaa !1
  %93 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv51, i64 1
  %94 = load float* %93, align 4, !tbaa !1
  %95 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %80, i64 1
  %96 = load float* %95, align 4, !tbaa !1
  store float %96, float* %93, align 4, !tbaa !1
  store float %94, float* %95, align 4, !tbaa !1
  %97 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv51, i64 2
  %98 = load float* %97, align 8, !tbaa !1
  %99 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %80, i64 2
  %100 = load float* %99, align 8, !tbaa !1
  store float %100, float* %97, align 8, !tbaa !1
  store float %98, float* %99, align 8, !tbaa !1
  %101 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv51, i64 2
  %102 = load float* %101, align 8, !tbaa !1
  %103 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %80, i64 2
  %104 = load float* %103, align 8, !tbaa !1
  store float %104, float* %101, align 8, !tbaa !1
  store float %102, float* %103, align 8, !tbaa !1
  %105 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv51, i64 3
  %106 = load float* %105, align 4, !tbaa !1
  %107 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %80, i64 3
  %108 = load float* %107, align 4, !tbaa !1
  store float %108, float* %105, align 4, !tbaa !1
  store float %106, float* %107, align 4, !tbaa !1
  %109 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv51, i64 3
  %110 = load float* %109, align 4, !tbaa !1
  %111 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %80, i64 3
  %112 = load float* %111, align 4, !tbaa !1
  store float %112, float* %109, align 4, !tbaa !1
  store float %110, float* %111, align 4, !tbaa !1
  br label %.lr.ph21

.lr.ph21:                                         ; preds = %78, %.preheader6
  %113 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv51, i64 0
  br label %114

; <label>:114                                     ; preds = %114, %.lr.ph21
  %indvars.iv47 = phi i64 [ %indvars.iv45, %.lr.ph21 ], [ %indvars.iv.next48, %114 ]
  %115 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv47, i64 %indvars.iv51
  %116 = load float* %115, align 4, !tbaa !1
  %117 = load float* %33, align 4, !tbaa !1
  %118 = fdiv float %116, %117
  %119 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv47, i64 0
  %120 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv51, i64 0
  %121 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv47, i64 0
  %122 = bitcast float* %113 to <4 x float>*
  %123 = load <4 x float>* %122, align 16, !tbaa !1
  %124 = insertelement <4 x float> undef, float %118, i32 0
  %125 = insertelement <4 x float> %124, float %118, i32 1
  %126 = insertelement <4 x float> %125, float %118, i32 2
  %127 = insertelement <4 x float> %126, float %118, i32 3
  %128 = fmul <4 x float> %127, %123
  %129 = bitcast float* %119 to <4 x float>*
  %130 = load <4 x float>* %129, align 16, !tbaa !1
  %131 = fsub <4 x float> %130, %128
  %132 = bitcast float* %119 to <4 x float>*
  store <4 x float> %131, <4 x float>* %132, align 16, !tbaa !1
  %133 = bitcast float* %120 to <4 x float>*
  %134 = load <4 x float>* %133, align 16, !tbaa !1
  %135 = fmul <4 x float> %127, %134
  %136 = bitcast float* %121 to <4 x float>*
  %137 = load <4 x float>* %136, align 16, !tbaa !1
  %138 = fsub <4 x float> %137, %135
  %139 = bitcast float* %121 to <4 x float>*
  store <4 x float> %138, <4 x float>* %139, align 16, !tbaa !1
  %indvars.iv.next48 = add nuw nsw i64 %indvars.iv47, 1
  %lftr.wideiv49 = trunc i64 %indvars.iv.next48 to i32
  %exitcond50 = icmp eq i32 %lftr.wideiv49, 4
  br i1 %exitcond50, label %.loopexit, label %114

.preheader4:                                      ; preds = %.preheader4.preheader, %._crit_edge
  %indvars.iv26 = phi i64 [ %indvars.iv.next27, %._crit_edge ], [ 3, %.preheader4.preheader ]
  %indvars.iv24 = phi i32 [ %indvars.iv.next25, %._crit_edge ], [ 3, %.preheader4.preheader ]
  %140 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv26, i64 %indvars.iv26
  %141 = load float* %140, align 4, !tbaa !1
  %142 = fcmp oeq float %141, 0.000000e+00
  br i1 %142, label %159, label %.preheader3

.preheader3:                                      ; preds = %.preheader4
  %143 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv26, i64 0
  %144 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv26, i64 0
  %145 = bitcast float* %143 to <4 x float>*
  %146 = load <4 x float>* %145, align 16, !tbaa !1
  %147 = insertelement <4 x float> undef, float %141, i32 0
  %148 = insertelement <4 x float> %147, float %141, i32 1
  %149 = insertelement <4 x float> %148, float %141, i32 2
  %150 = insertelement <4 x float> %149, float %141, i32 3
  %151 = fdiv <4 x float> %146, %150
  %152 = bitcast float* %143 to <4 x float>*
  store <4 x float> %151, <4 x float>* %152, align 16, !tbaa !1
  %153 = bitcast float* %144 to <4 x float>*
  %154 = load <4 x float>* %153, align 16, !tbaa !1
  %155 = fdiv <4 x float> %154, %150
  %156 = bitcast float* %144 to <4 x float>*
  store <4 x float> %155, <4 x float>* %156, align 16, !tbaa !1
  %157 = trunc i64 %indvars.iv26 to i32
  %158 = icmp sgt i32 %157, 0
  br i1 %158, label %.lr.ph.preheader, label %._crit_edge.thread

.lr.ph.preheader:                                 ; preds = %.preheader3
  br label %.lr.ph

; <label>:159                                     ; preds = %.preheader4
  br i1 %singExc, label %160, label %164

; <label>:160                                     ; preds = %159
  %161 = tail call i8* @__cxa_allocate_exception(i64 24) #2
  %162 = bitcast i8* %161 to %"class.Iex_2_2::BaseExc"*
  tail call void @_ZN7Iex_2_27BaseExcC2EPKc(%"class.Iex_2_2::BaseExc"* %162, i8* getelementptr inbounds ([31 x i8]* @.str, i64 0, i64 0)) #2
  %163 = bitcast i8* %161 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ([5 x i8*]* @_ZTVN9Imath_2_213SingMatrixExcE, i64 0, i64 2) to i32 (...)**), i32 (...)*** %163, align 8, !tbaa !13
  invoke void @__cxa_throw(i8* %161, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN9Imath_2_213SingMatrixExcE to i8*), i8* bitcast (void (%"class.Iex_2_2::BaseExc"*)* @_ZN7Iex_2_27BaseExcD2Ev to i8*)) #13
          to label %214 unwind label %37

; <label>:164                                     ; preds = %159
  %165 = bitcast %"class.Imath_2_2::Matrix44"* %agg.result to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %165, i8 0, i64 60, i32 4, i1 false) #2
  %166 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 0, i64 0
  store float 1.000000e+00, float* %166, align 4, !tbaa !1
  %167 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 1, i64 1
  store float 1.000000e+00, float* %167, align 4, !tbaa !1
  %168 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 2, i64 2
  store float 1.000000e+00, float* %168, align 4, !tbaa !1
  %169 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 3, i64 3
  store float 1.000000e+00, float* %169, align 4, !tbaa !1
  br label %212

.lr.ph:                                           ; preds = %.lr.ph.preheader, %._crit_edge54
  %indvars.iv = phi i64 [ %indvars.iv.next, %._crit_edge54 ], [ 0, %.lr.ph.preheader ]
  %170 = phi <4 x float> [ %191, %._crit_edge54 ], [ %151, %.lr.ph.preheader ]
  %171 = phi <4 x float> [ %193, %._crit_edge54 ], [ %155, %.lr.ph.preheader ]
  %172 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv, i64 %indvars.iv26
  %173 = load float* %172, align 4, !tbaa !1
  %174 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %t, i64 0, i32 0, i64 %indvars.iv, i64 0
  %175 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 %indvars.iv, i64 0
  %176 = insertelement <4 x float> undef, float %173, i32 0
  %177 = insertelement <4 x float> %176, float %173, i32 1
  %178 = insertelement <4 x float> %177, float %173, i32 2
  %179 = insertelement <4 x float> %178, float %173, i32 3
  %180 = fmul <4 x float> %179, %170
  %181 = bitcast float* %174 to <4 x float>*
  %182 = load <4 x float>* %181, align 16, !tbaa !1
  %183 = fsub <4 x float> %182, %180
  %184 = bitcast float* %174 to <4 x float>*
  store <4 x float> %183, <4 x float>* %184, align 16, !tbaa !1
  %185 = fmul <4 x float> %179, %171
  %186 = bitcast float* %175 to <4 x float>*
  %187 = load <4 x float>* %186, align 16, !tbaa !1
  %188 = fsub <4 x float> %187, %185
  %189 = bitcast float* %175 to <4 x float>*
  store <4 x float> %188, <4 x float>* %189, align 16, !tbaa !1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %indvars.iv24
  br i1 %exitcond, label %._crit_edge, label %._crit_edge54

._crit_edge54:                                    ; preds = %.lr.ph
  %190 = bitcast float* %143 to <4 x float>*
  %191 = load <4 x float>* %190, align 16, !tbaa !1
  %192 = bitcast float* %144 to <4 x float>*
  %193 = load <4 x float>* %192, align 16, !tbaa !1
  br label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph
  %indvars.iv.next27 = add nsw i64 %indvars.iv26, -1
  %indvars.iv.next25 = add nsw i32 %indvars.iv24, -1
  br i1 %158, label %.preheader4, label %._crit_edge.thread

._crit_edge.thread:                               ; preds = %.preheader3, %._crit_edge
  %194 = bitcast %"class.Imath_2_2::Matrix44"* %s to <4 x float>*
  %195 = load <4 x float>* %194, align 16, !tbaa !1
  %196 = bitcast %"class.Imath_2_2::Matrix44"* %agg.result to <4 x float>*
  store <4 x float> %195, <4 x float>* %196, align 4, !tbaa !1
  %197 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 1, i64 0
  %198 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 1, i64 0
  %199 = bitcast float* %197 to <4 x float>*
  %200 = load <4 x float>* %199, align 16, !tbaa !1
  %201 = bitcast float* %198 to <4 x float>*
  store <4 x float> %200, <4 x float>* %201, align 4, !tbaa !1
  %202 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 2, i64 0
  %203 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 2, i64 0
  %204 = bitcast float* %202 to <4 x float>*
  %205 = load <4 x float>* %204, align 16, !tbaa !1
  %206 = bitcast float* %203 to <4 x float>*
  store <4 x float> %205, <4 x float>* %206, align 4, !tbaa !1
  %207 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %s, i64 0, i32 0, i64 3, i64 0
  %208 = getelementptr inbounds %"class.Imath_2_2::Matrix44"* %agg.result, i64 0, i32 0, i64 3, i64 0
  %209 = bitcast float* %207 to <4 x float>*
  %210 = load <4 x float>* %209, align 16, !tbaa !1
  %211 = bitcast float* %208 to <4 x float>*
  store <4 x float> %210, <4 x float>* %211, align 4, !tbaa !1
  br label %212

; <label>:212                                     ; preds = %._crit_edge.thread, %164, %72
  call void @llvm.lifetime.end(i64 64, i8* %6) #2
  call void @llvm.lifetime.end(i64 64, i8* %1) #2
  ret void

; <label>:213                                     ; preds = %37
  resume { i8*, i32 } %38

; <label>:214                                     ; preds = %160, %68
  unreachable

; <label>:215                                     ; preds = %60
  %216 = fsub float -0.000000e+00, %64
  br label %217

; <label>:217                                     ; preds = %215, %60
  %tmp.0.1 = phi float [ %216, %215 ], [ %64, %60 ]
  %218 = fcmp ogt float %tmp.0.1, %pivotsize.2
  %219 = trunc i64 %indvars.iv.next31 to i32
  %pivot.1.1 = select i1 %218, i32 %219, i32 %pivot.1
  %pivotsize.2.1 = select i1 %218, float %tmp.0.1, float %pivotsize.2
  %indvars.iv.next31.1 = add nuw nsw i64 %indvars.iv.next31, 1
  %lftr.wideiv32.1 = trunc i64 %indvars.iv.next31.1 to i32
  %exitcond33.1 = icmp eq i32 %lftr.wideiv32.1, 4
  br i1 %exitcond33.1, label %.unr-lcssa, label %54
}

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

declare void @__cxa_call_unexpected(i8*)

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #2

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN9Imath_2_213SingMatrixExcD0Ev(%"class.Imath_2_2::SingMatrixExc"* %this) unnamed_addr #4 align 2 {
  %1 = getelementptr inbounds %"class.Imath_2_2::SingMatrixExc"* %this, i64 0, i32 0, i32 0
  tail call void @_ZN7Iex_2_27BaseExcD2Ev(%"class.Iex_2_2::BaseExc"* %1) #2
  %2 = bitcast %"class.Imath_2_2::SingMatrixExc"* %this to i8*
  tail call void @_ZdlPv(i8* %2) #14
  ret void
}

; Function Attrs: nounwind
declare i8* @_ZNK7Iex_2_27BaseExc4whatEv(%"class.Iex_2_2::BaseExc"*) #1

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) #11

; Function Attrs: nounwind
declare void @_ZN7Iex_2_27BaseExcC2EPKc(%"class.Iex_2_2::BaseExc"*, i8*) #1

; Function Attrs: nounwind
declare void @_ZN7Iex_2_27BaseExcD2Ev(%"class.Iex_2_2::BaseExc"*) #1

; Function Attrs: nounwind readnone
declare float @erfcf(float) #7

; Function Attrs: nounwind readnone
declare float @expf(float) #7

; Function Attrs: nounwind readnone
declare float @erff(float) #7

; Function Attrs: nounwind readnone
declare float @fmodf(float, float) #7

; Function Attrs: nounwind readnone
declare float @copysignf(float, float) #7

define internal void @_GLOBAL__sub_I_llvm_ops.cpp() section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*)) #2
  ret void
}

attributes #0 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { inlinehint nounwind readonly uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { inlinehint nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { nounwind readonly uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { nobuiltin nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { nounwind readnone }
attributes #13 = { noreturn }
attributes #14 = { builtin nounwind }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 (tags/RELEASE_350/final)"}
!1 = metadata !{metadata !2, metadata !2, i64 0}
!2 = metadata !{metadata !"float", metadata !3, i64 0}
!3 = metadata !{metadata !"omnipotent char", metadata !4, i64 0}
!4 = metadata !{metadata !"Simple C/C++ TBAA"}
!5 = metadata !{metadata !6, metadata !2, i64 0}
!6 = metadata !{metadata !"_ZTSN9Imath_2_24Vec3IfEE", metadata !2, i64 0, metadata !2, i64 4, metadata !2, i64 8}
!7 = metadata !{metadata !6, metadata !2, i64 4}
!8 = metadata !{metadata !6, metadata !2, i64 8}
!9 = metadata !{metadata !10, metadata !2, i64 0}
!10 = metadata !{metadata !"_ZTSN3OSL5Dual2IfEE", metadata !2, i64 0, metadata !2, i64 4, metadata !2, i64 8}
!11 = metadata !{metadata !10, metadata !2, i64 4}
!12 = metadata !{metadata !10, metadata !2, i64 8}
!13 = metadata !{metadata !14, metadata !14, i64 0}
!14 = metadata !{metadata !"vtable pointer", metadata !4, i64 0}
!15 = metadata !{metadata !16, metadata !18, i64 288}
!16 = metadata !{metadata !"_ZTSN3OSL13ShaderGlobalsE", metadata !6, i64 0, metadata !6, i64 12, metadata !6, i64 24, metadata !6, i64 36, metadata !6, i64 48, metadata !6, i64 60, metadata !6, i64 72, metadata !6, i64 84, metadata !6, i64 96, metadata !2, i64 108, metadata !2, i64 112, metadata !2, i64 116, metadata !2, i64 120, metadata !2, i64 124, metadata !2, i64 128, metadata !6, i64 132, metadata !6, i64 144, metadata !2, i64 156, metadata !2, i64 160, metadata !6, i64 164, metadata !6, i64 176, metadata !6, i64 188, metadata !6, i64 200, metadata !17, i64 216, metadata !17, i64 224, metadata !17, i64 232, metadata !17, i64 240, metadata !17, i64 248, metadata !17, i64 256, metadata !17, i64 264, metadata !17, i64 272, metadata !2, i64 280, metadata !18, i64 284, metadata !18, i64 288, metadata !18, i64 292}
!17 = metadata !{metadata !"any pointer", metadata !3, i64 0}
!18 = metadata !{metadata !"int", metadata !3, i64 0}
!19 = metadata !{metadata !16, metadata !18, i64 284}
