; ModuleID = 'llvm_ops.cpp'
source_filename = "llvm_ops.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.OSL::Dual2.0" = type { %"class.Imath_2_2::Vec3", %"class.Imath_2_2::Vec3", %"class.Imath_2_2::Vec3" }
%"class.Imath_2_2::Vec3" = type { float, float, float }
%"class.OSL::Dual2" = type { float, float, float }
%"class.Imath_2_2::Matrix44" = type { [4 x [4 x float]] }
%"class.Iex_2_2::BaseExc" = type { %"class.std::exception", %"class.std::basic_string", %"class.std::basic_string" }
%"class.std::exception" = type { i32 (...)** }
%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%"class.Imath_2_2::SingMatrixExc" = type { %"class.Iex_2_2::MathExc" }
%"class.Iex_2_2::MathExc" = type { %"class.Iex_2_2::BaseExc" }

$_ZN11OpenImageIO4v1_713fast_safe_powEff = comdat any

$_ZN3OSL20robust_multVecMatrixERKN9Imath_2_28Matrix44IfEERKNS_5Dual2INS0_4Vec3IfEEEERS8_ = comdat any

$_ZNK9Imath_2_28Matrix44IfE7inverseEb = comdat any

$_ZN3OSL9normalizeERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE = comdat any

$_ZNK9Imath_2_28Matrix44IfE9gjInverseEb = comdat any

$_ZN9Imath_2_213SingMatrixExcD0Ev = comdat any

$_ZTSN7Iex_2_27MathExcE = comdat any

$_ZTIN7Iex_2_27MathExcE = comdat any

$_ZTSN9Imath_2_213SingMatrixExcE = comdat any

$_ZTIN9Imath_2_213SingMatrixExcE = comdat any

$_ZTVN9Imath_2_213SingMatrixExcE = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = global i8* null, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTSN7Iex_2_27MathExcE = linkonce_odr constant [19 x i8] c"N7Iex_2_27MathExcE\00", comdat
@_ZTIN7Iex_2_27BaseExcE = external constant i8*
@_ZTIN7Iex_2_27MathExcE = linkonce_odr constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([19 x i8], [19 x i8]* @_ZTSN7Iex_2_27MathExcE, i32 0, i32 0), i8* bitcast (i8** @_ZTIN7Iex_2_27BaseExcE to i8*) }, comdat
@.str = private unnamed_addr constant [31 x i8] c"Cannot invert singular matrix.\00", align 1
@_ZTSN9Imath_2_213SingMatrixExcE = linkonce_odr constant [28 x i8] c"N9Imath_2_213SingMatrixExcE\00", comdat
@_ZTIN9Imath_2_213SingMatrixExcE = linkonce_odr constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([28 x i8], [28 x i8]* @_ZTSN9Imath_2_213SingMatrixExcE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN7Iex_2_27MathExcE to i8*) }, comdat
@_ZTVN9Imath_2_213SingMatrixExcE = linkonce_odr unnamed_addr constant [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN9Imath_2_213SingMatrixExcE to i8*), i8* bitcast (void (%"class.Iex_2_2::BaseExc"*)* @_ZN7Iex_2_27BaseExcD2Ev to i8*), i8* bitcast (void (%"class.Imath_2_2::SingMatrixExc"*)* @_ZN9Imath_2_213SingMatrixExcD0Ev to i8*), i8* bitcast (i8* (%"class.Iex_2_2::BaseExc"*)* @_ZNK7Iex_2_27BaseExc4whatEv to i8*)], comdat, align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_llvm_ops.cpp, i8* null }]

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: nounwind readnone uwtable
define float @osl_sin_ff(float) local_unnamed_addr #3 {
  %2 = fmul float %0, 0x3FD45F3060000000
  %3 = tail call float @copysignf(float 5.000000e-01, float %2) #13
  %4 = fadd float %2, %3
  %5 = fptosi float %4 to i32
  %6 = sitofp i32 %5 to float
  %7 = fmul float %6, 3.140625e+00
  %8 = fsub float %0, %7
  %9 = fmul float %6, 0x3F4FB40000000000
  %10 = fsub float %8, %9
  %11 = fmul float %6, 0x3E84440000000000
  %12 = fsub float %10, %11
  %13 = fmul float %6, 0x3D968C2340000000
  %14 = fsub float %12, %13
  %15 = fsub float 0x3FF921FB60000000, %14
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = fmul float %16, %16
  %18 = and i32 %5, 1
  %19 = icmp eq i32 %18, 0
  %20 = fsub float -0.000000e+00, %16
  %21 = select i1 %19, float %16, float %20
  %22 = fmul float %17, 0x3EC5E150E0000000
  %23 = fadd float %22, 0xBF29F75D60000000
  %24 = fmul float %17, %23
  %25 = fadd float %24, 0x3F8110EEE0000000
  %26 = fmul float %17, %25
  %27 = fadd float %26, 0xBFC55554C0000000
  %28 = fmul float %21, %27
  %29 = fmul float %17, %28
  %30 = fadd float %21, %29
  %31 = tail call float @fabsf(float %30) #13
  %32 = fcmp ogt float %31, 1.000000e+00
  %33 = select i1 %32, float 0.000000e+00, float %30
  ret float %33
}

; Function Attrs: nounwind uwtable
define void @osl_sin_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fmul float %4, 0x3FD45F3060000000
  %6 = tail call float @copysignf(float 5.000000e-01, float %5) #13
  %7 = fadd float %5, %6
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fmul float %9, 3.140625e+00
  %11 = fsub float %4, %10
  %12 = fmul float %9, 0x3F4FB40000000000
  %13 = fsub float %11, %12
  %14 = fmul float %9, 0x3E84440000000000
  %15 = fsub float %13, %14
  %16 = fmul float %9, 0x3D968C2340000000
  %17 = fsub float %15, %16
  %18 = fsub float 0x3FF921FB60000000, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fmul float %19, %19
  %21 = and i32 %8, 1
  %22 = icmp ne i32 %21, 0
  %23 = fsub float -0.000000e+00, %19
  %24 = select i1 %22, float %23, float %19
  %25 = fmul float %20, 0x3EC5E150E0000000
  %26 = fadd float %25, 0xBF29F75D60000000
  %27 = fmul float %20, %26
  %28 = fadd float %27, 0x3F8110EEE0000000
  %29 = fmul float %20, %28
  %30 = fadd float %29, 0xBFC55554C0000000
  %31 = fmul float %24, %30
  %32 = fmul float %20, %31
  %33 = fadd float %24, %32
  %34 = fmul float %20, 0x3E923DB120000000
  %35 = fsub float 0x3EFA00F160000000, %34
  %36 = fmul float %20, %35
  %37 = fadd float %36, 0xBF56C16B00000000
  %38 = fmul float %20, %37
  %39 = fadd float %38, 0x3FA5555540000000
  %40 = fmul float %20, %39
  %41 = fadd float %40, -5.000000e-01
  %42 = fmul float %20, %41
  %43 = fadd float %42, 1.000000e+00
  %44 = fsub float -0.000000e+00, %43
  %45 = select i1 %22, float %44, float %43
  %46 = tail call float @fabsf(float %33) #13
  %47 = fcmp ogt float %46, 1.000000e+00
  %48 = tail call float @fabsf(float %45) #13
  %49 = fcmp ogt float %48, 1.000000e+00
  %50 = select i1 %49, float 0.000000e+00, float %45
  %51 = getelementptr inbounds i8, i8* %1, i64 4
  %52 = bitcast i8* %51 to float*
  %53 = load float, float* %52, align 4, !tbaa !1
  %54 = fmul float %53, %50
  %55 = getelementptr inbounds i8, i8* %1, i64 8
  %56 = bitcast i8* %55 to float*
  %57 = load float, float* %56, align 4, !tbaa !1
  %58 = fmul float %57, %50
  %59 = select i1 %47, float 0.000000e+00, float %33
  %60 = insertelement <2 x float> undef, float %59, i32 0
  %61 = insertelement <2 x float> %60, float %54, i32 1
  %62 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %61, <2 x float>* %62, align 4
  %63 = getelementptr inbounds i8, i8* %0, i64 8
  %64 = bitcast i8* %63 to float*
  store float %58, float* %64, align 4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #5

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #5

; Function Attrs: nounwind uwtable
define void @osl_sin_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fmul float %4, 0x3FD45F3060000000
  %6 = tail call float @copysignf(float 5.000000e-01, float %5) #13
  %7 = fadd float %5, %6
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fmul float %9, 3.140625e+00
  %11 = fsub float %4, %10
  %12 = fmul float %9, 0x3F4FB40000000000
  %13 = fsub float %11, %12
  %14 = fmul float %9, 0x3E84440000000000
  %15 = fsub float %13, %14
  %16 = fmul float %9, 0x3D968C2340000000
  %17 = fsub float %15, %16
  %18 = fsub float 0x3FF921FB60000000, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fmul float %19, %19
  %21 = and i32 %8, 1
  %22 = icmp eq i32 %21, 0
  %23 = fsub float -0.000000e+00, %19
  %24 = select i1 %22, float %19, float %23
  %25 = fmul float %20, 0x3EC5E150E0000000
  %26 = fadd float %25, 0xBF29F75D60000000
  %27 = fmul float %20, %26
  %28 = fadd float %27, 0x3F8110EEE0000000
  %29 = fmul float %20, %28
  %30 = fadd float %29, 0xBFC55554C0000000
  %31 = fmul float %24, %30
  %32 = fmul float %20, %31
  %33 = fadd float %24, %32
  %34 = tail call float @fabsf(float %33) #13
  %35 = fcmp ogt float %34, 1.000000e+00
  %36 = select i1 %35, float 0.000000e+00, float %33
  %37 = bitcast i8* %0 to float*
  store float %36, float* %37, align 4, !tbaa !1
  %38 = getelementptr inbounds i8, i8* %1, i64 4
  %39 = bitcast i8* %38 to float*
  %40 = load float, float* %39, align 4, !tbaa !1
  %41 = fmul float %40, 0x3FD45F3060000000
  %42 = tail call float @copysignf(float 5.000000e-01, float %41) #13
  %43 = fadd float %41, %42
  %44 = fptosi float %43 to i32
  %45 = sitofp i32 %44 to float
  %46 = fmul float %45, 3.140625e+00
  %47 = fsub float %40, %46
  %48 = fmul float %45, 0x3F4FB40000000000
  %49 = fsub float %47, %48
  %50 = fmul float %45, 0x3E84440000000000
  %51 = fsub float %49, %50
  %52 = fmul float %45, 0x3D968C2340000000
  %53 = fsub float %51, %52
  %54 = fsub float 0x3FF921FB60000000, %53
  %55 = fsub float 0x3FF921FB60000000, %54
  %56 = fmul float %55, %55
  %57 = and i32 %44, 1
  %58 = icmp eq i32 %57, 0
  %59 = fsub float -0.000000e+00, %55
  %60 = select i1 %58, float %55, float %59
  %61 = fmul float %56, 0x3EC5E150E0000000
  %62 = fadd float %61, 0xBF29F75D60000000
  %63 = fmul float %56, %62
  %64 = fadd float %63, 0x3F8110EEE0000000
  %65 = fmul float %56, %64
  %66 = fadd float %65, 0xBFC55554C0000000
  %67 = fmul float %60, %66
  %68 = fmul float %56, %67
  %69 = fadd float %60, %68
  %70 = tail call float @fabsf(float %69) #13
  %71 = fcmp ogt float %70, 1.000000e+00
  %72 = select i1 %71, float 0.000000e+00, float %69
  %73 = getelementptr inbounds i8, i8* %0, i64 4
  %74 = bitcast i8* %73 to float*
  store float %72, float* %74, align 4, !tbaa !1
  %75 = getelementptr inbounds i8, i8* %1, i64 8
  %76 = bitcast i8* %75 to float*
  %77 = load float, float* %76, align 4, !tbaa !1
  %78 = fmul float %77, 0x3FD45F3060000000
  %79 = tail call float @copysignf(float 5.000000e-01, float %78) #13
  %80 = fadd float %78, %79
  %81 = fptosi float %80 to i32
  %82 = sitofp i32 %81 to float
  %83 = fmul float %82, 3.140625e+00
  %84 = fsub float %77, %83
  %85 = fmul float %82, 0x3F4FB40000000000
  %86 = fsub float %84, %85
  %87 = fmul float %82, 0x3E84440000000000
  %88 = fsub float %86, %87
  %89 = fmul float %82, 0x3D968C2340000000
  %90 = fsub float %88, %89
  %91 = fsub float 0x3FF921FB60000000, %90
  %92 = fsub float 0x3FF921FB60000000, %91
  %93 = fmul float %92, %92
  %94 = and i32 %81, 1
  %95 = icmp eq i32 %94, 0
  %96 = fsub float -0.000000e+00, %92
  %97 = select i1 %95, float %92, float %96
  %98 = fmul float %93, 0x3EC5E150E0000000
  %99 = fadd float %98, 0xBF29F75D60000000
  %100 = fmul float %93, %99
  %101 = fadd float %100, 0x3F8110EEE0000000
  %102 = fmul float %93, %101
  %103 = fadd float %102, 0xBFC55554C0000000
  %104 = fmul float %97, %103
  %105 = fmul float %93, %104
  %106 = fadd float %97, %105
  %107 = tail call float @fabsf(float %106) #13
  %108 = fcmp ogt float %107, 1.000000e+00
  %109 = select i1 %108, float 0.000000e+00, float %106
  %110 = getelementptr inbounds i8, i8* %0, i64 8
  %111 = bitcast i8* %110 to float*
  store float %109, float* %111, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sin_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fmul float %5, 0x3FD45F3060000000
  %7 = tail call float @copysignf(float 5.000000e-01, float %6) #13
  %8 = fadd float %6, %7
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fmul float %10, 3.140625e+00
  %12 = fsub float %5, %11
  %13 = fmul float %10, 0x3F4FB40000000000
  %14 = fsub float %12, %13
  %15 = fmul float %10, 0x3E84440000000000
  %16 = fsub float %14, %15
  %17 = fmul float %10, 0x3D968C2340000000
  %18 = fsub float %16, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fsub float 0x3FF921FB60000000, %19
  %21 = fmul float %20, %20
  %22 = and i32 %9, 1
  %23 = icmp ne i32 %22, 0
  %24 = fsub float -0.000000e+00, %20
  %25 = select i1 %23, float %24, float %20
  %26 = fmul float %21, 0x3EC5E150E0000000
  %27 = fadd float %26, 0xBF29F75D60000000
  %28 = fmul float %21, %27
  %29 = fadd float %28, 0x3F8110EEE0000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 0xBFC55554C0000000
  %32 = fmul float %25, %31
  %33 = fmul float %21, %32
  %34 = fadd float %25, %33
  %35 = fmul float %21, 0x3E923DB120000000
  %36 = fsub float 0x3EFA00F160000000, %35
  %37 = fmul float %21, %36
  %38 = fadd float %37, 0xBF56C16B00000000
  %39 = fmul float %21, %38
  %40 = fadd float %39, 0x3FA5555540000000
  %41 = fmul float %21, %40
  %42 = fadd float %41, -5.000000e-01
  %43 = fmul float %21, %42
  %44 = fadd float %43, 1.000000e+00
  %45 = fsub float -0.000000e+00, %44
  %46 = select i1 %23, float %45, float %44
  %47 = tail call float @fabsf(float %34) #13
  %48 = fcmp ogt float %47, 1.000000e+00
  %49 = tail call float @fabsf(float %46) #13
  %50 = fcmp ogt float %49, 1.000000e+00
  %51 = select i1 %50, float 0.000000e+00, float %46
  %52 = getelementptr inbounds i8, i8* %1, i64 4
  %53 = getelementptr inbounds i8, i8* %1, i64 28
  %54 = bitcast i8* %52 to float*
  %55 = load float, float* %54, align 4, !tbaa !1
  %56 = bitcast i8* %53 to float*
  %57 = load float, float* %56, align 4, !tbaa !1
  %58 = fmul float %55, 0x3FD45F3060000000
  %59 = tail call float @copysignf(float 5.000000e-01, float %58) #13
  %60 = fadd float %58, %59
  %61 = fptosi float %60 to i32
  %62 = sitofp i32 %61 to float
  %63 = fmul float %62, 3.140625e+00
  %64 = fsub float %55, %63
  %65 = fmul float %62, 0x3F4FB40000000000
  %66 = fsub float %64, %65
  %67 = fmul float %62, 0x3E84440000000000
  %68 = fsub float %66, %67
  %69 = fmul float %62, 0x3D968C2340000000
  %70 = fsub float %68, %69
  %71 = fsub float 0x3FF921FB60000000, %70
  %72 = fsub float 0x3FF921FB60000000, %71
  %73 = fmul float %72, %72
  %74 = and i32 %61, 1
  %75 = icmp ne i32 %74, 0
  %76 = fsub float -0.000000e+00, %72
  %77 = select i1 %75, float %76, float %72
  %78 = fmul float %73, 0x3EC5E150E0000000
  %79 = fadd float %78, 0xBF29F75D60000000
  %80 = fmul float %73, %79
  %81 = fadd float %80, 0x3F8110EEE0000000
  %82 = fmul float %73, %81
  %83 = fadd float %82, 0xBFC55554C0000000
  %84 = fmul float %77, %83
  %85 = fmul float %73, %84
  %86 = fadd float %77, %85
  %87 = fmul float %73, 0x3E923DB120000000
  %88 = fsub float 0x3EFA00F160000000, %87
  %89 = fmul float %73, %88
  %90 = fadd float %89, 0xBF56C16B00000000
  %91 = fmul float %73, %90
  %92 = fadd float %91, 0x3FA5555540000000
  %93 = fmul float %73, %92
  %94 = fadd float %93, -5.000000e-01
  %95 = fmul float %73, %94
  %96 = fadd float %95, 1.000000e+00
  %97 = fsub float -0.000000e+00, %96
  %98 = select i1 %75, float %97, float %96
  %99 = tail call float @fabsf(float %86) #13
  %100 = fcmp ogt float %99, 1.000000e+00
  %101 = tail call float @fabsf(float %98) #13
  %102 = fcmp ogt float %101, 1.000000e+00
  %103 = select i1 %102, float 0.000000e+00, float %98
  %104 = fmul float %57, %103
  %105 = getelementptr inbounds i8, i8* %1, i64 8
  %106 = getelementptr inbounds i8, i8* %1, i64 32
  %107 = bitcast i8* %105 to float*
  %108 = load float, float* %107, align 4, !tbaa !1
  %109 = bitcast i8* %3 to <4 x float>*
  %110 = load <4 x float>, <4 x float>* %109, align 4, !tbaa !1
  %111 = bitcast i8* %106 to float*
  %112 = load float, float* %111, align 4, !tbaa !1
  %113 = fmul float %108, 0x3FD45F3060000000
  %114 = tail call float @copysignf(float 5.000000e-01, float %113) #13
  %115 = fadd float %113, %114
  %116 = fptosi float %115 to i32
  %117 = sitofp i32 %116 to float
  %118 = fmul float %117, 3.140625e+00
  %119 = fsub float %108, %118
  %120 = fmul float %117, 0x3F4FB40000000000
  %121 = fsub float %119, %120
  %122 = fmul float %117, 0x3E84440000000000
  %123 = fsub float %121, %122
  %124 = fmul float %117, 0x3D968C2340000000
  %125 = fsub float %123, %124
  %126 = fsub float 0x3FF921FB60000000, %125
  %127 = fsub float 0x3FF921FB60000000, %126
  %128 = fmul float %127, %127
  %129 = and i32 %116, 1
  %130 = icmp ne i32 %129, 0
  %131 = fsub float -0.000000e+00, %127
  %132 = select i1 %130, float %131, float %127
  %133 = fmul float %128, 0x3EC5E150E0000000
  %134 = fadd float %133, 0xBF29F75D60000000
  %135 = fmul float %128, %134
  %136 = fadd float %135, 0x3F8110EEE0000000
  %137 = fmul float %128, %136
  %138 = fadd float %137, 0xBFC55554C0000000
  %139 = fmul float %132, %138
  %140 = fmul float %128, %139
  %141 = fadd float %132, %140
  %142 = fmul float %128, 0x3E923DB120000000
  %143 = fsub float 0x3EFA00F160000000, %142
  %144 = fmul float %128, %143
  %145 = fadd float %144, 0xBF56C16B00000000
  %146 = fmul float %128, %145
  %147 = fadd float %146, 0x3FA5555540000000
  %148 = fmul float %128, %147
  %149 = fadd float %148, -5.000000e-01
  %150 = fmul float %128, %149
  %151 = fadd float %150, 1.000000e+00
  %152 = fsub float -0.000000e+00, %151
  %153 = select i1 %130, float %152, float %151
  %154 = tail call float @fabsf(float %141) #13
  %155 = fcmp ogt float %154, 1.000000e+00
  %156 = tail call float @fabsf(float %153) #13
  %157 = fcmp ogt float %156, 1.000000e+00
  %158 = select i1 %157, float 0.000000e+00, float %153
  %159 = insertelement <4 x float> undef, float %51, i32 0
  %160 = insertelement <4 x float> %159, float %103, i32 1
  %161 = insertelement <4 x float> %160, float %158, i32 2
  %162 = insertelement <4 x float> %161, float %51, i32 3
  %163 = fmul <4 x float> %110, %162
  %164 = fmul float %112, %158
  %165 = bitcast float %34 to i32
  %166 = select i1 %48, i32 0, i32 %165
  %167 = bitcast float %86 to i32
  %168 = select i1 %100, i32 0, i32 %167
  %169 = bitcast float %141 to i32
  %170 = select i1 %155, i32 0, i32 %169
  %171 = bitcast i8* %0 to i32*
  store i32 %166, i32* %171, align 4, !tbaa !5
  %172 = getelementptr inbounds i8, i8* %0, i64 4
  %173 = bitcast i8* %172 to i32*
  store i32 %168, i32* %173, align 4, !tbaa !7
  %174 = getelementptr inbounds i8, i8* %0, i64 8
  %175 = bitcast i8* %174 to i32*
  store i32 %170, i32* %175, align 4, !tbaa !8
  %176 = getelementptr inbounds i8, i8* %0, i64 12
  %177 = bitcast i8* %176 to <4 x float>*
  store <4 x float> %163, <4 x float>* %177, align 4, !tbaa !1
  %178 = getelementptr inbounds i8, i8* %0, i64 28
  %179 = bitcast i8* %178 to float*
  store float %104, float* %179, align 4, !tbaa !7
  %180 = getelementptr inbounds i8, i8* %0, i64 32
  %181 = bitcast i8* %180 to float*
  store float %164, float* %181, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_cos_ff(float) local_unnamed_addr #3 {
  %2 = fmul float %0, 0x3FD45F3060000000
  %3 = tail call float @copysignf(float 5.000000e-01, float %2) #13
  %4 = fadd float %2, %3
  %5 = fptosi float %4 to i32
  %6 = sitofp i32 %5 to float
  %7 = fmul float %6, 3.140625e+00
  %8 = fsub float %0, %7
  %9 = fmul float %6, 0x3F4FB40000000000
  %10 = fsub float %8, %9
  %11 = fmul float %6, 0x3E84440000000000
  %12 = fsub float %10, %11
  %13 = fmul float %6, 0x3D968C2340000000
  %14 = fsub float %12, %13
  %15 = fsub float 0x3FF921FB60000000, %14
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = fmul float %16, %16
  %18 = fmul float %17, 0x3E923DB120000000
  %19 = fsub float 0x3EFA00F160000000, %18
  %20 = fmul float %17, %19
  %21 = fadd float %20, 0xBF56C16B00000000
  %22 = fmul float %17, %21
  %23 = fadd float %22, 0x3FA5555540000000
  %24 = fmul float %17, %23
  %25 = fadd float %24, -5.000000e-01
  %26 = fmul float %17, %25
  %27 = fadd float %26, 1.000000e+00
  %28 = and i32 %5, 1
  %29 = icmp eq i32 %28, 0
  %30 = fsub float -0.000000e+00, %27
  %31 = select i1 %29, float %27, float %30
  %32 = tail call float @fabsf(float %31) #13
  %33 = fcmp ogt float %32, 1.000000e+00
  %34 = select i1 %33, float 0.000000e+00, float %31
  ret float %34
}

; Function Attrs: nounwind uwtable
define void @osl_cos_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fmul float %4, 0x3FD45F3060000000
  %6 = tail call float @copysignf(float 5.000000e-01, float %5) #13
  %7 = fadd float %5, %6
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fmul float %9, 3.140625e+00
  %11 = fsub float %4, %10
  %12 = fmul float %9, 0x3F4FB40000000000
  %13 = fsub float %11, %12
  %14 = fmul float %9, 0x3E84440000000000
  %15 = fsub float %13, %14
  %16 = fmul float %9, 0x3D968C2340000000
  %17 = fsub float %15, %16
  %18 = fsub float 0x3FF921FB60000000, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fmul float %19, %19
  %21 = and i32 %8, 1
  %22 = icmp ne i32 %21, 0
  %23 = fsub float -0.000000e+00, %19
  %24 = select i1 %22, float %23, float %19
  %25 = fmul float %20, 0x3EC5E150E0000000
  %26 = fadd float %25, 0xBF29F75D60000000
  %27 = fmul float %20, %26
  %28 = fadd float %27, 0x3F8110EEE0000000
  %29 = fmul float %20, %28
  %30 = fadd float %29, 0xBFC55554C0000000
  %31 = fmul float %24, %30
  %32 = fmul float %20, %31
  %33 = fadd float %24, %32
  %34 = fmul float %20, 0x3E923DB120000000
  %35 = fsub float 0x3EFA00F160000000, %34
  %36 = fmul float %20, %35
  %37 = fadd float %36, 0xBF56C16B00000000
  %38 = fmul float %20, %37
  %39 = fadd float %38, 0x3FA5555540000000
  %40 = fmul float %20, %39
  %41 = fadd float %40, -5.000000e-01
  %42 = fmul float %20, %41
  %43 = fadd float %42, 1.000000e+00
  %44 = fsub float -0.000000e+00, %43
  %45 = select i1 %22, float %44, float %43
  %46 = tail call float @fabsf(float %33) #13
  %47 = fcmp ogt float %46, 1.000000e+00
  %48 = select i1 %47, float 0.000000e+00, float %33
  %49 = tail call float @fabsf(float %45) #13
  %50 = fcmp ogt float %49, 1.000000e+00
  %51 = getelementptr inbounds i8, i8* %1, i64 4
  %52 = bitcast i8* %51 to float*
  %53 = load float, float* %52, align 4, !tbaa !1
  %54 = fmul float %53, %48
  %55 = fsub float -0.000000e+00, %54
  %56 = getelementptr inbounds i8, i8* %1, i64 8
  %57 = bitcast i8* %56 to float*
  %58 = load float, float* %57, align 4, !tbaa !1
  %59 = fmul float %58, %48
  %60 = fsub float -0.000000e+00, %59
  %61 = select i1 %50, float 0.000000e+00, float %45
  %62 = insertelement <2 x float> undef, float %61, i32 0
  %63 = insertelement <2 x float> %62, float %55, i32 1
  %64 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %63, <2 x float>* %64, align 4
  %65 = getelementptr inbounds i8, i8* %0, i64 8
  %66 = bitcast i8* %65 to float*
  store float %60, float* %66, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cos_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fmul float %4, 0x3FD45F3060000000
  %6 = tail call float @copysignf(float 5.000000e-01, float %5) #13
  %7 = fadd float %5, %6
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fmul float %9, 3.140625e+00
  %11 = fsub float %4, %10
  %12 = fmul float %9, 0x3F4FB40000000000
  %13 = fsub float %11, %12
  %14 = fmul float %9, 0x3E84440000000000
  %15 = fsub float %13, %14
  %16 = fmul float %9, 0x3D968C2340000000
  %17 = fsub float %15, %16
  %18 = fsub float 0x3FF921FB60000000, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fmul float %19, %19
  %21 = fmul float %20, 0x3E923DB120000000
  %22 = fsub float 0x3EFA00F160000000, %21
  %23 = fmul float %20, %22
  %24 = fadd float %23, 0xBF56C16B00000000
  %25 = fmul float %20, %24
  %26 = fadd float %25, 0x3FA5555540000000
  %27 = fmul float %20, %26
  %28 = fadd float %27, -5.000000e-01
  %29 = fmul float %20, %28
  %30 = fadd float %29, 1.000000e+00
  %31 = and i32 %8, 1
  %32 = icmp eq i32 %31, 0
  %33 = fsub float -0.000000e+00, %30
  %34 = select i1 %32, float %30, float %33
  %35 = tail call float @fabsf(float %34) #13
  %36 = fcmp ogt float %35, 1.000000e+00
  %37 = select i1 %36, float 0.000000e+00, float %34
  %38 = bitcast i8* %0 to float*
  store float %37, float* %38, align 4, !tbaa !1
  %39 = getelementptr inbounds i8, i8* %1, i64 4
  %40 = bitcast i8* %39 to float*
  %41 = load float, float* %40, align 4, !tbaa !1
  %42 = fmul float %41, 0x3FD45F3060000000
  %43 = tail call float @copysignf(float 5.000000e-01, float %42) #13
  %44 = fadd float %42, %43
  %45 = fptosi float %44 to i32
  %46 = sitofp i32 %45 to float
  %47 = fmul float %46, 3.140625e+00
  %48 = fsub float %41, %47
  %49 = fmul float %46, 0x3F4FB40000000000
  %50 = fsub float %48, %49
  %51 = fmul float %46, 0x3E84440000000000
  %52 = fsub float %50, %51
  %53 = fmul float %46, 0x3D968C2340000000
  %54 = fsub float %52, %53
  %55 = fsub float 0x3FF921FB60000000, %54
  %56 = fsub float 0x3FF921FB60000000, %55
  %57 = fmul float %56, %56
  %58 = fmul float %57, 0x3E923DB120000000
  %59 = fsub float 0x3EFA00F160000000, %58
  %60 = fmul float %57, %59
  %61 = fadd float %60, 0xBF56C16B00000000
  %62 = fmul float %57, %61
  %63 = fadd float %62, 0x3FA5555540000000
  %64 = fmul float %57, %63
  %65 = fadd float %64, -5.000000e-01
  %66 = fmul float %57, %65
  %67 = fadd float %66, 1.000000e+00
  %68 = and i32 %45, 1
  %69 = icmp eq i32 %68, 0
  %70 = fsub float -0.000000e+00, %67
  %71 = select i1 %69, float %67, float %70
  %72 = tail call float @fabsf(float %71) #13
  %73 = fcmp ogt float %72, 1.000000e+00
  %74 = select i1 %73, float 0.000000e+00, float %71
  %75 = getelementptr inbounds i8, i8* %0, i64 4
  %76 = bitcast i8* %75 to float*
  store float %74, float* %76, align 4, !tbaa !1
  %77 = getelementptr inbounds i8, i8* %1, i64 8
  %78 = bitcast i8* %77 to float*
  %79 = load float, float* %78, align 4, !tbaa !1
  %80 = fmul float %79, 0x3FD45F3060000000
  %81 = tail call float @copysignf(float 5.000000e-01, float %80) #13
  %82 = fadd float %80, %81
  %83 = fptosi float %82 to i32
  %84 = sitofp i32 %83 to float
  %85 = fmul float %84, 3.140625e+00
  %86 = fsub float %79, %85
  %87 = fmul float %84, 0x3F4FB40000000000
  %88 = fsub float %86, %87
  %89 = fmul float %84, 0x3E84440000000000
  %90 = fsub float %88, %89
  %91 = fmul float %84, 0x3D968C2340000000
  %92 = fsub float %90, %91
  %93 = fsub float 0x3FF921FB60000000, %92
  %94 = fsub float 0x3FF921FB60000000, %93
  %95 = fmul float %94, %94
  %96 = fmul float %95, 0x3E923DB120000000
  %97 = fsub float 0x3EFA00F160000000, %96
  %98 = fmul float %95, %97
  %99 = fadd float %98, 0xBF56C16B00000000
  %100 = fmul float %95, %99
  %101 = fadd float %100, 0x3FA5555540000000
  %102 = fmul float %95, %101
  %103 = fadd float %102, -5.000000e-01
  %104 = fmul float %95, %103
  %105 = fadd float %104, 1.000000e+00
  %106 = and i32 %83, 1
  %107 = icmp eq i32 %106, 0
  %108 = fsub float -0.000000e+00, %105
  %109 = select i1 %107, float %105, float %108
  %110 = tail call float @fabsf(float %109) #13
  %111 = fcmp ogt float %110, 1.000000e+00
  %112 = select i1 %111, float 0.000000e+00, float %109
  %113 = getelementptr inbounds i8, i8* %0, i64 8
  %114 = bitcast i8* %113 to float*
  store float %112, float* %114, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cos_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fmul float %5, 0x3FD45F3060000000
  %7 = tail call float @copysignf(float 5.000000e-01, float %6) #13
  %8 = fadd float %6, %7
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fmul float %10, 3.140625e+00
  %12 = fsub float %5, %11
  %13 = fmul float %10, 0x3F4FB40000000000
  %14 = fsub float %12, %13
  %15 = fmul float %10, 0x3E84440000000000
  %16 = fsub float %14, %15
  %17 = fmul float %10, 0x3D968C2340000000
  %18 = fsub float %16, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fsub float 0x3FF921FB60000000, %19
  %21 = fmul float %20, %20
  %22 = and i32 %9, 1
  %23 = icmp ne i32 %22, 0
  %24 = fsub float -0.000000e+00, %20
  %25 = select i1 %23, float %24, float %20
  %26 = fmul float %21, 0x3EC5E150E0000000
  %27 = fadd float %26, 0xBF29F75D60000000
  %28 = fmul float %21, %27
  %29 = fadd float %28, 0x3F8110EEE0000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 0xBFC55554C0000000
  %32 = fmul float %25, %31
  %33 = fmul float %21, %32
  %34 = fadd float %25, %33
  %35 = fmul float %21, 0x3E923DB120000000
  %36 = fsub float 0x3EFA00F160000000, %35
  %37 = fmul float %21, %36
  %38 = fadd float %37, 0xBF56C16B00000000
  %39 = fmul float %21, %38
  %40 = fadd float %39, 0x3FA5555540000000
  %41 = fmul float %21, %40
  %42 = fadd float %41, -5.000000e-01
  %43 = fmul float %21, %42
  %44 = fadd float %43, 1.000000e+00
  %45 = fsub float -0.000000e+00, %44
  %46 = select i1 %23, float %45, float %44
  %47 = tail call float @fabsf(float %34) #13
  %48 = fcmp ogt float %47, 1.000000e+00
  %49 = select i1 %48, float 0.000000e+00, float %34
  %50 = tail call float @fabsf(float %46) #13
  %51 = fcmp ogt float %50, 1.000000e+00
  %52 = getelementptr inbounds i8, i8* %1, i64 4
  %53 = getelementptr inbounds i8, i8* %1, i64 28
  %54 = bitcast i8* %52 to float*
  %55 = load float, float* %54, align 4, !tbaa !1
  %56 = bitcast i8* %53 to float*
  %57 = load float, float* %56, align 4, !tbaa !1
  %58 = fmul float %55, 0x3FD45F3060000000
  %59 = tail call float @copysignf(float 5.000000e-01, float %58) #13
  %60 = fadd float %58, %59
  %61 = fptosi float %60 to i32
  %62 = sitofp i32 %61 to float
  %63 = fmul float %62, 3.140625e+00
  %64 = fsub float %55, %63
  %65 = fmul float %62, 0x3F4FB40000000000
  %66 = fsub float %64, %65
  %67 = fmul float %62, 0x3E84440000000000
  %68 = fsub float %66, %67
  %69 = fmul float %62, 0x3D968C2340000000
  %70 = fsub float %68, %69
  %71 = fsub float 0x3FF921FB60000000, %70
  %72 = fsub float 0x3FF921FB60000000, %71
  %73 = fmul float %72, %72
  %74 = and i32 %61, 1
  %75 = icmp ne i32 %74, 0
  %76 = fsub float -0.000000e+00, %72
  %77 = select i1 %75, float %76, float %72
  %78 = fmul float %73, 0x3EC5E150E0000000
  %79 = fadd float %78, 0xBF29F75D60000000
  %80 = fmul float %73, %79
  %81 = fadd float %80, 0x3F8110EEE0000000
  %82 = fmul float %73, %81
  %83 = fadd float %82, 0xBFC55554C0000000
  %84 = fmul float %77, %83
  %85 = fmul float %73, %84
  %86 = fadd float %77, %85
  %87 = fmul float %73, 0x3E923DB120000000
  %88 = fsub float 0x3EFA00F160000000, %87
  %89 = fmul float %73, %88
  %90 = fadd float %89, 0xBF56C16B00000000
  %91 = fmul float %73, %90
  %92 = fadd float %91, 0x3FA5555540000000
  %93 = fmul float %73, %92
  %94 = fadd float %93, -5.000000e-01
  %95 = fmul float %73, %94
  %96 = fadd float %95, 1.000000e+00
  %97 = fsub float -0.000000e+00, %96
  %98 = select i1 %75, float %97, float %96
  %99 = tail call float @fabsf(float %86) #13
  %100 = fcmp ogt float %99, 1.000000e+00
  %101 = select i1 %100, float 0.000000e+00, float %86
  %102 = tail call float @fabsf(float %98) #13
  %103 = fcmp ogt float %102, 1.000000e+00
  %104 = fmul float %57, %101
  %105 = fsub float -0.000000e+00, %104
  %106 = getelementptr inbounds i8, i8* %1, i64 8
  %107 = getelementptr inbounds i8, i8* %1, i64 32
  %108 = bitcast i8* %106 to float*
  %109 = load float, float* %108, align 4, !tbaa !1
  %110 = bitcast i8* %3 to <4 x float>*
  %111 = load <4 x float>, <4 x float>* %110, align 4, !tbaa !1
  %112 = bitcast i8* %107 to float*
  %113 = load float, float* %112, align 4, !tbaa !1
  %114 = fmul float %109, 0x3FD45F3060000000
  %115 = tail call float @copysignf(float 5.000000e-01, float %114) #13
  %116 = fadd float %114, %115
  %117 = fptosi float %116 to i32
  %118 = sitofp i32 %117 to float
  %119 = fmul float %118, 3.140625e+00
  %120 = fsub float %109, %119
  %121 = fmul float %118, 0x3F4FB40000000000
  %122 = fsub float %120, %121
  %123 = fmul float %118, 0x3E84440000000000
  %124 = fsub float %122, %123
  %125 = fmul float %118, 0x3D968C2340000000
  %126 = fsub float %124, %125
  %127 = fsub float 0x3FF921FB60000000, %126
  %128 = fsub float 0x3FF921FB60000000, %127
  %129 = fmul float %128, %128
  %130 = and i32 %117, 1
  %131 = icmp ne i32 %130, 0
  %132 = fsub float -0.000000e+00, %128
  %133 = select i1 %131, float %132, float %128
  %134 = fmul float %129, 0x3EC5E150E0000000
  %135 = fadd float %134, 0xBF29F75D60000000
  %136 = fmul float %129, %135
  %137 = fadd float %136, 0x3F8110EEE0000000
  %138 = fmul float %129, %137
  %139 = fadd float %138, 0xBFC55554C0000000
  %140 = fmul float %133, %139
  %141 = fmul float %129, %140
  %142 = fadd float %133, %141
  %143 = fmul float %129, 0x3E923DB120000000
  %144 = fsub float 0x3EFA00F160000000, %143
  %145 = fmul float %129, %144
  %146 = fadd float %145, 0xBF56C16B00000000
  %147 = fmul float %129, %146
  %148 = fadd float %147, 0x3FA5555540000000
  %149 = fmul float %129, %148
  %150 = fadd float %149, -5.000000e-01
  %151 = fmul float %129, %150
  %152 = fadd float %151, 1.000000e+00
  %153 = fsub float -0.000000e+00, %152
  %154 = select i1 %131, float %153, float %152
  %155 = tail call float @fabsf(float %142) #13
  %156 = fcmp ogt float %155, 1.000000e+00
  %157 = select i1 %156, float 0.000000e+00, float %142
  %158 = tail call float @fabsf(float %154) #13
  %159 = fcmp ogt float %158, 1.000000e+00
  %160 = insertelement <4 x float> undef, float %49, i32 0
  %161 = insertelement <4 x float> %160, float %101, i32 1
  %162 = insertelement <4 x float> %161, float %157, i32 2
  %163 = insertelement <4 x float> %162, float %49, i32 3
  %164 = fmul <4 x float> %111, %163
  %165 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %164
  %166 = fmul float %113, %157
  %167 = fsub float -0.000000e+00, %166
  %168 = bitcast float %46 to i32
  %169 = select i1 %51, i32 0, i32 %168
  %170 = bitcast float %98 to i32
  %171 = select i1 %103, i32 0, i32 %170
  %172 = bitcast float %154 to i32
  %173 = select i1 %159, i32 0, i32 %172
  %174 = bitcast i8* %0 to i32*
  store i32 %169, i32* %174, align 4, !tbaa !5
  %175 = getelementptr inbounds i8, i8* %0, i64 4
  %176 = bitcast i8* %175 to i32*
  store i32 %171, i32* %176, align 4, !tbaa !7
  %177 = getelementptr inbounds i8, i8* %0, i64 8
  %178 = bitcast i8* %177 to i32*
  store i32 %173, i32* %178, align 4, !tbaa !8
  %179 = getelementptr inbounds i8, i8* %0, i64 12
  %180 = bitcast i8* %179 to <4 x float>*
  store <4 x float> %165, <4 x float>* %180, align 4, !tbaa !1
  %181 = getelementptr inbounds i8, i8* %0, i64 28
  %182 = bitcast i8* %181 to float*
  store float %105, float* %182, align 4, !tbaa !7
  %183 = getelementptr inbounds i8, i8* %0, i64 32
  %184 = bitcast i8* %183 to float*
  store float %167, float* %184, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_tan_ff(float) local_unnamed_addr #3 {
  %2 = fmul float %0, 0x3FE45F3060000000
  %3 = tail call float @copysignf(float 5.000000e-01, float %2) #13
  %4 = fadd float %2, %3
  %5 = fptosi float %4 to i32
  %6 = sitofp i32 %5 to float
  %7 = fmul float %6, 0x3FF9200000000000
  %8 = fsub float %0, %7
  %9 = fmul float %6, 0x3F3FB40000000000
  %10 = fsub float %8, %9
  %11 = fmul float %6, 0x3E74440000000000
  %12 = fsub float %10, %11
  %13 = fmul float %6, 0x3D868C2340000000
  %14 = fsub float %12, %13
  %15 = and i32 %5, 1
  %16 = icmp eq i32 %15, 0
  %17 = fsub float 0x3FE921FB60000000, %14
  %18 = fsub float 0x3FE921FB60000000, %17
  %19 = select i1 %16, float %18, float %14
  %20 = fmul float %19, %19
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
  %31 = fmul float %19, %30
  %32 = fmul float %20, %31
  %33 = fadd float %19, %32
  %34 = fdiv float -1.000000e+00, %33
  %35 = select i1 %16, float %33, float %34
  ret float %35
}

; Function Attrs: nounwind uwtable
define void @osl_tan_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fmul float %4, 0x3FE45F3060000000
  %6 = tail call float @copysignf(float 5.000000e-01, float %5) #13
  %7 = fadd float %5, %6
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fmul float %9, 0x3FF9200000000000
  %11 = fsub float %4, %10
  %12 = fmul float %9, 0x3F3FB40000000000
  %13 = fsub float %11, %12
  %14 = fmul float %9, 0x3E74440000000000
  %15 = fsub float %13, %14
  %16 = fmul float %9, 0x3D868C2340000000
  %17 = fsub float %15, %16
  %18 = and i32 %8, 1
  %19 = icmp eq i32 %18, 0
  %20 = fsub float 0x3FE921FB60000000, %17
  %21 = fsub float 0x3FE921FB60000000, %20
  %22 = select i1 %19, float %21, float %17
  %23 = fmul float %22, %22
  %24 = fmul float %23, 0x3F82FD7040000000
  %25 = fadd float %24, 0x3F6B323AE0000000
  %26 = fmul float %23, %25
  %27 = fadd float %26, 0x3F98E20C80000000
  %28 = fmul float %23, %27
  %29 = fadd float %28, 0x3FAB5DBCA0000000
  %30 = fmul float %23, %29
  %31 = fadd float %30, 0x3FC112B1C0000000
  %32 = fmul float %23, %31
  %33 = fadd float %32, 0x3FD5554F20000000
  %34 = fmul float %22, %33
  %35 = fmul float %23, %34
  %36 = fadd float %22, %35
  %37 = fdiv float -1.000000e+00, %36
  %38 = select i1 %19, float %36, float %37
  %39 = fmul float %4, 0x3FD45F3060000000
  %40 = tail call float @copysignf(float 5.000000e-01, float %39) #13
  %41 = fadd float %39, %40
  %42 = fptosi float %41 to i32
  %43 = sitofp i32 %42 to float
  %44 = fmul float %43, 3.140625e+00
  %45 = fsub float %4, %44
  %46 = fmul float %43, 0x3F4FB40000000000
  %47 = fsub float %45, %46
  %48 = fmul float %43, 0x3E84440000000000
  %49 = fsub float %47, %48
  %50 = fmul float %43, 0x3D968C2340000000
  %51 = fsub float %49, %50
  %52 = fsub float 0x3FF921FB60000000, %51
  %53 = fsub float 0x3FF921FB60000000, %52
  %54 = fmul float %53, %53
  %55 = fmul float %54, 0x3E923DB120000000
  %56 = fsub float 0x3EFA00F160000000, %55
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
  %67 = fsub float -0.000000e+00, %64
  %68 = select i1 %66, float %64, float %67
  %69 = tail call float @fabsf(float %68) #13
  %70 = fcmp ogt float %69, 1.000000e+00
  %71 = select i1 %70, float 0.000000e+00, float %68
  %72 = fmul float %71, %71
  %73 = fdiv float 1.000000e+00, %72
  %74 = getelementptr inbounds i8, i8* %1, i64 4
  %75 = bitcast i8* %74 to float*
  %76 = load float, float* %75, align 4, !tbaa !1
  %77 = fmul float %76, %73
  %78 = getelementptr inbounds i8, i8* %1, i64 8
  %79 = bitcast i8* %78 to float*
  %80 = load float, float* %79, align 4, !tbaa !1
  %81 = fmul float %80, %73
  %82 = insertelement <2 x float> undef, float %38, i32 0
  %83 = insertelement <2 x float> %82, float %77, i32 1
  %84 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %83, <2 x float>* %84, align 4
  %85 = getelementptr inbounds i8, i8* %0, i64 8
  %86 = bitcast i8* %85 to float*
  store float %81, float* %86, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_tan_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fmul float %4, 0x3FE45F3060000000
  %6 = tail call float @copysignf(float 5.000000e-01, float %5) #13
  %7 = fadd float %5, %6
  %8 = fptosi float %7 to i32
  %9 = sitofp i32 %8 to float
  %10 = fmul float %9, 0x3FF9200000000000
  %11 = fsub float %4, %10
  %12 = fmul float %9, 0x3F3FB40000000000
  %13 = fsub float %11, %12
  %14 = fmul float %9, 0x3E74440000000000
  %15 = fsub float %13, %14
  %16 = fmul float %9, 0x3D868C2340000000
  %17 = fsub float %15, %16
  %18 = and i32 %8, 1
  %19 = icmp eq i32 %18, 0
  %20 = fsub float 0x3FE921FB60000000, %17
  %21 = fsub float 0x3FE921FB60000000, %20
  %22 = select i1 %19, float %21, float %17
  %23 = fmul float %22, %22
  %24 = fmul float %23, 0x3F82FD7040000000
  %25 = fadd float %24, 0x3F6B323AE0000000
  %26 = fmul float %23, %25
  %27 = fadd float %26, 0x3F98E20C80000000
  %28 = fmul float %23, %27
  %29 = fadd float %28, 0x3FAB5DBCA0000000
  %30 = fmul float %23, %29
  %31 = fadd float %30, 0x3FC112B1C0000000
  %32 = fmul float %23, %31
  %33 = fadd float %32, 0x3FD5554F20000000
  %34 = fmul float %22, %33
  %35 = fmul float %23, %34
  %36 = fadd float %22, %35
  %37 = fdiv float -1.000000e+00, %36
  %38 = select i1 %19, float %36, float %37
  %39 = bitcast i8* %0 to float*
  store float %38, float* %39, align 4, !tbaa !1
  %40 = getelementptr inbounds i8, i8* %1, i64 4
  %41 = bitcast i8* %40 to float*
  %42 = load float, float* %41, align 4, !tbaa !1
  %43 = fmul float %42, 0x3FE45F3060000000
  %44 = tail call float @copysignf(float 5.000000e-01, float %43) #13
  %45 = fadd float %43, %44
  %46 = fptosi float %45 to i32
  %47 = sitofp i32 %46 to float
  %48 = fmul float %47, 0x3FF9200000000000
  %49 = fsub float %42, %48
  %50 = fmul float %47, 0x3F3FB40000000000
  %51 = fsub float %49, %50
  %52 = fmul float %47, 0x3E74440000000000
  %53 = fsub float %51, %52
  %54 = fmul float %47, 0x3D868C2340000000
  %55 = fsub float %53, %54
  %56 = and i32 %46, 1
  %57 = icmp eq i32 %56, 0
  %58 = fsub float 0x3FE921FB60000000, %55
  %59 = fsub float 0x3FE921FB60000000, %58
  %60 = select i1 %57, float %59, float %55
  %61 = fmul float %60, %60
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
  %72 = fmul float %60, %71
  %73 = fmul float %61, %72
  %74 = fadd float %60, %73
  %75 = fdiv float -1.000000e+00, %74
  %76 = select i1 %57, float %74, float %75
  %77 = getelementptr inbounds i8, i8* %0, i64 4
  %78 = bitcast i8* %77 to float*
  store float %76, float* %78, align 4, !tbaa !1
  %79 = getelementptr inbounds i8, i8* %1, i64 8
  %80 = bitcast i8* %79 to float*
  %81 = load float, float* %80, align 4, !tbaa !1
  %82 = fmul float %81, 0x3FE45F3060000000
  %83 = tail call float @copysignf(float 5.000000e-01, float %82) #13
  %84 = fadd float %82, %83
  %85 = fptosi float %84 to i32
  %86 = sitofp i32 %85 to float
  %87 = fmul float %86, 0x3FF9200000000000
  %88 = fsub float %81, %87
  %89 = fmul float %86, 0x3F3FB40000000000
  %90 = fsub float %88, %89
  %91 = fmul float %86, 0x3E74440000000000
  %92 = fsub float %90, %91
  %93 = fmul float %86, 0x3D868C2340000000
  %94 = fsub float %92, %93
  %95 = and i32 %85, 1
  %96 = icmp eq i32 %95, 0
  %97 = fsub float 0x3FE921FB60000000, %94
  %98 = fsub float 0x3FE921FB60000000, %97
  %99 = select i1 %96, float %98, float %94
  %100 = fmul float %99, %99
  %101 = fmul float %100, 0x3F82FD7040000000
  %102 = fadd float %101, 0x3F6B323AE0000000
  %103 = fmul float %100, %102
  %104 = fadd float %103, 0x3F98E20C80000000
  %105 = fmul float %100, %104
  %106 = fadd float %105, 0x3FAB5DBCA0000000
  %107 = fmul float %100, %106
  %108 = fadd float %107, 0x3FC112B1C0000000
  %109 = fmul float %100, %108
  %110 = fadd float %109, 0x3FD5554F20000000
  %111 = fmul float %99, %110
  %112 = fmul float %100, %111
  %113 = fadd float %99, %112
  %114 = fdiv float -1.000000e+00, %113
  %115 = select i1 %96, float %113, float %114
  %116 = getelementptr inbounds i8, i8* %0, i64 8
  %117 = bitcast i8* %116 to float*
  store float %115, float* %117, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_tan_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fmul float %5, 0x3FE45F3060000000
  %7 = tail call float @copysignf(float 5.000000e-01, float %6) #13
  %8 = fadd float %6, %7
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fmul float %10, 0x3FF9200000000000
  %12 = fsub float %5, %11
  %13 = fmul float %10, 0x3F3FB40000000000
  %14 = fsub float %12, %13
  %15 = fmul float %10, 0x3E74440000000000
  %16 = fsub float %14, %15
  %17 = fmul float %10, 0x3D868C2340000000
  %18 = fsub float %16, %17
  %19 = and i32 %9, 1
  %20 = icmp eq i32 %19, 0
  %21 = fsub float 0x3FE921FB60000000, %18
  %22 = fsub float 0x3FE921FB60000000, %21
  %23 = select i1 %20, float %22, float %18
  %24 = fmul float %23, %23
  %25 = fmul float %24, 0x3F82FD7040000000
  %26 = fadd float %25, 0x3F6B323AE0000000
  %27 = fmul float %24, %26
  %28 = fadd float %27, 0x3F98E20C80000000
  %29 = fmul float %24, %28
  %30 = fadd float %29, 0x3FAB5DBCA0000000
  %31 = fmul float %24, %30
  %32 = fadd float %31, 0x3FC112B1C0000000
  %33 = fmul float %24, %32
  %34 = fadd float %33, 0x3FD5554F20000000
  %35 = fmul float %23, %34
  %36 = fmul float %24, %35
  %37 = fadd float %23, %36
  %38 = fdiv float -1.000000e+00, %37
  %39 = select i1 %20, float %37, float %38
  %40 = fmul float %5, 0x3FD45F3060000000
  %41 = tail call float @copysignf(float 5.000000e-01, float %40) #13
  %42 = fadd float %40, %41
  %43 = fptosi float %42 to i32
  %44 = sitofp i32 %43 to float
  %45 = fmul float %44, 3.140625e+00
  %46 = fsub float %5, %45
  %47 = fmul float %44, 0x3F4FB40000000000
  %48 = fsub float %46, %47
  %49 = fmul float %44, 0x3E84440000000000
  %50 = fsub float %48, %49
  %51 = fmul float %44, 0x3D968C2340000000
  %52 = fsub float %50, %51
  %53 = fsub float 0x3FF921FB60000000, %52
  %54 = fsub float 0x3FF921FB60000000, %53
  %55 = fmul float %54, %54
  %56 = fmul float %55, 0x3E923DB120000000
  %57 = fsub float 0x3EFA00F160000000, %56
  %58 = fmul float %55, %57
  %59 = fadd float %58, 0xBF56C16B00000000
  %60 = fmul float %55, %59
  %61 = fadd float %60, 0x3FA5555540000000
  %62 = fmul float %55, %61
  %63 = fadd float %62, -5.000000e-01
  %64 = fmul float %55, %63
  %65 = fadd float %64, 1.000000e+00
  %66 = and i32 %43, 1
  %67 = icmp eq i32 %66, 0
  %68 = fsub float -0.000000e+00, %65
  %69 = select i1 %67, float %65, float %68
  %70 = tail call float @fabsf(float %69) #13
  %71 = fcmp ogt float %70, 1.000000e+00
  %72 = select i1 %71, float 0.000000e+00, float %69
  %73 = fmul float %72, %72
  %74 = fdiv float 1.000000e+00, %73
  %75 = getelementptr inbounds i8, i8* %1, i64 4
  %76 = getelementptr inbounds i8, i8* %1, i64 28
  %77 = bitcast i8* %75 to float*
  %78 = load float, float* %77, align 4, !tbaa !1
  %79 = bitcast i8* %76 to float*
  %80 = load float, float* %79, align 4, !tbaa !1
  %81 = fmul float %78, 0x3FE45F3060000000
  %82 = tail call float @copysignf(float 5.000000e-01, float %81) #13
  %83 = fadd float %81, %82
  %84 = fptosi float %83 to i32
  %85 = sitofp i32 %84 to float
  %86 = fmul float %85, 0x3FF9200000000000
  %87 = fsub float %78, %86
  %88 = fmul float %85, 0x3F3FB40000000000
  %89 = fsub float %87, %88
  %90 = fmul float %85, 0x3E74440000000000
  %91 = fsub float %89, %90
  %92 = fmul float %85, 0x3D868C2340000000
  %93 = fsub float %91, %92
  %94 = and i32 %84, 1
  %95 = icmp eq i32 %94, 0
  %96 = fsub float 0x3FE921FB60000000, %93
  %97 = fsub float 0x3FE921FB60000000, %96
  %98 = select i1 %95, float %97, float %93
  %99 = fmul float %98, %98
  %100 = fmul float %99, 0x3F82FD7040000000
  %101 = fadd float %100, 0x3F6B323AE0000000
  %102 = fmul float %99, %101
  %103 = fadd float %102, 0x3F98E20C80000000
  %104 = fmul float %99, %103
  %105 = fadd float %104, 0x3FAB5DBCA0000000
  %106 = fmul float %99, %105
  %107 = fadd float %106, 0x3FC112B1C0000000
  %108 = fmul float %99, %107
  %109 = fadd float %108, 0x3FD5554F20000000
  %110 = fmul float %98, %109
  %111 = fmul float %99, %110
  %112 = fadd float %98, %111
  %113 = fdiv float -1.000000e+00, %112
  %114 = select i1 %95, float %112, float %113
  %115 = fmul float %78, 0x3FD45F3060000000
  %116 = tail call float @copysignf(float 5.000000e-01, float %115) #13
  %117 = fadd float %115, %116
  %118 = fptosi float %117 to i32
  %119 = sitofp i32 %118 to float
  %120 = fmul float %119, 3.140625e+00
  %121 = fsub float %78, %120
  %122 = fmul float %119, 0x3F4FB40000000000
  %123 = fsub float %121, %122
  %124 = fmul float %119, 0x3E84440000000000
  %125 = fsub float %123, %124
  %126 = fmul float %119, 0x3D968C2340000000
  %127 = fsub float %125, %126
  %128 = fsub float 0x3FF921FB60000000, %127
  %129 = fsub float 0x3FF921FB60000000, %128
  %130 = fmul float %129, %129
  %131 = fmul float %130, 0x3E923DB120000000
  %132 = fsub float 0x3EFA00F160000000, %131
  %133 = fmul float %130, %132
  %134 = fadd float %133, 0xBF56C16B00000000
  %135 = fmul float %130, %134
  %136 = fadd float %135, 0x3FA5555540000000
  %137 = fmul float %130, %136
  %138 = fadd float %137, -5.000000e-01
  %139 = fmul float %130, %138
  %140 = fadd float %139, 1.000000e+00
  %141 = and i32 %118, 1
  %142 = icmp eq i32 %141, 0
  %143 = fsub float -0.000000e+00, %140
  %144 = select i1 %142, float %140, float %143
  %145 = tail call float @fabsf(float %144) #13
  %146 = fcmp ogt float %145, 1.000000e+00
  %147 = select i1 %146, float 0.000000e+00, float %144
  %148 = fmul float %147, %147
  %149 = fdiv float 1.000000e+00, %148
  %150 = fmul float %80, %149
  %151 = getelementptr inbounds i8, i8* %1, i64 8
  %152 = getelementptr inbounds i8, i8* %1, i64 32
  %153 = bitcast i8* %151 to float*
  %154 = load float, float* %153, align 4, !tbaa !1
  %155 = bitcast i8* %3 to <4 x float>*
  %156 = load <4 x float>, <4 x float>* %155, align 4, !tbaa !1
  %157 = bitcast i8* %152 to float*
  %158 = load float, float* %157, align 4, !tbaa !1
  %159 = fmul float %154, 0x3FE45F3060000000
  %160 = tail call float @copysignf(float 5.000000e-01, float %159) #13
  %161 = fadd float %159, %160
  %162 = fptosi float %161 to i32
  %163 = sitofp i32 %162 to float
  %164 = fmul float %163, 0x3FF9200000000000
  %165 = fsub float %154, %164
  %166 = fmul float %163, 0x3F3FB40000000000
  %167 = fsub float %165, %166
  %168 = fmul float %163, 0x3E74440000000000
  %169 = fsub float %167, %168
  %170 = fmul float %163, 0x3D868C2340000000
  %171 = fsub float %169, %170
  %172 = and i32 %162, 1
  %173 = icmp eq i32 %172, 0
  %174 = fsub float 0x3FE921FB60000000, %171
  %175 = fsub float 0x3FE921FB60000000, %174
  %176 = select i1 %173, float %175, float %171
  %177 = fmul float %176, %176
  %178 = fmul float %177, 0x3F82FD7040000000
  %179 = fadd float %178, 0x3F6B323AE0000000
  %180 = fmul float %177, %179
  %181 = fadd float %180, 0x3F98E20C80000000
  %182 = fmul float %177, %181
  %183 = fadd float %182, 0x3FAB5DBCA0000000
  %184 = fmul float %177, %183
  %185 = fadd float %184, 0x3FC112B1C0000000
  %186 = fmul float %177, %185
  %187 = fadd float %186, 0x3FD5554F20000000
  %188 = fmul float %176, %187
  %189 = fmul float %177, %188
  %190 = fadd float %176, %189
  %191 = fdiv float -1.000000e+00, %190
  %192 = select i1 %173, float %190, float %191
  %193 = fmul float %154, 0x3FD45F3060000000
  %194 = tail call float @copysignf(float 5.000000e-01, float %193) #13
  %195 = fadd float %193, %194
  %196 = fptosi float %195 to i32
  %197 = sitofp i32 %196 to float
  %198 = fmul float %197, 3.140625e+00
  %199 = fsub float %154, %198
  %200 = fmul float %197, 0x3F4FB40000000000
  %201 = fsub float %199, %200
  %202 = fmul float %197, 0x3E84440000000000
  %203 = fsub float %201, %202
  %204 = fmul float %197, 0x3D968C2340000000
  %205 = fsub float %203, %204
  %206 = fsub float 0x3FF921FB60000000, %205
  %207 = fsub float 0x3FF921FB60000000, %206
  %208 = fmul float %207, %207
  %209 = fmul float %208, 0x3E923DB120000000
  %210 = fsub float 0x3EFA00F160000000, %209
  %211 = fmul float %208, %210
  %212 = fadd float %211, 0xBF56C16B00000000
  %213 = fmul float %208, %212
  %214 = fadd float %213, 0x3FA5555540000000
  %215 = fmul float %208, %214
  %216 = fadd float %215, -5.000000e-01
  %217 = fmul float %208, %216
  %218 = fadd float %217, 1.000000e+00
  %219 = and i32 %196, 1
  %220 = icmp eq i32 %219, 0
  %221 = fsub float -0.000000e+00, %218
  %222 = select i1 %220, float %218, float %221
  %223 = tail call float @fabsf(float %222) #13
  %224 = fcmp ogt float %223, 1.000000e+00
  %225 = select i1 %224, float 0.000000e+00, float %222
  %226 = fmul float %225, %225
  %227 = fdiv float 1.000000e+00, %226
  %228 = insertelement <4 x float> undef, float %74, i32 0
  %229 = insertelement <4 x float> %228, float %149, i32 1
  %230 = insertelement <4 x float> %229, float %227, i32 2
  %231 = insertelement <4 x float> %230, float %74, i32 3
  %232 = fmul <4 x float> %156, %231
  %233 = fmul float %158, %227
  %234 = bitcast i8* %0 to float*
  store float %39, float* %234, align 4, !tbaa !5
  %235 = getelementptr inbounds i8, i8* %0, i64 4
  %236 = bitcast i8* %235 to float*
  store float %114, float* %236, align 4, !tbaa !7
  %237 = getelementptr inbounds i8, i8* %0, i64 8
  %238 = bitcast i8* %237 to float*
  store float %192, float* %238, align 4, !tbaa !8
  %239 = getelementptr inbounds i8, i8* %0, i64 12
  %240 = bitcast i8* %239 to <4 x float>*
  store <4 x float> %232, <4 x float>* %240, align 4, !tbaa !1
  %241 = getelementptr inbounds i8, i8* %0, i64 28
  %242 = bitcast i8* %241 to float*
  store float %150, float* %242, align 4, !tbaa !7
  %243 = getelementptr inbounds i8, i8* %0, i64 32
  %244 = bitcast i8* %243 to float*
  store float %233, float* %244, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_asin_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fcmp olt float %2, 1.000000e+00
  %4 = fsub float 1.000000e+00, %2
  %5 = fsub float 1.000000e+00, %4
  %6 = select i1 %3, float %5, float 1.000000e+00
  %7 = fsub float 1.000000e+00, %6
  %8 = tail call float @sqrtf(float %7) #13
  %9 = fmul float %6, 0x3F96290BA0000000
  %10 = fsub float 0x3FB3F68760000000, %9
  %11 = fmul float %6, %10
  %12 = fadd float %11, 0xBFCB4D7260000000
  %13 = fmul float %6, %12
  %14 = fadd float %13, 0x3FF921FB60000000
  %15 = fmul float %8, %14
  %16 = fsub float 0x3FF921FB60000000, %15
  %17 = tail call float @copysignf(float %16, float %0) #13
  ret float %17
}

; Function Attrs: nounwind uwtable
define void @osl_asin_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp olt float %5, 1.000000e+00
  %7 = fsub float 1.000000e+00, %5
  %8 = fsub float 1.000000e+00, %7
  %9 = select i1 %6, float %8, float 1.000000e+00
  %10 = fsub float 1.000000e+00, %9
  %11 = tail call float @sqrtf(float %10) #13
  %12 = fmul float %9, 0x3F96290BA0000000
  %13 = fsub float 0x3FB3F68760000000, %12
  %14 = fmul float %9, %13
  %15 = fadd float %14, 0xBFCB4D7260000000
  %16 = fmul float %9, %15
  %17 = fadd float %16, 0x3FF921FB60000000
  %18 = fmul float %11, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = tail call float @copysignf(float %19, float %4) #13
  br i1 %6, label %21, label %26

; <label>:21:                                     ; preds = %2
  %22 = fmul float %4, %4
  %23 = fsub float 1.000000e+00, %22
  %24 = tail call float @sqrtf(float %23) #13
  %25 = fdiv float 1.000000e+00, %24
  br label %26

; <label>:26:                                     ; preds = %2, %21
  %27 = phi float [ %25, %21 ], [ 0.000000e+00, %2 ]
  %28 = getelementptr inbounds i8, i8* %1, i64 4
  %29 = bitcast i8* %28 to float*
  %30 = load float, float* %29, align 4, !tbaa !1
  %31 = fmul float %27, %30
  %32 = getelementptr inbounds i8, i8* %1, i64 8
  %33 = bitcast i8* %32 to float*
  %34 = load float, float* %33, align 4, !tbaa !1
  %35 = fmul float %27, %34
  %36 = insertelement <2 x float> undef, float %20, i32 0
  %37 = insertelement <2 x float> %36, float %31, i32 1
  %38 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %37, <2 x float>* %38, align 4
  %39 = getelementptr inbounds i8, i8* %0, i64 8
  %40 = bitcast i8* %39 to float*
  store float %35, float* %40, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_asin_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp olt float %5, 1.000000e+00
  %7 = fsub float 1.000000e+00, %5
  %8 = fsub float 1.000000e+00, %7
  %9 = select i1 %6, float %8, float 1.000000e+00
  %10 = fsub float 1.000000e+00, %9
  %11 = tail call float @sqrtf(float %10) #13
  %12 = fmul float %9, 0x3F96290BA0000000
  %13 = fsub float 0x3FB3F68760000000, %12
  %14 = fmul float %9, %13
  %15 = fadd float %14, 0xBFCB4D7260000000
  %16 = fmul float %9, %15
  %17 = fadd float %16, 0x3FF921FB60000000
  %18 = fmul float %11, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = tail call float @copysignf(float %19, float %4) #13
  %21 = bitcast i8* %0 to float*
  store float %20, float* %21, align 4, !tbaa !1
  %22 = getelementptr inbounds i8, i8* %1, i64 4
  %23 = bitcast i8* %22 to float*
  %24 = load float, float* %23, align 4, !tbaa !1
  %25 = tail call float @fabsf(float %24) #13
  %26 = fcmp olt float %25, 1.000000e+00
  %27 = fsub float 1.000000e+00, %25
  %28 = fsub float 1.000000e+00, %27
  %29 = select i1 %26, float %28, float 1.000000e+00
  %30 = fsub float 1.000000e+00, %29
  %31 = tail call float @sqrtf(float %30) #13
  %32 = fmul float %29, 0x3F96290BA0000000
  %33 = fsub float 0x3FB3F68760000000, %32
  %34 = fmul float %29, %33
  %35 = fadd float %34, 0xBFCB4D7260000000
  %36 = fmul float %29, %35
  %37 = fadd float %36, 0x3FF921FB60000000
  %38 = fmul float %31, %37
  %39 = fsub float 0x3FF921FB60000000, %38
  %40 = tail call float @copysignf(float %39, float %24) #13
  %41 = getelementptr inbounds i8, i8* %0, i64 4
  %42 = bitcast i8* %41 to float*
  store float %40, float* %42, align 4, !tbaa !1
  %43 = getelementptr inbounds i8, i8* %1, i64 8
  %44 = bitcast i8* %43 to float*
  %45 = load float, float* %44, align 4, !tbaa !1
  %46 = tail call float @fabsf(float %45) #13
  %47 = fcmp olt float %46, 1.000000e+00
  %48 = fsub float 1.000000e+00, %46
  %49 = fsub float 1.000000e+00, %48
  %50 = select i1 %47, float %49, float 1.000000e+00
  %51 = fsub float 1.000000e+00, %50
  %52 = tail call float @sqrtf(float %51) #13
  %53 = fmul float %50, 0x3F96290BA0000000
  %54 = fsub float 0x3FB3F68760000000, %53
  %55 = fmul float %50, %54
  %56 = fadd float %55, 0xBFCB4D7260000000
  %57 = fmul float %50, %56
  %58 = fadd float %57, 0x3FF921FB60000000
  %59 = fmul float %52, %58
  %60 = fsub float 0x3FF921FB60000000, %59
  %61 = tail call float @copysignf(float %60, float %45) #13
  %62 = getelementptr inbounds i8, i8* %0, i64 8
  %63 = bitcast i8* %62 to float*
  store float %61, float* %63, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_asin_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = getelementptr inbounds i8, i8* %1, i64 24
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = bitcast i8* %3 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = bitcast i8* %4 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = tail call float @fabsf(float %6) #13
  %12 = fcmp olt float %11, 1.000000e+00
  %13 = fsub float 1.000000e+00, %11
  %14 = fsub float 1.000000e+00, %13
  %15 = select i1 %12, float %14, float 1.000000e+00
  %16 = fsub float 1.000000e+00, %15
  %17 = tail call float @sqrtf(float %16) #13
  %18 = fmul float %15, 0x3F96290BA0000000
  %19 = fsub float 0x3FB3F68760000000, %18
  %20 = fmul float %15, %19
  %21 = fadd float %20, 0xBFCB4D7260000000
  %22 = fmul float %15, %21
  %23 = fadd float %22, 0x3FF921FB60000000
  %24 = fmul float %17, %23
  %25 = fsub float 0x3FF921FB60000000, %24
  %26 = tail call float @copysignf(float %25, float %6) #13
  br i1 %12, label %27, label %32

; <label>:27:                                     ; preds = %2
  %28 = fmul float %6, %6
  %29 = fsub float 1.000000e+00, %28
  %30 = tail call float @sqrtf(float %29) #13
  %31 = fdiv float 1.000000e+00, %30
  br label %32

; <label>:32:                                     ; preds = %2, %27
  %33 = phi float [ %31, %27 ], [ 0.000000e+00, %2 ]
  %34 = fmul float %8, %33
  %35 = fmul float %10, %33
  %36 = getelementptr inbounds i8, i8* %1, i64 4
  %37 = getelementptr inbounds i8, i8* %1, i64 16
  %38 = getelementptr inbounds i8, i8* %1, i64 28
  %39 = bitcast i8* %36 to float*
  %40 = load float, float* %39, align 4, !tbaa !1
  %41 = bitcast i8* %37 to float*
  %42 = load float, float* %41, align 4, !tbaa !1
  %43 = bitcast i8* %38 to float*
  %44 = load float, float* %43, align 4, !tbaa !1
  %45 = tail call float @fabsf(float %40) #13
  %46 = fcmp olt float %45, 1.000000e+00
  %47 = fsub float 1.000000e+00, %45
  %48 = fsub float 1.000000e+00, %47
  %49 = select i1 %46, float %48, float 1.000000e+00
  %50 = fsub float 1.000000e+00, %49
  %51 = tail call float @sqrtf(float %50) #13
  %52 = fmul float %49, 0x3F96290BA0000000
  %53 = fsub float 0x3FB3F68760000000, %52
  %54 = fmul float %49, %53
  %55 = fadd float %54, 0xBFCB4D7260000000
  %56 = fmul float %49, %55
  %57 = fadd float %56, 0x3FF921FB60000000
  %58 = fmul float %51, %57
  %59 = fsub float 0x3FF921FB60000000, %58
  %60 = tail call float @copysignf(float %59, float %40) #13
  br i1 %46, label %61, label %66

; <label>:61:                                     ; preds = %32
  %62 = fmul float %40, %40
  %63 = fsub float 1.000000e+00, %62
  %64 = tail call float @sqrtf(float %63) #13
  %65 = fdiv float 1.000000e+00, %64
  br label %66

; <label>:66:                                     ; preds = %32, %61
  %67 = phi float [ %65, %61 ], [ 0.000000e+00, %32 ]
  %68 = fmul float %42, %67
  %69 = fmul float %44, %67
  %70 = getelementptr inbounds i8, i8* %1, i64 8
  %71 = getelementptr inbounds i8, i8* %1, i64 20
  %72 = getelementptr inbounds i8, i8* %1, i64 32
  %73 = bitcast i8* %70 to float*
  %74 = load float, float* %73, align 4, !tbaa !1
  %75 = bitcast i8* %71 to float*
  %76 = load float, float* %75, align 4, !tbaa !1
  %77 = bitcast i8* %72 to float*
  %78 = load float, float* %77, align 4, !tbaa !1
  %79 = tail call float @fabsf(float %74) #13
  %80 = fcmp olt float %79, 1.000000e+00
  %81 = fsub float 1.000000e+00, %79
  %82 = fsub float 1.000000e+00, %81
  %83 = select i1 %80, float %82, float 1.000000e+00
  %84 = fsub float 1.000000e+00, %83
  %85 = tail call float @sqrtf(float %84) #13
  %86 = fmul float %83, 0x3F96290BA0000000
  %87 = fsub float 0x3FB3F68760000000, %86
  %88 = fmul float %83, %87
  %89 = fadd float %88, 0xBFCB4D7260000000
  %90 = fmul float %83, %89
  %91 = fadd float %90, 0x3FF921FB60000000
  %92 = fmul float %85, %91
  %93 = fsub float 0x3FF921FB60000000, %92
  %94 = tail call float @copysignf(float %93, float %74) #13
  br i1 %80, label %95, label %100

; <label>:95:                                     ; preds = %66
  %96 = fmul float %74, %74
  %97 = fsub float 1.000000e+00, %96
  %98 = tail call float @sqrtf(float %97) #13
  %99 = fdiv float 1.000000e+00, %98
  br label %100

; <label>:100:                                    ; preds = %66, %95
  %101 = phi float [ %99, %95 ], [ 0.000000e+00, %66 ]
  %102 = fmul float %76, %101
  %103 = fmul float %78, %101
  %104 = bitcast i8* %0 to float*
  store float %26, float* %104, align 4, !tbaa !5
  %105 = getelementptr inbounds i8, i8* %0, i64 4
  %106 = bitcast i8* %105 to float*
  store float %60, float* %106, align 4, !tbaa !7
  %107 = getelementptr inbounds i8, i8* %0, i64 8
  %108 = bitcast i8* %107 to float*
  store float %94, float* %108, align 4, !tbaa !8
  %109 = getelementptr inbounds i8, i8* %0, i64 12
  %110 = bitcast i8* %109 to float*
  store float %34, float* %110, align 4, !tbaa !5
  %111 = getelementptr inbounds i8, i8* %0, i64 16
  %112 = bitcast i8* %111 to float*
  store float %68, float* %112, align 4, !tbaa !7
  %113 = getelementptr inbounds i8, i8* %0, i64 20
  %114 = bitcast i8* %113 to float*
  store float %102, float* %114, align 4, !tbaa !8
  %115 = getelementptr inbounds i8, i8* %0, i64 24
  %116 = bitcast i8* %115 to float*
  store float %35, float* %116, align 4, !tbaa !5
  %117 = getelementptr inbounds i8, i8* %0, i64 28
  %118 = bitcast i8* %117 to float*
  store float %69, float* %118, align 4, !tbaa !7
  %119 = getelementptr inbounds i8, i8* %0, i64 32
  %120 = bitcast i8* %119 to float*
  store float %103, float* %120, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_acos_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fcmp olt float %2, 1.000000e+00
  %4 = fsub float 1.000000e+00, %2
  %5 = fsub float 1.000000e+00, %4
  %6 = select i1 %3, float %5, float 1.000000e+00
  %7 = fsub float 1.000000e+00, %6
  %8 = tail call float @sqrtf(float %7) #13
  %9 = fmul float %6, 0x3F96290BA0000000
  %10 = fsub float 0x3FB3F68760000000, %9
  %11 = fmul float %6, %10
  %12 = fadd float %11, 0xBFCB4D7260000000
  %13 = fmul float %6, %12
  %14 = fadd float %13, 0x3FF921FB60000000
  %15 = fmul float %8, %14
  %16 = fcmp olt float %0, 0.000000e+00
  %17 = fsub float 0x400921FB60000000, %15
  %18 = select i1 %16, float %17, float %15
  ret float %18
}

; Function Attrs: nounwind uwtable
define void @osl_acos_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp olt float %5, 1.000000e+00
  %7 = fsub float 1.000000e+00, %5
  %8 = fsub float 1.000000e+00, %7
  %9 = select i1 %6, float %8, float 1.000000e+00
  %10 = fsub float 1.000000e+00, %9
  %11 = tail call float @sqrtf(float %10) #13
  %12 = fmul float %9, 0x3F96290BA0000000
  %13 = fsub float 0x3FB3F68760000000, %12
  %14 = fmul float %9, %13
  %15 = fadd float %14, 0xBFCB4D7260000000
  %16 = fmul float %9, %15
  %17 = fadd float %16, 0x3FF921FB60000000
  %18 = fmul float %11, %17
  %19 = fcmp olt float %4, 0.000000e+00
  %20 = fsub float 0x400921FB60000000, %18
  %21 = select i1 %19, float %20, float %18
  br i1 %6, label %22, label %27

; <label>:22:                                     ; preds = %2
  %23 = fmul float %4, %4
  %24 = fsub float 1.000000e+00, %23
  %25 = tail call float @sqrtf(float %24) #13
  %26 = fdiv float -1.000000e+00, %25
  br label %27

; <label>:27:                                     ; preds = %2, %22
  %28 = phi float [ %26, %22 ], [ 0.000000e+00, %2 ]
  %29 = getelementptr inbounds i8, i8* %1, i64 4
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !1
  %32 = fmul float %28, %31
  %33 = getelementptr inbounds i8, i8* %1, i64 8
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %34, align 4, !tbaa !1
  %36 = fmul float %28, %35
  %37 = insertelement <2 x float> undef, float %21, i32 0
  %38 = insertelement <2 x float> %37, float %32, i32 1
  %39 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %38, <2 x float>* %39, align 4
  %40 = getelementptr inbounds i8, i8* %0, i64 8
  %41 = bitcast i8* %40 to float*
  store float %36, float* %41, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_acos_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp olt float %5, 1.000000e+00
  %7 = fsub float 1.000000e+00, %5
  %8 = fsub float 1.000000e+00, %7
  %9 = select i1 %6, float %8, float 1.000000e+00
  %10 = fsub float 1.000000e+00, %9
  %11 = tail call float @sqrtf(float %10) #13
  %12 = fmul float %9, 0x3F96290BA0000000
  %13 = fsub float 0x3FB3F68760000000, %12
  %14 = fmul float %9, %13
  %15 = fadd float %14, 0xBFCB4D7260000000
  %16 = fmul float %9, %15
  %17 = fadd float %16, 0x3FF921FB60000000
  %18 = fmul float %11, %17
  %19 = fcmp olt float %4, 0.000000e+00
  %20 = fsub float 0x400921FB60000000, %18
  %21 = select i1 %19, float %20, float %18
  %22 = bitcast i8* %0 to float*
  store float %21, float* %22, align 4, !tbaa !1
  %23 = getelementptr inbounds i8, i8* %1, i64 4
  %24 = bitcast i8* %23 to float*
  %25 = load float, float* %24, align 4, !tbaa !1
  %26 = tail call float @fabsf(float %25) #13
  %27 = fcmp olt float %26, 1.000000e+00
  %28 = fsub float 1.000000e+00, %26
  %29 = fsub float 1.000000e+00, %28
  %30 = select i1 %27, float %29, float 1.000000e+00
  %31 = fsub float 1.000000e+00, %30
  %32 = tail call float @sqrtf(float %31) #13
  %33 = fmul float %30, 0x3F96290BA0000000
  %34 = fsub float 0x3FB3F68760000000, %33
  %35 = fmul float %30, %34
  %36 = fadd float %35, 0xBFCB4D7260000000
  %37 = fmul float %30, %36
  %38 = fadd float %37, 0x3FF921FB60000000
  %39 = fmul float %32, %38
  %40 = fcmp olt float %25, 0.000000e+00
  %41 = fsub float 0x400921FB60000000, %39
  %42 = select i1 %40, float %41, float %39
  %43 = getelementptr inbounds i8, i8* %0, i64 4
  %44 = bitcast i8* %43 to float*
  store float %42, float* %44, align 4, !tbaa !1
  %45 = getelementptr inbounds i8, i8* %1, i64 8
  %46 = bitcast i8* %45 to float*
  %47 = load float, float* %46, align 4, !tbaa !1
  %48 = tail call float @fabsf(float %47) #13
  %49 = fcmp olt float %48, 1.000000e+00
  %50 = fsub float 1.000000e+00, %48
  %51 = fsub float 1.000000e+00, %50
  %52 = select i1 %49, float %51, float 1.000000e+00
  %53 = fsub float 1.000000e+00, %52
  %54 = tail call float @sqrtf(float %53) #13
  %55 = fmul float %52, 0x3F96290BA0000000
  %56 = fsub float 0x3FB3F68760000000, %55
  %57 = fmul float %52, %56
  %58 = fadd float %57, 0xBFCB4D7260000000
  %59 = fmul float %52, %58
  %60 = fadd float %59, 0x3FF921FB60000000
  %61 = fmul float %54, %60
  %62 = fcmp olt float %47, 0.000000e+00
  %63 = fsub float 0x400921FB60000000, %61
  %64 = select i1 %62, float %63, float %61
  %65 = getelementptr inbounds i8, i8* %0, i64 8
  %66 = bitcast i8* %65 to float*
  store float %64, float* %66, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_acos_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = getelementptr inbounds i8, i8* %1, i64 24
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = bitcast i8* %3 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = bitcast i8* %4 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = tail call float @fabsf(float %6) #13
  %12 = fcmp olt float %11, 1.000000e+00
  %13 = fsub float 1.000000e+00, %11
  %14 = fsub float 1.000000e+00, %13
  %15 = select i1 %12, float %14, float 1.000000e+00
  %16 = fsub float 1.000000e+00, %15
  %17 = tail call float @sqrtf(float %16) #13
  %18 = fmul float %15, 0x3F96290BA0000000
  %19 = fsub float 0x3FB3F68760000000, %18
  %20 = fmul float %15, %19
  %21 = fadd float %20, 0xBFCB4D7260000000
  %22 = fmul float %15, %21
  %23 = fadd float %22, 0x3FF921FB60000000
  %24 = fmul float %17, %23
  %25 = fcmp olt float %6, 0.000000e+00
  %26 = fsub float 0x400921FB60000000, %24
  %27 = select i1 %25, float %26, float %24
  br i1 %12, label %28, label %33

; <label>:28:                                     ; preds = %2
  %29 = fmul float %6, %6
  %30 = fsub float 1.000000e+00, %29
  %31 = tail call float @sqrtf(float %30) #13
  %32 = fdiv float -1.000000e+00, %31
  br label %33

; <label>:33:                                     ; preds = %2, %28
  %34 = phi float [ %32, %28 ], [ 0.000000e+00, %2 ]
  %35 = fmul float %8, %34
  %36 = fmul float %10, %34
  %37 = getelementptr inbounds i8, i8* %1, i64 4
  %38 = getelementptr inbounds i8, i8* %1, i64 16
  %39 = getelementptr inbounds i8, i8* %1, i64 28
  %40 = bitcast i8* %37 to float*
  %41 = load float, float* %40, align 4, !tbaa !1
  %42 = bitcast i8* %38 to float*
  %43 = load float, float* %42, align 4, !tbaa !1
  %44 = bitcast i8* %39 to float*
  %45 = load float, float* %44, align 4, !tbaa !1
  %46 = tail call float @fabsf(float %41) #13
  %47 = fcmp olt float %46, 1.000000e+00
  %48 = fsub float 1.000000e+00, %46
  %49 = fsub float 1.000000e+00, %48
  %50 = select i1 %47, float %49, float 1.000000e+00
  %51 = fsub float 1.000000e+00, %50
  %52 = tail call float @sqrtf(float %51) #13
  %53 = fmul float %50, 0x3F96290BA0000000
  %54 = fsub float 0x3FB3F68760000000, %53
  %55 = fmul float %50, %54
  %56 = fadd float %55, 0xBFCB4D7260000000
  %57 = fmul float %50, %56
  %58 = fadd float %57, 0x3FF921FB60000000
  %59 = fmul float %52, %58
  %60 = fcmp olt float %41, 0.000000e+00
  %61 = fsub float 0x400921FB60000000, %59
  %62 = select i1 %60, float %61, float %59
  br i1 %47, label %63, label %68

; <label>:63:                                     ; preds = %33
  %64 = fmul float %41, %41
  %65 = fsub float 1.000000e+00, %64
  %66 = tail call float @sqrtf(float %65) #13
  %67 = fdiv float -1.000000e+00, %66
  br label %68

; <label>:68:                                     ; preds = %33, %63
  %69 = phi float [ %67, %63 ], [ 0.000000e+00, %33 ]
  %70 = fmul float %43, %69
  %71 = fmul float %45, %69
  %72 = getelementptr inbounds i8, i8* %1, i64 8
  %73 = getelementptr inbounds i8, i8* %1, i64 20
  %74 = getelementptr inbounds i8, i8* %1, i64 32
  %75 = bitcast i8* %72 to float*
  %76 = load float, float* %75, align 4, !tbaa !1
  %77 = bitcast i8* %73 to float*
  %78 = load float, float* %77, align 4, !tbaa !1
  %79 = bitcast i8* %74 to float*
  %80 = load float, float* %79, align 4, !tbaa !1
  %81 = tail call float @fabsf(float %76) #13
  %82 = fcmp olt float %81, 1.000000e+00
  %83 = fsub float 1.000000e+00, %81
  %84 = fsub float 1.000000e+00, %83
  %85 = select i1 %82, float %84, float 1.000000e+00
  %86 = fsub float 1.000000e+00, %85
  %87 = tail call float @sqrtf(float %86) #13
  %88 = fmul float %85, 0x3F96290BA0000000
  %89 = fsub float 0x3FB3F68760000000, %88
  %90 = fmul float %85, %89
  %91 = fadd float %90, 0xBFCB4D7260000000
  %92 = fmul float %85, %91
  %93 = fadd float %92, 0x3FF921FB60000000
  %94 = fmul float %87, %93
  %95 = fcmp olt float %76, 0.000000e+00
  %96 = fsub float 0x400921FB60000000, %94
  %97 = select i1 %95, float %96, float %94
  br i1 %82, label %98, label %103

; <label>:98:                                     ; preds = %68
  %99 = fmul float %76, %76
  %100 = fsub float 1.000000e+00, %99
  %101 = tail call float @sqrtf(float %100) #13
  %102 = fdiv float -1.000000e+00, %101
  br label %103

; <label>:103:                                    ; preds = %68, %98
  %104 = phi float [ %102, %98 ], [ 0.000000e+00, %68 ]
  %105 = fmul float %78, %104
  %106 = fmul float %80, %104
  %107 = bitcast i8* %0 to float*
  store float %27, float* %107, align 4, !tbaa !5
  %108 = getelementptr inbounds i8, i8* %0, i64 4
  %109 = bitcast i8* %108 to float*
  store float %62, float* %109, align 4, !tbaa !7
  %110 = getelementptr inbounds i8, i8* %0, i64 8
  %111 = bitcast i8* %110 to float*
  store float %97, float* %111, align 4, !tbaa !8
  %112 = getelementptr inbounds i8, i8* %0, i64 12
  %113 = bitcast i8* %112 to float*
  store float %35, float* %113, align 4, !tbaa !5
  %114 = getelementptr inbounds i8, i8* %0, i64 16
  %115 = bitcast i8* %114 to float*
  store float %70, float* %115, align 4, !tbaa !7
  %116 = getelementptr inbounds i8, i8* %0, i64 20
  %117 = bitcast i8* %116 to float*
  store float %105, float* %117, align 4, !tbaa !8
  %118 = getelementptr inbounds i8, i8* %0, i64 24
  %119 = bitcast i8* %118 to float*
  store float %36, float* %119, align 4, !tbaa !5
  %120 = getelementptr inbounds i8, i8* %0, i64 28
  %121 = bitcast i8* %120 to float*
  store float %71, float* %121, align 4, !tbaa !7
  %122 = getelementptr inbounds i8, i8* %0, i64 32
  %123 = bitcast i8* %122 to float*
  store float %106, float* %123, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_atan_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fcmp ogt float %2, 1.000000e+00
  %4 = fdiv float 1.000000e+00, %2
  %5 = select i1 %3, float %4, float %2
  %6 = fsub float 1.000000e+00, %5
  %7 = fsub float 1.000000e+00, %6
  %8 = fmul float %7, %7
  %9 = fmul float %8, 0x3FDB9F00A0000000
  %10 = fadd float %9, 1.000000e+00
  %11 = fmul float %7, %10
  %12 = fmul float %8, 0x3FADDC09A0000000
  %13 = fadd float %12, 0x3FE87649C0000000
  %14 = fmul float %8, %13
  %15 = fadd float %14, 1.000000e+00
  %16 = fdiv float %11, %15
  %17 = fsub float 0x3FF921FB60000000, %16
  %18 = select i1 %3, float %17, float %16
  %19 = tail call float @copysignf(float %18, float %0) #13
  ret float %19
}

; Function Attrs: nounwind uwtable
define void @osl_atan_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp ogt float %5, 1.000000e+00
  %7 = fdiv float 1.000000e+00, %5
  %8 = select i1 %6, float %7, float %5
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
  %20 = fsub float 0x3FF921FB60000000, %19
  %21 = select i1 %6, float %20, float %19
  %22 = tail call float @copysignf(float %21, float %4) #13
  %23 = fmul float %4, %4
  %24 = fadd float %23, 1.000000e+00
  %25 = fdiv float 1.000000e+00, %24
  %26 = getelementptr inbounds i8, i8* %1, i64 4
  %27 = bitcast i8* %26 to float*
  %28 = load float, float* %27, align 4, !tbaa !1
  %29 = fmul float %25, %28
  %30 = getelementptr inbounds i8, i8* %1, i64 8
  %31 = bitcast i8* %30 to float*
  %32 = load float, float* %31, align 4, !tbaa !1
  %33 = fmul float %25, %32
  %34 = insertelement <2 x float> undef, float %22, i32 0
  %35 = insertelement <2 x float> %34, float %29, i32 1
  %36 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %35, <2 x float>* %36, align 4
  %37 = getelementptr inbounds i8, i8* %0, i64 8
  %38 = bitcast i8* %37 to float*
  store float %33, float* %38, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp ogt float %5, 1.000000e+00
  %7 = fdiv float 1.000000e+00, %5
  %8 = select i1 %6, float %7, float %5
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
  %20 = fsub float 0x3FF921FB60000000, %19
  %21 = select i1 %6, float %20, float %19
  %22 = tail call float @copysignf(float %21, float %4) #13
  %23 = bitcast i8* %0 to float*
  store float %22, float* %23, align 4, !tbaa !1
  %24 = getelementptr inbounds i8, i8* %1, i64 4
  %25 = bitcast i8* %24 to float*
  %26 = load float, float* %25, align 4, !tbaa !1
  %27 = tail call float @fabsf(float %26) #13
  %28 = fcmp ogt float %27, 1.000000e+00
  %29 = fdiv float 1.000000e+00, %27
  %30 = select i1 %28, float %29, float %27
  %31 = fsub float 1.000000e+00, %30
  %32 = fsub float 1.000000e+00, %31
  %33 = fmul float %32, %32
  %34 = fmul float %33, 0x3FDB9F00A0000000
  %35 = fadd float %34, 1.000000e+00
  %36 = fmul float %32, %35
  %37 = fmul float %33, 0x3FADDC09A0000000
  %38 = fadd float %37, 0x3FE87649C0000000
  %39 = fmul float %33, %38
  %40 = fadd float %39, 1.000000e+00
  %41 = fdiv float %36, %40
  %42 = fsub float 0x3FF921FB60000000, %41
  %43 = select i1 %28, float %42, float %41
  %44 = tail call float @copysignf(float %43, float %26) #13
  %45 = getelementptr inbounds i8, i8* %0, i64 4
  %46 = bitcast i8* %45 to float*
  store float %44, float* %46, align 4, !tbaa !1
  %47 = getelementptr inbounds i8, i8* %1, i64 8
  %48 = bitcast i8* %47 to float*
  %49 = load float, float* %48, align 4, !tbaa !1
  %50 = tail call float @fabsf(float %49) #13
  %51 = fcmp ogt float %50, 1.000000e+00
  %52 = fdiv float 1.000000e+00, %50
  %53 = select i1 %51, float %52, float %50
  %54 = fsub float 1.000000e+00, %53
  %55 = fsub float 1.000000e+00, %54
  %56 = fmul float %55, %55
  %57 = fmul float %56, 0x3FDB9F00A0000000
  %58 = fadd float %57, 1.000000e+00
  %59 = fmul float %55, %58
  %60 = fmul float %56, 0x3FADDC09A0000000
  %61 = fadd float %60, 0x3FE87649C0000000
  %62 = fmul float %56, %61
  %63 = fadd float %62, 1.000000e+00
  %64 = fdiv float %59, %63
  %65 = fsub float 0x3FF921FB60000000, %64
  %66 = select i1 %51, float %65, float %64
  %67 = tail call float @copysignf(float %66, float %49) #13
  %68 = getelementptr inbounds i8, i8* %0, i64 8
  %69 = bitcast i8* %68 to float*
  store float %67, float* %69, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = tail call float @fabsf(float %5) #13
  %7 = fcmp ogt float %6, 1.000000e+00
  %8 = fdiv float 1.000000e+00, %6
  %9 = select i1 %7, float %8, float %6
  %10 = fsub float 1.000000e+00, %9
  %11 = fsub float 1.000000e+00, %10
  %12 = fmul float %11, %11
  %13 = fmul float %12, 0x3FDB9F00A0000000
  %14 = fadd float %13, 1.000000e+00
  %15 = fmul float %11, %14
  %16 = fmul float %12, 0x3FADDC09A0000000
  %17 = fadd float %16, 0x3FE87649C0000000
  %18 = fmul float %12, %17
  %19 = fadd float %18, 1.000000e+00
  %20 = fdiv float %15, %19
  %21 = fsub float 0x3FF921FB60000000, %20
  %22 = select i1 %7, float %21, float %20
  %23 = tail call float @copysignf(float %22, float %5) #13
  %24 = fmul float %5, %5
  %25 = fadd float %24, 1.000000e+00
  %26 = fdiv float 1.000000e+00, %25
  %27 = getelementptr inbounds i8, i8* %1, i64 4
  %28 = getelementptr inbounds i8, i8* %1, i64 28
  %29 = bitcast i8* %27 to float*
  %30 = load float, float* %29, align 4, !tbaa !1
  %31 = bitcast i8* %28 to float*
  %32 = load float, float* %31, align 4, !tbaa !1
  %33 = tail call float @fabsf(float %30) #13
  %34 = fcmp ogt float %33, 1.000000e+00
  %35 = fdiv float 1.000000e+00, %33
  %36 = select i1 %34, float %35, float %33
  %37 = fsub float 1.000000e+00, %36
  %38 = fsub float 1.000000e+00, %37
  %39 = fmul float %38, %38
  %40 = fmul float %39, 0x3FDB9F00A0000000
  %41 = fadd float %40, 1.000000e+00
  %42 = fmul float %38, %41
  %43 = fmul float %39, 0x3FADDC09A0000000
  %44 = fadd float %43, 0x3FE87649C0000000
  %45 = fmul float %39, %44
  %46 = fadd float %45, 1.000000e+00
  %47 = fdiv float %42, %46
  %48 = fsub float 0x3FF921FB60000000, %47
  %49 = select i1 %34, float %48, float %47
  %50 = tail call float @copysignf(float %49, float %30) #13
  %51 = fmul float %30, %30
  %52 = fadd float %51, 1.000000e+00
  %53 = fdiv float 1.000000e+00, %52
  %54 = fmul float %32, %53
  %55 = getelementptr inbounds i8, i8* %1, i64 8
  %56 = getelementptr inbounds i8, i8* %1, i64 32
  %57 = bitcast i8* %55 to float*
  %58 = load float, float* %57, align 4, !tbaa !1
  %59 = bitcast i8* %3 to <4 x float>*
  %60 = load <4 x float>, <4 x float>* %59, align 4, !tbaa !1
  %61 = bitcast i8* %56 to float*
  %62 = load float, float* %61, align 4, !tbaa !1
  %63 = tail call float @fabsf(float %58) #13
  %64 = fcmp ogt float %63, 1.000000e+00
  %65 = fdiv float 1.000000e+00, %63
  %66 = select i1 %64, float %65, float %63
  %67 = fsub float 1.000000e+00, %66
  %68 = fsub float 1.000000e+00, %67
  %69 = fmul float %68, %68
  %70 = fmul float %69, 0x3FDB9F00A0000000
  %71 = fadd float %70, 1.000000e+00
  %72 = fmul float %68, %71
  %73 = fmul float %69, 0x3FADDC09A0000000
  %74 = fadd float %73, 0x3FE87649C0000000
  %75 = fmul float %69, %74
  %76 = fadd float %75, 1.000000e+00
  %77 = fdiv float %72, %76
  %78 = fsub float 0x3FF921FB60000000, %77
  %79 = select i1 %64, float %78, float %77
  %80 = tail call float @copysignf(float %79, float %58) #13
  %81 = fmul float %58, %58
  %82 = fadd float %81, 1.000000e+00
  %83 = fdiv float 1.000000e+00, %82
  %84 = insertelement <4 x float> undef, float %26, i32 0
  %85 = insertelement <4 x float> %84, float %53, i32 1
  %86 = insertelement <4 x float> %85, float %83, i32 2
  %87 = insertelement <4 x float> %86, float %26, i32 3
  %88 = fmul <4 x float> %60, %87
  %89 = fmul float %62, %83
  %90 = bitcast i8* %0 to float*
  store float %23, float* %90, align 4, !tbaa !5
  %91 = getelementptr inbounds i8, i8* %0, i64 4
  %92 = bitcast i8* %91 to float*
  store float %50, float* %92, align 4, !tbaa !7
  %93 = getelementptr inbounds i8, i8* %0, i64 8
  %94 = bitcast i8* %93 to float*
  store float %80, float* %94, align 4, !tbaa !8
  %95 = getelementptr inbounds i8, i8* %0, i64 12
  %96 = bitcast i8* %95 to <4 x float>*
  store <4 x float> %88, <4 x float>* %96, align 4, !tbaa !1
  %97 = getelementptr inbounds i8, i8* %0, i64 28
  %98 = bitcast i8* %97 to float*
  store float %54, float* %98, align 4, !tbaa !7
  %99 = getelementptr inbounds i8, i8* %0, i64 32
  %100 = bitcast i8* %99 to float*
  store float %89, float* %100, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_atan2_fff(float, float) local_unnamed_addr #3 {
  %3 = tail call float @fabsf(float %1) #13
  %4 = tail call float @fabsf(float %0) #13
  %5 = fcmp oeq float %0, 0.000000e+00
  br i1 %5, label %14, label %6

; <label>:6:                                      ; preds = %2
  %7 = fcmp oeq float %3, %4
  br i1 %7, label %14, label %8

; <label>:8:                                      ; preds = %6
  %9 = fcmp ogt float %4, %3
  br i1 %9, label %10, label %12

; <label>:10:                                     ; preds = %8
  %11 = fdiv float %3, %4
  br label %14

; <label>:12:                                     ; preds = %8
  %13 = fdiv float %4, %3
  br label %14

; <label>:14:                                     ; preds = %2, %6, %10, %12
  %15 = phi float [ 0.000000e+00, %2 ], [ 1.000000e+00, %6 ], [ %11, %10 ], [ %13, %12 ]
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
  %28 = fsub float 0x3FF921FB60000000, %26
  %29 = select i1 %27, float %28, float %26
  %30 = bitcast float %1 to i32
  %31 = icmp slt i32 %30, 0
  %32 = fsub float 0x400921FB60000000, %29
  %33 = select i1 %31, float %32, float %29
  %34 = tail call float @copysignf(float %33, float %0) #13
  ret float %34
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dfdfdf(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %2 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = tail call float @fabsf(float %7) #13
  %9 = tail call float @fabsf(float %5) #13
  %10 = fcmp oeq float %5, 0.000000e+00
  br i1 %10, label %19, label %11

; <label>:11:                                     ; preds = %3
  %12 = fcmp oeq float %8, %9
  br i1 %12, label %19, label %13

; <label>:13:                                     ; preds = %11
  %14 = fcmp ogt float %9, %8
  br i1 %14, label %15, label %17

; <label>:15:                                     ; preds = %13
  %16 = fdiv float %8, %9
  br label %19

; <label>:17:                                     ; preds = %13
  %18 = fdiv float %9, %8
  br label %19

; <label>:19:                                     ; preds = %17, %15, %11, %3
  %20 = phi float [ 0.000000e+00, %3 ], [ 1.000000e+00, %11 ], [ %16, %15 ], [ %18, %17 ]
  %21 = fsub float 1.000000e+00, %20
  %22 = fsub float 1.000000e+00, %21
  %23 = fmul float %22, %22
  %24 = fmul float %23, 0x3FDB9F00A0000000
  %25 = fadd float %24, 1.000000e+00
  %26 = fmul float %22, %25
  %27 = fmul float %23, 0x3FADDC09A0000000
  %28 = fadd float %27, 0x3FE87649C0000000
  %29 = fmul float %23, %28
  %30 = fadd float %29, 1.000000e+00
  %31 = fdiv float %26, %30
  %32 = fcmp ogt float %9, %8
  %33 = fsub float 0x3FF921FB60000000, %31
  %34 = select i1 %32, float %33, float %31
  %35 = bitcast float %7 to i32
  %36 = icmp slt i32 %35, 0
  %37 = fsub float 0x400921FB60000000, %34
  %38 = select i1 %36, float %37, float %34
  %39 = tail call float @copysignf(float %38, float %5) #13
  %40 = fcmp oeq float %7, 0.000000e+00
  %41 = and i1 %10, %40
  br i1 %41, label %47, label %42

; <label>:42:                                     ; preds = %19
  %43 = fmul float %7, %7
  %44 = fmul float %5, %5
  %45 = fadd float %44, %43
  %46 = fdiv float 1.000000e+00, %45
  br label %47

; <label>:47:                                     ; preds = %19, %42
  %48 = phi float [ %46, %42 ], [ 0.000000e+00, %19 ]
  %49 = getelementptr inbounds i8, i8* %2, i64 4
  %50 = bitcast i8* %49 to float*
  %51 = load float, float* %50, align 4, !tbaa !1
  %52 = fmul float %5, %51
  %53 = getelementptr inbounds i8, i8* %1, i64 4
  %54 = bitcast i8* %53 to float*
  %55 = load float, float* %54, align 4, !tbaa !1
  %56 = fmul float %7, %55
  %57 = fsub float %52, %56
  %58 = fmul float %48, %57
  %59 = getelementptr inbounds i8, i8* %2, i64 8
  %60 = bitcast i8* %59 to float*
  %61 = load float, float* %60, align 4, !tbaa !1
  %62 = fmul float %5, %61
  %63 = getelementptr inbounds i8, i8* %1, i64 8
  %64 = bitcast i8* %63 to float*
  %65 = load float, float* %64, align 4, !tbaa !1
  %66 = fmul float %7, %65
  %67 = fsub float %62, %66
  %68 = fmul float %48, %67
  %69 = insertelement <2 x float> undef, float %39, i32 0
  %70 = insertelement <2 x float> %69, float %58, i32 1
  %71 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %70, <2 x float>* %71, align 4
  %72 = getelementptr inbounds i8, i8* %0, i64 8
  %73 = bitcast i8* %72 to float*
  store float %68, float* %73, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dffdf(i8* nocapture, float, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = tail call float @fabsf(float %5) #13
  %7 = tail call float @fabsf(float %1) #13
  %8 = fcmp oeq float %1, 0.000000e+00
  br i1 %8, label %17, label %9

; <label>:9:                                      ; preds = %3
  %10 = fcmp oeq float %6, %7
  br i1 %10, label %17, label %11

; <label>:11:                                     ; preds = %9
  %12 = fcmp ogt float %7, %6
  br i1 %12, label %13, label %15

; <label>:13:                                     ; preds = %11
  %14 = fdiv float %6, %7
  br label %17

; <label>:15:                                     ; preds = %11
  %16 = fdiv float %7, %6
  br label %17

; <label>:17:                                     ; preds = %15, %13, %9, %3
  %18 = phi float [ 0.000000e+00, %3 ], [ 1.000000e+00, %9 ], [ %14, %13 ], [ %16, %15 ]
  %19 = fsub float 1.000000e+00, %18
  %20 = fsub float 1.000000e+00, %19
  %21 = fmul float %20, %20
  %22 = fmul float %21, 0x3FDB9F00A0000000
  %23 = fadd float %22, 1.000000e+00
  %24 = fmul float %20, %23
  %25 = fmul float %21, 0x3FADDC09A0000000
  %26 = fadd float %25, 0x3FE87649C0000000
  %27 = fmul float %21, %26
  %28 = fadd float %27, 1.000000e+00
  %29 = fdiv float %24, %28
  %30 = fcmp ogt float %7, %6
  %31 = fsub float 0x3FF921FB60000000, %29
  %32 = select i1 %30, float %31, float %29
  %33 = bitcast float %5 to i32
  %34 = icmp slt i32 %33, 0
  %35 = fsub float 0x400921FB60000000, %32
  %36 = select i1 %34, float %35, float %32
  %37 = tail call float @copysignf(float %36, float %1) #13
  %38 = fcmp oeq float %5, 0.000000e+00
  %39 = and i1 %8, %38
  br i1 %39, label %45, label %40

; <label>:40:                                     ; preds = %17
  %41 = fmul float %5, %5
  %42 = fmul float %1, %1
  %43 = fadd float %42, %41
  %44 = fdiv float 1.000000e+00, %43
  br label %45

; <label>:45:                                     ; preds = %17, %40
  %46 = phi float [ %44, %40 ], [ 0.000000e+00, %17 ]
  %47 = getelementptr inbounds i8, i8* %2, i64 4
  %48 = bitcast i8* %47 to float*
  %49 = load float, float* %48, align 4, !tbaa !1
  %50 = fmul float %49, %1
  %51 = fmul float %5, 0.000000e+00
  %52 = fsub float %50, %51
  %53 = fmul float %46, %52
  %54 = getelementptr inbounds i8, i8* %2, i64 8
  %55 = bitcast i8* %54 to float*
  %56 = load float, float* %55, align 4, !tbaa !1
  %57 = fmul float %56, %1
  %58 = fsub float %57, %51
  %59 = fmul float %46, %58
  %60 = insertelement <2 x float> undef, float %37, i32 0
  %61 = insertelement <2 x float> %60, float %53, i32 1
  %62 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %61, <2 x float>* %62, align 4
  %63 = getelementptr inbounds i8, i8* %0, i64 8
  %64 = bitcast i8* %63 to float*
  store float %59, float* %64, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dfdff(i8* nocapture, i8* nocapture readonly, float) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = tail call float @fabsf(float %2) #13
  %7 = tail call float @fabsf(float %5) #13
  %8 = fcmp oeq float %5, 0.000000e+00
  br i1 %8, label %17, label %9

; <label>:9:                                      ; preds = %3
  %10 = fcmp oeq float %6, %7
  br i1 %10, label %17, label %11

; <label>:11:                                     ; preds = %9
  %12 = fcmp ogt float %7, %6
  br i1 %12, label %13, label %15

; <label>:13:                                     ; preds = %11
  %14 = fdiv float %6, %7
  br label %17

; <label>:15:                                     ; preds = %11
  %16 = fdiv float %7, %6
  br label %17

; <label>:17:                                     ; preds = %15, %13, %9, %3
  %18 = phi float [ 0.000000e+00, %3 ], [ 1.000000e+00, %9 ], [ %14, %13 ], [ %16, %15 ]
  %19 = fsub float 1.000000e+00, %18
  %20 = fsub float 1.000000e+00, %19
  %21 = fmul float %20, %20
  %22 = fmul float %21, 0x3FDB9F00A0000000
  %23 = fadd float %22, 1.000000e+00
  %24 = fmul float %20, %23
  %25 = fmul float %21, 0x3FADDC09A0000000
  %26 = fadd float %25, 0x3FE87649C0000000
  %27 = fmul float %21, %26
  %28 = fadd float %27, 1.000000e+00
  %29 = fdiv float %24, %28
  %30 = fcmp ogt float %7, %6
  %31 = fsub float 0x3FF921FB60000000, %29
  %32 = select i1 %30, float %31, float %29
  %33 = bitcast float %2 to i32
  %34 = icmp slt i32 %33, 0
  %35 = fsub float 0x400921FB60000000, %32
  %36 = select i1 %34, float %35, float %32
  %37 = tail call float @copysignf(float %36, float %5) #13
  %38 = fcmp oeq float %2, 0.000000e+00
  %39 = and i1 %38, %8
  br i1 %39, label %45, label %40

; <label>:40:                                     ; preds = %17
  %41 = fmul float %2, %2
  %42 = fmul float %5, %5
  %43 = fadd float %41, %42
  %44 = fdiv float 1.000000e+00, %43
  br label %45

; <label>:45:                                     ; preds = %17, %40
  %46 = phi float [ %44, %40 ], [ 0.000000e+00, %17 ]
  %47 = fmul float %5, 0.000000e+00
  %48 = getelementptr inbounds i8, i8* %1, i64 4
  %49 = bitcast i8* %48 to float*
  %50 = load float, float* %49, align 4, !tbaa !1
  %51 = fmul float %50, %2
  %52 = fsub float %47, %51
  %53 = fmul float %46, %52
  %54 = getelementptr inbounds i8, i8* %1, i64 8
  %55 = bitcast i8* %54 to float*
  %56 = load float, float* %55, align 4, !tbaa !1
  %57 = fmul float %56, %2
  %58 = fsub float %47, %57
  %59 = fmul float %46, %58
  %60 = insertelement <2 x float> undef, float %37, i32 0
  %61 = insertelement <2 x float> %60, float %53, i32 1
  %62 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %61, <2 x float>* %62, align 4
  %63 = getelementptr inbounds i8, i8* %0, i64 8
  %64 = bitcast i8* %63 to float*
  store float %59, float* %64, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_vvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %2 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = tail call float @fabsf(float %7) #13
  %9 = tail call float @fabsf(float %5) #13
  %10 = fcmp oeq float %5, 0.000000e+00
  br i1 %10, label %19, label %11

; <label>:11:                                     ; preds = %3
  %12 = fcmp oeq float %8, %9
  br i1 %12, label %19, label %13

; <label>:13:                                     ; preds = %11
  %14 = fcmp ogt float %9, %8
  br i1 %14, label %15, label %17

; <label>:15:                                     ; preds = %13
  %16 = fdiv float %8, %9
  br label %19

; <label>:17:                                     ; preds = %13
  %18 = fdiv float %9, %8
  br label %19

; <label>:19:                                     ; preds = %3, %11, %15, %17
  %20 = phi float [ 0.000000e+00, %3 ], [ 1.000000e+00, %11 ], [ %16, %15 ], [ %18, %17 ]
  %21 = fsub float 1.000000e+00, %20
  %22 = fsub float 1.000000e+00, %21
  %23 = fmul float %22, %22
  %24 = fmul float %23, 0x3FDB9F00A0000000
  %25 = fadd float %24, 1.000000e+00
  %26 = fmul float %22, %25
  %27 = fmul float %23, 0x3FADDC09A0000000
  %28 = fadd float %27, 0x3FE87649C0000000
  %29 = fmul float %23, %28
  %30 = fadd float %29, 1.000000e+00
  %31 = fdiv float %26, %30
  %32 = fcmp ogt float %9, %8
  %33 = fsub float 0x3FF921FB60000000, %31
  %34 = select i1 %32, float %33, float %31
  %35 = bitcast float %7 to i32
  %36 = icmp slt i32 %35, 0
  %37 = fsub float 0x400921FB60000000, %34
  %38 = select i1 %36, float %37, float %34
  %39 = tail call float @copysignf(float %38, float %5) #13
  %40 = bitcast i8* %0 to float*
  store float %39, float* %40, align 4, !tbaa !1
  %41 = getelementptr inbounds i8, i8* %1, i64 4
  %42 = bitcast i8* %41 to float*
  %43 = load float, float* %42, align 4, !tbaa !1
  %44 = getelementptr inbounds i8, i8* %2, i64 4
  %45 = bitcast i8* %44 to float*
  %46 = load float, float* %45, align 4, !tbaa !1
  %47 = tail call float @fabsf(float %46) #13
  %48 = tail call float @fabsf(float %43) #13
  %49 = fcmp oeq float %43, 0.000000e+00
  br i1 %49, label %58, label %50

; <label>:50:                                     ; preds = %19
  %51 = fcmp oeq float %47, %48
  br i1 %51, label %58, label %52

; <label>:52:                                     ; preds = %50
  %53 = fcmp ogt float %48, %47
  br i1 %53, label %54, label %56

; <label>:54:                                     ; preds = %52
  %55 = fdiv float %47, %48
  br label %58

; <label>:56:                                     ; preds = %52
  %57 = fdiv float %48, %47
  br label %58

; <label>:58:                                     ; preds = %19, %50, %54, %56
  %59 = phi float [ 0.000000e+00, %19 ], [ 1.000000e+00, %50 ], [ %55, %54 ], [ %57, %56 ]
  %60 = fsub float 1.000000e+00, %59
  %61 = fsub float 1.000000e+00, %60
  %62 = fmul float %61, %61
  %63 = fmul float %62, 0x3FDB9F00A0000000
  %64 = fadd float %63, 1.000000e+00
  %65 = fmul float %61, %64
  %66 = fmul float %62, 0x3FADDC09A0000000
  %67 = fadd float %66, 0x3FE87649C0000000
  %68 = fmul float %62, %67
  %69 = fadd float %68, 1.000000e+00
  %70 = fdiv float %65, %69
  %71 = fcmp ogt float %48, %47
  %72 = fsub float 0x3FF921FB60000000, %70
  %73 = select i1 %71, float %72, float %70
  %74 = bitcast float %46 to i32
  %75 = icmp slt i32 %74, 0
  %76 = fsub float 0x400921FB60000000, %73
  %77 = select i1 %75, float %76, float %73
  %78 = tail call float @copysignf(float %77, float %43) #13
  %79 = getelementptr inbounds i8, i8* %0, i64 4
  %80 = bitcast i8* %79 to float*
  store float %78, float* %80, align 4, !tbaa !1
  %81 = getelementptr inbounds i8, i8* %1, i64 8
  %82 = bitcast i8* %81 to float*
  %83 = load float, float* %82, align 4, !tbaa !1
  %84 = getelementptr inbounds i8, i8* %2, i64 8
  %85 = bitcast i8* %84 to float*
  %86 = load float, float* %85, align 4, !tbaa !1
  %87 = tail call float @fabsf(float %86) #13
  %88 = tail call float @fabsf(float %83) #13
  %89 = fcmp oeq float %83, 0.000000e+00
  br i1 %89, label %98, label %90

; <label>:90:                                     ; preds = %58
  %91 = fcmp oeq float %87, %88
  br i1 %91, label %98, label %92

; <label>:92:                                     ; preds = %90
  %93 = fcmp ogt float %88, %87
  br i1 %93, label %94, label %96

; <label>:94:                                     ; preds = %92
  %95 = fdiv float %87, %88
  br label %98

; <label>:96:                                     ; preds = %92
  %97 = fdiv float %88, %87
  br label %98

; <label>:98:                                     ; preds = %58, %90, %94, %96
  %99 = phi float [ 0.000000e+00, %58 ], [ 1.000000e+00, %90 ], [ %95, %94 ], [ %97, %96 ]
  %100 = fsub float 1.000000e+00, %99
  %101 = fsub float 1.000000e+00, %100
  %102 = fmul float %101, %101
  %103 = fmul float %102, 0x3FDB9F00A0000000
  %104 = fadd float %103, 1.000000e+00
  %105 = fmul float %101, %104
  %106 = fmul float %102, 0x3FADDC09A0000000
  %107 = fadd float %106, 0x3FE87649C0000000
  %108 = fmul float %102, %107
  %109 = fadd float %108, 1.000000e+00
  %110 = fdiv float %105, %109
  %111 = fcmp ogt float %88, %87
  %112 = fsub float 0x3FF921FB60000000, %110
  %113 = select i1 %111, float %112, float %110
  %114 = bitcast float %86 to i32
  %115 = icmp slt i32 %114, 0
  %116 = fsub float 0x400921FB60000000, %113
  %117 = select i1 %115, float %116, float %113
  %118 = tail call float @copysignf(float %117, float %83) #13
  %119 = getelementptr inbounds i8, i8* %0, i64 8
  %120 = bitcast i8* %119 to float*
  store float %118, float* %120, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dvdvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = getelementptr inbounds i8, i8* %1, i64 12
  %5 = getelementptr inbounds i8, i8* %1, i64 24
  %6 = bitcast i8* %1 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = bitcast i8* %4 to float*
  %9 = load float, float* %8, align 4, !tbaa !1
  %10 = bitcast i8* %5 to float*
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = getelementptr inbounds i8, i8* %2, i64 12
  %13 = getelementptr inbounds i8, i8* %2, i64 24
  %14 = bitcast i8* %2 to float*
  %15 = load float, float* %14, align 4, !tbaa !1
  %16 = bitcast i8* %12 to float*
  %17 = load float, float* %16, align 4, !tbaa !1
  %18 = bitcast i8* %13 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = tail call float @fabsf(float %15) #13
  %21 = tail call float @fabsf(float %7) #13
  %22 = fcmp oeq float %7, 0.000000e+00
  br i1 %22, label %31, label %23

; <label>:23:                                     ; preds = %3
  %24 = fcmp oeq float %20, %21
  br i1 %24, label %31, label %25

; <label>:25:                                     ; preds = %23
  %26 = fcmp ogt float %21, %20
  br i1 %26, label %27, label %29

; <label>:27:                                     ; preds = %25
  %28 = fdiv float %20, %21
  br label %31

; <label>:29:                                     ; preds = %25
  %30 = fdiv float %21, %20
  br label %31

; <label>:31:                                     ; preds = %29, %27, %23, %3
  %32 = phi float [ 0.000000e+00, %3 ], [ 1.000000e+00, %23 ], [ %28, %27 ], [ %30, %29 ]
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
  %44 = fcmp ogt float %21, %20
  %45 = fsub float 0x3FF921FB60000000, %43
  %46 = select i1 %44, float %45, float %43
  %47 = bitcast float %15 to i32
  %48 = icmp slt i32 %47, 0
  %49 = fsub float 0x400921FB60000000, %46
  %50 = select i1 %48, float %49, float %46
  %51 = tail call float @copysignf(float %50, float %7) #13
  %52 = fcmp oeq float %15, 0.000000e+00
  %53 = and i1 %22, %52
  br i1 %53, label %59, label %54

; <label>:54:                                     ; preds = %31
  %55 = fmul float %15, %15
  %56 = fmul float %7, %7
  %57 = fadd float %56, %55
  %58 = fdiv float 1.000000e+00, %57
  br label %59

; <label>:59:                                     ; preds = %31, %54
  %60 = phi float [ %58, %54 ], [ 0.000000e+00, %31 ]
  %61 = fmul float %7, %17
  %62 = fmul float %9, %15
  %63 = fsub float %61, %62
  %64 = fmul float %63, %60
  %65 = fmul float %7, %19
  %66 = fmul float %11, %15
  %67 = fsub float %65, %66
  %68 = fmul float %67, %60
  %69 = getelementptr inbounds i8, i8* %1, i64 4
  %70 = getelementptr inbounds i8, i8* %1, i64 16
  %71 = getelementptr inbounds i8, i8* %1, i64 28
  %72 = bitcast i8* %69 to float*
  %73 = load float, float* %72, align 4, !tbaa !1
  %74 = bitcast i8* %70 to float*
  %75 = load float, float* %74, align 4, !tbaa !1
  %76 = bitcast i8* %71 to float*
  %77 = load float, float* %76, align 4, !tbaa !1
  %78 = getelementptr inbounds i8, i8* %2, i64 4
  %79 = getelementptr inbounds i8, i8* %2, i64 16
  %80 = getelementptr inbounds i8, i8* %2, i64 28
  %81 = bitcast i8* %78 to float*
  %82 = load float, float* %81, align 4, !tbaa !1
  %83 = bitcast i8* %79 to float*
  %84 = load float, float* %83, align 4, !tbaa !1
  %85 = bitcast i8* %80 to float*
  %86 = load float, float* %85, align 4, !tbaa !1
  %87 = tail call float @fabsf(float %82) #13
  %88 = tail call float @fabsf(float %73) #13
  %89 = fcmp oeq float %73, 0.000000e+00
  br i1 %89, label %98, label %90

; <label>:90:                                     ; preds = %59
  %91 = fcmp oeq float %87, %88
  br i1 %91, label %98, label %92

; <label>:92:                                     ; preds = %90
  %93 = fcmp ogt float %88, %87
  br i1 %93, label %94, label %96

; <label>:94:                                     ; preds = %92
  %95 = fdiv float %87, %88
  br label %98

; <label>:96:                                     ; preds = %92
  %97 = fdiv float %88, %87
  br label %98

; <label>:98:                                     ; preds = %96, %94, %90, %59
  %99 = phi float [ 0.000000e+00, %59 ], [ 1.000000e+00, %90 ], [ %95, %94 ], [ %97, %96 ]
  %100 = fsub float 1.000000e+00, %99
  %101 = fsub float 1.000000e+00, %100
  %102 = fmul float %101, %101
  %103 = fmul float %102, 0x3FDB9F00A0000000
  %104 = fadd float %103, 1.000000e+00
  %105 = fmul float %101, %104
  %106 = fmul float %102, 0x3FADDC09A0000000
  %107 = fadd float %106, 0x3FE87649C0000000
  %108 = fmul float %102, %107
  %109 = fadd float %108, 1.000000e+00
  %110 = fdiv float %105, %109
  %111 = fcmp ogt float %88, %87
  %112 = fsub float 0x3FF921FB60000000, %110
  %113 = select i1 %111, float %112, float %110
  %114 = bitcast float %82 to i32
  %115 = icmp slt i32 %114, 0
  %116 = fsub float 0x400921FB60000000, %113
  %117 = select i1 %115, float %116, float %113
  %118 = tail call float @copysignf(float %117, float %73) #13
  %119 = fcmp oeq float %82, 0.000000e+00
  %120 = and i1 %89, %119
  br i1 %120, label %126, label %121

; <label>:121:                                    ; preds = %98
  %122 = fmul float %82, %82
  %123 = fmul float %73, %73
  %124 = fadd float %123, %122
  %125 = fdiv float 1.000000e+00, %124
  br label %126

; <label>:126:                                    ; preds = %98, %121
  %127 = phi float [ %125, %121 ], [ 0.000000e+00, %98 ]
  %128 = fmul float %73, %84
  %129 = fmul float %75, %82
  %130 = fsub float %128, %129
  %131 = fmul float %130, %127
  %132 = fmul float %73, %86
  %133 = fmul float %77, %82
  %134 = fsub float %132, %133
  %135 = fmul float %134, %127
  %136 = getelementptr inbounds i8, i8* %1, i64 8
  %137 = getelementptr inbounds i8, i8* %1, i64 20
  %138 = getelementptr inbounds i8, i8* %1, i64 32
  %139 = bitcast i8* %136 to float*
  %140 = load float, float* %139, align 4, !tbaa !1
  %141 = bitcast i8* %137 to float*
  %142 = load float, float* %141, align 4, !tbaa !1
  %143 = bitcast i8* %138 to float*
  %144 = load float, float* %143, align 4, !tbaa !1
  %145 = getelementptr inbounds i8, i8* %2, i64 8
  %146 = getelementptr inbounds i8, i8* %2, i64 20
  %147 = getelementptr inbounds i8, i8* %2, i64 32
  %148 = bitcast i8* %145 to float*
  %149 = load float, float* %148, align 4, !tbaa !1
  %150 = bitcast i8* %146 to float*
  %151 = load float, float* %150, align 4, !tbaa !1
  %152 = bitcast i8* %147 to float*
  %153 = load float, float* %152, align 4, !tbaa !1
  %154 = tail call float @fabsf(float %149) #13
  %155 = tail call float @fabsf(float %140) #13
  %156 = fcmp oeq float %140, 0.000000e+00
  br i1 %156, label %165, label %157

; <label>:157:                                    ; preds = %126
  %158 = fcmp oeq float %154, %155
  br i1 %158, label %165, label %159

; <label>:159:                                    ; preds = %157
  %160 = fcmp ogt float %155, %154
  br i1 %160, label %161, label %163

; <label>:161:                                    ; preds = %159
  %162 = fdiv float %154, %155
  br label %165

; <label>:163:                                    ; preds = %159
  %164 = fdiv float %155, %154
  br label %165

; <label>:165:                                    ; preds = %163, %161, %157, %126
  %166 = phi float [ 0.000000e+00, %126 ], [ 1.000000e+00, %157 ], [ %162, %161 ], [ %164, %163 ]
  %167 = fsub float 1.000000e+00, %166
  %168 = fsub float 1.000000e+00, %167
  %169 = fmul float %168, %168
  %170 = fmul float %169, 0x3FDB9F00A0000000
  %171 = fadd float %170, 1.000000e+00
  %172 = fmul float %168, %171
  %173 = fmul float %169, 0x3FADDC09A0000000
  %174 = fadd float %173, 0x3FE87649C0000000
  %175 = fmul float %169, %174
  %176 = fadd float %175, 1.000000e+00
  %177 = fdiv float %172, %176
  %178 = fcmp ogt float %155, %154
  %179 = fsub float 0x3FF921FB60000000, %177
  %180 = select i1 %178, float %179, float %177
  %181 = bitcast float %149 to i32
  %182 = icmp slt i32 %181, 0
  %183 = fsub float 0x400921FB60000000, %180
  %184 = select i1 %182, float %183, float %180
  %185 = tail call float @copysignf(float %184, float %140) #13
  %186 = fcmp oeq float %149, 0.000000e+00
  %187 = and i1 %156, %186
  br i1 %187, label %193, label %188

; <label>:188:                                    ; preds = %165
  %189 = fmul float %149, %149
  %190 = fmul float %140, %140
  %191 = fadd float %190, %189
  %192 = fdiv float 1.000000e+00, %191
  br label %193

; <label>:193:                                    ; preds = %165, %188
  %194 = phi float [ %192, %188 ], [ 0.000000e+00, %165 ]
  %195 = fmul float %140, %151
  %196 = fmul float %142, %149
  %197 = fsub float %195, %196
  %198 = fmul float %197, %194
  %199 = fmul float %140, %153
  %200 = fmul float %144, %149
  %201 = fsub float %199, %200
  %202 = fmul float %201, %194
  %203 = bitcast i8* %0 to float*
  store float %51, float* %203, align 4, !tbaa !5
  %204 = getelementptr inbounds i8, i8* %0, i64 4
  %205 = bitcast i8* %204 to float*
  store float %118, float* %205, align 4, !tbaa !7
  %206 = getelementptr inbounds i8, i8* %0, i64 8
  %207 = bitcast i8* %206 to float*
  store float %185, float* %207, align 4, !tbaa !8
  %208 = getelementptr inbounds i8, i8* %0, i64 12
  %209 = bitcast i8* %208 to float*
  store float %64, float* %209, align 4, !tbaa !5
  %210 = getelementptr inbounds i8, i8* %0, i64 16
  %211 = bitcast i8* %210 to float*
  store float %131, float* %211, align 4, !tbaa !7
  %212 = getelementptr inbounds i8, i8* %0, i64 20
  %213 = bitcast i8* %212 to float*
  store float %198, float* %213, align 4, !tbaa !8
  %214 = getelementptr inbounds i8, i8* %0, i64 24
  %215 = bitcast i8* %214 to float*
  store float %68, float* %215, align 4, !tbaa !5
  %216 = getelementptr inbounds i8, i8* %0, i64 28
  %217 = bitcast i8* %216 to float*
  store float %135, float* %217, align 4, !tbaa !7
  %218 = getelementptr inbounds i8, i8* %0, i64 32
  %219 = bitcast i8* %218 to float*
  store float %202, float* %219, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dvvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = alloca %"class.OSL::Dual2.0", align 4
  %5 = bitcast %"class.OSL::Dual2.0"* %4 to i8*
  call void @llvm.lifetime.start(i64 36, i8* %5) #2
  %6 = bitcast i8* %1 to i32*
  %7 = load i32, i32* %6, align 4, !tbaa !5
  %8 = bitcast %"class.OSL::Dual2.0"* %4 to i32*
  store i32 %7, i32* %8, align 4, !tbaa !5
  %9 = getelementptr inbounds i8, i8* %1, i64 4
  %10 = bitcast i8* %9 to i32*
  %11 = load i32, i32* %10, align 4, !tbaa !7
  %12 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 1
  %13 = bitcast float* %12 to i32*
  store i32 %11, i32* %13, align 4, !tbaa !7
  %14 = getelementptr inbounds i8, i8* %1, i64 8
  %15 = bitcast i8* %14 to i32*
  %16 = load i32, i32* %15, align 4, !tbaa !8
  %17 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 2
  %18 = bitcast float* %17 to i32*
  store i32 %16, i32* %18, align 4, !tbaa !8
  %19 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 1, i32 0
  %20 = bitcast float* %19 to i8*
  call void @llvm.memset.p0i8.i64(i8* %20, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_atan2_dvdvdv(i8* %0, i8* %5, i8* %2)
  call void @llvm.lifetime.end(i64 36, i8* %5) #2
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_atan2_dvdvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = alloca %"class.OSL::Dual2.0", align 4
  %5 = bitcast %"class.OSL::Dual2.0"* %4 to i8*
  call void @llvm.lifetime.start(i64 36, i8* %5) #2
  %6 = bitcast i8* %2 to i32*
  %7 = load i32, i32* %6, align 4, !tbaa !5
  %8 = bitcast %"class.OSL::Dual2.0"* %4 to i32*
  store i32 %7, i32* %8, align 4, !tbaa !5
  %9 = getelementptr inbounds i8, i8* %2, i64 4
  %10 = bitcast i8* %9 to i32*
  %11 = load i32, i32* %10, align 4, !tbaa !7
  %12 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 1
  %13 = bitcast float* %12 to i32*
  store i32 %11, i32* %13, align 4, !tbaa !7
  %14 = getelementptr inbounds i8, i8* %2, i64 8
  %15 = bitcast i8* %14 to i32*
  %16 = load i32, i32* %15, align 4, !tbaa !8
  %17 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 2
  %18 = bitcast float* %17 to i32*
  store i32 %16, i32* %18, align 4, !tbaa !8
  %19 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 1, i32 0
  %20 = bitcast float* %19 to i8*
  call void @llvm.memset.p0i8.i64(i8* %20, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_atan2_dvdvdv(i8* %0, i8* %1, i8* %5)
  call void @llvm.lifetime.end(i64 36, i8* %5) #2
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_sinh_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fcmp ogt float %2, 1.000000e+00
  br i1 %3, label %4, label %32

; <label>:4:                                      ; preds = %1
  %5 = fmul float %2, 0x3FF7154760000000
  %6 = fcmp olt float %5, -1.260000e+02
  %7 = fcmp ogt float %5, 1.260000e+02
  %8 = select i1 %7, float 1.260000e+02, float %5
  %9 = select i1 %6, float -1.260000e+02, float %8
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
  br label %43

; <label>:32:                                     ; preds = %1
  %33 = fsub float 1.000000e+00, %2
  %34 = fsub float 1.000000e+00, %33
  %35 = fmul float %34, %34
  %36 = fmul float %35, 0x3F2ABB46A0000000
  %37 = fadd float %36, 0x3F810F44A0000000
  %38 = fmul float %35, %37
  %39 = fadd float %38, 0x3FC5555B00000000
  %40 = fmul float %34, %39
  %41 = fmul float %35, %40
  %42 = fadd float %34, %41
  br label %43

; <label>:43:                                     ; preds = %4, %32
  %44 = phi float [ %31, %4 ], [ %42, %32 ]
  %45 = tail call float @copysignf(float %44, float %0) #13
  ret float %45
}

; Function Attrs: nounwind uwtable
define void @osl_sinh_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fmul float %5, 0x3FF7154760000000
  %7 = fcmp olt float %6, -1.260000e+02
  %8 = fcmp ogt float %6, 1.260000e+02
  %9 = select i1 %8, float 1.260000e+02, float %6
  %10 = select i1 %7, float -1.260000e+02, float %9
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
  %30 = fmul float %29, 5.000000e-01
  %31 = fdiv float 5.000000e-01, %29
  %32 = fadd float %30, %31
  %33 = fcmp ogt float %5, 1.000000e+00
  br i1 %33, label %34, label %36

; <label>:34:                                     ; preds = %2
  %35 = fsub float %30, %31
  br label %47

; <label>:36:                                     ; preds = %2
  %37 = fsub float 1.000000e+00, %5
  %38 = fsub float 1.000000e+00, %37
  %39 = fmul float %38, %38
  %40 = fmul float %39, 0x3F2ABB46A0000000
  %41 = fadd float %40, 0x3F810F44A0000000
  %42 = fmul float %39, %41
  %43 = fadd float %42, 0x3FC5555B00000000
  %44 = fmul float %38, %43
  %45 = fmul float %39, %44
  %46 = fadd float %38, %45
  br label %47

; <label>:47:                                     ; preds = %34, %36
  %48 = phi float [ %35, %34 ], [ %46, %36 ]
  %49 = tail call float @copysignf(float %48, float %4) #13
  %50 = getelementptr inbounds i8, i8* %1, i64 4
  %51 = bitcast i8* %50 to float*
  %52 = load float, float* %51, align 4, !tbaa !1
  %53 = fmul float %32, %52
  %54 = getelementptr inbounds i8, i8* %1, i64 8
  %55 = bitcast i8* %54 to float*
  %56 = load float, float* %55, align 4, !tbaa !1
  %57 = fmul float %32, %56
  %58 = insertelement <2 x float> undef, float %49, i32 0
  %59 = insertelement <2 x float> %58, float %53, i32 1
  %60 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %59, <2 x float>* %60, align 4
  %61 = getelementptr inbounds i8, i8* %0, i64 8
  %62 = bitcast i8* %61 to float*
  store float %57, float* %62, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sinh_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp ogt float %5, 1.000000e+00
  br i1 %6, label %7, label %35

; <label>:7:                                      ; preds = %2
  %8 = fmul float %5, 0x3FF7154760000000
  %9 = fcmp olt float %8, -1.260000e+02
  %10 = fcmp ogt float %8, 1.260000e+02
  %11 = select i1 %10, float 1.260000e+02, float %8
  %12 = select i1 %9, float -1.260000e+02, float %11
  %13 = fptosi float %12 to i32
  %14 = sitofp i32 %13 to float
  %15 = fsub float %12, %14
  %16 = fsub float 1.000000e+00, %15
  %17 = fsub float 1.000000e+00, %16
  %18 = fmul float %17, 0x3F55D889C0000000
  %19 = fadd float %18, 0x3F84177340000000
  %20 = fmul float %17, %19
  %21 = fadd float %20, 0x3FAC6CE660000000
  %22 = fmul float %17, %21
  %23 = fadd float %22, 0x3FCEBE3240000000
  %24 = fmul float %17, %23
  %25 = fadd float %24, 0x3FE62E3E20000000
  %26 = fmul float %17, %25
  %27 = fadd float %26, 1.000000e+00
  %28 = bitcast float %27 to i32
  %29 = shl i32 %13, 23
  %30 = add i32 %28, %29
  %31 = bitcast i32 %30 to float
  %32 = fmul float %31, 5.000000e-01
  %33 = fdiv float 5.000000e-01, %31
  %34 = fsub float %32, %33
  br label %46

; <label>:35:                                     ; preds = %2
  %36 = fsub float 1.000000e+00, %5
  %37 = fsub float 1.000000e+00, %36
  %38 = fmul float %37, %37
  %39 = fmul float %38, 0x3F2ABB46A0000000
  %40 = fadd float %39, 0x3F810F44A0000000
  %41 = fmul float %38, %40
  %42 = fadd float %41, 0x3FC5555B00000000
  %43 = fmul float %37, %42
  %44 = fmul float %38, %43
  %45 = fadd float %37, %44
  br label %46

; <label>:46:                                     ; preds = %7, %35
  %47 = phi float [ %34, %7 ], [ %45, %35 ]
  %48 = tail call float @copysignf(float %47, float %4) #13
  %49 = bitcast i8* %0 to float*
  store float %48, float* %49, align 4, !tbaa !1
  %50 = getelementptr inbounds i8, i8* %1, i64 4
  %51 = bitcast i8* %50 to float*
  %52 = load float, float* %51, align 4, !tbaa !1
  %53 = tail call float @fabsf(float %52) #13
  %54 = fcmp ogt float %53, 1.000000e+00
  br i1 %54, label %55, label %83

; <label>:55:                                     ; preds = %46
  %56 = fmul float %53, 0x3FF7154760000000
  %57 = fcmp olt float %56, -1.260000e+02
  %58 = fcmp ogt float %56, 1.260000e+02
  %59 = select i1 %58, float 1.260000e+02, float %56
  %60 = select i1 %57, float -1.260000e+02, float %59
  %61 = fptosi float %60 to i32
  %62 = sitofp i32 %61 to float
  %63 = fsub float %60, %62
  %64 = fsub float 1.000000e+00, %63
  %65 = fsub float 1.000000e+00, %64
  %66 = fmul float %65, 0x3F55D889C0000000
  %67 = fadd float %66, 0x3F84177340000000
  %68 = fmul float %65, %67
  %69 = fadd float %68, 0x3FAC6CE660000000
  %70 = fmul float %65, %69
  %71 = fadd float %70, 0x3FCEBE3240000000
  %72 = fmul float %65, %71
  %73 = fadd float %72, 0x3FE62E3E20000000
  %74 = fmul float %65, %73
  %75 = fadd float %74, 1.000000e+00
  %76 = bitcast float %75 to i32
  %77 = shl i32 %61, 23
  %78 = add i32 %76, %77
  %79 = bitcast i32 %78 to float
  %80 = fmul float %79, 5.000000e-01
  %81 = fdiv float 5.000000e-01, %79
  %82 = fsub float %80, %81
  br label %94

; <label>:83:                                     ; preds = %46
  %84 = fsub float 1.000000e+00, %53
  %85 = fsub float 1.000000e+00, %84
  %86 = fmul float %85, %85
  %87 = fmul float %86, 0x3F2ABB46A0000000
  %88 = fadd float %87, 0x3F810F44A0000000
  %89 = fmul float %86, %88
  %90 = fadd float %89, 0x3FC5555B00000000
  %91 = fmul float %85, %90
  %92 = fmul float %86, %91
  %93 = fadd float %85, %92
  br label %94

; <label>:94:                                     ; preds = %55, %83
  %95 = phi float [ %82, %55 ], [ %93, %83 ]
  %96 = tail call float @copysignf(float %95, float %52) #13
  %97 = getelementptr inbounds i8, i8* %0, i64 4
  %98 = bitcast i8* %97 to float*
  store float %96, float* %98, align 4, !tbaa !1
  %99 = getelementptr inbounds i8, i8* %1, i64 8
  %100 = bitcast i8* %99 to float*
  %101 = load float, float* %100, align 4, !tbaa !1
  %102 = tail call float @fabsf(float %101) #13
  %103 = fcmp ogt float %102, 1.000000e+00
  br i1 %103, label %104, label %132

; <label>:104:                                    ; preds = %94
  %105 = fmul float %102, 0x3FF7154760000000
  %106 = fcmp olt float %105, -1.260000e+02
  %107 = fcmp ogt float %105, 1.260000e+02
  %108 = select i1 %107, float 1.260000e+02, float %105
  %109 = select i1 %106, float -1.260000e+02, float %108
  %110 = fptosi float %109 to i32
  %111 = sitofp i32 %110 to float
  %112 = fsub float %109, %111
  %113 = fsub float 1.000000e+00, %112
  %114 = fsub float 1.000000e+00, %113
  %115 = fmul float %114, 0x3F55D889C0000000
  %116 = fadd float %115, 0x3F84177340000000
  %117 = fmul float %114, %116
  %118 = fadd float %117, 0x3FAC6CE660000000
  %119 = fmul float %114, %118
  %120 = fadd float %119, 0x3FCEBE3240000000
  %121 = fmul float %114, %120
  %122 = fadd float %121, 0x3FE62E3E20000000
  %123 = fmul float %114, %122
  %124 = fadd float %123, 1.000000e+00
  %125 = bitcast float %124 to i32
  %126 = shl i32 %110, 23
  %127 = add i32 %125, %126
  %128 = bitcast i32 %127 to float
  %129 = fmul float %128, 5.000000e-01
  %130 = fdiv float 5.000000e-01, %128
  %131 = fsub float %129, %130
  br label %143

; <label>:132:                                    ; preds = %94
  %133 = fsub float 1.000000e+00, %102
  %134 = fsub float 1.000000e+00, %133
  %135 = fmul float %134, %134
  %136 = fmul float %135, 0x3F2ABB46A0000000
  %137 = fadd float %136, 0x3F810F44A0000000
  %138 = fmul float %135, %137
  %139 = fadd float %138, 0x3FC5555B00000000
  %140 = fmul float %134, %139
  %141 = fmul float %135, %140
  %142 = fadd float %134, %141
  br label %143

; <label>:143:                                    ; preds = %104, %132
  %144 = phi float [ %131, %104 ], [ %142, %132 ]
  %145 = tail call float @copysignf(float %144, float %101) #13
  %146 = getelementptr inbounds i8, i8* %0, i64 8
  %147 = bitcast i8* %146 to float*
  store float %145, float* %147, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sinh_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = getelementptr inbounds i8, i8* %1, i64 24
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = bitcast i8* %3 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = bitcast i8* %4 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = tail call float @fabsf(float %6) #13
  %12 = fmul float %11, 0x3FF7154760000000
  %13 = fcmp olt float %12, -1.260000e+02
  %14 = fcmp ogt float %12, 1.260000e+02
  %15 = select i1 %14, float 1.260000e+02, float %12
  %16 = select i1 %13, float -1.260000e+02, float %15
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
  %36 = fmul float %35, 5.000000e-01
  %37 = fdiv float 5.000000e-01, %35
  %38 = fadd float %36, %37
  %39 = fcmp ogt float %11, 1.000000e+00
  br i1 %39, label %40, label %42

; <label>:40:                                     ; preds = %2
  %41 = fsub float %36, %37
  br label %53

; <label>:42:                                     ; preds = %2
  %43 = fsub float 1.000000e+00, %11
  %44 = fsub float 1.000000e+00, %43
  %45 = fmul float %44, %44
  %46 = fmul float %45, 0x3F2ABB46A0000000
  %47 = fadd float %46, 0x3F810F44A0000000
  %48 = fmul float %45, %47
  %49 = fadd float %48, 0x3FC5555B00000000
  %50 = fmul float %44, %49
  %51 = fmul float %45, %50
  %52 = fadd float %44, %51
  br label %53

; <label>:53:                                     ; preds = %40, %42
  %54 = phi float [ %41, %40 ], [ %52, %42 ]
  %55 = tail call float @copysignf(float %54, float %6) #13
  %56 = fmul float %8, %38
  %57 = fmul float %10, %38
  %58 = getelementptr inbounds i8, i8* %1, i64 4
  %59 = getelementptr inbounds i8, i8* %1, i64 16
  %60 = getelementptr inbounds i8, i8* %1, i64 28
  %61 = bitcast i8* %58 to float*
  %62 = load float, float* %61, align 4, !tbaa !1
  %63 = bitcast i8* %59 to float*
  %64 = load float, float* %63, align 4, !tbaa !1
  %65 = bitcast i8* %60 to float*
  %66 = load float, float* %65, align 4, !tbaa !1
  %67 = tail call float @fabsf(float %62) #13
  %68 = fmul float %67, 0x3FF7154760000000
  %69 = fcmp olt float %68, -1.260000e+02
  %70 = fcmp ogt float %68, 1.260000e+02
  %71 = select i1 %70, float 1.260000e+02, float %68
  %72 = select i1 %69, float -1.260000e+02, float %71
  %73 = fptosi float %72 to i32
  %74 = sitofp i32 %73 to float
  %75 = fsub float %72, %74
  %76 = fsub float 1.000000e+00, %75
  %77 = fsub float 1.000000e+00, %76
  %78 = fmul float %77, 0x3F55D889C0000000
  %79 = fadd float %78, 0x3F84177340000000
  %80 = fmul float %77, %79
  %81 = fadd float %80, 0x3FAC6CE660000000
  %82 = fmul float %77, %81
  %83 = fadd float %82, 0x3FCEBE3240000000
  %84 = fmul float %77, %83
  %85 = fadd float %84, 0x3FE62E3E20000000
  %86 = fmul float %77, %85
  %87 = fadd float %86, 1.000000e+00
  %88 = bitcast float %87 to i32
  %89 = shl i32 %73, 23
  %90 = add i32 %88, %89
  %91 = bitcast i32 %90 to float
  %92 = fmul float %91, 5.000000e-01
  %93 = fdiv float 5.000000e-01, %91
  %94 = fadd float %92, %93
  %95 = fcmp ogt float %67, 1.000000e+00
  br i1 %95, label %96, label %98

; <label>:96:                                     ; preds = %53
  %97 = fsub float %92, %93
  br label %109

; <label>:98:                                     ; preds = %53
  %99 = fsub float 1.000000e+00, %67
  %100 = fsub float 1.000000e+00, %99
  %101 = fmul float %100, %100
  %102 = fmul float %101, 0x3F2ABB46A0000000
  %103 = fadd float %102, 0x3F810F44A0000000
  %104 = fmul float %101, %103
  %105 = fadd float %104, 0x3FC5555B00000000
  %106 = fmul float %100, %105
  %107 = fmul float %101, %106
  %108 = fadd float %100, %107
  br label %109

; <label>:109:                                    ; preds = %96, %98
  %110 = phi float [ %97, %96 ], [ %108, %98 ]
  %111 = tail call float @copysignf(float %110, float %62) #13
  %112 = fmul float %64, %94
  %113 = fmul float %66, %94
  %114 = getelementptr inbounds i8, i8* %1, i64 8
  %115 = getelementptr inbounds i8, i8* %1, i64 20
  %116 = getelementptr inbounds i8, i8* %1, i64 32
  %117 = bitcast i8* %114 to float*
  %118 = load float, float* %117, align 4, !tbaa !1
  %119 = bitcast i8* %115 to float*
  %120 = load float, float* %119, align 4, !tbaa !1
  %121 = bitcast i8* %116 to float*
  %122 = load float, float* %121, align 4, !tbaa !1
  %123 = tail call float @fabsf(float %118) #13
  %124 = fmul float %123, 0x3FF7154760000000
  %125 = fcmp olt float %124, -1.260000e+02
  %126 = fcmp ogt float %124, 1.260000e+02
  %127 = select i1 %126, float 1.260000e+02, float %124
  %128 = select i1 %125, float -1.260000e+02, float %127
  %129 = fptosi float %128 to i32
  %130 = sitofp i32 %129 to float
  %131 = fsub float %128, %130
  %132 = fsub float 1.000000e+00, %131
  %133 = fsub float 1.000000e+00, %132
  %134 = fmul float %133, 0x3F55D889C0000000
  %135 = fadd float %134, 0x3F84177340000000
  %136 = fmul float %133, %135
  %137 = fadd float %136, 0x3FAC6CE660000000
  %138 = fmul float %133, %137
  %139 = fadd float %138, 0x3FCEBE3240000000
  %140 = fmul float %133, %139
  %141 = fadd float %140, 0x3FE62E3E20000000
  %142 = fmul float %133, %141
  %143 = fadd float %142, 1.000000e+00
  %144 = bitcast float %143 to i32
  %145 = shl i32 %129, 23
  %146 = add i32 %144, %145
  %147 = bitcast i32 %146 to float
  %148 = fmul float %147, 5.000000e-01
  %149 = fdiv float 5.000000e-01, %147
  %150 = fadd float %148, %149
  %151 = fcmp ogt float %123, 1.000000e+00
  br i1 %151, label %152, label %154

; <label>:152:                                    ; preds = %109
  %153 = fsub float %148, %149
  br label %165

; <label>:154:                                    ; preds = %109
  %155 = fsub float 1.000000e+00, %123
  %156 = fsub float 1.000000e+00, %155
  %157 = fmul float %156, %156
  %158 = fmul float %157, 0x3F2ABB46A0000000
  %159 = fadd float %158, 0x3F810F44A0000000
  %160 = fmul float %157, %159
  %161 = fadd float %160, 0x3FC5555B00000000
  %162 = fmul float %156, %161
  %163 = fmul float %157, %162
  %164 = fadd float %156, %163
  br label %165

; <label>:165:                                    ; preds = %152, %154
  %166 = phi float [ %153, %152 ], [ %164, %154 ]
  %167 = tail call float @copysignf(float %166, float %118) #13
  %168 = fmul float %120, %150
  %169 = fmul float %122, %150
  %170 = bitcast i8* %0 to float*
  store float %55, float* %170, align 4, !tbaa !5
  %171 = getelementptr inbounds i8, i8* %0, i64 4
  %172 = bitcast i8* %171 to float*
  store float %111, float* %172, align 4, !tbaa !7
  %173 = getelementptr inbounds i8, i8* %0, i64 8
  %174 = bitcast i8* %173 to float*
  store float %167, float* %174, align 4, !tbaa !8
  %175 = getelementptr inbounds i8, i8* %0, i64 12
  %176 = bitcast i8* %175 to float*
  store float %56, float* %176, align 4, !tbaa !5
  %177 = getelementptr inbounds i8, i8* %0, i64 16
  %178 = bitcast i8* %177 to float*
  store float %112, float* %178, align 4, !tbaa !7
  %179 = getelementptr inbounds i8, i8* %0, i64 20
  %180 = bitcast i8* %179 to float*
  store float %168, float* %180, align 4, !tbaa !8
  %181 = getelementptr inbounds i8, i8* %0, i64 24
  %182 = bitcast i8* %181 to float*
  store float %57, float* %182, align 4, !tbaa !5
  %183 = getelementptr inbounds i8, i8* %0, i64 28
  %184 = bitcast i8* %183 to float*
  store float %113, float* %184, align 4, !tbaa !7
  %185 = getelementptr inbounds i8, i8* %0, i64 32
  %186 = bitcast i8* %185 to float*
  store float %169, float* %186, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_cosh_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fmul float %2, 0x3FF7154760000000
  %4 = fcmp olt float %3, -1.260000e+02
  %5 = fcmp ogt float %3, 1.260000e+02
  %6 = select i1 %5, float 1.260000e+02, float %3
  %7 = select i1 %4, float -1.260000e+02, float %6
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
  %29 = fadd float %27, %28
  ret float %29
}

; Function Attrs: nounwind uwtable
define void @osl_cosh_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fmul float %5, 0x3FF7154760000000
  %7 = fcmp olt float %6, -1.260000e+02
  %8 = fcmp ogt float %6, 1.260000e+02
  %9 = select i1 %8, float 1.260000e+02, float %6
  %10 = select i1 %7, float -1.260000e+02, float %9
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
  %30 = fmul float %29, 5.000000e-01
  %31 = fdiv float 5.000000e-01, %29
  %32 = fadd float %30, %31
  %33 = fcmp ogt float %5, 1.000000e+00
  br i1 %33, label %34, label %36

; <label>:34:                                     ; preds = %2
  %35 = fsub float %30, %31
  br label %47

; <label>:36:                                     ; preds = %2
  %37 = fsub float 1.000000e+00, %5
  %38 = fsub float 1.000000e+00, %37
  %39 = fmul float %38, %38
  %40 = fmul float %39, 0x3F2ABB46A0000000
  %41 = fadd float %40, 0x3F810F44A0000000
  %42 = fmul float %39, %41
  %43 = fadd float %42, 0x3FC5555B00000000
  %44 = fmul float %38, %43
  %45 = fmul float %39, %44
  %46 = fadd float %38, %45
  br label %47

; <label>:47:                                     ; preds = %34, %36
  %48 = phi float [ %35, %34 ], [ %46, %36 ]
  %49 = tail call float @copysignf(float %48, float %4) #13
  %50 = getelementptr inbounds i8, i8* %1, i64 4
  %51 = bitcast i8* %50 to float*
  %52 = load float, float* %51, align 4, !tbaa !1
  %53 = fmul float %49, %52
  %54 = getelementptr inbounds i8, i8* %1, i64 8
  %55 = bitcast i8* %54 to float*
  %56 = load float, float* %55, align 4, !tbaa !1
  %57 = fmul float %49, %56
  %58 = insertelement <2 x float> undef, float %32, i32 0
  %59 = insertelement <2 x float> %58, float %53, i32 1
  %60 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %59, <2 x float>* %60, align 4
  %61 = getelementptr inbounds i8, i8* %0, i64 8
  %62 = bitcast i8* %61 to float*
  store float %57, float* %62, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cosh_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fmul float %5, 0x3FF7154760000000
  %7 = fcmp olt float %6, -1.260000e+02
  %8 = fcmp ogt float %6, 1.260000e+02
  %9 = select i1 %8, float 1.260000e+02, float %6
  %10 = select i1 %7, float -1.260000e+02, float %9
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
  %30 = fmul float %29, 5.000000e-01
  %31 = fdiv float 5.000000e-01, %29
  %32 = fadd float %30, %31
  %33 = bitcast i8* %0 to float*
  store float %32, float* %33, align 4, !tbaa !1
  %34 = getelementptr inbounds i8, i8* %1, i64 4
  %35 = bitcast i8* %34 to float*
  %36 = load float, float* %35, align 4, !tbaa !1
  %37 = tail call float @fabsf(float %36) #13
  %38 = fmul float %37, 0x3FF7154760000000
  %39 = fcmp olt float %38, -1.260000e+02
  %40 = fcmp ogt float %38, 1.260000e+02
  %41 = select i1 %40, float 1.260000e+02, float %38
  %42 = select i1 %39, float -1.260000e+02, float %41
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
  %64 = fadd float %62, %63
  %65 = getelementptr inbounds i8, i8* %0, i64 4
  %66 = bitcast i8* %65 to float*
  store float %64, float* %66, align 4, !tbaa !1
  %67 = getelementptr inbounds i8, i8* %1, i64 8
  %68 = bitcast i8* %67 to float*
  %69 = load float, float* %68, align 4, !tbaa !1
  %70 = tail call float @fabsf(float %69) #13
  %71 = fmul float %70, 0x3FF7154760000000
  %72 = fcmp olt float %71, -1.260000e+02
  %73 = fcmp ogt float %71, 1.260000e+02
  %74 = select i1 %73, float 1.260000e+02, float %71
  %75 = select i1 %72, float -1.260000e+02, float %74
  %76 = fptosi float %75 to i32
  %77 = sitofp i32 %76 to float
  %78 = fsub float %75, %77
  %79 = fsub float 1.000000e+00, %78
  %80 = fsub float 1.000000e+00, %79
  %81 = fmul float %80, 0x3F55D889C0000000
  %82 = fadd float %81, 0x3F84177340000000
  %83 = fmul float %80, %82
  %84 = fadd float %83, 0x3FAC6CE660000000
  %85 = fmul float %80, %84
  %86 = fadd float %85, 0x3FCEBE3240000000
  %87 = fmul float %80, %86
  %88 = fadd float %87, 0x3FE62E3E20000000
  %89 = fmul float %80, %88
  %90 = fadd float %89, 1.000000e+00
  %91 = bitcast float %90 to i32
  %92 = shl i32 %76, 23
  %93 = add i32 %91, %92
  %94 = bitcast i32 %93 to float
  %95 = fmul float %94, 5.000000e-01
  %96 = fdiv float 5.000000e-01, %94
  %97 = fadd float %95, %96
  %98 = getelementptr inbounds i8, i8* %0, i64 8
  %99 = bitcast i8* %98 to float*
  store float %97, float* %99, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cosh_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = getelementptr inbounds i8, i8* %1, i64 24
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = bitcast i8* %3 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = bitcast i8* %4 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = tail call float @fabsf(float %6) #13
  %12 = fmul float %11, 0x3FF7154760000000
  %13 = fcmp olt float %12, -1.260000e+02
  %14 = fcmp ogt float %12, 1.260000e+02
  %15 = select i1 %14, float 1.260000e+02, float %12
  %16 = select i1 %13, float -1.260000e+02, float %15
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
  %36 = fmul float %35, 5.000000e-01
  %37 = fdiv float 5.000000e-01, %35
  %38 = fadd float %36, %37
  %39 = fcmp ogt float %11, 1.000000e+00
  br i1 %39, label %40, label %42

; <label>:40:                                     ; preds = %2
  %41 = fsub float %36, %37
  br label %53

; <label>:42:                                     ; preds = %2
  %43 = fsub float 1.000000e+00, %11
  %44 = fsub float 1.000000e+00, %43
  %45 = fmul float %44, %44
  %46 = fmul float %45, 0x3F2ABB46A0000000
  %47 = fadd float %46, 0x3F810F44A0000000
  %48 = fmul float %45, %47
  %49 = fadd float %48, 0x3FC5555B00000000
  %50 = fmul float %44, %49
  %51 = fmul float %45, %50
  %52 = fadd float %44, %51
  br label %53

; <label>:53:                                     ; preds = %40, %42
  %54 = phi float [ %41, %40 ], [ %52, %42 ]
  %55 = tail call float @copysignf(float %54, float %6) #13
  %56 = fmul float %8, %55
  %57 = fmul float %10, %55
  %58 = getelementptr inbounds i8, i8* %1, i64 4
  %59 = getelementptr inbounds i8, i8* %1, i64 16
  %60 = getelementptr inbounds i8, i8* %1, i64 28
  %61 = bitcast i8* %58 to float*
  %62 = load float, float* %61, align 4, !tbaa !1
  %63 = bitcast i8* %59 to float*
  %64 = load float, float* %63, align 4, !tbaa !1
  %65 = bitcast i8* %60 to float*
  %66 = load float, float* %65, align 4, !tbaa !1
  %67 = tail call float @fabsf(float %62) #13
  %68 = fmul float %67, 0x3FF7154760000000
  %69 = fcmp olt float %68, -1.260000e+02
  %70 = fcmp ogt float %68, 1.260000e+02
  %71 = select i1 %70, float 1.260000e+02, float %68
  %72 = select i1 %69, float -1.260000e+02, float %71
  %73 = fptosi float %72 to i32
  %74 = sitofp i32 %73 to float
  %75 = fsub float %72, %74
  %76 = fsub float 1.000000e+00, %75
  %77 = fsub float 1.000000e+00, %76
  %78 = fmul float %77, 0x3F55D889C0000000
  %79 = fadd float %78, 0x3F84177340000000
  %80 = fmul float %77, %79
  %81 = fadd float %80, 0x3FAC6CE660000000
  %82 = fmul float %77, %81
  %83 = fadd float %82, 0x3FCEBE3240000000
  %84 = fmul float %77, %83
  %85 = fadd float %84, 0x3FE62E3E20000000
  %86 = fmul float %77, %85
  %87 = fadd float %86, 1.000000e+00
  %88 = bitcast float %87 to i32
  %89 = shl i32 %73, 23
  %90 = add i32 %88, %89
  %91 = bitcast i32 %90 to float
  %92 = fmul float %91, 5.000000e-01
  %93 = fdiv float 5.000000e-01, %91
  %94 = fadd float %92, %93
  %95 = fcmp ogt float %67, 1.000000e+00
  br i1 %95, label %96, label %98

; <label>:96:                                     ; preds = %53
  %97 = fsub float %92, %93
  br label %109

; <label>:98:                                     ; preds = %53
  %99 = fsub float 1.000000e+00, %67
  %100 = fsub float 1.000000e+00, %99
  %101 = fmul float %100, %100
  %102 = fmul float %101, 0x3F2ABB46A0000000
  %103 = fadd float %102, 0x3F810F44A0000000
  %104 = fmul float %101, %103
  %105 = fadd float %104, 0x3FC5555B00000000
  %106 = fmul float %100, %105
  %107 = fmul float %101, %106
  %108 = fadd float %100, %107
  br label %109

; <label>:109:                                    ; preds = %96, %98
  %110 = phi float [ %97, %96 ], [ %108, %98 ]
  %111 = tail call float @copysignf(float %110, float %62) #13
  %112 = fmul float %64, %111
  %113 = fmul float %66, %111
  %114 = getelementptr inbounds i8, i8* %1, i64 8
  %115 = getelementptr inbounds i8, i8* %1, i64 20
  %116 = getelementptr inbounds i8, i8* %1, i64 32
  %117 = bitcast i8* %114 to float*
  %118 = load float, float* %117, align 4, !tbaa !1
  %119 = bitcast i8* %115 to float*
  %120 = load float, float* %119, align 4, !tbaa !1
  %121 = bitcast i8* %116 to float*
  %122 = load float, float* %121, align 4, !tbaa !1
  %123 = tail call float @fabsf(float %118) #13
  %124 = fmul float %123, 0x3FF7154760000000
  %125 = fcmp olt float %124, -1.260000e+02
  %126 = fcmp ogt float %124, 1.260000e+02
  %127 = select i1 %126, float 1.260000e+02, float %124
  %128 = select i1 %125, float -1.260000e+02, float %127
  %129 = fptosi float %128 to i32
  %130 = sitofp i32 %129 to float
  %131 = fsub float %128, %130
  %132 = fsub float 1.000000e+00, %131
  %133 = fsub float 1.000000e+00, %132
  %134 = fmul float %133, 0x3F55D889C0000000
  %135 = fadd float %134, 0x3F84177340000000
  %136 = fmul float %133, %135
  %137 = fadd float %136, 0x3FAC6CE660000000
  %138 = fmul float %133, %137
  %139 = fadd float %138, 0x3FCEBE3240000000
  %140 = fmul float %133, %139
  %141 = fadd float %140, 0x3FE62E3E20000000
  %142 = fmul float %133, %141
  %143 = fadd float %142, 1.000000e+00
  %144 = bitcast float %143 to i32
  %145 = shl i32 %129, 23
  %146 = add i32 %144, %145
  %147 = bitcast i32 %146 to float
  %148 = fmul float %147, 5.000000e-01
  %149 = fdiv float 5.000000e-01, %147
  %150 = fadd float %148, %149
  %151 = fcmp ogt float %123, 1.000000e+00
  br i1 %151, label %152, label %154

; <label>:152:                                    ; preds = %109
  %153 = fsub float %148, %149
  br label %165

; <label>:154:                                    ; preds = %109
  %155 = fsub float 1.000000e+00, %123
  %156 = fsub float 1.000000e+00, %155
  %157 = fmul float %156, %156
  %158 = fmul float %157, 0x3F2ABB46A0000000
  %159 = fadd float %158, 0x3F810F44A0000000
  %160 = fmul float %157, %159
  %161 = fadd float %160, 0x3FC5555B00000000
  %162 = fmul float %156, %161
  %163 = fmul float %157, %162
  %164 = fadd float %156, %163
  br label %165

; <label>:165:                                    ; preds = %152, %154
  %166 = phi float [ %153, %152 ], [ %164, %154 ]
  %167 = tail call float @copysignf(float %166, float %118) #13
  %168 = fmul float %120, %167
  %169 = fmul float %122, %167
  %170 = bitcast i8* %0 to float*
  store float %38, float* %170, align 4, !tbaa !5
  %171 = getelementptr inbounds i8, i8* %0, i64 4
  %172 = bitcast i8* %171 to float*
  store float %94, float* %172, align 4, !tbaa !7
  %173 = getelementptr inbounds i8, i8* %0, i64 8
  %174 = bitcast i8* %173 to float*
  store float %150, float* %174, align 4, !tbaa !8
  %175 = getelementptr inbounds i8, i8* %0, i64 12
  %176 = bitcast i8* %175 to float*
  store float %56, float* %176, align 4, !tbaa !5
  %177 = getelementptr inbounds i8, i8* %0, i64 16
  %178 = bitcast i8* %177 to float*
  store float %112, float* %178, align 4, !tbaa !7
  %179 = getelementptr inbounds i8, i8* %0, i64 20
  %180 = bitcast i8* %179 to float*
  store float %168, float* %180, align 4, !tbaa !8
  %181 = getelementptr inbounds i8, i8* %0, i64 24
  %182 = bitcast i8* %181 to float*
  store float %57, float* %182, align 4, !tbaa !5
  %183 = getelementptr inbounds i8, i8* %0, i64 28
  %184 = bitcast i8* %183 to float*
  store float %113, float* %184, align 4, !tbaa !7
  %185 = getelementptr inbounds i8, i8* %0, i64 32
  %186 = bitcast i8* %185 to float*
  store float %169, float* %186, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_tanh_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fmul float %2, 2.000000e+00
  %4 = fmul float %3, 0x3FF7154760000000
  %5 = fcmp olt float %4, -1.260000e+02
  %6 = fcmp ogt float %4, 1.260000e+02
  %7 = select i1 %6, float 1.260000e+02, float %4
  %8 = select i1 %5, float -1.260000e+02, float %7
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
  %31 = tail call float @copysignf(float %30, float %0) #13
  ret float %31
}

; Function Attrs: nounwind uwtable
define void @osl_tanh_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fmul float %5, 2.000000e+00
  %7 = fmul float %6, 0x3FF7154760000000
  %8 = fcmp olt float %7, -1.260000e+02
  %9 = fcmp ogt float %7, 1.260000e+02
  %10 = select i1 %9, float 1.260000e+02, float %7
  %11 = select i1 %8, float -1.260000e+02, float %10
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
  %31 = fadd float %30, 1.000000e+00
  %32 = fdiv float 2.000000e+00, %31
  %33 = fsub float 1.000000e+00, %32
  %34 = tail call float @copysignf(float %33, float %4) #13
  %35 = fmul float %5, 0x3FF7154760000000
  %36 = fcmp olt float %35, -1.260000e+02
  %37 = fcmp ogt float %35, 1.260000e+02
  %38 = select i1 %37, float 1.260000e+02, float %35
  %39 = select i1 %36, float -1.260000e+02, float %38
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
  %61 = fadd float %59, %60
  %62 = fmul float %61, %61
  %63 = fdiv float 1.000000e+00, %62
  %64 = getelementptr inbounds i8, i8* %1, i64 4
  %65 = bitcast i8* %64 to float*
  %66 = load float, float* %65, align 4, !tbaa !1
  %67 = fmul float %66, %63
  %68 = getelementptr inbounds i8, i8* %1, i64 8
  %69 = bitcast i8* %68 to float*
  %70 = load float, float* %69, align 4, !tbaa !1
  %71 = fmul float %70, %63
  %72 = insertelement <2 x float> undef, float %34, i32 0
  %73 = insertelement <2 x float> %72, float %67, i32 1
  %74 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %73, <2 x float>* %74, align 4
  %75 = getelementptr inbounds i8, i8* %0, i64 8
  %76 = bitcast i8* %75 to float*
  store float %71, float* %76, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_tanh_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fmul float %5, 2.000000e+00
  %7 = fmul float %6, 0x3FF7154760000000
  %8 = fcmp olt float %7, -1.260000e+02
  %9 = fcmp ogt float %7, 1.260000e+02
  %10 = select i1 %9, float 1.260000e+02, float %7
  %11 = select i1 %8, float -1.260000e+02, float %10
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
  %31 = fadd float %30, 1.000000e+00
  %32 = fdiv float 2.000000e+00, %31
  %33 = fsub float 1.000000e+00, %32
  %34 = tail call float @copysignf(float %33, float %4) #13
  %35 = bitcast i8* %0 to float*
  store float %34, float* %35, align 4, !tbaa !1
  %36 = getelementptr inbounds i8, i8* %1, i64 4
  %37 = bitcast i8* %36 to float*
  %38 = load float, float* %37, align 4, !tbaa !1
  %39 = tail call float @fabsf(float %38) #13
  %40 = fmul float %39, 2.000000e+00
  %41 = fmul float %40, 0x3FF7154760000000
  %42 = fcmp olt float %41, -1.260000e+02
  %43 = fcmp ogt float %41, 1.260000e+02
  %44 = select i1 %43, float 1.260000e+02, float %41
  %45 = select i1 %42, float -1.260000e+02, float %44
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
  %68 = tail call float @copysignf(float %67, float %38) #13
  %69 = getelementptr inbounds i8, i8* %0, i64 4
  %70 = bitcast i8* %69 to float*
  store float %68, float* %70, align 4, !tbaa !1
  %71 = getelementptr inbounds i8, i8* %1, i64 8
  %72 = bitcast i8* %71 to float*
  %73 = load float, float* %72, align 4, !tbaa !1
  %74 = tail call float @fabsf(float %73) #13
  %75 = fmul float %74, 2.000000e+00
  %76 = fmul float %75, 0x3FF7154760000000
  %77 = fcmp olt float %76, -1.260000e+02
  %78 = fcmp ogt float %76, 1.260000e+02
  %79 = select i1 %78, float 1.260000e+02, float %76
  %80 = select i1 %77, float -1.260000e+02, float %79
  %81 = fptosi float %80 to i32
  %82 = sitofp i32 %81 to float
  %83 = fsub float %80, %82
  %84 = fsub float 1.000000e+00, %83
  %85 = fsub float 1.000000e+00, %84
  %86 = fmul float %85, 0x3F55D889C0000000
  %87 = fadd float %86, 0x3F84177340000000
  %88 = fmul float %85, %87
  %89 = fadd float %88, 0x3FAC6CE660000000
  %90 = fmul float %85, %89
  %91 = fadd float %90, 0x3FCEBE3240000000
  %92 = fmul float %85, %91
  %93 = fadd float %92, 0x3FE62E3E20000000
  %94 = fmul float %85, %93
  %95 = fadd float %94, 1.000000e+00
  %96 = bitcast float %95 to i32
  %97 = shl i32 %81, 23
  %98 = add i32 %96, %97
  %99 = bitcast i32 %98 to float
  %100 = fadd float %99, 1.000000e+00
  %101 = fdiv float 2.000000e+00, %100
  %102 = fsub float 1.000000e+00, %101
  %103 = tail call float @copysignf(float %102, float %73) #13
  %104 = getelementptr inbounds i8, i8* %0, i64 8
  %105 = bitcast i8* %104 to float*
  store float %103, float* %105, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_tanh_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = tail call float @fabsf(float %5) #13
  %7 = fmul float %6, 2.000000e+00
  %8 = fmul float %7, 0x3FF7154760000000
  %9 = fcmp olt float %8, -1.260000e+02
  %10 = fcmp ogt float %8, 1.260000e+02
  %11 = select i1 %10, float 1.260000e+02, float %8
  %12 = select i1 %9, float -1.260000e+02, float %11
  %13 = fptosi float %12 to i32
  %14 = sitofp i32 %13 to float
  %15 = fsub float %12, %14
  %16 = fsub float 1.000000e+00, %15
  %17 = fsub float 1.000000e+00, %16
  %18 = fmul float %17, 0x3F55D889C0000000
  %19 = fadd float %18, 0x3F84177340000000
  %20 = fmul float %17, %19
  %21 = fadd float %20, 0x3FAC6CE660000000
  %22 = fmul float %17, %21
  %23 = fadd float %22, 0x3FCEBE3240000000
  %24 = fmul float %17, %23
  %25 = fadd float %24, 0x3FE62E3E20000000
  %26 = fmul float %17, %25
  %27 = fadd float %26, 1.000000e+00
  %28 = bitcast float %27 to i32
  %29 = shl i32 %13, 23
  %30 = add i32 %28, %29
  %31 = bitcast i32 %30 to float
  %32 = fadd float %31, 1.000000e+00
  %33 = fdiv float 2.000000e+00, %32
  %34 = fsub float 1.000000e+00, %33
  %35 = tail call float @copysignf(float %34, float %5) #13
  %36 = fmul float %6, 0x3FF7154760000000
  %37 = fcmp olt float %36, -1.260000e+02
  %38 = fcmp ogt float %36, 1.260000e+02
  %39 = select i1 %38, float 1.260000e+02, float %36
  %40 = select i1 %37, float -1.260000e+02, float %39
  %41 = fptosi float %40 to i32
  %42 = sitofp i32 %41 to float
  %43 = fsub float %40, %42
  %44 = fsub float 1.000000e+00, %43
  %45 = fsub float 1.000000e+00, %44
  %46 = fmul float %45, 0x3F55D889C0000000
  %47 = fadd float %46, 0x3F84177340000000
  %48 = fmul float %45, %47
  %49 = fadd float %48, 0x3FAC6CE660000000
  %50 = fmul float %45, %49
  %51 = fadd float %50, 0x3FCEBE3240000000
  %52 = fmul float %45, %51
  %53 = fadd float %52, 0x3FE62E3E20000000
  %54 = fmul float %45, %53
  %55 = fadd float %54, 1.000000e+00
  %56 = bitcast float %55 to i32
  %57 = shl i32 %41, 23
  %58 = add i32 %56, %57
  %59 = bitcast i32 %58 to float
  %60 = fmul float %59, 5.000000e-01
  %61 = fdiv float 5.000000e-01, %59
  %62 = fadd float %60, %61
  %63 = fmul float %62, %62
  %64 = fdiv float 1.000000e+00, %63
  %65 = getelementptr inbounds i8, i8* %1, i64 4
  %66 = getelementptr inbounds i8, i8* %1, i64 28
  %67 = bitcast i8* %65 to float*
  %68 = load float, float* %67, align 4, !tbaa !1
  %69 = bitcast i8* %66 to float*
  %70 = load float, float* %69, align 4, !tbaa !1
  %71 = tail call float @fabsf(float %68) #13
  %72 = fmul float %71, 2.000000e+00
  %73 = fmul float %72, 0x3FF7154760000000
  %74 = fcmp olt float %73, -1.260000e+02
  %75 = fcmp ogt float %73, 1.260000e+02
  %76 = select i1 %75, float 1.260000e+02, float %73
  %77 = select i1 %74, float -1.260000e+02, float %76
  %78 = fptosi float %77 to i32
  %79 = sitofp i32 %78 to float
  %80 = fsub float %77, %79
  %81 = fsub float 1.000000e+00, %80
  %82 = fsub float 1.000000e+00, %81
  %83 = fmul float %82, 0x3F55D889C0000000
  %84 = fadd float %83, 0x3F84177340000000
  %85 = fmul float %82, %84
  %86 = fadd float %85, 0x3FAC6CE660000000
  %87 = fmul float %82, %86
  %88 = fadd float %87, 0x3FCEBE3240000000
  %89 = fmul float %82, %88
  %90 = fadd float %89, 0x3FE62E3E20000000
  %91 = fmul float %82, %90
  %92 = fadd float %91, 1.000000e+00
  %93 = bitcast float %92 to i32
  %94 = shl i32 %78, 23
  %95 = add i32 %93, %94
  %96 = bitcast i32 %95 to float
  %97 = fadd float %96, 1.000000e+00
  %98 = fdiv float 2.000000e+00, %97
  %99 = fsub float 1.000000e+00, %98
  %100 = tail call float @copysignf(float %99, float %68) #13
  %101 = fmul float %71, 0x3FF7154760000000
  %102 = fcmp olt float %101, -1.260000e+02
  %103 = fcmp ogt float %101, 1.260000e+02
  %104 = select i1 %103, float 1.260000e+02, float %101
  %105 = select i1 %102, float -1.260000e+02, float %104
  %106 = fptosi float %105 to i32
  %107 = sitofp i32 %106 to float
  %108 = fsub float %105, %107
  %109 = fsub float 1.000000e+00, %108
  %110 = fsub float 1.000000e+00, %109
  %111 = fmul float %110, 0x3F55D889C0000000
  %112 = fadd float %111, 0x3F84177340000000
  %113 = fmul float %110, %112
  %114 = fadd float %113, 0x3FAC6CE660000000
  %115 = fmul float %110, %114
  %116 = fadd float %115, 0x3FCEBE3240000000
  %117 = fmul float %110, %116
  %118 = fadd float %117, 0x3FE62E3E20000000
  %119 = fmul float %110, %118
  %120 = fadd float %119, 1.000000e+00
  %121 = bitcast float %120 to i32
  %122 = shl i32 %106, 23
  %123 = add i32 %121, %122
  %124 = bitcast i32 %123 to float
  %125 = fmul float %124, 5.000000e-01
  %126 = fdiv float 5.000000e-01, %124
  %127 = fadd float %125, %126
  %128 = fmul float %127, %127
  %129 = fdiv float 1.000000e+00, %128
  %130 = fmul float %70, %129
  %131 = getelementptr inbounds i8, i8* %1, i64 8
  %132 = getelementptr inbounds i8, i8* %1, i64 32
  %133 = bitcast i8* %131 to float*
  %134 = load float, float* %133, align 4, !tbaa !1
  %135 = bitcast i8* %3 to <4 x float>*
  %136 = load <4 x float>, <4 x float>* %135, align 4, !tbaa !1
  %137 = bitcast i8* %132 to float*
  %138 = load float, float* %137, align 4, !tbaa !1
  %139 = tail call float @fabsf(float %134) #13
  %140 = fmul float %139, 2.000000e+00
  %141 = fmul float %140, 0x3FF7154760000000
  %142 = fcmp olt float %141, -1.260000e+02
  %143 = fcmp ogt float %141, 1.260000e+02
  %144 = select i1 %143, float 1.260000e+02, float %141
  %145 = select i1 %142, float -1.260000e+02, float %144
  %146 = fptosi float %145 to i32
  %147 = sitofp i32 %146 to float
  %148 = fsub float %145, %147
  %149 = fsub float 1.000000e+00, %148
  %150 = fsub float 1.000000e+00, %149
  %151 = fmul float %150, 0x3F55D889C0000000
  %152 = fadd float %151, 0x3F84177340000000
  %153 = fmul float %150, %152
  %154 = fadd float %153, 0x3FAC6CE660000000
  %155 = fmul float %150, %154
  %156 = fadd float %155, 0x3FCEBE3240000000
  %157 = fmul float %150, %156
  %158 = fadd float %157, 0x3FE62E3E20000000
  %159 = fmul float %150, %158
  %160 = fadd float %159, 1.000000e+00
  %161 = bitcast float %160 to i32
  %162 = shl i32 %146, 23
  %163 = add i32 %161, %162
  %164 = bitcast i32 %163 to float
  %165 = fadd float %164, 1.000000e+00
  %166 = fdiv float 2.000000e+00, %165
  %167 = fsub float 1.000000e+00, %166
  %168 = tail call float @copysignf(float %167, float %134) #13
  %169 = fmul float %139, 0x3FF7154760000000
  %170 = fcmp olt float %169, -1.260000e+02
  %171 = fcmp ogt float %169, 1.260000e+02
  %172 = select i1 %171, float 1.260000e+02, float %169
  %173 = select i1 %170, float -1.260000e+02, float %172
  %174 = fptosi float %173 to i32
  %175 = sitofp i32 %174 to float
  %176 = fsub float %173, %175
  %177 = fsub float 1.000000e+00, %176
  %178 = fsub float 1.000000e+00, %177
  %179 = fmul float %178, 0x3F55D889C0000000
  %180 = fadd float %179, 0x3F84177340000000
  %181 = fmul float %178, %180
  %182 = fadd float %181, 0x3FAC6CE660000000
  %183 = fmul float %178, %182
  %184 = fadd float %183, 0x3FCEBE3240000000
  %185 = fmul float %178, %184
  %186 = fadd float %185, 0x3FE62E3E20000000
  %187 = fmul float %178, %186
  %188 = fadd float %187, 1.000000e+00
  %189 = bitcast float %188 to i32
  %190 = shl i32 %174, 23
  %191 = add i32 %189, %190
  %192 = bitcast i32 %191 to float
  %193 = fmul float %192, 5.000000e-01
  %194 = fdiv float 5.000000e-01, %192
  %195 = fadd float %193, %194
  %196 = fmul float %195, %195
  %197 = fdiv float 1.000000e+00, %196
  %198 = insertelement <4 x float> undef, float %64, i32 0
  %199 = insertelement <4 x float> %198, float %129, i32 1
  %200 = insertelement <4 x float> %199, float %197, i32 2
  %201 = insertelement <4 x float> %200, float %64, i32 3
  %202 = fmul <4 x float> %136, %201
  %203 = fmul float %138, %197
  %204 = bitcast i8* %0 to float*
  store float %35, float* %204, align 4, !tbaa !5
  %205 = getelementptr inbounds i8, i8* %0, i64 4
  %206 = bitcast i8* %205 to float*
  store float %100, float* %206, align 4, !tbaa !7
  %207 = getelementptr inbounds i8, i8* %0, i64 8
  %208 = bitcast i8* %207 to float*
  store float %168, float* %208, align 4, !tbaa !8
  %209 = getelementptr inbounds i8, i8* %0, i64 12
  %210 = bitcast i8* %209 to <4 x float>*
  store <4 x float> %202, <4 x float>* %210, align 4, !tbaa !1
  %211 = getelementptr inbounds i8, i8* %0, i64 28
  %212 = bitcast i8* %211 to float*
  store float %130, float* %212, align 4, !tbaa !7
  %213 = getelementptr inbounds i8, i8* %0, i64 32
  %214 = bitcast i8* %213 to float*
  store float %203, float* %214, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_fff(float, i8* nocapture, i8* nocapture) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = bitcast i8* %2 to float*
  %6 = fmul float %0, 0x3FD45F3060000000
  %7 = tail call float @copysignf(float 5.000000e-01, float %6) #13
  %8 = fadd float %6, %7
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fmul float %10, 3.140625e+00
  %12 = fsub float %0, %11
  %13 = fmul float %10, 0x3F4FB40000000000
  %14 = fsub float %12, %13
  %15 = fmul float %10, 0x3E84440000000000
  %16 = fsub float %14, %15
  %17 = fmul float %10, 0x3D968C2340000000
  %18 = fsub float %16, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fsub float 0x3FF921FB60000000, %19
  %21 = fmul float %20, %20
  %22 = and i32 %9, 1
  %23 = icmp ne i32 %22, 0
  %24 = fsub float -0.000000e+00, %20
  %25 = select i1 %23, float %24, float %20
  %26 = fmul float %21, 0x3EC5E150E0000000
  %27 = fadd float %26, 0xBF29F75D60000000
  %28 = fmul float %21, %27
  %29 = fadd float %28, 0x3F8110EEE0000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 0xBFC55554C0000000
  %32 = fmul float %25, %31
  %33 = fmul float %21, %32
  %34 = fadd float %25, %33
  %35 = fmul float %21, 0x3E923DB120000000
  %36 = fsub float 0x3EFA00F160000000, %35
  %37 = fmul float %21, %36
  %38 = fadd float %37, 0xBF56C16B00000000
  %39 = fmul float %21, %38
  %40 = fadd float %39, 0x3FA5555540000000
  %41 = fmul float %21, %40
  %42 = fadd float %41, -5.000000e-01
  %43 = fmul float %21, %42
  %44 = fadd float %43, 1.000000e+00
  %45 = fsub float -0.000000e+00, %44
  %46 = select i1 %23, float %45, float %44
  %47 = tail call float @fabsf(float %34) #13
  %48 = fcmp ogt float %47, 1.000000e+00
  %49 = select i1 %48, float 0.000000e+00, float %34
  %50 = tail call float @fabsf(float %46) #13
  %51 = fcmp ogt float %50, 1.000000e+00
  %52 = select i1 %51, float 0.000000e+00, float %46
  store float %49, float* %4, align 4, !tbaa !1
  store float %52, float* %5, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dfdff(i8* nocapture readonly, i8* nocapture, i8* nocapture) local_unnamed_addr #4 {
  %4 = bitcast i8* %0 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fmul float %5, 0x3FD45F3060000000
  %7 = tail call float @copysignf(float 5.000000e-01, float %6) #13
  %8 = fadd float %6, %7
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fmul float %10, 3.140625e+00
  %12 = fsub float %5, %11
  %13 = fmul float %10, 0x3F4FB40000000000
  %14 = fsub float %12, %13
  %15 = fmul float %10, 0x3E84440000000000
  %16 = fsub float %14, %15
  %17 = fmul float %10, 0x3D968C2340000000
  %18 = fsub float %16, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fsub float 0x3FF921FB60000000, %19
  %21 = fmul float %20, %20
  %22 = and i32 %9, 1
  %23 = icmp ne i32 %22, 0
  %24 = fsub float -0.000000e+00, %20
  %25 = select i1 %23, float %24, float %20
  %26 = fmul float %21, 0x3EC5E150E0000000
  %27 = fadd float %26, 0xBF29F75D60000000
  %28 = fmul float %21, %27
  %29 = fadd float %28, 0x3F8110EEE0000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 0xBFC55554C0000000
  %32 = fmul float %25, %31
  %33 = fmul float %21, %32
  %34 = fadd float %25, %33
  %35 = fmul float %21, 0x3E923DB120000000
  %36 = fsub float 0x3EFA00F160000000, %35
  %37 = fmul float %21, %36
  %38 = fadd float %37, 0xBF56C16B00000000
  %39 = fmul float %21, %38
  %40 = fadd float %39, 0x3FA5555540000000
  %41 = fmul float %21, %40
  %42 = fadd float %41, -5.000000e-01
  %43 = fmul float %21, %42
  %44 = fadd float %43, 1.000000e+00
  %45 = fsub float -0.000000e+00, %44
  %46 = select i1 %23, float %45, float %44
  %47 = tail call float @fabsf(float %34) #13
  %48 = fcmp ogt float %47, 1.000000e+00
  %49 = tail call float @fabsf(float %46) #13
  %50 = fcmp ogt float %49, 1.000000e+00
  %51 = bitcast float %34 to i32
  %52 = select i1 %48, i32 0, i32 %51
  %53 = bitcast float %46 to i32
  %54 = select i1 %50, i32 0, i32 %53
  %55 = getelementptr inbounds i8, i8* %0, i64 4
  %56 = bitcast i8* %55 to float*
  %57 = load float, float* %56, align 4, !tbaa !1
  %58 = getelementptr inbounds i8, i8* %0, i64 8
  %59 = bitcast i8* %58 to float*
  %60 = load float, float* %59, align 4, !tbaa !1
  %61 = bitcast i32 %54 to float
  %62 = fmul float %57, %61
  %63 = fmul float %60, %61
  %64 = bitcast i8* %1 to i32*
  store i32 %52, i32* %64, align 4
  %65 = getelementptr inbounds i8, i8* %1, i64 4
  %66 = bitcast i8* %65 to float*
  store float %62, float* %66, align 4
  %67 = getelementptr inbounds i8, i8* %1, i64 8
  %68 = bitcast i8* %67 to float*
  store float %63, float* %68, align 4
  %69 = bitcast i8* %2 to i32*
  store i32 %54, i32* %69, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dffdf(i8* nocapture readonly, i8* nocapture, i8* nocapture) local_unnamed_addr #4 {
  %4 = bitcast i8* %0 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fmul float %5, 0x3FD45F3060000000
  %7 = tail call float @copysignf(float 5.000000e-01, float %6) #13
  %8 = fadd float %6, %7
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fmul float %10, 3.140625e+00
  %12 = fsub float %5, %11
  %13 = fmul float %10, 0x3F4FB40000000000
  %14 = fsub float %12, %13
  %15 = fmul float %10, 0x3E84440000000000
  %16 = fsub float %14, %15
  %17 = fmul float %10, 0x3D968C2340000000
  %18 = fsub float %16, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fsub float 0x3FF921FB60000000, %19
  %21 = fmul float %20, %20
  %22 = and i32 %9, 1
  %23 = icmp ne i32 %22, 0
  %24 = fsub float -0.000000e+00, %20
  %25 = select i1 %23, float %24, float %20
  %26 = fmul float %21, 0x3EC5E150E0000000
  %27 = fadd float %26, 0xBF29F75D60000000
  %28 = fmul float %21, %27
  %29 = fadd float %28, 0x3F8110EEE0000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 0xBFC55554C0000000
  %32 = fmul float %25, %31
  %33 = fmul float %21, %32
  %34 = fadd float %25, %33
  %35 = fmul float %21, 0x3E923DB120000000
  %36 = fsub float 0x3EFA00F160000000, %35
  %37 = fmul float %21, %36
  %38 = fadd float %37, 0xBF56C16B00000000
  %39 = fmul float %21, %38
  %40 = fadd float %39, 0x3FA5555540000000
  %41 = fmul float %21, %40
  %42 = fadd float %41, -5.000000e-01
  %43 = fmul float %21, %42
  %44 = fadd float %43, 1.000000e+00
  %45 = fsub float -0.000000e+00, %44
  %46 = select i1 %23, float %45, float %44
  %47 = tail call float @fabsf(float %34) #13
  %48 = fcmp ogt float %47, 1.000000e+00
  %49 = tail call float @fabsf(float %46) #13
  %50 = fcmp ogt float %49, 1.000000e+00
  %51 = bitcast float %34 to i32
  %52 = select i1 %48, i32 0, i32 %51
  %53 = bitcast float %46 to i32
  %54 = select i1 %50, i32 0, i32 %53
  %55 = getelementptr inbounds i8, i8* %0, i64 4
  %56 = bitcast i8* %55 to float*
  %57 = load float, float* %56, align 4, !tbaa !1
  %58 = getelementptr inbounds i8, i8* %0, i64 8
  %59 = bitcast i8* %58 to float*
  %60 = load float, float* %59, align 4, !tbaa !1
  %61 = bitcast i8* %1 to i32*
  store i32 %52, i32* %61, align 4, !tbaa !1
  %62 = bitcast i32 %52 to float
  %63 = fmul float %57, %62
  %64 = fsub float -0.000000e+00, %63
  %65 = fmul float %60, %62
  %66 = fsub float -0.000000e+00, %65
  %67 = bitcast i8* %2 to i32*
  store i32 %54, i32* %67, align 4
  %68 = getelementptr inbounds i8, i8* %2, i64 4
  %69 = bitcast i8* %68 to float*
  store float %64, float* %69, align 4
  %70 = getelementptr inbounds i8, i8* %2, i64 8
  %71 = bitcast i8* %70 to float*
  store float %66, float* %71, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dfdfdf(i8* nocapture readonly, i8* nocapture, i8* nocapture) local_unnamed_addr #4 {
  %4 = bitcast i8* %0 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fmul float %5, 0x3FD45F3060000000
  %7 = tail call float @copysignf(float 5.000000e-01, float %6) #13
  %8 = fadd float %6, %7
  %9 = fptosi float %8 to i32
  %10 = sitofp i32 %9 to float
  %11 = fmul float %10, 3.140625e+00
  %12 = fsub float %5, %11
  %13 = fmul float %10, 0x3F4FB40000000000
  %14 = fsub float %12, %13
  %15 = fmul float %10, 0x3E84440000000000
  %16 = fsub float %14, %15
  %17 = fmul float %10, 0x3D968C2340000000
  %18 = fsub float %16, %17
  %19 = fsub float 0x3FF921FB60000000, %18
  %20 = fsub float 0x3FF921FB60000000, %19
  %21 = fmul float %20, %20
  %22 = and i32 %9, 1
  %23 = icmp ne i32 %22, 0
  %24 = fsub float -0.000000e+00, %20
  %25 = select i1 %23, float %24, float %20
  %26 = fmul float %21, 0x3EC5E150E0000000
  %27 = fadd float %26, 0xBF29F75D60000000
  %28 = fmul float %21, %27
  %29 = fadd float %28, 0x3F8110EEE0000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 0xBFC55554C0000000
  %32 = fmul float %25, %31
  %33 = fmul float %21, %32
  %34 = fadd float %25, %33
  %35 = fmul float %21, 0x3E923DB120000000
  %36 = fsub float 0x3EFA00F160000000, %35
  %37 = fmul float %21, %36
  %38 = fadd float %37, 0xBF56C16B00000000
  %39 = fmul float %21, %38
  %40 = fadd float %39, 0x3FA5555540000000
  %41 = fmul float %21, %40
  %42 = fadd float %41, -5.000000e-01
  %43 = fmul float %21, %42
  %44 = fadd float %43, 1.000000e+00
  %45 = fsub float -0.000000e+00, %44
  %46 = select i1 %23, float %45, float %44
  %47 = tail call float @fabsf(float %34) #13
  %48 = fcmp ogt float %47, 1.000000e+00
  %49 = tail call float @fabsf(float %46) #13
  %50 = fcmp ogt float %49, 1.000000e+00
  %51 = bitcast float %34 to i32
  %52 = select i1 %48, i32 0, i32 %51
  %53 = bitcast float %46 to i32
  %54 = select i1 %50, i32 0, i32 %53
  %55 = getelementptr inbounds i8, i8* %0, i64 4
  %56 = bitcast i8* %55 to float*
  %57 = load float, float* %56, align 4, !tbaa !1
  %58 = getelementptr inbounds i8, i8* %0, i64 8
  %59 = bitcast i8* %58 to float*
  %60 = load float, float* %59, align 4, !tbaa !1
  %61 = bitcast i32 %54 to float
  %62 = fmul float %57, %61
  %63 = fmul float %60, %61
  %64 = bitcast i8* %1 to i32*
  store i32 %52, i32* %64, align 4
  %65 = getelementptr inbounds i8, i8* %1, i64 4
  %66 = bitcast i8* %65 to float*
  store float %62, float* %66, align 4
  %67 = getelementptr inbounds i8, i8* %1, i64 8
  %68 = bitcast i8* %67 to float*
  store float %63, float* %68, align 4
  %69 = bitcast i32 %52 to float
  %70 = fmul float %57, %69
  %71 = fsub float -0.000000e+00, %70
  %72 = fmul float %60, %69
  %73 = fsub float -0.000000e+00, %72
  %74 = bitcast i8* %2 to i32*
  store i32 %54, i32* %74, align 4
  %75 = getelementptr inbounds i8, i8* %2, i64 4
  %76 = bitcast i8* %75 to float*
  store float %71, float* %76, align 4
  %77 = getelementptr inbounds i8, i8* %2, i64 8
  %78 = bitcast i8* %77 to float*
  store float %73, float* %78, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_vvv(i8* nocapture readonly, i8* nocapture, i8* nocapture) local_unnamed_addr #4 {
  %4 = bitcast i8* %0 to float*
  %5 = bitcast i8* %1 to float*
  %6 = bitcast i8* %2 to float*
  br label %8

; <label>:7:                                      ; preds = %8
  ret void

; <label>:8:                                      ; preds = %8, %3
  %9 = phi i64 [ 0, %3 ], [ %61, %8 ]
  %10 = getelementptr inbounds float, float* %4, i64 %9
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = getelementptr inbounds float, float* %5, i64 %9
  %13 = getelementptr inbounds float, float* %6, i64 %9
  %14 = fmul float %11, 0x3FD45F3060000000
  %15 = tail call float @copysignf(float 5.000000e-01, float %14) #13
  %16 = fadd float %14, %15
  %17 = fptosi float %16 to i32
  %18 = sitofp i32 %17 to float
  %19 = fmul float %18, 3.140625e+00
  %20 = fsub float %11, %19
  %21 = fmul float %18, 0x3F4FB40000000000
  %22 = fsub float %20, %21
  %23 = fmul float %18, 0x3E84440000000000
  %24 = fsub float %22, %23
  %25 = fmul float %18, 0x3D968C2340000000
  %26 = fsub float %24, %25
  %27 = fsub float 0x3FF921FB60000000, %26
  %28 = fsub float 0x3FF921FB60000000, %27
  %29 = fmul float %28, %28
  %30 = and i32 %17, 1
  %31 = icmp ne i32 %30, 0
  %32 = fsub float -0.000000e+00, %28
  %33 = select i1 %31, float %32, float %28
  %34 = fmul float %29, 0x3EC5E150E0000000
  %35 = fadd float %34, 0xBF29F75D60000000
  %36 = fmul float %29, %35
  %37 = fadd float %36, 0x3F8110EEE0000000
  %38 = fmul float %29, %37
  %39 = fadd float %38, 0xBFC55554C0000000
  %40 = fmul float %33, %39
  %41 = fmul float %29, %40
  %42 = fadd float %33, %41
  %43 = fmul float %29, 0x3E923DB120000000
  %44 = fsub float 0x3EFA00F160000000, %43
  %45 = fmul float %29, %44
  %46 = fadd float %45, 0xBF56C16B00000000
  %47 = fmul float %29, %46
  %48 = fadd float %47, 0x3FA5555540000000
  %49 = fmul float %29, %48
  %50 = fadd float %49, -5.000000e-01
  %51 = fmul float %29, %50
  %52 = fadd float %51, 1.000000e+00
  %53 = fsub float -0.000000e+00, %52
  %54 = select i1 %31, float %53, float %52
  %55 = tail call float @fabsf(float %42) #13
  %56 = fcmp ogt float %55, 1.000000e+00
  %57 = select i1 %56, float 0.000000e+00, float %42
  %58 = tail call float @fabsf(float %54) #13
  %59 = fcmp ogt float %58, 1.000000e+00
  %60 = select i1 %59, float 0.000000e+00, float %54
  store float %57, float* %12, align 4, !tbaa !1
  store float %60, float* %13, align 4, !tbaa !1
  %61 = add nuw nsw i64 %9, 1
  %62 = icmp eq i64 %61, 3
  br i1 %62, label %7, label %8
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dvdvv(i8* nocapture readonly, i8* nocapture, i8* nocapture) local_unnamed_addr #4 {
  %4 = bitcast i8* %0 to float*
  %5 = getelementptr inbounds i8, i8* %0, i64 12
  %6 = bitcast i8* %5 to float*
  %7 = getelementptr inbounds i8, i8* %0, i64 24
  %8 = bitcast i8* %7 to float*
  %9 = bitcast i8* %1 to float*
  %10 = getelementptr inbounds i8, i8* %1, i64 12
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8, i8* %1, i64 24
  %13 = bitcast i8* %12 to float*
  %14 = bitcast i8* %2 to float*
  br label %16

; <label>:15:                                     ; preds = %16
  ret void

; <label>:16:                                     ; preds = %16, %3
  %17 = phi i64 [ 0, %3 ], [ %82, %16 ]
  %18 = getelementptr inbounds float, float* %4, i64 %17
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = fmul float %19, 0x3FD45F3060000000
  %21 = tail call float @copysignf(float 5.000000e-01, float %20) #13
  %22 = fadd float %20, %21
  %23 = fptosi float %22 to i32
  %24 = sitofp i32 %23 to float
  %25 = fmul float %24, 3.140625e+00
  %26 = fsub float %19, %25
  %27 = fmul float %24, 0x3F4FB40000000000
  %28 = fsub float %26, %27
  %29 = fmul float %24, 0x3E84440000000000
  %30 = fsub float %28, %29
  %31 = fmul float %24, 0x3D968C2340000000
  %32 = fsub float %30, %31
  %33 = fsub float 0x3FF921FB60000000, %32
  %34 = fsub float 0x3FF921FB60000000, %33
  %35 = fmul float %34, %34
  %36 = and i32 %23, 1
  %37 = icmp ne i32 %36, 0
  %38 = fsub float -0.000000e+00, %34
  %39 = select i1 %37, float %38, float %34
  %40 = fmul float %35, 0x3EC5E150E0000000
  %41 = fadd float %40, 0xBF29F75D60000000
  %42 = fmul float %35, %41
  %43 = fadd float %42, 0x3F8110EEE0000000
  %44 = fmul float %35, %43
  %45 = fadd float %44, 0xBFC55554C0000000
  %46 = fmul float %39, %45
  %47 = fmul float %35, %46
  %48 = fadd float %39, %47
  %49 = fmul float %35, 0x3E923DB120000000
  %50 = fsub float 0x3EFA00F160000000, %49
  %51 = fmul float %35, %50
  %52 = fadd float %51, 0xBF56C16B00000000
  %53 = fmul float %35, %52
  %54 = fadd float %53, 0x3FA5555540000000
  %55 = fmul float %35, %54
  %56 = fadd float %55, -5.000000e-01
  %57 = fmul float %35, %56
  %58 = fadd float %57, 1.000000e+00
  %59 = fsub float -0.000000e+00, %58
  %60 = select i1 %37, float %59, float %58
  %61 = tail call float @fabsf(float %48) #13
  %62 = fcmp ogt float %61, 1.000000e+00
  %63 = tail call float @fabsf(float %60) #13
  %64 = fcmp ogt float %63, 1.000000e+00
  %65 = bitcast float %48 to i32
  %66 = select i1 %62, i32 0, i32 %65
  %67 = bitcast float %60 to i32
  %68 = select i1 %64, i32 0, i32 %67
  %69 = getelementptr inbounds float, float* %6, i64 %17
  %70 = load float, float* %69, align 4, !tbaa !1
  %71 = getelementptr inbounds float, float* %8, i64 %17
  %72 = load float, float* %71, align 4, !tbaa !1
  %73 = getelementptr inbounds float, float* %9, i64 %17
  %74 = bitcast float* %73 to i32*
  store i32 %66, i32* %74, align 4, !tbaa !1
  %75 = bitcast i32 %68 to float
  %76 = fmul float %70, %75
  %77 = getelementptr inbounds float, float* %11, i64 %17
  store float %76, float* %77, align 4, !tbaa !1
  %78 = fmul float %72, %75
  %79 = getelementptr inbounds float, float* %13, i64 %17
  store float %78, float* %79, align 4, !tbaa !1
  %80 = getelementptr inbounds float, float* %14, i64 %17
  %81 = bitcast float* %80 to i32*
  store i32 %68, i32* %81, align 4, !tbaa !1
  %82 = add nuw nsw i64 %17, 1
  %83 = icmp eq i64 %82, 3
  br i1 %83, label %15, label %16
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dvvdv(i8* nocapture readonly, i8* nocapture, i8* nocapture) local_unnamed_addr #4 {
  %4 = bitcast i8* %0 to float*
  %5 = getelementptr inbounds i8, i8* %0, i64 12
  %6 = bitcast i8* %5 to float*
  %7 = getelementptr inbounds i8, i8* %0, i64 24
  %8 = bitcast i8* %7 to float*
  %9 = bitcast i8* %1 to float*
  %10 = bitcast i8* %2 to float*
  %11 = getelementptr inbounds i8, i8* %2, i64 12
  %12 = bitcast i8* %11 to float*
  %13 = getelementptr inbounds i8, i8* %2, i64 24
  %14 = bitcast i8* %13 to float*
  br label %16

; <label>:15:                                     ; preds = %16
  ret void

; <label>:16:                                     ; preds = %16, %3
  %17 = phi i64 [ 0, %3 ], [ %84, %16 ]
  %18 = getelementptr inbounds float, float* %4, i64 %17
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = fmul float %19, 0x3FD45F3060000000
  %21 = tail call float @copysignf(float 5.000000e-01, float %20) #13
  %22 = fadd float %20, %21
  %23 = fptosi float %22 to i32
  %24 = sitofp i32 %23 to float
  %25 = fmul float %24, 3.140625e+00
  %26 = fsub float %19, %25
  %27 = fmul float %24, 0x3F4FB40000000000
  %28 = fsub float %26, %27
  %29 = fmul float %24, 0x3E84440000000000
  %30 = fsub float %28, %29
  %31 = fmul float %24, 0x3D968C2340000000
  %32 = fsub float %30, %31
  %33 = fsub float 0x3FF921FB60000000, %32
  %34 = fsub float 0x3FF921FB60000000, %33
  %35 = fmul float %34, %34
  %36 = and i32 %23, 1
  %37 = icmp ne i32 %36, 0
  %38 = fsub float -0.000000e+00, %34
  %39 = select i1 %37, float %38, float %34
  %40 = fmul float %35, 0x3EC5E150E0000000
  %41 = fadd float %40, 0xBF29F75D60000000
  %42 = fmul float %35, %41
  %43 = fadd float %42, 0x3F8110EEE0000000
  %44 = fmul float %35, %43
  %45 = fadd float %44, 0xBFC55554C0000000
  %46 = fmul float %39, %45
  %47 = fmul float %35, %46
  %48 = fadd float %39, %47
  %49 = fmul float %35, 0x3E923DB120000000
  %50 = fsub float 0x3EFA00F160000000, %49
  %51 = fmul float %35, %50
  %52 = fadd float %51, 0xBF56C16B00000000
  %53 = fmul float %35, %52
  %54 = fadd float %53, 0x3FA5555540000000
  %55 = fmul float %35, %54
  %56 = fadd float %55, -5.000000e-01
  %57 = fmul float %35, %56
  %58 = fadd float %57, 1.000000e+00
  %59 = fsub float -0.000000e+00, %58
  %60 = select i1 %37, float %59, float %58
  %61 = tail call float @fabsf(float %48) #13
  %62 = fcmp ogt float %61, 1.000000e+00
  %63 = tail call float @fabsf(float %60) #13
  %64 = fcmp ogt float %63, 1.000000e+00
  %65 = bitcast float %48 to i32
  %66 = select i1 %62, i32 0, i32 %65
  %67 = bitcast float %60 to i32
  %68 = select i1 %64, i32 0, i32 %67
  %69 = getelementptr inbounds float, float* %6, i64 %17
  %70 = load float, float* %69, align 4, !tbaa !1
  %71 = getelementptr inbounds float, float* %8, i64 %17
  %72 = load float, float* %71, align 4, !tbaa !1
  %73 = getelementptr inbounds float, float* %9, i64 %17
  %74 = bitcast float* %73 to i32*
  store i32 %66, i32* %74, align 4, !tbaa !1
  %75 = getelementptr inbounds float, float* %10, i64 %17
  %76 = bitcast float* %75 to i32*
  store i32 %68, i32* %76, align 4, !tbaa !1
  %77 = bitcast i32 %66 to float
  %78 = fmul float %70, %77
  %79 = fsub float -0.000000e+00, %78
  %80 = getelementptr inbounds float, float* %12, i64 %17
  store float %79, float* %80, align 4, !tbaa !1
  %81 = fmul float %72, %77
  %82 = fsub float -0.000000e+00, %81
  %83 = getelementptr inbounds float, float* %14, i64 %17
  store float %82, float* %83, align 4, !tbaa !1
  %84 = add nuw nsw i64 %17, 1
  %85 = icmp eq i64 %84, 3
  br i1 %85, label %15, label %16
}

; Function Attrs: nounwind uwtable
define void @osl_sincos_dvdvdv(i8* nocapture readonly, i8* nocapture, i8* nocapture) local_unnamed_addr #4 {
  %4 = bitcast i8* %0 to float*
  %5 = getelementptr inbounds i8, i8* %0, i64 12
  %6 = bitcast i8* %5 to float*
  %7 = getelementptr inbounds i8, i8* %0, i64 24
  %8 = bitcast i8* %7 to float*
  %9 = bitcast i8* %1 to float*
  %10 = getelementptr inbounds i8, i8* %1, i64 12
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8, i8* %1, i64 24
  %13 = bitcast i8* %12 to float*
  %14 = bitcast i8* %2 to float*
  %15 = getelementptr inbounds i8, i8* %2, i64 12
  %16 = bitcast i8* %15 to float*
  %17 = getelementptr inbounds i8, i8* %2, i64 24
  %18 = bitcast i8* %17 to float*
  br label %20

; <label>:19:                                     ; preds = %20
  ret void

; <label>:20:                                     ; preds = %20, %3
  %21 = phi i64 [ 0, %3 ], [ %93, %20 ]
  %22 = getelementptr inbounds float, float* %4, i64 %21
  %23 = load float, float* %22, align 4, !tbaa !1
  %24 = fmul float %23, 0x3FD45F3060000000
  %25 = tail call float @copysignf(float 5.000000e-01, float %24) #13
  %26 = fadd float %24, %25
  %27 = fptosi float %26 to i32
  %28 = sitofp i32 %27 to float
  %29 = fmul float %28, 3.140625e+00
  %30 = fsub float %23, %29
  %31 = fmul float %28, 0x3F4FB40000000000
  %32 = fsub float %30, %31
  %33 = fmul float %28, 0x3E84440000000000
  %34 = fsub float %32, %33
  %35 = fmul float %28, 0x3D968C2340000000
  %36 = fsub float %34, %35
  %37 = fsub float 0x3FF921FB60000000, %36
  %38 = fsub float 0x3FF921FB60000000, %37
  %39 = fmul float %38, %38
  %40 = and i32 %27, 1
  %41 = icmp ne i32 %40, 0
  %42 = fsub float -0.000000e+00, %38
  %43 = select i1 %41, float %42, float %38
  %44 = fmul float %39, 0x3EC5E150E0000000
  %45 = fadd float %44, 0xBF29F75D60000000
  %46 = fmul float %39, %45
  %47 = fadd float %46, 0x3F8110EEE0000000
  %48 = fmul float %39, %47
  %49 = fadd float %48, 0xBFC55554C0000000
  %50 = fmul float %43, %49
  %51 = fmul float %39, %50
  %52 = fadd float %43, %51
  %53 = fmul float %39, 0x3E923DB120000000
  %54 = fsub float 0x3EFA00F160000000, %53
  %55 = fmul float %39, %54
  %56 = fadd float %55, 0xBF56C16B00000000
  %57 = fmul float %39, %56
  %58 = fadd float %57, 0x3FA5555540000000
  %59 = fmul float %39, %58
  %60 = fadd float %59, -5.000000e-01
  %61 = fmul float %39, %60
  %62 = fadd float %61, 1.000000e+00
  %63 = fsub float -0.000000e+00, %62
  %64 = select i1 %41, float %63, float %62
  %65 = tail call float @fabsf(float %52) #13
  %66 = fcmp ogt float %65, 1.000000e+00
  %67 = tail call float @fabsf(float %64) #13
  %68 = fcmp ogt float %67, 1.000000e+00
  %69 = bitcast float %52 to i32
  %70 = select i1 %66, i32 0, i32 %69
  %71 = bitcast float %64 to i32
  %72 = select i1 %68, i32 0, i32 %71
  %73 = getelementptr inbounds float, float* %6, i64 %21
  %74 = load float, float* %73, align 4, !tbaa !1
  %75 = getelementptr inbounds float, float* %8, i64 %21
  %76 = load float, float* %75, align 4, !tbaa !1
  %77 = getelementptr inbounds float, float* %9, i64 %21
  %78 = bitcast float* %77 to i32*
  store i32 %70, i32* %78, align 4, !tbaa !1
  %79 = bitcast i32 %72 to float
  %80 = fmul float %74, %79
  %81 = getelementptr inbounds float, float* %11, i64 %21
  store float %80, float* %81, align 4, !tbaa !1
  %82 = fmul float %76, %79
  %83 = getelementptr inbounds float, float* %13, i64 %21
  store float %82, float* %83, align 4, !tbaa !1
  %84 = getelementptr inbounds float, float* %14, i64 %21
  %85 = bitcast float* %84 to i32*
  store i32 %72, i32* %85, align 4, !tbaa !1
  %86 = bitcast i32 %70 to float
  %87 = fmul float %74, %86
  %88 = fsub float -0.000000e+00, %87
  %89 = getelementptr inbounds float, float* %16, i64 %21
  store float %88, float* %89, align 4, !tbaa !1
  %90 = fmul float %76, %86
  %91 = fsub float -0.000000e+00, %90
  %92 = getelementptr inbounds float, float* %18, i64 %21
  store float %91, float* %92, align 4, !tbaa !1
  %93 = add nuw nsw i64 %21, 1
  %94 = icmp eq i64 %93, 3
  br i1 %94, label %19, label %20
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_log_ff(float) local_unnamed_addr #6 {
  %2 = fcmp olt float %0, 0x3810000000000000
  %3 = fcmp ogt float %0, 0x47EFFFFFE0000000
  %4 = select i1 %3, float 0x47EFFFFFE0000000, float %0
  %5 = bitcast float %4 to i32
  %6 = select i1 %2, i32 8388608, i32 %5
  %7 = lshr i32 %6, 23
  %8 = add nsw i32 %7, -127
  %9 = and i32 %6, 8388607
  %10 = or i32 %9, 1065353216
  %11 = bitcast i32 %10 to float
  %12 = fadd float %11, -1.000000e+00
  %13 = fmul float %12, %12
  %14 = fmul float %13, %13
  %15 = fmul float %12, 0x3F831161A0000000
  %16 = fsub float 0x3FAAA83920000000, %15
  %17 = fmul float %12, 0x3FDEA2C5A0000000
  %18 = fadd float %17, 0xBFE713CA80000000
  %19 = fmul float %12, %16
  %20 = fadd float %19, 0xBFC19A9FA0000000
  %21 = fmul float %12, %20
  %22 = fadd float %21, 0x3FCEF5B7A0000000
  %23 = fmul float %12, %22
  %24 = fadd float %23, 0xBFD63A40C0000000
  %25 = fmul float %12, %18
  %26 = fadd float %25, 0x3FF7154200000000
  %27 = fmul float %14, %24
  %28 = fmul float %12, %26
  %29 = fadd float %28, %27
  %30 = sitofp i32 %8 to float
  %31 = fadd float %30, %29
  %32 = fmul float %31, 0x3FE62E4300000000
  ret float %32
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_log_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, 0x3810000000000000
  %6 = fcmp ogt float %4, 0x47EFFFFFE0000000
  %7 = select i1 %6, float 0x47EFFFFFE0000000, float %4
  %8 = bitcast float %7 to i32
  %9 = select i1 %5, i32 8388608, i32 %8
  %10 = lshr i32 %9, 23
  %11 = add nsw i32 %10, -127
  %12 = and i32 %9, 8388607
  %13 = or i32 %12, 1065353216
  %14 = bitcast i32 %13 to float
  %15 = fadd float %14, -1.000000e+00
  %16 = fmul float %15, %15
  %17 = fmul float %16, %16
  %18 = fmul float %15, 0x3F831161A0000000
  %19 = fsub float 0x3FAAA83920000000, %18
  %20 = fmul float %15, 0x3FDEA2C5A0000000
  %21 = fadd float %20, 0xBFE713CA80000000
  %22 = fmul float %15, %19
  %23 = fadd float %22, 0xBFC19A9FA0000000
  %24 = fmul float %15, %23
  %25 = fadd float %24, 0x3FCEF5B7A0000000
  %26 = fmul float %15, %25
  %27 = fadd float %26, 0xBFD63A40C0000000
  %28 = fmul float %15, %21
  %29 = fadd float %28, 0x3FF7154200000000
  %30 = fmul float %17, %27
  %31 = fmul float %15, %29
  %32 = fadd float %31, %30
  %33 = sitofp i32 %11 to float
  %34 = fadd float %33, %32
  %35 = fmul float %34, 0x3FE62E4300000000
  %36 = fdiv float 1.000000e+00, %4
  %37 = select i1 %5, float 0.000000e+00, float %36
  %38 = getelementptr inbounds i8, i8* %1, i64 4
  %39 = bitcast i8* %38 to float*
  %40 = load float, float* %39, align 4, !tbaa !1
  %41 = fmul float %40, %37
  %42 = getelementptr inbounds i8, i8* %1, i64 8
  %43 = bitcast i8* %42 to float*
  %44 = load float, float* %43, align 4, !tbaa !1
  %45 = fmul float %37, %44
  %46 = insertelement <2 x float> undef, float %35, i32 0
  %47 = insertelement <2 x float> %46, float %41, i32 1
  %48 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %47, <2 x float>* %48, align 4
  %49 = getelementptr inbounds i8, i8* %0, i64 8
  %50 = bitcast i8* %49 to float*
  store float %45, float* %50, align 4
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_log_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, 0x3810000000000000
  %6 = fcmp ogt float %4, 0x47EFFFFFE0000000
  %7 = select i1 %6, float 0x47EFFFFFE0000000, float %4
  %8 = bitcast float %7 to i32
  %9 = select i1 %5, i32 8388608, i32 %8
  %10 = lshr i32 %9, 23
  %11 = add nsw i32 %10, -127
  %12 = and i32 %9, 8388607
  %13 = or i32 %12, 1065353216
  %14 = bitcast i32 %13 to float
  %15 = fadd float %14, -1.000000e+00
  %16 = fmul float %15, %15
  %17 = fmul float %16, %16
  %18 = fmul float %15, 0x3F831161A0000000
  %19 = fsub float 0x3FAAA83920000000, %18
  %20 = fmul float %15, 0x3FDEA2C5A0000000
  %21 = fadd float %20, 0xBFE713CA80000000
  %22 = fmul float %15, %19
  %23 = fadd float %22, 0xBFC19A9FA0000000
  %24 = fmul float %15, %23
  %25 = fadd float %24, 0x3FCEF5B7A0000000
  %26 = fmul float %15, %25
  %27 = fadd float %26, 0xBFD63A40C0000000
  %28 = fmul float %15, %21
  %29 = fadd float %28, 0x3FF7154200000000
  %30 = fmul float %17, %27
  %31 = fmul float %15, %29
  %32 = fadd float %31, %30
  %33 = sitofp i32 %11 to float
  %34 = fadd float %33, %32
  %35 = fmul float %34, 0x3FE62E4300000000
  %36 = bitcast i8* %0 to float*
  store float %35, float* %36, align 4, !tbaa !1
  %37 = getelementptr inbounds i8, i8* %1, i64 4
  %38 = bitcast i8* %37 to float*
  %39 = load float, float* %38, align 4, !tbaa !1
  %40 = fcmp olt float %39, 0x3810000000000000
  %41 = fcmp ogt float %39, 0x47EFFFFFE0000000
  %42 = select i1 %41, float 0x47EFFFFFE0000000, float %39
  %43 = bitcast float %42 to i32
  %44 = select i1 %40, i32 8388608, i32 %43
  %45 = lshr i32 %44, 23
  %46 = add nsw i32 %45, -127
  %47 = and i32 %44, 8388607
  %48 = or i32 %47, 1065353216
  %49 = bitcast i32 %48 to float
  %50 = fadd float %49, -1.000000e+00
  %51 = fmul float %50, %50
  %52 = fmul float %51, %51
  %53 = fmul float %50, 0x3F831161A0000000
  %54 = fsub float 0x3FAAA83920000000, %53
  %55 = fmul float %50, 0x3FDEA2C5A0000000
  %56 = fadd float %55, 0xBFE713CA80000000
  %57 = fmul float %50, %54
  %58 = fadd float %57, 0xBFC19A9FA0000000
  %59 = fmul float %50, %58
  %60 = fadd float %59, 0x3FCEF5B7A0000000
  %61 = fmul float %50, %60
  %62 = fadd float %61, 0xBFD63A40C0000000
  %63 = fmul float %50, %56
  %64 = fadd float %63, 0x3FF7154200000000
  %65 = fmul float %52, %62
  %66 = fmul float %50, %64
  %67 = fadd float %66, %65
  %68 = sitofp i32 %46 to float
  %69 = fadd float %68, %67
  %70 = fmul float %69, 0x3FE62E4300000000
  %71 = getelementptr inbounds i8, i8* %0, i64 4
  %72 = bitcast i8* %71 to float*
  store float %70, float* %72, align 4, !tbaa !1
  %73 = getelementptr inbounds i8, i8* %1, i64 8
  %74 = bitcast i8* %73 to float*
  %75 = load float, float* %74, align 4, !tbaa !1
  %76 = fcmp olt float %75, 0x3810000000000000
  %77 = fcmp ogt float %75, 0x47EFFFFFE0000000
  %78 = select i1 %77, float 0x47EFFFFFE0000000, float %75
  %79 = bitcast float %78 to i32
  %80 = select i1 %76, i32 8388608, i32 %79
  %81 = lshr i32 %80, 23
  %82 = add nsw i32 %81, -127
  %83 = and i32 %80, 8388607
  %84 = or i32 %83, 1065353216
  %85 = bitcast i32 %84 to float
  %86 = fadd float %85, -1.000000e+00
  %87 = fmul float %86, %86
  %88 = fmul float %87, %87
  %89 = fmul float %86, 0x3F831161A0000000
  %90 = fsub float 0x3FAAA83920000000, %89
  %91 = fmul float %86, 0x3FDEA2C5A0000000
  %92 = fadd float %91, 0xBFE713CA80000000
  %93 = fmul float %86, %90
  %94 = fadd float %93, 0xBFC19A9FA0000000
  %95 = fmul float %86, %94
  %96 = fadd float %95, 0x3FCEF5B7A0000000
  %97 = fmul float %86, %96
  %98 = fadd float %97, 0xBFD63A40C0000000
  %99 = fmul float %86, %92
  %100 = fadd float %99, 0x3FF7154200000000
  %101 = fmul float %88, %98
  %102 = fmul float %86, %100
  %103 = fadd float %102, %101
  %104 = sitofp i32 %82 to float
  %105 = fadd float %104, %103
  %106 = fmul float %105, 0x3FE62E4300000000
  %107 = getelementptr inbounds i8, i8* %0, i64 8
  %108 = bitcast i8* %107 to float*
  store float %106, float* %108, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %3 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fcmp olt float %5, 0x3810000000000000
  %9 = fcmp ogt float %5, 0x47EFFFFFE0000000
  %10 = select i1 %9, float 0x47EFFFFFE0000000, float %5
  %11 = bitcast float %10 to i32
  %12 = select i1 %8, i32 8388608, i32 %11
  %13 = lshr i32 %12, 23
  %14 = add nsw i32 %13, -127
  %15 = and i32 %12, 8388607
  %16 = or i32 %15, 1065353216
  %17 = bitcast i32 %16 to float
  %18 = fadd float %17, -1.000000e+00
  %19 = fmul float %18, %18
  %20 = fmul float %19, %19
  %21 = fmul float %18, 0x3F831161A0000000
  %22 = fsub float 0x3FAAA83920000000, %21
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
  %38 = fdiv float 1.000000e+00, %5
  %39 = select i1 %8, float 0.000000e+00, float %38
  %40 = getelementptr inbounds i8, i8* %1, i64 4
  %41 = getelementptr inbounds i8, i8* %1, i64 16
  %42 = bitcast i8* %40 to float*
  %43 = load float, float* %42, align 4, !tbaa !1
  %44 = fcmp olt float %43, 0x3810000000000000
  %45 = fcmp ogt float %43, 0x47EFFFFFE0000000
  %46 = select i1 %45, float 0x47EFFFFFE0000000, float %43
  %47 = bitcast float %46 to i32
  %48 = select i1 %44, i32 8388608, i32 %47
  %49 = lshr i32 %48, 23
  %50 = add nsw i32 %49, -127
  %51 = and i32 %48, 8388607
  %52 = or i32 %51, 1065353216
  %53 = bitcast i32 %52 to float
  %54 = fadd float %53, -1.000000e+00
  %55 = fmul float %54, %54
  %56 = fmul float %55, %55
  %57 = fmul float %54, 0x3F831161A0000000
  %58 = fsub float 0x3FAAA83920000000, %57
  %59 = fmul float %54, 0x3FDEA2C5A0000000
  %60 = fadd float %59, 0xBFE713CA80000000
  %61 = fmul float %54, %58
  %62 = fadd float %61, 0xBFC19A9FA0000000
  %63 = fmul float %54, %62
  %64 = fadd float %63, 0x3FCEF5B7A0000000
  %65 = fmul float %54, %64
  %66 = fadd float %65, 0xBFD63A40C0000000
  %67 = fmul float %54, %60
  %68 = fadd float %67, 0x3FF7154200000000
  %69 = fmul float %56, %66
  %70 = fmul float %54, %68
  %71 = fadd float %70, %69
  %72 = sitofp i32 %50 to float
  %73 = fadd float %72, %71
  %74 = fdiv float 1.000000e+00, %43
  %75 = select i1 %44, float 0.000000e+00, float %74
  %76 = getelementptr inbounds i8, i8* %1, i64 8
  %77 = getelementptr inbounds i8, i8* %1, i64 32
  %78 = bitcast i8* %76 to float*
  %79 = load float, float* %78, align 4, !tbaa !1
  %80 = bitcast i8* %41 to <4 x float>*
  %81 = load <4 x float>, <4 x float>* %80, align 4, !tbaa !1
  %82 = bitcast i8* %77 to float*
  %83 = load float, float* %82, align 4, !tbaa !1
  %84 = fcmp olt float %79, 0x3810000000000000
  %85 = fcmp ogt float %79, 0x47EFFFFFE0000000
  %86 = select i1 %85, float 0x47EFFFFFE0000000, float %79
  %87 = bitcast float %86 to i32
  %88 = select i1 %84, i32 8388608, i32 %87
  %89 = lshr i32 %88, 23
  %90 = add nsw i32 %89, -127
  %91 = and i32 %88, 8388607
  %92 = or i32 %91, 1065353216
  %93 = bitcast i32 %92 to float
  %94 = fadd float %93, -1.000000e+00
  %95 = fmul float %94, %94
  %96 = fmul float %95, %95
  %97 = fmul float %94, 0x3F831161A0000000
  %98 = fsub float 0x3FAAA83920000000, %97
  %99 = fmul float %94, 0x3FDEA2C5A0000000
  %100 = fadd float %99, 0xBFE713CA80000000
  %101 = fmul float %94, %98
  %102 = fadd float %101, 0xBFC19A9FA0000000
  %103 = fmul float %94, %102
  %104 = fadd float %103, 0x3FCEF5B7A0000000
  %105 = fmul float %94, %104
  %106 = fadd float %105, 0xBFD63A40C0000000
  %107 = fmul float %94, %100
  %108 = fadd float %107, 0x3FF7154200000000
  %109 = fmul float %96, %106
  %110 = fmul float %94, %108
  %111 = fadd float %110, %109
  %112 = sitofp i32 %90 to float
  %113 = fadd float %112, %111
  %114 = insertelement <4 x float> <float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float undef>, float %7, i32 3
  %115 = insertelement <4 x float> undef, float %37, i32 0
  %116 = insertelement <4 x float> %115, float %73, i32 1
  %117 = insertelement <4 x float> %116, float %113, i32 2
  %118 = insertelement <4 x float> %117, float %39, i32 3
  %119 = fmul <4 x float> %114, %118
  %120 = fdiv float 1.000000e+00, %79
  %121 = select i1 %84, float 0.000000e+00, float %120
  %122 = insertelement <4 x float> undef, float %75, i32 0
  %123 = insertelement <4 x float> %122, float %121, i32 1
  %124 = insertelement <4 x float> %123, float %39, i32 2
  %125 = insertelement <4 x float> %124, float %75, i32 3
  %126 = fmul <4 x float> %81, %125
  %127 = fmul float %121, %83
  %128 = bitcast i8* %0 to <4 x float>*
  store <4 x float> %119, <4 x float>* %128, align 4, !tbaa !1
  %129 = getelementptr inbounds i8, i8* %0, i64 16
  %130 = bitcast i8* %129 to <4 x float>*
  store <4 x float> %126, <4 x float>* %130, align 4, !tbaa !1
  %131 = getelementptr inbounds i8, i8* %0, i64 32
  %132 = bitcast i8* %131 to float*
  store float %127, float* %132, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_log2_ff(float) local_unnamed_addr #6 {
  %2 = fcmp olt float %0, 0x3810000000000000
  %3 = fcmp ogt float %0, 0x47EFFFFFE0000000
  %4 = select i1 %3, float 0x47EFFFFFE0000000, float %0
  %5 = bitcast float %4 to i32
  %6 = select i1 %2, i32 8388608, i32 %5
  %7 = lshr i32 %6, 23
  %8 = add nsw i32 %7, -127
  %9 = and i32 %6, 8388607
  %10 = or i32 %9, 1065353216
  %11 = bitcast i32 %10 to float
  %12 = fadd float %11, -1.000000e+00
  %13 = fmul float %12, %12
  %14 = fmul float %13, %13
  %15 = fmul float %12, 0x3F831161A0000000
  %16 = fsub float 0x3FAAA83920000000, %15
  %17 = fmul float %12, 0x3FDEA2C5A0000000
  %18 = fadd float %17, 0xBFE713CA80000000
  %19 = fmul float %12, %16
  %20 = fadd float %19, 0xBFC19A9FA0000000
  %21 = fmul float %12, %20
  %22 = fadd float %21, 0x3FCEF5B7A0000000
  %23 = fmul float %12, %22
  %24 = fadd float %23, 0xBFD63A40C0000000
  %25 = fmul float %12, %18
  %26 = fadd float %25, 0x3FF7154200000000
  %27 = fmul float %14, %24
  %28 = fmul float %12, %26
  %29 = fadd float %28, %27
  %30 = sitofp i32 %8 to float
  %31 = fadd float %30, %29
  ret float %31
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_log2_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, 0x3810000000000000
  %6 = fcmp ogt float %4, 0x47EFFFFFE0000000
  %7 = select i1 %6, float 0x47EFFFFFE0000000, float %4
  %8 = bitcast float %7 to i32
  %9 = select i1 %5, i32 8388608, i32 %8
  %10 = lshr i32 %9, 23
  %11 = add nsw i32 %10, -127
  %12 = and i32 %9, 8388607
  %13 = or i32 %12, 1065353216
  %14 = bitcast i32 %13 to float
  %15 = fadd float %14, -1.000000e+00
  %16 = fmul float %15, %15
  %17 = fmul float %16, %16
  %18 = fmul float %15, 0x3F831161A0000000
  %19 = fsub float 0x3FAAA83920000000, %18
  %20 = fmul float %15, 0x3FDEA2C5A0000000
  %21 = fadd float %20, 0xBFE713CA80000000
  %22 = fmul float %15, %19
  %23 = fadd float %22, 0xBFC19A9FA0000000
  %24 = fmul float %15, %23
  %25 = fadd float %24, 0x3FCEF5B7A0000000
  %26 = fmul float %15, %25
  %27 = fadd float %26, 0xBFD63A40C0000000
  %28 = fmul float %15, %21
  %29 = fadd float %28, 0x3FF7154200000000
  %30 = fmul float %17, %27
  %31 = fmul float %15, %29
  %32 = fadd float %31, %30
  %33 = sitofp i32 %11 to float
  %34 = fadd float %33, %32
  %35 = fmul float %4, 0x3FE62E4300000000
  %36 = fcmp olt float %35, 0x3810000000000000
  %37 = fdiv float 1.000000e+00, %35
  %38 = select i1 %36, float 0.000000e+00, float %37
  %39 = getelementptr inbounds i8, i8* %1, i64 4
  %40 = bitcast i8* %39 to float*
  %41 = load float, float* %40, align 4, !tbaa !1
  %42 = fmul float %41, %38
  %43 = getelementptr inbounds i8, i8* %1, i64 8
  %44 = bitcast i8* %43 to float*
  %45 = load float, float* %44, align 4, !tbaa !1
  %46 = fmul float %45, %38
  %47 = insertelement <2 x float> undef, float %34, i32 0
  %48 = insertelement <2 x float> %47, float %42, i32 1
  %49 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %48, <2 x float>* %49, align 4
  %50 = getelementptr inbounds i8, i8* %0, i64 8
  %51 = bitcast i8* %50 to float*
  store float %46, float* %51, align 4
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_log2_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, 0x3810000000000000
  %6 = fcmp ogt float %4, 0x47EFFFFFE0000000
  %7 = select i1 %6, float 0x47EFFFFFE0000000, float %4
  %8 = bitcast float %7 to i32
  %9 = select i1 %5, i32 8388608, i32 %8
  %10 = lshr i32 %9, 23
  %11 = add nsw i32 %10, -127
  %12 = and i32 %9, 8388607
  %13 = or i32 %12, 1065353216
  %14 = bitcast i32 %13 to float
  %15 = fadd float %14, -1.000000e+00
  %16 = fmul float %15, %15
  %17 = fmul float %16, %16
  %18 = fmul float %15, 0x3F831161A0000000
  %19 = fsub float 0x3FAAA83920000000, %18
  %20 = fmul float %15, 0x3FDEA2C5A0000000
  %21 = fadd float %20, 0xBFE713CA80000000
  %22 = fmul float %15, %19
  %23 = fadd float %22, 0xBFC19A9FA0000000
  %24 = fmul float %15, %23
  %25 = fadd float %24, 0x3FCEF5B7A0000000
  %26 = fmul float %15, %25
  %27 = fadd float %26, 0xBFD63A40C0000000
  %28 = fmul float %15, %21
  %29 = fadd float %28, 0x3FF7154200000000
  %30 = fmul float %17, %27
  %31 = fmul float %15, %29
  %32 = fadd float %31, %30
  %33 = sitofp i32 %11 to float
  %34 = fadd float %33, %32
  %35 = bitcast i8* %0 to float*
  store float %34, float* %35, align 4, !tbaa !1
  %36 = getelementptr inbounds i8, i8* %1, i64 4
  %37 = bitcast i8* %36 to float*
  %38 = load float, float* %37, align 4, !tbaa !1
  %39 = fcmp olt float %38, 0x3810000000000000
  %40 = fcmp ogt float %38, 0x47EFFFFFE0000000
  %41 = select i1 %40, float 0x47EFFFFFE0000000, float %38
  %42 = bitcast float %41 to i32
  %43 = select i1 %39, i32 8388608, i32 %42
  %44 = lshr i32 %43, 23
  %45 = add nsw i32 %44, -127
  %46 = and i32 %43, 8388607
  %47 = or i32 %46, 1065353216
  %48 = bitcast i32 %47 to float
  %49 = fadd float %48, -1.000000e+00
  %50 = fmul float %49, %49
  %51 = fmul float %50, %50
  %52 = fmul float %49, 0x3F831161A0000000
  %53 = fsub float 0x3FAAA83920000000, %52
  %54 = fmul float %49, 0x3FDEA2C5A0000000
  %55 = fadd float %54, 0xBFE713CA80000000
  %56 = fmul float %49, %53
  %57 = fadd float %56, 0xBFC19A9FA0000000
  %58 = fmul float %49, %57
  %59 = fadd float %58, 0x3FCEF5B7A0000000
  %60 = fmul float %49, %59
  %61 = fadd float %60, 0xBFD63A40C0000000
  %62 = fmul float %49, %55
  %63 = fadd float %62, 0x3FF7154200000000
  %64 = fmul float %51, %61
  %65 = fmul float %49, %63
  %66 = fadd float %65, %64
  %67 = sitofp i32 %45 to float
  %68 = fadd float %67, %66
  %69 = getelementptr inbounds i8, i8* %0, i64 4
  %70 = bitcast i8* %69 to float*
  store float %68, float* %70, align 4, !tbaa !1
  %71 = getelementptr inbounds i8, i8* %1, i64 8
  %72 = bitcast i8* %71 to float*
  %73 = load float, float* %72, align 4, !tbaa !1
  %74 = fcmp olt float %73, 0x3810000000000000
  %75 = fcmp ogt float %73, 0x47EFFFFFE0000000
  %76 = select i1 %75, float 0x47EFFFFFE0000000, float %73
  %77 = bitcast float %76 to i32
  %78 = select i1 %74, i32 8388608, i32 %77
  %79 = lshr i32 %78, 23
  %80 = add nsw i32 %79, -127
  %81 = and i32 %78, 8388607
  %82 = or i32 %81, 1065353216
  %83 = bitcast i32 %82 to float
  %84 = fadd float %83, -1.000000e+00
  %85 = fmul float %84, %84
  %86 = fmul float %85, %85
  %87 = fmul float %84, 0x3F831161A0000000
  %88 = fsub float 0x3FAAA83920000000, %87
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
  %104 = getelementptr inbounds i8, i8* %0, i64 8
  %105 = bitcast i8* %104 to float*
  store float %103, float* %105, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log2_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fcmp olt float %5, 0x3810000000000000
  %7 = fcmp ogt float %5, 0x47EFFFFFE0000000
  %8 = select i1 %7, float 0x47EFFFFFE0000000, float %5
  %9 = bitcast float %8 to i32
  %10 = select i1 %6, i32 8388608, i32 %9
  %11 = lshr i32 %10, 23
  %12 = add nsw i32 %11, -127
  %13 = and i32 %10, 8388607
  %14 = or i32 %13, 1065353216
  %15 = bitcast i32 %14 to float
  %16 = fadd float %15, -1.000000e+00
  %17 = fmul float %16, %16
  %18 = fmul float %17, %17
  %19 = fmul float %16, 0x3F831161A0000000
  %20 = fsub float 0x3FAAA83920000000, %19
  %21 = fmul float %16, 0x3FDEA2C5A0000000
  %22 = fadd float %21, 0xBFE713CA80000000
  %23 = fmul float %16, %20
  %24 = fadd float %23, 0xBFC19A9FA0000000
  %25 = fmul float %16, %24
  %26 = fadd float %25, 0x3FCEF5B7A0000000
  %27 = fmul float %16, %26
  %28 = fadd float %27, 0xBFD63A40C0000000
  %29 = fmul float %16, %22
  %30 = fadd float %29, 0x3FF7154200000000
  %31 = fmul float %18, %28
  %32 = fmul float %16, %30
  %33 = fadd float %32, %31
  %34 = sitofp i32 %12 to float
  %35 = fadd float %34, %33
  %36 = fmul float %5, 0x3FE62E4300000000
  %37 = fcmp olt float %36, 0x3810000000000000
  %38 = fdiv float 1.000000e+00, %36
  %39 = select i1 %37, float 0.000000e+00, float %38
  %40 = getelementptr inbounds i8, i8* %1, i64 4
  %41 = getelementptr inbounds i8, i8* %1, i64 28
  %42 = bitcast i8* %40 to float*
  %43 = load float, float* %42, align 4, !tbaa !1
  %44 = bitcast i8* %41 to float*
  %45 = load float, float* %44, align 4, !tbaa !1
  %46 = fcmp olt float %43, 0x3810000000000000
  %47 = fcmp ogt float %43, 0x47EFFFFFE0000000
  %48 = select i1 %47, float 0x47EFFFFFE0000000, float %43
  %49 = bitcast float %48 to i32
  %50 = select i1 %46, i32 8388608, i32 %49
  %51 = lshr i32 %50, 23
  %52 = add nsw i32 %51, -127
  %53 = and i32 %50, 8388607
  %54 = or i32 %53, 1065353216
  %55 = bitcast i32 %54 to float
  %56 = fadd float %55, -1.000000e+00
  %57 = fmul float %56, %56
  %58 = fmul float %57, %57
  %59 = fmul float %56, 0x3F831161A0000000
  %60 = fsub float 0x3FAAA83920000000, %59
  %61 = fmul float %56, 0x3FDEA2C5A0000000
  %62 = fadd float %61, 0xBFE713CA80000000
  %63 = fmul float %56, %60
  %64 = fadd float %63, 0xBFC19A9FA0000000
  %65 = fmul float %56, %64
  %66 = fadd float %65, 0x3FCEF5B7A0000000
  %67 = fmul float %56, %66
  %68 = fadd float %67, 0xBFD63A40C0000000
  %69 = fmul float %56, %62
  %70 = fadd float %69, 0x3FF7154200000000
  %71 = fmul float %58, %68
  %72 = fmul float %56, %70
  %73 = fadd float %72, %71
  %74 = sitofp i32 %52 to float
  %75 = fadd float %74, %73
  %76 = fmul float %43, 0x3FE62E4300000000
  %77 = fcmp olt float %76, 0x3810000000000000
  %78 = fdiv float 1.000000e+00, %76
  %79 = select i1 %77, float 0.000000e+00, float %78
  %80 = fmul float %45, %79
  %81 = getelementptr inbounds i8, i8* %1, i64 8
  %82 = getelementptr inbounds i8, i8* %1, i64 32
  %83 = bitcast i8* %81 to float*
  %84 = load float, float* %83, align 4, !tbaa !1
  %85 = bitcast i8* %3 to <4 x float>*
  %86 = load <4 x float>, <4 x float>* %85, align 4, !tbaa !1
  %87 = bitcast i8* %82 to float*
  %88 = load float, float* %87, align 4, !tbaa !1
  %89 = fcmp olt float %84, 0x3810000000000000
  %90 = fcmp ogt float %84, 0x47EFFFFFE0000000
  %91 = select i1 %90, float 0x47EFFFFFE0000000, float %84
  %92 = bitcast float %91 to i32
  %93 = select i1 %89, i32 8388608, i32 %92
  %94 = lshr i32 %93, 23
  %95 = add nsw i32 %94, -127
  %96 = and i32 %93, 8388607
  %97 = or i32 %96, 1065353216
  %98 = bitcast i32 %97 to float
  %99 = fadd float %98, -1.000000e+00
  %100 = fmul float %99, %99
  %101 = fmul float %100, %100
  %102 = fmul float %99, 0x3F831161A0000000
  %103 = fsub float 0x3FAAA83920000000, %102
  %104 = fmul float %99, 0x3FDEA2C5A0000000
  %105 = fadd float %104, 0xBFE713CA80000000
  %106 = fmul float %99, %103
  %107 = fadd float %106, 0xBFC19A9FA0000000
  %108 = fmul float %99, %107
  %109 = fadd float %108, 0x3FCEF5B7A0000000
  %110 = fmul float %99, %109
  %111 = fadd float %110, 0xBFD63A40C0000000
  %112 = fmul float %99, %105
  %113 = fadd float %112, 0x3FF7154200000000
  %114 = fmul float %101, %111
  %115 = fmul float %99, %113
  %116 = fadd float %115, %114
  %117 = sitofp i32 %95 to float
  %118 = fadd float %117, %116
  %119 = fmul float %84, 0x3FE62E4300000000
  %120 = fcmp olt float %119, 0x3810000000000000
  %121 = fdiv float 1.000000e+00, %119
  %122 = select i1 %120, float 0.000000e+00, float %121
  %123 = insertelement <4 x float> undef, float %39, i32 0
  %124 = insertelement <4 x float> %123, float %79, i32 1
  %125 = insertelement <4 x float> %124, float %122, i32 2
  %126 = insertelement <4 x float> %125, float %39, i32 3
  %127 = fmul <4 x float> %86, %126
  %128 = fmul float %88, %122
  %129 = bitcast i8* %0 to float*
  store float %35, float* %129, align 4, !tbaa !5
  %130 = getelementptr inbounds i8, i8* %0, i64 4
  %131 = bitcast i8* %130 to float*
  store float %75, float* %131, align 4, !tbaa !7
  %132 = getelementptr inbounds i8, i8* %0, i64 8
  %133 = bitcast i8* %132 to float*
  store float %118, float* %133, align 4, !tbaa !8
  %134 = getelementptr inbounds i8, i8* %0, i64 12
  %135 = bitcast i8* %134 to <4 x float>*
  store <4 x float> %127, <4 x float>* %135, align 4, !tbaa !1
  %136 = getelementptr inbounds i8, i8* %0, i64 28
  %137 = bitcast i8* %136 to float*
  store float %80, float* %137, align 4, !tbaa !7
  %138 = getelementptr inbounds i8, i8* %0, i64 32
  %139 = bitcast i8* %138 to float*
  store float %128, float* %139, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_log10_ff(float) local_unnamed_addr #6 {
  %2 = fcmp olt float %0, 0x3810000000000000
  %3 = fcmp ogt float %0, 0x47EFFFFFE0000000
  %4 = select i1 %3, float 0x47EFFFFFE0000000, float %0
  %5 = bitcast float %4 to i32
  %6 = select i1 %2, i32 8388608, i32 %5
  %7 = lshr i32 %6, 23
  %8 = add nsw i32 %7, -127
  %9 = and i32 %6, 8388607
  %10 = or i32 %9, 1065353216
  %11 = bitcast i32 %10 to float
  %12 = fadd float %11, -1.000000e+00
  %13 = fmul float %12, %12
  %14 = fmul float %13, %13
  %15 = fmul float %12, 0x3F831161A0000000
  %16 = fsub float 0x3FAAA83920000000, %15
  %17 = fmul float %12, 0x3FDEA2C5A0000000
  %18 = fadd float %17, 0xBFE713CA80000000
  %19 = fmul float %12, %16
  %20 = fadd float %19, 0xBFC19A9FA0000000
  %21 = fmul float %12, %20
  %22 = fadd float %21, 0x3FCEF5B7A0000000
  %23 = fmul float %12, %22
  %24 = fadd float %23, 0xBFD63A40C0000000
  %25 = fmul float %12, %18
  %26 = fadd float %25, 0x3FF7154200000000
  %27 = fmul float %14, %24
  %28 = fmul float %12, %26
  %29 = fadd float %28, %27
  %30 = sitofp i32 %8 to float
  %31 = fadd float %30, %29
  %32 = fmul float %31, 0x3FD3441360000000
  ret float %32
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_log10_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, 0x3810000000000000
  %6 = fcmp ogt float %4, 0x47EFFFFFE0000000
  %7 = select i1 %6, float 0x47EFFFFFE0000000, float %4
  %8 = bitcast float %7 to i32
  %9 = select i1 %5, i32 8388608, i32 %8
  %10 = lshr i32 %9, 23
  %11 = add nsw i32 %10, -127
  %12 = and i32 %9, 8388607
  %13 = or i32 %12, 1065353216
  %14 = bitcast i32 %13 to float
  %15 = fadd float %14, -1.000000e+00
  %16 = fmul float %15, %15
  %17 = fmul float %16, %16
  %18 = fmul float %15, 0x3F831161A0000000
  %19 = fsub float 0x3FAAA83920000000, %18
  %20 = fmul float %15, 0x3FDEA2C5A0000000
  %21 = fadd float %20, 0xBFE713CA80000000
  %22 = fmul float %15, %19
  %23 = fadd float %22, 0xBFC19A9FA0000000
  %24 = fmul float %15, %23
  %25 = fadd float %24, 0x3FCEF5B7A0000000
  %26 = fmul float %15, %25
  %27 = fadd float %26, 0xBFD63A40C0000000
  %28 = fmul float %15, %21
  %29 = fadd float %28, 0x3FF7154200000000
  %30 = fmul float %17, %27
  %31 = fmul float %15, %29
  %32 = fadd float %31, %30
  %33 = sitofp i32 %11 to float
  %34 = fadd float %33, %32
  %35 = fmul float %34, 0x3FD3441360000000
  %36 = fmul float %4, 0x40026BB1C0000000
  %37 = fcmp olt float %36, 0x3810000000000000
  %38 = fdiv float 1.000000e+00, %36
  %39 = select i1 %37, float 0.000000e+00, float %38
  %40 = getelementptr inbounds i8, i8* %1, i64 4
  %41 = bitcast i8* %40 to float*
  %42 = load float, float* %41, align 4, !tbaa !1
  %43 = fmul float %42, %39
  %44 = getelementptr inbounds i8, i8* %1, i64 8
  %45 = bitcast i8* %44 to float*
  %46 = load float, float* %45, align 4, !tbaa !1
  %47 = fmul float %46, %39
  %48 = insertelement <2 x float> undef, float %35, i32 0
  %49 = insertelement <2 x float> %48, float %43, i32 1
  %50 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %49, <2 x float>* %50, align 4
  %51 = getelementptr inbounds i8, i8* %0, i64 8
  %52 = bitcast i8* %51 to float*
  store float %47, float* %52, align 4
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_log10_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, 0x3810000000000000
  %6 = fcmp ogt float %4, 0x47EFFFFFE0000000
  %7 = select i1 %6, float 0x47EFFFFFE0000000, float %4
  %8 = bitcast float %7 to i32
  %9 = select i1 %5, i32 8388608, i32 %8
  %10 = lshr i32 %9, 23
  %11 = add nsw i32 %10, -127
  %12 = and i32 %9, 8388607
  %13 = or i32 %12, 1065353216
  %14 = bitcast i32 %13 to float
  %15 = fadd float %14, -1.000000e+00
  %16 = fmul float %15, %15
  %17 = fmul float %16, %16
  %18 = fmul float %15, 0x3F831161A0000000
  %19 = fsub float 0x3FAAA83920000000, %18
  %20 = fmul float %15, 0x3FDEA2C5A0000000
  %21 = fadd float %20, 0xBFE713CA80000000
  %22 = fmul float %15, %19
  %23 = fadd float %22, 0xBFC19A9FA0000000
  %24 = fmul float %15, %23
  %25 = fadd float %24, 0x3FCEF5B7A0000000
  %26 = fmul float %15, %25
  %27 = fadd float %26, 0xBFD63A40C0000000
  %28 = fmul float %15, %21
  %29 = fadd float %28, 0x3FF7154200000000
  %30 = fmul float %17, %27
  %31 = fmul float %15, %29
  %32 = fadd float %31, %30
  %33 = sitofp i32 %11 to float
  %34 = fadd float %33, %32
  %35 = fmul float %34, 0x3FD3441360000000
  %36 = bitcast i8* %0 to float*
  store float %35, float* %36, align 4, !tbaa !1
  %37 = getelementptr inbounds i8, i8* %1, i64 4
  %38 = bitcast i8* %37 to float*
  %39 = load float, float* %38, align 4, !tbaa !1
  %40 = fcmp olt float %39, 0x3810000000000000
  %41 = fcmp ogt float %39, 0x47EFFFFFE0000000
  %42 = select i1 %41, float 0x47EFFFFFE0000000, float %39
  %43 = bitcast float %42 to i32
  %44 = select i1 %40, i32 8388608, i32 %43
  %45 = lshr i32 %44, 23
  %46 = add nsw i32 %45, -127
  %47 = and i32 %44, 8388607
  %48 = or i32 %47, 1065353216
  %49 = bitcast i32 %48 to float
  %50 = fadd float %49, -1.000000e+00
  %51 = fmul float %50, %50
  %52 = fmul float %51, %51
  %53 = fmul float %50, 0x3F831161A0000000
  %54 = fsub float 0x3FAAA83920000000, %53
  %55 = fmul float %50, 0x3FDEA2C5A0000000
  %56 = fadd float %55, 0xBFE713CA80000000
  %57 = fmul float %50, %54
  %58 = fadd float %57, 0xBFC19A9FA0000000
  %59 = fmul float %50, %58
  %60 = fadd float %59, 0x3FCEF5B7A0000000
  %61 = fmul float %50, %60
  %62 = fadd float %61, 0xBFD63A40C0000000
  %63 = fmul float %50, %56
  %64 = fadd float %63, 0x3FF7154200000000
  %65 = fmul float %52, %62
  %66 = fmul float %50, %64
  %67 = fadd float %66, %65
  %68 = sitofp i32 %46 to float
  %69 = fadd float %68, %67
  %70 = fmul float %69, 0x3FD3441360000000
  %71 = getelementptr inbounds i8, i8* %0, i64 4
  %72 = bitcast i8* %71 to float*
  store float %70, float* %72, align 4, !tbaa !1
  %73 = getelementptr inbounds i8, i8* %1, i64 8
  %74 = bitcast i8* %73 to float*
  %75 = load float, float* %74, align 4, !tbaa !1
  %76 = fcmp olt float %75, 0x3810000000000000
  %77 = fcmp ogt float %75, 0x47EFFFFFE0000000
  %78 = select i1 %77, float 0x47EFFFFFE0000000, float %75
  %79 = bitcast float %78 to i32
  %80 = select i1 %76, i32 8388608, i32 %79
  %81 = lshr i32 %80, 23
  %82 = add nsw i32 %81, -127
  %83 = and i32 %80, 8388607
  %84 = or i32 %83, 1065353216
  %85 = bitcast i32 %84 to float
  %86 = fadd float %85, -1.000000e+00
  %87 = fmul float %86, %86
  %88 = fmul float %87, %87
  %89 = fmul float %86, 0x3F831161A0000000
  %90 = fsub float 0x3FAAA83920000000, %89
  %91 = fmul float %86, 0x3FDEA2C5A0000000
  %92 = fadd float %91, 0xBFE713CA80000000
  %93 = fmul float %86, %90
  %94 = fadd float %93, 0xBFC19A9FA0000000
  %95 = fmul float %86, %94
  %96 = fadd float %95, 0x3FCEF5B7A0000000
  %97 = fmul float %86, %96
  %98 = fadd float %97, 0xBFD63A40C0000000
  %99 = fmul float %86, %92
  %100 = fadd float %99, 0x3FF7154200000000
  %101 = fmul float %88, %98
  %102 = fmul float %86, %100
  %103 = fadd float %102, %101
  %104 = sitofp i32 %82 to float
  %105 = fadd float %104, %103
  %106 = fmul float %105, 0x3FD3441360000000
  %107 = getelementptr inbounds i8, i8* %0, i64 8
  %108 = bitcast i8* %107 to float*
  store float %106, float* %108, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_log10_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %3 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fcmp olt float %5, 0x3810000000000000
  %9 = fcmp ogt float %5, 0x47EFFFFFE0000000
  %10 = select i1 %9, float 0x47EFFFFFE0000000, float %5
  %11 = bitcast float %10 to i32
  %12 = select i1 %8, i32 8388608, i32 %11
  %13 = lshr i32 %12, 23
  %14 = add nsw i32 %13, -127
  %15 = and i32 %12, 8388607
  %16 = or i32 %15, 1065353216
  %17 = bitcast i32 %16 to float
  %18 = fadd float %17, -1.000000e+00
  %19 = fmul float %18, %18
  %20 = fmul float %19, %19
  %21 = fmul float %18, 0x3F831161A0000000
  %22 = fsub float 0x3FAAA83920000000, %21
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
  %38 = fmul float %5, 0x40026BB1C0000000
  %39 = fcmp olt float %38, 0x3810000000000000
  %40 = fdiv float 1.000000e+00, %38
  %41 = select i1 %39, float 0.000000e+00, float %40
  %42 = getelementptr inbounds i8, i8* %1, i64 4
  %43 = getelementptr inbounds i8, i8* %1, i64 16
  %44 = bitcast i8* %42 to float*
  %45 = load float, float* %44, align 4, !tbaa !1
  %46 = fcmp olt float %45, 0x3810000000000000
  %47 = fcmp ogt float %45, 0x47EFFFFFE0000000
  %48 = select i1 %47, float 0x47EFFFFFE0000000, float %45
  %49 = bitcast float %48 to i32
  %50 = select i1 %46, i32 8388608, i32 %49
  %51 = lshr i32 %50, 23
  %52 = add nsw i32 %51, -127
  %53 = and i32 %50, 8388607
  %54 = or i32 %53, 1065353216
  %55 = bitcast i32 %54 to float
  %56 = fadd float %55, -1.000000e+00
  %57 = fmul float %56, %56
  %58 = fmul float %57, %57
  %59 = fmul float %56, 0x3F831161A0000000
  %60 = fsub float 0x3FAAA83920000000, %59
  %61 = fmul float %56, 0x3FDEA2C5A0000000
  %62 = fadd float %61, 0xBFE713CA80000000
  %63 = fmul float %56, %60
  %64 = fadd float %63, 0xBFC19A9FA0000000
  %65 = fmul float %56, %64
  %66 = fadd float %65, 0x3FCEF5B7A0000000
  %67 = fmul float %56, %66
  %68 = fadd float %67, 0xBFD63A40C0000000
  %69 = fmul float %56, %62
  %70 = fadd float %69, 0x3FF7154200000000
  %71 = fmul float %58, %68
  %72 = fmul float %56, %70
  %73 = fadd float %72, %71
  %74 = sitofp i32 %52 to float
  %75 = fadd float %74, %73
  %76 = fmul float %45, 0x40026BB1C0000000
  %77 = fcmp olt float %76, 0x3810000000000000
  %78 = fdiv float 1.000000e+00, %76
  %79 = select i1 %77, float 0.000000e+00, float %78
  %80 = getelementptr inbounds i8, i8* %1, i64 8
  %81 = getelementptr inbounds i8, i8* %1, i64 32
  %82 = bitcast i8* %80 to float*
  %83 = load float, float* %82, align 4, !tbaa !1
  %84 = bitcast i8* %43 to <4 x float>*
  %85 = load <4 x float>, <4 x float>* %84, align 4, !tbaa !1
  %86 = bitcast i8* %81 to float*
  %87 = load float, float* %86, align 4, !tbaa !1
  %88 = fcmp olt float %83, 0x3810000000000000
  %89 = fcmp ogt float %83, 0x47EFFFFFE0000000
  %90 = select i1 %89, float 0x47EFFFFFE0000000, float %83
  %91 = bitcast float %90 to i32
  %92 = select i1 %88, i32 8388608, i32 %91
  %93 = lshr i32 %92, 23
  %94 = add nsw i32 %93, -127
  %95 = and i32 %92, 8388607
  %96 = or i32 %95, 1065353216
  %97 = bitcast i32 %96 to float
  %98 = fadd float %97, -1.000000e+00
  %99 = fmul float %98, %98
  %100 = fmul float %99, %99
  %101 = fmul float %98, 0x3F831161A0000000
  %102 = fsub float 0x3FAAA83920000000, %101
  %103 = fmul float %98, 0x3FDEA2C5A0000000
  %104 = fadd float %103, 0xBFE713CA80000000
  %105 = fmul float %98, %102
  %106 = fadd float %105, 0xBFC19A9FA0000000
  %107 = fmul float %98, %106
  %108 = fadd float %107, 0x3FCEF5B7A0000000
  %109 = fmul float %98, %108
  %110 = fadd float %109, 0xBFD63A40C0000000
  %111 = fmul float %98, %104
  %112 = fadd float %111, 0x3FF7154200000000
  %113 = fmul float %100, %110
  %114 = fmul float %98, %112
  %115 = fadd float %114, %113
  %116 = sitofp i32 %94 to float
  %117 = fadd float %116, %115
  %118 = insertelement <4 x float> <float 0x3FD3441360000000, float 0x3FD3441360000000, float 0x3FD3441360000000, float undef>, float %7, i32 3
  %119 = insertelement <4 x float> undef, float %37, i32 0
  %120 = insertelement <4 x float> %119, float %75, i32 1
  %121 = insertelement <4 x float> %120, float %117, i32 2
  %122 = insertelement <4 x float> %121, float %41, i32 3
  %123 = fmul <4 x float> %118, %122
  %124 = fmul float %83, 0x40026BB1C0000000
  %125 = fcmp olt float %124, 0x3810000000000000
  %126 = fdiv float 1.000000e+00, %124
  %127 = select i1 %125, float 0.000000e+00, float %126
  %128 = insertelement <4 x float> undef, float %79, i32 0
  %129 = insertelement <4 x float> %128, float %127, i32 1
  %130 = insertelement <4 x float> %129, float %41, i32 2
  %131 = insertelement <4 x float> %130, float %79, i32 3
  %132 = fmul <4 x float> %85, %131
  %133 = fmul float %87, %127
  %134 = bitcast i8* %0 to <4 x float>*
  store <4 x float> %123, <4 x float>* %134, align 4, !tbaa !1
  %135 = getelementptr inbounds i8, i8* %0, i64 16
  %136 = bitcast i8* %135 to <4 x float>*
  store <4 x float> %132, <4 x float>* %136, align 4, !tbaa !1
  %137 = getelementptr inbounds i8, i8* %0, i64 32
  %138 = bitcast i8* %137 to float*
  store float %133, float* %138, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_exp_ff(float) local_unnamed_addr #6 {
  %2 = fmul float %0, 0x3FF7154760000000
  %3 = fcmp olt float %2, -1.260000e+02
  %4 = fcmp ogt float %2, 1.260000e+02
  %5 = select i1 %4, float 1.260000e+02, float %2
  %6 = select i1 %3, float -1.260000e+02, float %5
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

; Function Attrs: norecurse nounwind uwtable
define void @osl_exp_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fmul float %4, 0x3FF7154760000000
  %6 = fcmp olt float %5, -1.260000e+02
  %7 = fcmp ogt float %5, 1.260000e+02
  %8 = select i1 %7, float 1.260000e+02, float %5
  %9 = select i1 %6, float -1.260000e+02, float %8
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
  %29 = getelementptr inbounds i8, i8* %1, i64 4
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !1
  %32 = fmul float %31, %28
  %33 = getelementptr inbounds i8, i8* %1, i64 8
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %34, align 4, !tbaa !1
  %36 = fmul float %35, %28
  %37 = insertelement <2 x float> undef, float %28, i32 0
  %38 = insertelement <2 x float> %37, float %32, i32 1
  %39 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %38, <2 x float>* %39, align 4
  %40 = getelementptr inbounds i8, i8* %0, i64 8
  %41 = bitcast i8* %40 to float*
  store float %36, float* %41, align 4
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_exp_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fmul float %4, 0x3FF7154760000000
  %6 = fcmp olt float %5, -1.260000e+02
  %7 = fcmp ogt float %5, 1.260000e+02
  %8 = select i1 %7, float 1.260000e+02, float %5
  %9 = select i1 %6, float -1.260000e+02, float %8
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
  %28 = bitcast i8* %0 to i32*
  store i32 %27, i32* %28, align 4, !tbaa !1
  %29 = getelementptr inbounds i8, i8* %1, i64 4
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !1
  %32 = fmul float %31, 0x3FF7154760000000
  %33 = fcmp olt float %32, -1.260000e+02
  %34 = fcmp ogt float %32, 1.260000e+02
  %35 = select i1 %34, float 1.260000e+02, float %32
  %36 = select i1 %33, float -1.260000e+02, float %35
  %37 = fptosi float %36 to i32
  %38 = sitofp i32 %37 to float
  %39 = fsub float %36, %38
  %40 = fsub float 1.000000e+00, %39
  %41 = fsub float 1.000000e+00, %40
  %42 = fmul float %41, 0x3F55D889C0000000
  %43 = fadd float %42, 0x3F84177340000000
  %44 = fmul float %41, %43
  %45 = fadd float %44, 0x3FAC6CE660000000
  %46 = fmul float %41, %45
  %47 = fadd float %46, 0x3FCEBE3240000000
  %48 = fmul float %41, %47
  %49 = fadd float %48, 0x3FE62E3E20000000
  %50 = fmul float %41, %49
  %51 = fadd float %50, 1.000000e+00
  %52 = bitcast float %51 to i32
  %53 = shl i32 %37, 23
  %54 = add i32 %52, %53
  %55 = getelementptr inbounds i8, i8* %0, i64 4
  %56 = bitcast i8* %55 to i32*
  store i32 %54, i32* %56, align 4, !tbaa !1
  %57 = getelementptr inbounds i8, i8* %1, i64 8
  %58 = bitcast i8* %57 to float*
  %59 = load float, float* %58, align 4, !tbaa !1
  %60 = fmul float %59, 0x3FF7154760000000
  %61 = fcmp olt float %60, -1.260000e+02
  %62 = fcmp ogt float %60, 1.260000e+02
  %63 = select i1 %62, float 1.260000e+02, float %60
  %64 = select i1 %61, float -1.260000e+02, float %63
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
  %83 = getelementptr inbounds i8, i8* %0, i64 8
  %84 = bitcast i8* %83 to i32*
  store i32 %82, i32* %84, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_exp_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fmul float %5, 0x3FF7154760000000
  %7 = fcmp olt float %6, -1.260000e+02
  %8 = fcmp ogt float %6, 1.260000e+02
  %9 = select i1 %8, float 1.260000e+02, float %6
  %10 = select i1 %7, float -1.260000e+02, float %9
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
  %30 = getelementptr inbounds i8, i8* %1, i64 4
  %31 = getelementptr inbounds i8, i8* %1, i64 28
  %32 = bitcast i8* %30 to float*
  %33 = load float, float* %32, align 4, !tbaa !1
  %34 = bitcast i8* %31 to float*
  %35 = load float, float* %34, align 4, !tbaa !1
  %36 = fmul float %33, 0x3FF7154760000000
  %37 = fcmp olt float %36, -1.260000e+02
  %38 = fcmp ogt float %36, 1.260000e+02
  %39 = select i1 %38, float 1.260000e+02, float %36
  %40 = select i1 %37, float -1.260000e+02, float %39
  %41 = fptosi float %40 to i32
  %42 = sitofp i32 %41 to float
  %43 = fsub float %40, %42
  %44 = fsub float 1.000000e+00, %43
  %45 = fsub float 1.000000e+00, %44
  %46 = fmul float %45, 0x3F55D889C0000000
  %47 = fadd float %46, 0x3F84177340000000
  %48 = fmul float %45, %47
  %49 = fadd float %48, 0x3FAC6CE660000000
  %50 = fmul float %45, %49
  %51 = fadd float %50, 0x3FCEBE3240000000
  %52 = fmul float %45, %51
  %53 = fadd float %52, 0x3FE62E3E20000000
  %54 = fmul float %45, %53
  %55 = fadd float %54, 1.000000e+00
  %56 = bitcast float %55 to i32
  %57 = shl i32 %41, 23
  %58 = add i32 %56, %57
  %59 = bitcast i32 %58 to float
  %60 = fmul float %35, %59
  %61 = getelementptr inbounds i8, i8* %1, i64 8
  %62 = getelementptr inbounds i8, i8* %1, i64 32
  %63 = bitcast i8* %61 to float*
  %64 = load float, float* %63, align 4, !tbaa !1
  %65 = bitcast i8* %3 to <4 x float>*
  %66 = load <4 x float>, <4 x float>* %65, align 4, !tbaa !1
  %67 = bitcast i8* %62 to float*
  %68 = load float, float* %67, align 4, !tbaa !1
  %69 = fmul float %64, 0x3FF7154760000000
  %70 = fcmp olt float %69, -1.260000e+02
  %71 = fcmp ogt float %69, 1.260000e+02
  %72 = select i1 %71, float 1.260000e+02, float %69
  %73 = select i1 %70, float -1.260000e+02, float %72
  %74 = fptosi float %73 to i32
  %75 = sitofp i32 %74 to float
  %76 = fsub float %73, %75
  %77 = fsub float 1.000000e+00, %76
  %78 = fsub float 1.000000e+00, %77
  %79 = fmul float %78, 0x3F55D889C0000000
  %80 = fadd float %79, 0x3F84177340000000
  %81 = fmul float %78, %80
  %82 = fadd float %81, 0x3FAC6CE660000000
  %83 = fmul float %78, %82
  %84 = fadd float %83, 0x3FCEBE3240000000
  %85 = fmul float %78, %84
  %86 = fadd float %85, 0x3FE62E3E20000000
  %87 = fmul float %78, %86
  %88 = fadd float %87, 1.000000e+00
  %89 = bitcast float %88 to i32
  %90 = shl i32 %74, 23
  %91 = add i32 %89, %90
  %92 = bitcast i32 %91 to float
  %93 = insertelement <4 x float> undef, float %29, i32 0
  %94 = insertelement <4 x float> %93, float %59, i32 1
  %95 = insertelement <4 x float> %94, float %92, i32 2
  %96 = insertelement <4 x float> %95, float %29, i32 3
  %97 = fmul <4 x float> %66, %96
  %98 = fmul float %68, %92
  %99 = bitcast i8* %0 to i32*
  store i32 %28, i32* %99, align 4, !tbaa !5
  %100 = getelementptr inbounds i8, i8* %0, i64 4
  %101 = bitcast i8* %100 to i32*
  store i32 %58, i32* %101, align 4, !tbaa !7
  %102 = getelementptr inbounds i8, i8* %0, i64 8
  %103 = bitcast i8* %102 to i32*
  store i32 %91, i32* %103, align 4, !tbaa !8
  %104 = getelementptr inbounds i8, i8* %0, i64 12
  %105 = bitcast i8* %104 to <4 x float>*
  store <4 x float> %97, <4 x float>* %105, align 4, !tbaa !1
  %106 = getelementptr inbounds i8, i8* %0, i64 28
  %107 = bitcast i8* %106 to float*
  store float %60, float* %107, align 4, !tbaa !7
  %108 = getelementptr inbounds i8, i8* %0, i64 32
  %109 = bitcast i8* %108 to float*
  store float %98, float* %109, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_exp2_ff(float) local_unnamed_addr #6 {
  %2 = fcmp olt float %0, -1.260000e+02
  %3 = fcmp ogt float %0, 1.260000e+02
  %4 = select i1 %3, float 1.260000e+02, float %0
  %5 = select i1 %2, float -1.260000e+02, float %4
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

; Function Attrs: norecurse nounwind uwtable
define void @osl_exp2_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, -1.260000e+02
  %6 = fcmp ogt float %4, 1.260000e+02
  %7 = select i1 %6, float 1.260000e+02, float %4
  %8 = select i1 %5, float -1.260000e+02, float %7
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
  %28 = fmul float %27, 0x3FE62E4300000000
  %29 = getelementptr inbounds i8, i8* %1, i64 4
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !1
  %32 = fmul float %31, %28
  %33 = getelementptr inbounds i8, i8* %1, i64 8
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %34, align 4, !tbaa !1
  %36 = fmul float %35, %28
  %37 = insertelement <2 x float> undef, float %27, i32 0
  %38 = insertelement <2 x float> %37, float %32, i32 1
  %39 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %38, <2 x float>* %39, align 4
  %40 = getelementptr inbounds i8, i8* %0, i64 8
  %41 = bitcast i8* %40 to float*
  store float %36, float* %41, align 4
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_exp2_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, -1.260000e+02
  %6 = fcmp ogt float %4, 1.260000e+02
  %7 = select i1 %6, float 1.260000e+02, float %4
  %8 = select i1 %5, float -1.260000e+02, float %7
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
  %27 = bitcast i8* %0 to i32*
  store i32 %26, i32* %27, align 4, !tbaa !1
  %28 = getelementptr inbounds i8, i8* %1, i64 4
  %29 = bitcast i8* %28 to float*
  %30 = load float, float* %29, align 4, !tbaa !1
  %31 = fcmp olt float %30, -1.260000e+02
  %32 = fcmp ogt float %30, 1.260000e+02
  %33 = select i1 %32, float 1.260000e+02, float %30
  %34 = select i1 %31, float -1.260000e+02, float %33
  %35 = fptosi float %34 to i32
  %36 = sitofp i32 %35 to float
  %37 = fsub float %34, %36
  %38 = fsub float 1.000000e+00, %37
  %39 = fsub float 1.000000e+00, %38
  %40 = fmul float %39, 0x3F55D889C0000000
  %41 = fadd float %40, 0x3F84177340000000
  %42 = fmul float %39, %41
  %43 = fadd float %42, 0x3FAC6CE660000000
  %44 = fmul float %39, %43
  %45 = fadd float %44, 0x3FCEBE3240000000
  %46 = fmul float %39, %45
  %47 = fadd float %46, 0x3FE62E3E20000000
  %48 = fmul float %39, %47
  %49 = fadd float %48, 1.000000e+00
  %50 = bitcast float %49 to i32
  %51 = shl i32 %35, 23
  %52 = add i32 %50, %51
  %53 = getelementptr inbounds i8, i8* %0, i64 4
  %54 = bitcast i8* %53 to i32*
  store i32 %52, i32* %54, align 4, !tbaa !1
  %55 = getelementptr inbounds i8, i8* %1, i64 8
  %56 = bitcast i8* %55 to float*
  %57 = load float, float* %56, align 4, !tbaa !1
  %58 = fcmp olt float %57, -1.260000e+02
  %59 = fcmp ogt float %57, 1.260000e+02
  %60 = select i1 %59, float 1.260000e+02, float %57
  %61 = select i1 %58, float -1.260000e+02, float %60
  %62 = fptosi float %61 to i32
  %63 = sitofp i32 %62 to float
  %64 = fsub float %61, %63
  %65 = fsub float 1.000000e+00, %64
  %66 = fsub float 1.000000e+00, %65
  %67 = fmul float %66, 0x3F55D889C0000000
  %68 = fadd float %67, 0x3F84177340000000
  %69 = fmul float %66, %68
  %70 = fadd float %69, 0x3FAC6CE660000000
  %71 = fmul float %66, %70
  %72 = fadd float %71, 0x3FCEBE3240000000
  %73 = fmul float %66, %72
  %74 = fadd float %73, 0x3FE62E3E20000000
  %75 = fmul float %66, %74
  %76 = fadd float %75, 1.000000e+00
  %77 = bitcast float %76 to i32
  %78 = shl i32 %62, 23
  %79 = add i32 %77, %78
  %80 = getelementptr inbounds i8, i8* %0, i64 8
  %81 = bitcast i8* %80 to i32*
  store i32 %79, i32* %81, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_exp2_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fcmp olt float %5, -1.260000e+02
  %7 = fcmp ogt float %5, 1.260000e+02
  %8 = select i1 %7, float 1.260000e+02, float %5
  %9 = select i1 %6, float -1.260000e+02, float %8
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
  %29 = fmul float %28, 0x3FE62E4300000000
  %30 = getelementptr inbounds i8, i8* %1, i64 4
  %31 = getelementptr inbounds i8, i8* %1, i64 28
  %32 = bitcast i8* %30 to float*
  %33 = load float, float* %32, align 4, !tbaa !1
  %34 = bitcast i8* %31 to float*
  %35 = load float, float* %34, align 4, !tbaa !1
  %36 = fcmp olt float %33, -1.260000e+02
  %37 = fcmp ogt float %33, 1.260000e+02
  %38 = select i1 %37, float 1.260000e+02, float %33
  %39 = select i1 %36, float -1.260000e+02, float %38
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
  %59 = fmul float %58, 0x3FE62E4300000000
  %60 = fmul float %35, %59
  %61 = getelementptr inbounds i8, i8* %1, i64 8
  %62 = getelementptr inbounds i8, i8* %1, i64 32
  %63 = bitcast i8* %61 to float*
  %64 = load float, float* %63, align 4, !tbaa !1
  %65 = bitcast i8* %3 to <4 x float>*
  %66 = load <4 x float>, <4 x float>* %65, align 4, !tbaa !1
  %67 = bitcast i8* %62 to float*
  %68 = load float, float* %67, align 4, !tbaa !1
  %69 = fcmp olt float %64, -1.260000e+02
  %70 = fcmp ogt float %64, 1.260000e+02
  %71 = select i1 %70, float 1.260000e+02, float %64
  %72 = select i1 %69, float -1.260000e+02, float %71
  %73 = fptosi float %72 to i32
  %74 = sitofp i32 %73 to float
  %75 = fsub float %72, %74
  %76 = fsub float 1.000000e+00, %75
  %77 = fsub float 1.000000e+00, %76
  %78 = fmul float %77, 0x3F55D889C0000000
  %79 = fadd float %78, 0x3F84177340000000
  %80 = fmul float %77, %79
  %81 = fadd float %80, 0x3FAC6CE660000000
  %82 = fmul float %77, %81
  %83 = fadd float %82, 0x3FCEBE3240000000
  %84 = fmul float %77, %83
  %85 = fadd float %84, 0x3FE62E3E20000000
  %86 = fmul float %77, %85
  %87 = fadd float %86, 1.000000e+00
  %88 = bitcast float %87 to i32
  %89 = shl i32 %73, 23
  %90 = add i32 %88, %89
  %91 = bitcast i32 %90 to float
  %92 = fmul float %91, 0x3FE62E4300000000
  %93 = insertelement <4 x float> undef, float %29, i32 0
  %94 = insertelement <4 x float> %93, float %59, i32 1
  %95 = insertelement <4 x float> %94, float %92, i32 2
  %96 = insertelement <4 x float> %95, float %29, i32 3
  %97 = fmul <4 x float> %66, %96
  %98 = fmul float %68, %92
  %99 = bitcast i8* %0 to i32*
  store i32 %27, i32* %99, align 4, !tbaa !5
  %100 = getelementptr inbounds i8, i8* %0, i64 4
  %101 = bitcast i8* %100 to i32*
  store i32 %57, i32* %101, align 4, !tbaa !7
  %102 = getelementptr inbounds i8, i8* %0, i64 8
  %103 = bitcast i8* %102 to i32*
  store i32 %90, i32* %103, align 4, !tbaa !8
  %104 = getelementptr inbounds i8, i8* %0, i64 12
  %105 = bitcast i8* %104 to <4 x float>*
  store <4 x float> %97, <4 x float>* %105, align 4, !tbaa !1
  %106 = getelementptr inbounds i8, i8* %0, i64 28
  %107 = bitcast i8* %106 to float*
  store float %60, float* %107, align 4, !tbaa !7
  %108 = getelementptr inbounds i8, i8* %0, i64 32
  %109 = bitcast i8* %108 to float*
  store float %98, float* %109, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_expm1_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fcmp olt float %2, 0x3F9EB851E0000000
  br i1 %3, label %4, label %11

; <label>:4:                                      ; preds = %1
  %5 = fsub float 1.000000e+00, %0
  %6 = fsub float 1.000000e+00, %5
  %7 = fmul float %6, %6
  %8 = fmul float %7, 5.000000e-01
  %9 = fadd float %6, %8
  %10 = tail call float @copysignf(float %9, float %0) #13
  br label %37

; <label>:11:                                     ; preds = %1
  %12 = fmul float %0, 0x3FF7154760000000
  %13 = fcmp olt float %12, -1.260000e+02
  %14 = fcmp ogt float %12, 1.260000e+02
  %15 = select i1 %14, float 1.260000e+02, float %12
  %16 = select i1 %13, float -1.260000e+02, float %15
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
  br label %37

; <label>:37:                                     ; preds = %4, %11
  %38 = phi float [ %10, %4 ], [ %36, %11 ]
  ret float %38
}

; Function Attrs: nounwind uwtable
define void @osl_expm1_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp olt float %5, 0x3F9EB851E0000000
  br i1 %6, label %7, label %15

; <label>:7:                                      ; preds = %2
  %8 = fsub float 1.000000e+00, %4
  %9 = fsub float 1.000000e+00, %8
  %10 = fmul float %9, %9
  %11 = fmul float %10, 5.000000e-01
  %12 = fadd float %9, %11
  %13 = tail call float @copysignf(float %12, float %4) #13
  %14 = fmul float %4, 0x3FF7154760000000
  br label %41

; <label>:15:                                     ; preds = %2
  %16 = fmul float %4, 0x3FF7154760000000
  %17 = fcmp olt float %16, -1.260000e+02
  %18 = fcmp ogt float %16, 1.260000e+02
  %19 = select i1 %18, float 1.260000e+02, float %16
  %20 = select i1 %17, float -1.260000e+02, float %19
  %21 = fptosi float %20 to i32
  %22 = sitofp i32 %21 to float
  %23 = fsub float %20, %22
  %24 = fsub float 1.000000e+00, %23
  %25 = fsub float 1.000000e+00, %24
  %26 = fmul float %25, 0x3F55D889C0000000
  %27 = fadd float %26, 0x3F84177340000000
  %28 = fmul float %25, %27
  %29 = fadd float %28, 0x3FAC6CE660000000
  %30 = fmul float %25, %29
  %31 = fadd float %30, 0x3FCEBE3240000000
  %32 = fmul float %25, %31
  %33 = fadd float %32, 0x3FE62E3E20000000
  %34 = fmul float %25, %33
  %35 = fadd float %34, 1.000000e+00
  %36 = bitcast float %35 to i32
  %37 = shl i32 %21, 23
  %38 = add i32 %36, %37
  %39 = bitcast i32 %38 to float
  %40 = fadd float %39, -1.000000e+00
  br label %41

; <label>:41:                                     ; preds = %7, %15
  %42 = phi float [ %14, %7 ], [ %16, %15 ]
  %43 = phi float [ %13, %7 ], [ %40, %15 ]
  %44 = fcmp olt float %42, -1.260000e+02
  %45 = fcmp ogt float %42, 1.260000e+02
  %46 = select i1 %45, float 1.260000e+02, float %42
  %47 = select i1 %44, float -1.260000e+02, float %46
  %48 = fptosi float %47 to i32
  %49 = sitofp i32 %48 to float
  %50 = fsub float %47, %49
  %51 = fsub float 1.000000e+00, %50
  %52 = fsub float 1.000000e+00, %51
  %53 = fmul float %52, 0x3F55D889C0000000
  %54 = fadd float %53, 0x3F84177340000000
  %55 = fmul float %52, %54
  %56 = fadd float %55, 0x3FAC6CE660000000
  %57 = fmul float %52, %56
  %58 = fadd float %57, 0x3FCEBE3240000000
  %59 = fmul float %52, %58
  %60 = fadd float %59, 0x3FE62E3E20000000
  %61 = fmul float %52, %60
  %62 = fadd float %61, 1.000000e+00
  %63 = bitcast float %62 to i32
  %64 = shl i32 %48, 23
  %65 = add i32 %63, %64
  %66 = bitcast i32 %65 to float
  %67 = getelementptr inbounds i8, i8* %1, i64 4
  %68 = bitcast i8* %67 to float*
  %69 = load float, float* %68, align 4, !tbaa !1
  %70 = fmul float %69, %66
  %71 = getelementptr inbounds i8, i8* %1, i64 8
  %72 = bitcast i8* %71 to float*
  %73 = load float, float* %72, align 4, !tbaa !1
  %74 = fmul float %73, %66
  %75 = insertelement <2 x float> undef, float %43, i32 0
  %76 = insertelement <2 x float> %75, float %70, i32 1
  %77 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %76, <2 x float>* %77, align 4
  %78 = getelementptr inbounds i8, i8* %0, i64 8
  %79 = bitcast i8* %78 to float*
  store float %74, float* %79, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_expm1_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp olt float %5, 0x3F9EB851E0000000
  br i1 %6, label %7, label %14

; <label>:7:                                      ; preds = %2
  %8 = fsub float 1.000000e+00, %4
  %9 = fsub float 1.000000e+00, %8
  %10 = fmul float %9, %9
  %11 = fmul float %10, 5.000000e-01
  %12 = fadd float %9, %11
  %13 = tail call float @copysignf(float %12, float %4) #13
  br label %40

; <label>:14:                                     ; preds = %2
  %15 = fmul float %4, 0x3FF7154760000000
  %16 = fcmp olt float %15, -1.260000e+02
  %17 = fcmp ogt float %15, 1.260000e+02
  %18 = select i1 %17, float 1.260000e+02, float %15
  %19 = select i1 %16, float -1.260000e+02, float %18
  %20 = fptosi float %19 to i32
  %21 = sitofp i32 %20 to float
  %22 = fsub float %19, %21
  %23 = fsub float 1.000000e+00, %22
  %24 = fsub float 1.000000e+00, %23
  %25 = fmul float %24, 0x3F55D889C0000000
  %26 = fadd float %25, 0x3F84177340000000
  %27 = fmul float %24, %26
  %28 = fadd float %27, 0x3FAC6CE660000000
  %29 = fmul float %24, %28
  %30 = fadd float %29, 0x3FCEBE3240000000
  %31 = fmul float %24, %30
  %32 = fadd float %31, 0x3FE62E3E20000000
  %33 = fmul float %24, %32
  %34 = fadd float %33, 1.000000e+00
  %35 = bitcast float %34 to i32
  %36 = shl i32 %20, 23
  %37 = add i32 %35, %36
  %38 = bitcast i32 %37 to float
  %39 = fadd float %38, -1.000000e+00
  br label %40

; <label>:40:                                     ; preds = %7, %14
  %41 = phi float [ %13, %7 ], [ %39, %14 ]
  %42 = bitcast i8* %0 to float*
  store float %41, float* %42, align 4, !tbaa !1
  %43 = getelementptr inbounds i8, i8* %1, i64 4
  %44 = bitcast i8* %43 to float*
  %45 = load float, float* %44, align 4, !tbaa !1
  %46 = tail call float @fabsf(float %45) #13
  %47 = fcmp olt float %46, 0x3F9EB851E0000000
  br i1 %47, label %48, label %55

; <label>:48:                                     ; preds = %40
  %49 = fsub float 1.000000e+00, %45
  %50 = fsub float 1.000000e+00, %49
  %51 = fmul float %50, %50
  %52 = fmul float %51, 5.000000e-01
  %53 = fadd float %50, %52
  %54 = tail call float @copysignf(float %53, float %45) #13
  br label %81

; <label>:55:                                     ; preds = %40
  %56 = fmul float %45, 0x3FF7154760000000
  %57 = fcmp olt float %56, -1.260000e+02
  %58 = fcmp ogt float %56, 1.260000e+02
  %59 = select i1 %58, float 1.260000e+02, float %56
  %60 = select i1 %57, float -1.260000e+02, float %59
  %61 = fptosi float %60 to i32
  %62 = sitofp i32 %61 to float
  %63 = fsub float %60, %62
  %64 = fsub float 1.000000e+00, %63
  %65 = fsub float 1.000000e+00, %64
  %66 = fmul float %65, 0x3F55D889C0000000
  %67 = fadd float %66, 0x3F84177340000000
  %68 = fmul float %65, %67
  %69 = fadd float %68, 0x3FAC6CE660000000
  %70 = fmul float %65, %69
  %71 = fadd float %70, 0x3FCEBE3240000000
  %72 = fmul float %65, %71
  %73 = fadd float %72, 0x3FE62E3E20000000
  %74 = fmul float %65, %73
  %75 = fadd float %74, 1.000000e+00
  %76 = bitcast float %75 to i32
  %77 = shl i32 %61, 23
  %78 = add i32 %76, %77
  %79 = bitcast i32 %78 to float
  %80 = fadd float %79, -1.000000e+00
  br label %81

; <label>:81:                                     ; preds = %48, %55
  %82 = phi float [ %54, %48 ], [ %80, %55 ]
  %83 = getelementptr inbounds i8, i8* %0, i64 4
  %84 = bitcast i8* %83 to float*
  store float %82, float* %84, align 4, !tbaa !1
  %85 = getelementptr inbounds i8, i8* %1, i64 8
  %86 = bitcast i8* %85 to float*
  %87 = load float, float* %86, align 4, !tbaa !1
  %88 = tail call float @fabsf(float %87) #13
  %89 = fcmp olt float %88, 0x3F9EB851E0000000
  br i1 %89, label %90, label %97

; <label>:90:                                     ; preds = %81
  %91 = fsub float 1.000000e+00, %87
  %92 = fsub float 1.000000e+00, %91
  %93 = fmul float %92, %92
  %94 = fmul float %93, 5.000000e-01
  %95 = fadd float %92, %94
  %96 = tail call float @copysignf(float %95, float %87) #13
  br label %123

; <label>:97:                                     ; preds = %81
  %98 = fmul float %87, 0x3FF7154760000000
  %99 = fcmp olt float %98, -1.260000e+02
  %100 = fcmp ogt float %98, 1.260000e+02
  %101 = select i1 %100, float 1.260000e+02, float %98
  %102 = select i1 %99, float -1.260000e+02, float %101
  %103 = fptosi float %102 to i32
  %104 = sitofp i32 %103 to float
  %105 = fsub float %102, %104
  %106 = fsub float 1.000000e+00, %105
  %107 = fsub float 1.000000e+00, %106
  %108 = fmul float %107, 0x3F55D889C0000000
  %109 = fadd float %108, 0x3F84177340000000
  %110 = fmul float %107, %109
  %111 = fadd float %110, 0x3FAC6CE660000000
  %112 = fmul float %107, %111
  %113 = fadd float %112, 0x3FCEBE3240000000
  %114 = fmul float %107, %113
  %115 = fadd float %114, 0x3FE62E3E20000000
  %116 = fmul float %107, %115
  %117 = fadd float %116, 1.000000e+00
  %118 = bitcast float %117 to i32
  %119 = shl i32 %103, 23
  %120 = add i32 %118, %119
  %121 = bitcast i32 %120 to float
  %122 = fadd float %121, -1.000000e+00
  br label %123

; <label>:123:                                    ; preds = %90, %97
  %124 = phi float [ %96, %90 ], [ %122, %97 ]
  %125 = getelementptr inbounds i8, i8* %0, i64 8
  %126 = bitcast i8* %125 to float*
  store float %124, float* %126, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_expm1_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = getelementptr inbounds i8, i8* %1, i64 24
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = bitcast i8* %3 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = bitcast i8* %4 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = tail call float @fabsf(float %6) #13
  %12 = fcmp olt float %11, 0x3F9EB851E0000000
  br i1 %12, label %13, label %21

; <label>:13:                                     ; preds = %2
  %14 = fsub float 1.000000e+00, %6
  %15 = fsub float 1.000000e+00, %14
  %16 = fmul float %15, %15
  %17 = fmul float %16, 5.000000e-01
  %18 = fadd float %15, %17
  %19 = tail call float @copysignf(float %18, float %6) #13
  %20 = fmul float %6, 0x3FF7154760000000
  br label %47

; <label>:21:                                     ; preds = %2
  %22 = fmul float %6, 0x3FF7154760000000
  %23 = fcmp olt float %22, -1.260000e+02
  %24 = fcmp ogt float %22, 1.260000e+02
  %25 = select i1 %24, float 1.260000e+02, float %22
  %26 = select i1 %23, float -1.260000e+02, float %25
  %27 = fptosi float %26 to i32
  %28 = sitofp i32 %27 to float
  %29 = fsub float %26, %28
  %30 = fsub float 1.000000e+00, %29
  %31 = fsub float 1.000000e+00, %30
  %32 = fmul float %31, 0x3F55D889C0000000
  %33 = fadd float %32, 0x3F84177340000000
  %34 = fmul float %31, %33
  %35 = fadd float %34, 0x3FAC6CE660000000
  %36 = fmul float %31, %35
  %37 = fadd float %36, 0x3FCEBE3240000000
  %38 = fmul float %31, %37
  %39 = fadd float %38, 0x3FE62E3E20000000
  %40 = fmul float %31, %39
  %41 = fadd float %40, 1.000000e+00
  %42 = bitcast float %41 to i32
  %43 = shl i32 %27, 23
  %44 = add i32 %42, %43
  %45 = bitcast i32 %44 to float
  %46 = fadd float %45, -1.000000e+00
  br label %47

; <label>:47:                                     ; preds = %13, %21
  %48 = phi float [ %20, %13 ], [ %22, %21 ]
  %49 = phi float [ %19, %13 ], [ %46, %21 ]
  %50 = fcmp olt float %48, -1.260000e+02
  %51 = fcmp ogt float %48, 1.260000e+02
  %52 = select i1 %51, float 1.260000e+02, float %48
  %53 = select i1 %50, float -1.260000e+02, float %52
  %54 = fptosi float %53 to i32
  %55 = sitofp i32 %54 to float
  %56 = fsub float %53, %55
  %57 = fsub float 1.000000e+00, %56
  %58 = fsub float 1.000000e+00, %57
  %59 = fmul float %58, 0x3F55D889C0000000
  %60 = fadd float %59, 0x3F84177340000000
  %61 = fmul float %58, %60
  %62 = fadd float %61, 0x3FAC6CE660000000
  %63 = fmul float %58, %62
  %64 = fadd float %63, 0x3FCEBE3240000000
  %65 = fmul float %58, %64
  %66 = fadd float %65, 0x3FE62E3E20000000
  %67 = fmul float %58, %66
  %68 = fadd float %67, 1.000000e+00
  %69 = bitcast float %68 to i32
  %70 = shl i32 %54, 23
  %71 = add i32 %69, %70
  %72 = bitcast i32 %71 to float
  %73 = fmul float %8, %72
  %74 = fmul float %10, %72
  %75 = getelementptr inbounds i8, i8* %1, i64 4
  %76 = getelementptr inbounds i8, i8* %1, i64 16
  %77 = getelementptr inbounds i8, i8* %1, i64 28
  %78 = bitcast i8* %75 to float*
  %79 = load float, float* %78, align 4, !tbaa !1
  %80 = bitcast i8* %76 to float*
  %81 = load float, float* %80, align 4, !tbaa !1
  %82 = bitcast i8* %77 to float*
  %83 = load float, float* %82, align 4, !tbaa !1
  %84 = tail call float @fabsf(float %79) #13
  %85 = fcmp olt float %84, 0x3F9EB851E0000000
  br i1 %85, label %86, label %94

; <label>:86:                                     ; preds = %47
  %87 = fsub float 1.000000e+00, %79
  %88 = fsub float 1.000000e+00, %87
  %89 = fmul float %88, %88
  %90 = fmul float %89, 5.000000e-01
  %91 = fadd float %88, %90
  %92 = tail call float @copysignf(float %91, float %79) #13
  %93 = fmul float %79, 0x3FF7154760000000
  br label %120

; <label>:94:                                     ; preds = %47
  %95 = fmul float %79, 0x3FF7154760000000
  %96 = fcmp olt float %95, -1.260000e+02
  %97 = fcmp ogt float %95, 1.260000e+02
  %98 = select i1 %97, float 1.260000e+02, float %95
  %99 = select i1 %96, float -1.260000e+02, float %98
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
  br label %120

; <label>:120:                                    ; preds = %86, %94
  %121 = phi float [ %93, %86 ], [ %95, %94 ]
  %122 = phi float [ %92, %86 ], [ %119, %94 ]
  %123 = fcmp olt float %121, -1.260000e+02
  %124 = fcmp ogt float %121, 1.260000e+02
  %125 = select i1 %124, float 1.260000e+02, float %121
  %126 = select i1 %123, float -1.260000e+02, float %125
  %127 = fptosi float %126 to i32
  %128 = sitofp i32 %127 to float
  %129 = fsub float %126, %128
  %130 = fsub float 1.000000e+00, %129
  %131 = fsub float 1.000000e+00, %130
  %132 = fmul float %131, 0x3F55D889C0000000
  %133 = fadd float %132, 0x3F84177340000000
  %134 = fmul float %131, %133
  %135 = fadd float %134, 0x3FAC6CE660000000
  %136 = fmul float %131, %135
  %137 = fadd float %136, 0x3FCEBE3240000000
  %138 = fmul float %131, %137
  %139 = fadd float %138, 0x3FE62E3E20000000
  %140 = fmul float %131, %139
  %141 = fadd float %140, 1.000000e+00
  %142 = bitcast float %141 to i32
  %143 = shl i32 %127, 23
  %144 = add i32 %142, %143
  %145 = bitcast i32 %144 to float
  %146 = fmul float %81, %145
  %147 = fmul float %83, %145
  %148 = getelementptr inbounds i8, i8* %1, i64 8
  %149 = getelementptr inbounds i8, i8* %1, i64 20
  %150 = getelementptr inbounds i8, i8* %1, i64 32
  %151 = bitcast i8* %148 to float*
  %152 = load float, float* %151, align 4, !tbaa !1
  %153 = bitcast i8* %149 to float*
  %154 = load float, float* %153, align 4, !tbaa !1
  %155 = bitcast i8* %150 to float*
  %156 = load float, float* %155, align 4, !tbaa !1
  %157 = tail call float @fabsf(float %152) #13
  %158 = fcmp olt float %157, 0x3F9EB851E0000000
  br i1 %158, label %159, label %167

; <label>:159:                                    ; preds = %120
  %160 = fsub float 1.000000e+00, %152
  %161 = fsub float 1.000000e+00, %160
  %162 = fmul float %161, %161
  %163 = fmul float %162, 5.000000e-01
  %164 = fadd float %161, %163
  %165 = tail call float @copysignf(float %164, float %152) #13
  %166 = fmul float %152, 0x3FF7154760000000
  br label %193

; <label>:167:                                    ; preds = %120
  %168 = fmul float %152, 0x3FF7154760000000
  %169 = fcmp olt float %168, -1.260000e+02
  %170 = fcmp ogt float %168, 1.260000e+02
  %171 = select i1 %170, float 1.260000e+02, float %168
  %172 = select i1 %169, float -1.260000e+02, float %171
  %173 = fptosi float %172 to i32
  %174 = sitofp i32 %173 to float
  %175 = fsub float %172, %174
  %176 = fsub float 1.000000e+00, %175
  %177 = fsub float 1.000000e+00, %176
  %178 = fmul float %177, 0x3F55D889C0000000
  %179 = fadd float %178, 0x3F84177340000000
  %180 = fmul float %177, %179
  %181 = fadd float %180, 0x3FAC6CE660000000
  %182 = fmul float %177, %181
  %183 = fadd float %182, 0x3FCEBE3240000000
  %184 = fmul float %177, %183
  %185 = fadd float %184, 0x3FE62E3E20000000
  %186 = fmul float %177, %185
  %187 = fadd float %186, 1.000000e+00
  %188 = bitcast float %187 to i32
  %189 = shl i32 %173, 23
  %190 = add i32 %188, %189
  %191 = bitcast i32 %190 to float
  %192 = fadd float %191, -1.000000e+00
  br label %193

; <label>:193:                                    ; preds = %159, %167
  %194 = phi float [ %166, %159 ], [ %168, %167 ]
  %195 = phi float [ %165, %159 ], [ %192, %167 ]
  %196 = fcmp olt float %194, -1.260000e+02
  %197 = fcmp ogt float %194, 1.260000e+02
  %198 = select i1 %197, float 1.260000e+02, float %194
  %199 = select i1 %196, float -1.260000e+02, float %198
  %200 = fptosi float %199 to i32
  %201 = sitofp i32 %200 to float
  %202 = fsub float %199, %201
  %203 = fsub float 1.000000e+00, %202
  %204 = fsub float 1.000000e+00, %203
  %205 = fmul float %204, 0x3F55D889C0000000
  %206 = fadd float %205, 0x3F84177340000000
  %207 = fmul float %204, %206
  %208 = fadd float %207, 0x3FAC6CE660000000
  %209 = fmul float %204, %208
  %210 = fadd float %209, 0x3FCEBE3240000000
  %211 = fmul float %204, %210
  %212 = fadd float %211, 0x3FE62E3E20000000
  %213 = fmul float %204, %212
  %214 = fadd float %213, 1.000000e+00
  %215 = bitcast float %214 to i32
  %216 = shl i32 %200, 23
  %217 = add i32 %215, %216
  %218 = bitcast i32 %217 to float
  %219 = fmul float %154, %218
  %220 = fmul float %156, %218
  %221 = bitcast i8* %0 to float*
  store float %49, float* %221, align 4, !tbaa !5
  %222 = getelementptr inbounds i8, i8* %0, i64 4
  %223 = bitcast i8* %222 to float*
  store float %122, float* %223, align 4, !tbaa !7
  %224 = getelementptr inbounds i8, i8* %0, i64 8
  %225 = bitcast i8* %224 to float*
  store float %195, float* %225, align 4, !tbaa !8
  %226 = getelementptr inbounds i8, i8* %0, i64 12
  %227 = bitcast i8* %226 to float*
  store float %73, float* %227, align 4, !tbaa !5
  %228 = getelementptr inbounds i8, i8* %0, i64 16
  %229 = bitcast i8* %228 to float*
  store float %146, float* %229, align 4, !tbaa !7
  %230 = getelementptr inbounds i8, i8* %0, i64 20
  %231 = bitcast i8* %230 to float*
  store float %219, float* %231, align 4, !tbaa !8
  %232 = getelementptr inbounds i8, i8* %0, i64 24
  %233 = bitcast i8* %232 to float*
  store float %74, float* %233, align 4, !tbaa !5
  %234 = getelementptr inbounds i8, i8* %0, i64 28
  %235 = bitcast i8* %234 to float*
  store float %147, float* %235, align 4, !tbaa !7
  %236 = getelementptr inbounds i8, i8* %0, i64 32
  %237 = bitcast i8* %236 to float*
  store float %220, float* %237, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define float @osl_pow_fff(float, float) local_unnamed_addr #4 {
  %3 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %0, float %1)
  ret float %3
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float, float) local_unnamed_addr #8 comdat {
  %3 = fcmp oeq float %1, 0.000000e+00
  br i1 %3, label %90, label %4

; <label>:4:                                      ; preds = %2
  %5 = fcmp oeq float %0, 0.000000e+00
  br i1 %5, label %90, label %6

; <label>:6:                                      ; preds = %4
  %7 = fcmp oeq float %1, 1.000000e+00
  br i1 %7, label %90, label %8

; <label>:8:                                      ; preds = %6
  %9 = fcmp oeq float %1, 2.000000e+00
  br i1 %9, label %10, label %14

; <label>:10:                                     ; preds = %8
  %11 = fmul float %0, %0
  %12 = fcmp ogt float %11, 0x47EFFFFFE0000000
  %13 = select i1 %12, float 0x47EFFFFFE0000000, float %11
  br label %90

; <label>:14:                                     ; preds = %8
  %15 = fcmp olt float %0, 0.000000e+00
  br i1 %15, label %16, label %32

; <label>:16:                                     ; preds = %14
  %17 = bitcast float %1 to i32
  %18 = and i32 %17, 2147483647
  %19 = icmp ugt i32 %18, 1266679807
  br i1 %19, label %32, label %20

; <label>:20:                                     ; preds = %16
  %21 = icmp ugt i32 %18, 1065353215
  br i1 %21, label %22, label %90

; <label>:22:                                     ; preds = %20
  %23 = lshr i32 %18, 23
  %24 = sub nsw i32 150, %23
  %25 = lshr i32 %18, %24
  %26 = shl i32 %25, %24
  %27 = icmp eq i32 %26, %18
  br i1 %27, label %28, label %90

; <label>:28:                                     ; preds = %22
  %29 = shl i32 %25, 31
  %30 = or i32 %29, 1065353216
  %31 = bitcast i32 %30 to float
  br label %32

; <label>:32:                                     ; preds = %16, %28, %14
  %33 = phi float [ 1.000000e+00, %14 ], [ %31, %28 ], [ 1.000000e+00, %16 ]
  %34 = tail call float @fabsf(float %0) #13
  %35 = fcmp olt float %34, 0x3810000000000000
  %36 = fcmp ogt float %34, 0x47EFFFFFE0000000
  %37 = select i1 %36, float 0x47EFFFFFE0000000, float %34
  %38 = bitcast float %37 to i32
  %39 = select i1 %35, i32 8388608, i32 %38
  %40 = lshr i32 %39, 23
  %41 = add nsw i32 %40, -127
  %42 = and i32 %39, 8388607
  %43 = or i32 %42, 1065353216
  %44 = bitcast i32 %43 to float
  %45 = fadd float %44, -1.000000e+00
  %46 = fmul float %45, %45
  %47 = fmul float %46, %46
  %48 = fmul float %45, 0x3F831161A0000000
  %49 = fsub float 0x3FAAA83920000000, %48
  %50 = fmul float %45, 0x3FDEA2C5A0000000
  %51 = fadd float %50, 0xBFE713CA80000000
  %52 = fmul float %45, %49
  %53 = fadd float %52, 0xBFC19A9FA0000000
  %54 = fmul float %45, %53
  %55 = fadd float %54, 0x3FCEF5B7A0000000
  %56 = fmul float %45, %55
  %57 = fadd float %56, 0xBFD63A40C0000000
  %58 = fmul float %45, %51
  %59 = fadd float %58, 0x3FF7154200000000
  %60 = fmul float %47, %57
  %61 = fmul float %45, %59
  %62 = fadd float %61, %60
  %63 = sitofp i32 %41 to float
  %64 = fadd float %63, %62
  %65 = fmul float %64, %1
  %66 = fcmp olt float %65, -1.260000e+02
  %67 = fcmp ogt float %65, 1.260000e+02
  %68 = select i1 %67, float 1.260000e+02, float %65
  %69 = select i1 %66, float -1.260000e+02, float %68
  %70 = fptosi float %69 to i32
  %71 = sitofp i32 %70 to float
  %72 = fsub float %69, %71
  %73 = fsub float 1.000000e+00, %72
  %74 = fsub float 1.000000e+00, %73
  %75 = fmul float %74, 0x3F55D889C0000000
  %76 = fadd float %75, 0x3F84177340000000
  %77 = fmul float %74, %76
  %78 = fadd float %77, 0x3FAC6CE660000000
  %79 = fmul float %74, %78
  %80 = fadd float %79, 0x3FCEBE3240000000
  %81 = fmul float %74, %80
  %82 = fadd float %81, 0x3FE62E3E20000000
  %83 = fmul float %74, %82
  %84 = fadd float %83, 1.000000e+00
  %85 = bitcast float %84 to i32
  %86 = shl i32 %70, 23
  %87 = add i32 %85, %86
  %88 = bitcast i32 %87 to float
  %89 = fmul float %33, %88
  br label %90

; <label>:90:                                     ; preds = %20, %22, %32, %6, %4, %2, %10
  %91 = phi float [ %13, %10 ], [ 1.000000e+00, %2 ], [ 0.000000e+00, %4 ], [ %0, %6 ], [ %89, %32 ], [ 0.000000e+00, %22 ], [ 0.000000e+00, %20 ]
  ret float %91
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dfdfdf(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %2 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fadd float %7, -1.000000e+00
  %9 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %5, float %8) #2
  %10 = load float, float* %4, align 4, !tbaa !1
  %11 = fmul float %9, %10
  %12 = fcmp ogt float %10, 0.000000e+00
  br i1 %12, label %13, label %45

; <label>:13:                                     ; preds = %3
  %14 = fcmp olt float %10, 0x3810000000000000
  %15 = fcmp ogt float %10, 0x47EFFFFFE0000000
  %16 = select i1 %15, float 0x47EFFFFFE0000000, float %10
  %17 = bitcast float %16 to i32
  %18 = select i1 %14, i32 8388608, i32 %17
  %19 = lshr i32 %18, 23
  %20 = add nsw i32 %19, -127
  %21 = and i32 %18, 8388607
  %22 = or i32 %21, 1065353216
  %23 = bitcast i32 %22 to float
  %24 = fadd float %23, -1.000000e+00
  %25 = fmul float %24, %24
  %26 = fmul float %25, %25
  %27 = fmul float %24, 0x3F831161A0000000
  %28 = fsub float 0x3FAAA83920000000, %27
  %29 = fmul float %24, 0x3FDEA2C5A0000000
  %30 = fadd float %29, 0xBFE713CA80000000
  %31 = fmul float %24, %28
  %32 = fadd float %31, 0xBFC19A9FA0000000
  %33 = fmul float %24, %32
  %34 = fadd float %33, 0x3FCEF5B7A0000000
  %35 = fmul float %24, %34
  %36 = fadd float %35, 0xBFD63A40C0000000
  %37 = fmul float %24, %30
  %38 = fadd float %37, 0x3FF7154200000000
  %39 = fmul float %26, %36
  %40 = fmul float %24, %38
  %41 = fadd float %40, %39
  %42 = sitofp i32 %20 to float
  %43 = fadd float %42, %41
  %44 = fmul float %43, 0x3FE62E4300000000
  br label %45

; <label>:45:                                     ; preds = %3, %13
  %46 = phi float [ %44, %13 ], [ 0.000000e+00, %3 ]
  %47 = load float, float* %6, align 4, !tbaa !1
  %48 = fmul float %9, %47
  %49 = getelementptr inbounds i8, i8* %1, i64 4
  %50 = bitcast i8* %49 to float*
  %51 = load float, float* %50, align 4, !tbaa !1
  %52 = fmul float %48, %51
  %53 = fmul float %11, %46
  %54 = getelementptr inbounds i8, i8* %2, i64 4
  %55 = bitcast i8* %54 to float*
  %56 = load float, float* %55, align 4, !tbaa !1
  %57 = fmul float %53, %56
  %58 = fadd float %52, %57
  %59 = getelementptr inbounds i8, i8* %1, i64 8
  %60 = bitcast i8* %59 to float*
  %61 = load float, float* %60, align 4, !tbaa !1
  %62 = fmul float %48, %61
  %63 = getelementptr inbounds i8, i8* %2, i64 8
  %64 = bitcast i8* %63 to float*
  %65 = load float, float* %64, align 4, !tbaa !1
  %66 = fmul float %53, %65
  %67 = fadd float %62, %66
  %68 = insertelement <2 x float> undef, float %11, i32 0
  %69 = insertelement <2 x float> %68, float %58, i32 1
  %70 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %69, <2 x float>* %70, align 4
  %71 = getelementptr inbounds i8, i8* %0, i64 8
  %72 = bitcast i8* %71 to float*
  store float %67, float* %72, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dffdf(i8* nocapture, float, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fadd float %5, -1.000000e+00
  %7 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %1, float %6) #2
  %8 = fmul float %7, %1
  %9 = fcmp ogt float %1, 0.000000e+00
  br i1 %9, label %10, label %42

; <label>:10:                                     ; preds = %3
  %11 = fcmp olt float %1, 0x3810000000000000
  %12 = fcmp ogt float %1, 0x47EFFFFFE0000000
  %13 = select i1 %12, float 0x47EFFFFFE0000000, float %1
  %14 = bitcast float %13 to i32
  %15 = select i1 %11, i32 8388608, i32 %14
  %16 = lshr i32 %15, 23
  %17 = add nsw i32 %16, -127
  %18 = and i32 %15, 8388607
  %19 = or i32 %18, 1065353216
  %20 = bitcast i32 %19 to float
  %21 = fadd float %20, -1.000000e+00
  %22 = fmul float %21, %21
  %23 = fmul float %22, %22
  %24 = fmul float %21, 0x3F831161A0000000
  %25 = fsub float 0x3FAAA83920000000, %24
  %26 = fmul float %21, 0x3FDEA2C5A0000000
  %27 = fadd float %26, 0xBFE713CA80000000
  %28 = fmul float %21, %25
  %29 = fadd float %28, 0xBFC19A9FA0000000
  %30 = fmul float %21, %29
  %31 = fadd float %30, 0x3FCEF5B7A0000000
  %32 = fmul float %21, %31
  %33 = fadd float %32, 0xBFD63A40C0000000
  %34 = fmul float %21, %27
  %35 = fadd float %34, 0x3FF7154200000000
  %36 = fmul float %23, %33
  %37 = fmul float %21, %35
  %38 = fadd float %37, %36
  %39 = sitofp i32 %17 to float
  %40 = fadd float %39, %38
  %41 = fmul float %40, 0x3FE62E4300000000
  br label %42

; <label>:42:                                     ; preds = %3, %10
  %43 = phi float [ %41, %10 ], [ 0.000000e+00, %3 ]
  %44 = load float, float* %4, align 4, !tbaa !1
  %45 = fmul float %7, %44
  %46 = fmul float %45, 0.000000e+00
  %47 = fmul float %8, %43
  %48 = getelementptr inbounds i8, i8* %2, i64 4
  %49 = bitcast i8* %48 to float*
  %50 = load float, float* %49, align 4, !tbaa !1
  %51 = fmul float %47, %50
  %52 = fadd float %46, %51
  %53 = getelementptr inbounds i8, i8* %2, i64 8
  %54 = bitcast i8* %53 to float*
  %55 = load float, float* %54, align 4, !tbaa !1
  %56 = fmul float %47, %55
  %57 = fadd float %46, %56
  %58 = insertelement <2 x float> undef, float %8, i32 0
  %59 = insertelement <2 x float> %58, float %52, i32 1
  %60 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %59, <2 x float>* %60, align 4
  %61 = getelementptr inbounds i8, i8* %0, i64 8
  %62 = bitcast i8* %61 to float*
  store float %57, float* %62, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dfdff(i8* nocapture, i8* nocapture readonly, float) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fadd float %2, -1.000000e+00
  %7 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %5, float %6) #2
  %8 = load float, float* %4, align 4, !tbaa !1
  %9 = fmul float %7, %8
  %10 = fcmp ogt float %8, 0.000000e+00
  br i1 %10, label %11, label %43

; <label>:11:                                     ; preds = %3
  %12 = fcmp olt float %8, 0x3810000000000000
  %13 = fcmp ogt float %8, 0x47EFFFFFE0000000
  %14 = select i1 %13, float 0x47EFFFFFE0000000, float %8
  %15 = bitcast float %14 to i32
  %16 = select i1 %12, i32 8388608, i32 %15
  %17 = lshr i32 %16, 23
  %18 = add nsw i32 %17, -127
  %19 = and i32 %16, 8388607
  %20 = or i32 %19, 1065353216
  %21 = bitcast i32 %20 to float
  %22 = fadd float %21, -1.000000e+00
  %23 = fmul float %22, %22
  %24 = fmul float %23, %23
  %25 = fmul float %22, 0x3F831161A0000000
  %26 = fsub float 0x3FAAA83920000000, %25
  %27 = fmul float %22, 0x3FDEA2C5A0000000
  %28 = fadd float %27, 0xBFE713CA80000000
  %29 = fmul float %22, %26
  %30 = fadd float %29, 0xBFC19A9FA0000000
  %31 = fmul float %22, %30
  %32 = fadd float %31, 0x3FCEF5B7A0000000
  %33 = fmul float %22, %32
  %34 = fadd float %33, 0xBFD63A40C0000000
  %35 = fmul float %22, %28
  %36 = fadd float %35, 0x3FF7154200000000
  %37 = fmul float %24, %34
  %38 = fmul float %22, %36
  %39 = fadd float %38, %37
  %40 = sitofp i32 %18 to float
  %41 = fadd float %40, %39
  %42 = fmul float %41, 0x3FE62E4300000000
  br label %43

; <label>:43:                                     ; preds = %3, %11
  %44 = phi float [ %42, %11 ], [ 0.000000e+00, %3 ]
  %45 = fmul float %7, %2
  %46 = getelementptr inbounds i8, i8* %1, i64 4
  %47 = bitcast i8* %46 to float*
  %48 = load float, float* %47, align 4, !tbaa !1
  %49 = fmul float %45, %48
  %50 = fmul float %9, %44
  %51 = fmul float %50, 0.000000e+00
  %52 = fadd float %49, %51
  %53 = getelementptr inbounds i8, i8* %1, i64 8
  %54 = bitcast i8* %53 to float*
  %55 = load float, float* %54, align 4, !tbaa !1
  %56 = fmul float %45, %55
  %57 = fadd float %51, %56
  %58 = insertelement <2 x float> undef, float %9, i32 0
  %59 = insertelement <2 x float> %58, float %52, i32 1
  %60 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %59, <2 x float>* %60, align 4
  %61 = getelementptr inbounds i8, i8* %0, i64 8
  %62 = bitcast i8* %61 to float*
  store float %57, float* %62, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_vvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %2 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %5, float %7)
  %9 = bitcast i8* %0 to float*
  store float %8, float* %9, align 4, !tbaa !1
  %10 = getelementptr inbounds i8, i8* %1, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = getelementptr inbounds i8, i8* %2, i64 4
  %14 = bitcast i8* %13 to float*
  %15 = load float, float* %14, align 4, !tbaa !1
  %16 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %12, float %15)
  %17 = getelementptr inbounds i8, i8* %0, i64 4
  %18 = bitcast i8* %17 to float*
  store float %16, float* %18, align 4, !tbaa !1
  %19 = getelementptr inbounds i8, i8* %1, i64 8
  %20 = bitcast i8* %19 to float*
  %21 = load float, float* %20, align 4, !tbaa !1
  %22 = getelementptr inbounds i8, i8* %2, i64 8
  %23 = bitcast i8* %22 to float*
  %24 = load float, float* %23, align 4, !tbaa !1
  %25 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %21, float %24)
  %26 = getelementptr inbounds i8, i8* %0, i64 8
  %27 = bitcast i8* %26 to float*
  store float %25, float* %27, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvdvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = getelementptr inbounds i8, i8* %1, i64 12
  %5 = getelementptr inbounds i8, i8* %1, i64 24
  %6 = bitcast i8* %1 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = bitcast i8* %4 to float*
  %9 = load float, float* %8, align 4, !tbaa !1
  %10 = bitcast i8* %5 to float*
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = getelementptr inbounds i8, i8* %2, i64 12
  %13 = getelementptr inbounds i8, i8* %2, i64 24
  %14 = bitcast i8* %2 to float*
  %15 = load float, float* %14, align 4, !tbaa !1
  %16 = bitcast i8* %12 to float*
  %17 = load float, float* %16, align 4, !tbaa !1
  %18 = bitcast i8* %13 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = fadd float %15, -1.000000e+00
  %21 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %7, float %20) #2
  %22 = fmul float %7, %21
  %23 = fcmp ogt float %7, 0.000000e+00
  br i1 %23, label %24, label %56

; <label>:24:                                     ; preds = %3
  %25 = fcmp olt float %7, 0x3810000000000000
  %26 = fcmp ogt float %7, 0x47EFFFFFE0000000
  %27 = select i1 %26, float 0x47EFFFFFE0000000, float %7
  %28 = bitcast float %27 to i32
  %29 = select i1 %25, i32 8388608, i32 %28
  %30 = lshr i32 %29, 23
  %31 = add nsw i32 %30, -127
  %32 = and i32 %29, 8388607
  %33 = or i32 %32, 1065353216
  %34 = bitcast i32 %33 to float
  %35 = fadd float %34, -1.000000e+00
  %36 = fmul float %35, %35
  %37 = fmul float %36, %36
  %38 = fmul float %35, 0x3F831161A0000000
  %39 = fsub float 0x3FAAA83920000000, %38
  %40 = fmul float %35, 0x3FDEA2C5A0000000
  %41 = fadd float %40, 0xBFE713CA80000000
  %42 = fmul float %35, %39
  %43 = fadd float %42, 0xBFC19A9FA0000000
  %44 = fmul float %35, %43
  %45 = fadd float %44, 0x3FCEF5B7A0000000
  %46 = fmul float %35, %45
  %47 = fadd float %46, 0xBFD63A40C0000000
  %48 = fmul float %35, %41
  %49 = fadd float %48, 0x3FF7154200000000
  %50 = fmul float %37, %47
  %51 = fmul float %35, %49
  %52 = fadd float %51, %50
  %53 = sitofp i32 %31 to float
  %54 = fadd float %53, %52
  %55 = fmul float %54, 0x3FE62E4300000000
  br label %56

; <label>:56:                                     ; preds = %3, %24
  %57 = phi float [ %55, %24 ], [ 0.000000e+00, %3 ]
  %58 = fmul float %15, %21
  %59 = fmul float %9, %58
  %60 = fmul float %22, %57
  %61 = fmul float %17, %60
  %62 = fadd float %59, %61
  %63 = fmul float %11, %58
  %64 = fmul float %19, %60
  %65 = fadd float %63, %64
  %66 = getelementptr inbounds i8, i8* %1, i64 4
  %67 = getelementptr inbounds i8, i8* %1, i64 16
  %68 = getelementptr inbounds i8, i8* %1, i64 28
  %69 = bitcast i8* %66 to float*
  %70 = load float, float* %69, align 4, !tbaa !1
  %71 = bitcast i8* %67 to float*
  %72 = load float, float* %71, align 4, !tbaa !1
  %73 = bitcast i8* %68 to float*
  %74 = load float, float* %73, align 4, !tbaa !1
  %75 = getelementptr inbounds i8, i8* %2, i64 4
  %76 = getelementptr inbounds i8, i8* %2, i64 16
  %77 = getelementptr inbounds i8, i8* %2, i64 28
  %78 = bitcast i8* %75 to float*
  %79 = load float, float* %78, align 4, !tbaa !1
  %80 = bitcast i8* %76 to float*
  %81 = load float, float* %80, align 4, !tbaa !1
  %82 = bitcast i8* %77 to float*
  %83 = load float, float* %82, align 4, !tbaa !1
  %84 = fadd float %79, -1.000000e+00
  %85 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %70, float %84) #2
  %86 = fmul float %70, %85
  %87 = fcmp ogt float %70, 0.000000e+00
  br i1 %87, label %88, label %120

; <label>:88:                                     ; preds = %56
  %89 = fcmp olt float %70, 0x3810000000000000
  %90 = fcmp ogt float %70, 0x47EFFFFFE0000000
  %91 = select i1 %90, float 0x47EFFFFFE0000000, float %70
  %92 = bitcast float %91 to i32
  %93 = select i1 %89, i32 8388608, i32 %92
  %94 = lshr i32 %93, 23
  %95 = add nsw i32 %94, -127
  %96 = and i32 %93, 8388607
  %97 = or i32 %96, 1065353216
  %98 = bitcast i32 %97 to float
  %99 = fadd float %98, -1.000000e+00
  %100 = fmul float %99, %99
  %101 = fmul float %100, %100
  %102 = fmul float %99, 0x3F831161A0000000
  %103 = fsub float 0x3FAAA83920000000, %102
  %104 = fmul float %99, 0x3FDEA2C5A0000000
  %105 = fadd float %104, 0xBFE713CA80000000
  %106 = fmul float %99, %103
  %107 = fadd float %106, 0xBFC19A9FA0000000
  %108 = fmul float %99, %107
  %109 = fadd float %108, 0x3FCEF5B7A0000000
  %110 = fmul float %99, %109
  %111 = fadd float %110, 0xBFD63A40C0000000
  %112 = fmul float %99, %105
  %113 = fadd float %112, 0x3FF7154200000000
  %114 = fmul float %101, %111
  %115 = fmul float %99, %113
  %116 = fadd float %115, %114
  %117 = sitofp i32 %95 to float
  %118 = fadd float %117, %116
  %119 = fmul float %118, 0x3FE62E4300000000
  br label %120

; <label>:120:                                    ; preds = %56, %88
  %121 = phi float [ %119, %88 ], [ 0.000000e+00, %56 ]
  %122 = fmul float %79, %85
  %123 = fmul float %72, %122
  %124 = fmul float %86, %121
  %125 = fmul float %81, %124
  %126 = fadd float %123, %125
  %127 = fmul float %74, %122
  %128 = fmul float %83, %124
  %129 = fadd float %127, %128
  %130 = getelementptr inbounds i8, i8* %1, i64 8
  %131 = getelementptr inbounds i8, i8* %1, i64 20
  %132 = getelementptr inbounds i8, i8* %1, i64 32
  %133 = bitcast i8* %130 to float*
  %134 = load float, float* %133, align 4, !tbaa !1
  %135 = bitcast i8* %131 to float*
  %136 = load float, float* %135, align 4, !tbaa !1
  %137 = bitcast i8* %132 to float*
  %138 = load float, float* %137, align 4, !tbaa !1
  %139 = getelementptr inbounds i8, i8* %2, i64 8
  %140 = getelementptr inbounds i8, i8* %2, i64 20
  %141 = getelementptr inbounds i8, i8* %2, i64 32
  %142 = bitcast i8* %139 to float*
  %143 = load float, float* %142, align 4, !tbaa !1
  %144 = bitcast i8* %140 to float*
  %145 = load float, float* %144, align 4, !tbaa !1
  %146 = bitcast i8* %141 to float*
  %147 = load float, float* %146, align 4, !tbaa !1
  %148 = fadd float %143, -1.000000e+00
  %149 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %134, float %148) #2
  %150 = fmul float %134, %149
  %151 = fcmp ogt float %134, 0.000000e+00
  br i1 %151, label %152, label %184

; <label>:152:                                    ; preds = %120
  %153 = fcmp olt float %134, 0x3810000000000000
  %154 = fcmp ogt float %134, 0x47EFFFFFE0000000
  %155 = select i1 %154, float 0x47EFFFFFE0000000, float %134
  %156 = bitcast float %155 to i32
  %157 = select i1 %153, i32 8388608, i32 %156
  %158 = lshr i32 %157, 23
  %159 = add nsw i32 %158, -127
  %160 = and i32 %157, 8388607
  %161 = or i32 %160, 1065353216
  %162 = bitcast i32 %161 to float
  %163 = fadd float %162, -1.000000e+00
  %164 = fmul float %163, %163
  %165 = fmul float %164, %164
  %166 = fmul float %163, 0x3F831161A0000000
  %167 = fsub float 0x3FAAA83920000000, %166
  %168 = fmul float %163, 0x3FDEA2C5A0000000
  %169 = fadd float %168, 0xBFE713CA80000000
  %170 = fmul float %163, %167
  %171 = fadd float %170, 0xBFC19A9FA0000000
  %172 = fmul float %163, %171
  %173 = fadd float %172, 0x3FCEF5B7A0000000
  %174 = fmul float %163, %173
  %175 = fadd float %174, 0xBFD63A40C0000000
  %176 = fmul float %163, %169
  %177 = fadd float %176, 0x3FF7154200000000
  %178 = fmul float %165, %175
  %179 = fmul float %163, %177
  %180 = fadd float %179, %178
  %181 = sitofp i32 %159 to float
  %182 = fadd float %181, %180
  %183 = fmul float %182, 0x3FE62E4300000000
  br label %184

; <label>:184:                                    ; preds = %120, %152
  %185 = phi float [ %183, %152 ], [ 0.000000e+00, %120 ]
  %186 = fmul float %143, %149
  %187 = fmul float %136, %186
  %188 = fmul float %150, %185
  %189 = fmul float %145, %188
  %190 = fadd float %187, %189
  %191 = fmul float %138, %186
  %192 = fmul float %147, %188
  %193 = fadd float %191, %192
  %194 = bitcast i8* %0 to float*
  store float %22, float* %194, align 4, !tbaa !5
  %195 = getelementptr inbounds i8, i8* %0, i64 4
  %196 = bitcast i8* %195 to float*
  store float %86, float* %196, align 4, !tbaa !7
  %197 = getelementptr inbounds i8, i8* %0, i64 8
  %198 = bitcast i8* %197 to float*
  store float %150, float* %198, align 4, !tbaa !8
  %199 = getelementptr inbounds i8, i8* %0, i64 12
  %200 = bitcast i8* %199 to float*
  store float %62, float* %200, align 4, !tbaa !5
  %201 = getelementptr inbounds i8, i8* %0, i64 16
  %202 = bitcast i8* %201 to float*
  store float %126, float* %202, align 4, !tbaa !7
  %203 = getelementptr inbounds i8, i8* %0, i64 20
  %204 = bitcast i8* %203 to float*
  store float %190, float* %204, align 4, !tbaa !8
  %205 = getelementptr inbounds i8, i8* %0, i64 24
  %206 = bitcast i8* %205 to float*
  store float %65, float* %206, align 4, !tbaa !5
  %207 = getelementptr inbounds i8, i8* %0, i64 28
  %208 = bitcast i8* %207 to float*
  store float %129, float* %208, align 4, !tbaa !7
  %209 = getelementptr inbounds i8, i8* %0, i64 32
  %210 = bitcast i8* %209 to float*
  store float %193, float* %210, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = alloca %"class.OSL::Dual2.0", align 4
  %5 = bitcast %"class.OSL::Dual2.0"* %4 to i8*
  call void @llvm.lifetime.start(i64 36, i8* %5) #2
  %6 = bitcast i8* %1 to i32*
  %7 = load i32, i32* %6, align 4, !tbaa !5
  %8 = bitcast %"class.OSL::Dual2.0"* %4 to i32*
  store i32 %7, i32* %8, align 4, !tbaa !5
  %9 = getelementptr inbounds i8, i8* %1, i64 4
  %10 = bitcast i8* %9 to i32*
  %11 = load i32, i32* %10, align 4, !tbaa !7
  %12 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 1
  %13 = bitcast float* %12 to i32*
  store i32 %11, i32* %13, align 4, !tbaa !7
  %14 = getelementptr inbounds i8, i8* %1, i64 8
  %15 = bitcast i8* %14 to i32*
  %16 = load i32, i32* %15, align 4, !tbaa !8
  %17 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 2
  %18 = bitcast float* %17 to i32*
  store i32 %16, i32* %18, align 4, !tbaa !8
  %19 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 1, i32 0
  %20 = bitcast float* %19 to i8*
  call void @llvm.memset.p0i8.i64(i8* %20, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_pow_dvdvdv(i8* %0, i8* %5, i8* %2)
  call void @llvm.lifetime.end(i64 36, i8* %5) #2
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvdvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = alloca %"class.OSL::Dual2.0", align 4
  %5 = bitcast %"class.OSL::Dual2.0"* %4 to i8*
  call void @llvm.lifetime.start(i64 36, i8* %5) #2
  %6 = bitcast i8* %2 to i32*
  %7 = load i32, i32* %6, align 4, !tbaa !5
  %8 = bitcast %"class.OSL::Dual2.0"* %4 to i32*
  store i32 %7, i32* %8, align 4, !tbaa !5
  %9 = getelementptr inbounds i8, i8* %2, i64 4
  %10 = bitcast i8* %9 to i32*
  %11 = load i32, i32* %10, align 4, !tbaa !7
  %12 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 1
  %13 = bitcast float* %12 to i32*
  store i32 %11, i32* %13, align 4, !tbaa !7
  %14 = getelementptr inbounds i8, i8* %2, i64 8
  %15 = bitcast i8* %14 to i32*
  %16 = load i32, i32* %15, align 4, !tbaa !8
  %17 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 2
  %18 = bitcast float* %17 to i32*
  store i32 %16, i32* %18, align 4, !tbaa !8
  %19 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 1, i32 0
  %20 = bitcast float* %19 to i8*
  call void @llvm.memset.p0i8.i64(i8* %20, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_pow_dvdvdv(i8* %0, i8* %1, i8* %5)
  call void @llvm.lifetime.end(i64 36, i8* %5) #2
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_vvf(i8* nocapture, i8* nocapture readonly, float) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %5, float %2)
  %7 = bitcast i8* %0 to float*
  store float %6, float* %7, align 4, !tbaa !1
  %8 = getelementptr inbounds i8, i8* %1, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %10, float %2)
  %12 = getelementptr inbounds i8, i8* %0, i64 4
  %13 = bitcast i8* %12 to float*
  store float %11, float* %13, align 4, !tbaa !1
  %14 = getelementptr inbounds i8, i8* %1, i64 8
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !1
  %17 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %16, float %2)
  %18 = getelementptr inbounds i8, i8* %0, i64 8
  %19 = bitcast i8* %18 to float*
  store float %17, float* %19, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvdvdf(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = getelementptr inbounds i8, i8* %1, i64 12
  %5 = getelementptr inbounds i8, i8* %1, i64 24
  %6 = bitcast i8* %1 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = bitcast i8* %4 to float*
  %9 = load float, float* %8, align 4, !tbaa !1
  %10 = bitcast i8* %5 to float*
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = bitcast i8* %2 to float*
  %13 = load float, float* %12, align 4, !tbaa !1
  %14 = fadd float %13, -1.000000e+00
  %15 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %7, float %14) #2
  %16 = fmul float %7, %15
  %17 = fcmp ogt float %7, 0.000000e+00
  br i1 %17, label %18, label %50

; <label>:18:                                     ; preds = %3
  %19 = fcmp olt float %7, 0x3810000000000000
  %20 = fcmp ogt float %7, 0x47EFFFFFE0000000
  %21 = select i1 %20, float 0x47EFFFFFE0000000, float %7
  %22 = bitcast float %21 to i32
  %23 = select i1 %19, i32 8388608, i32 %22
  %24 = lshr i32 %23, 23
  %25 = add nsw i32 %24, -127
  %26 = and i32 %23, 8388607
  %27 = or i32 %26, 1065353216
  %28 = bitcast i32 %27 to float
  %29 = fadd float %28, -1.000000e+00
  %30 = fmul float %29, %29
  %31 = fmul float %30, %30
  %32 = fmul float %29, 0x3F831161A0000000
  %33 = fsub float 0x3FAAA83920000000, %32
  %34 = fmul float %29, 0x3FDEA2C5A0000000
  %35 = fadd float %34, 0xBFE713CA80000000
  %36 = fmul float %29, %33
  %37 = fadd float %36, 0xBFC19A9FA0000000
  %38 = fmul float %29, %37
  %39 = fadd float %38, 0x3FCEF5B7A0000000
  %40 = fmul float %29, %39
  %41 = fadd float %40, 0xBFD63A40C0000000
  %42 = fmul float %29, %35
  %43 = fadd float %42, 0x3FF7154200000000
  %44 = fmul float %31, %41
  %45 = fmul float %29, %43
  %46 = fadd float %45, %44
  %47 = sitofp i32 %25 to float
  %48 = fadd float %47, %46
  %49 = fmul float %48, 0x3FE62E4300000000
  br label %50

; <label>:50:                                     ; preds = %3, %18
  %51 = phi float [ %49, %18 ], [ 0.000000e+00, %3 ]
  %52 = load float, float* %12, align 4, !tbaa !1
  %53 = fmul float %15, %52
  %54 = fmul float %9, %53
  %55 = fmul float %16, %51
  %56 = getelementptr inbounds i8, i8* %2, i64 4
  %57 = bitcast i8* %56 to float*
  %58 = load float, float* %57, align 4, !tbaa !1
  %59 = fmul float %55, %58
  %60 = fadd float %54, %59
  %61 = fmul float %11, %53
  %62 = getelementptr inbounds i8, i8* %2, i64 8
  %63 = bitcast i8* %62 to float*
  %64 = load float, float* %63, align 4, !tbaa !1
  %65 = fmul float %55, %64
  %66 = fadd float %61, %65
  %67 = getelementptr inbounds i8, i8* %1, i64 4
  %68 = getelementptr inbounds i8, i8* %1, i64 16
  %69 = getelementptr inbounds i8, i8* %1, i64 28
  %70 = bitcast i8* %67 to float*
  %71 = load float, float* %70, align 4, !tbaa !1
  %72 = bitcast i8* %68 to float*
  %73 = load float, float* %72, align 4, !tbaa !1
  %74 = bitcast i8* %69 to float*
  %75 = load float, float* %74, align 4, !tbaa !1
  %76 = fadd float %52, -1.000000e+00
  %77 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %71, float %76) #2
  %78 = fmul float %71, %77
  %79 = fcmp ogt float %71, 0.000000e+00
  br i1 %79, label %80, label %112

; <label>:80:                                     ; preds = %50
  %81 = fcmp olt float %71, 0x3810000000000000
  %82 = fcmp ogt float %71, 0x47EFFFFFE0000000
  %83 = select i1 %82, float 0x47EFFFFFE0000000, float %71
  %84 = bitcast float %83 to i32
  %85 = select i1 %81, i32 8388608, i32 %84
  %86 = lshr i32 %85, 23
  %87 = add nsw i32 %86, -127
  %88 = and i32 %85, 8388607
  %89 = or i32 %88, 1065353216
  %90 = bitcast i32 %89 to float
  %91 = fadd float %90, -1.000000e+00
  %92 = fmul float %91, %91
  %93 = fmul float %92, %92
  %94 = fmul float %91, 0x3F831161A0000000
  %95 = fsub float 0x3FAAA83920000000, %94
  %96 = fmul float %91, 0x3FDEA2C5A0000000
  %97 = fadd float %96, 0xBFE713CA80000000
  %98 = fmul float %91, %95
  %99 = fadd float %98, 0xBFC19A9FA0000000
  %100 = fmul float %91, %99
  %101 = fadd float %100, 0x3FCEF5B7A0000000
  %102 = fmul float %91, %101
  %103 = fadd float %102, 0xBFD63A40C0000000
  %104 = fmul float %91, %97
  %105 = fadd float %104, 0x3FF7154200000000
  %106 = fmul float %93, %103
  %107 = fmul float %91, %105
  %108 = fadd float %107, %106
  %109 = sitofp i32 %87 to float
  %110 = fadd float %109, %108
  %111 = fmul float %110, 0x3FE62E4300000000
  br label %112

; <label>:112:                                    ; preds = %50, %80
  %113 = phi float [ %111, %80 ], [ 0.000000e+00, %50 ]
  %114 = load float, float* %12, align 4, !tbaa !1
  %115 = fmul float %77, %114
  %116 = fmul float %73, %115
  %117 = fmul float %78, %113
  %118 = load float, float* %57, align 4, !tbaa !1
  %119 = fmul float %117, %118
  %120 = fadd float %116, %119
  %121 = fmul float %75, %115
  %122 = load float, float* %63, align 4, !tbaa !1
  %123 = fmul float %117, %122
  %124 = fadd float %121, %123
  %125 = getelementptr inbounds i8, i8* %1, i64 8
  %126 = getelementptr inbounds i8, i8* %1, i64 20
  %127 = getelementptr inbounds i8, i8* %1, i64 32
  %128 = bitcast i8* %125 to float*
  %129 = load float, float* %128, align 4, !tbaa !1
  %130 = bitcast i8* %126 to float*
  %131 = load float, float* %130, align 4, !tbaa !1
  %132 = bitcast i8* %127 to float*
  %133 = load float, float* %132, align 4, !tbaa !1
  %134 = fadd float %114, -1.000000e+00
  %135 = tail call float @_ZN11OpenImageIO4v1_713fast_safe_powEff(float %129, float %134) #2
  %136 = fmul float %129, %135
  %137 = fcmp ogt float %129, 0.000000e+00
  br i1 %137, label %138, label %170

; <label>:138:                                    ; preds = %112
  %139 = fcmp olt float %129, 0x3810000000000000
  %140 = fcmp ogt float %129, 0x47EFFFFFE0000000
  %141 = select i1 %140, float 0x47EFFFFFE0000000, float %129
  %142 = bitcast float %141 to i32
  %143 = select i1 %139, i32 8388608, i32 %142
  %144 = lshr i32 %143, 23
  %145 = add nsw i32 %144, -127
  %146 = and i32 %143, 8388607
  %147 = or i32 %146, 1065353216
  %148 = bitcast i32 %147 to float
  %149 = fadd float %148, -1.000000e+00
  %150 = fmul float %149, %149
  %151 = fmul float %150, %150
  %152 = fmul float %149, 0x3F831161A0000000
  %153 = fsub float 0x3FAAA83920000000, %152
  %154 = fmul float %149, 0x3FDEA2C5A0000000
  %155 = fadd float %154, 0xBFE713CA80000000
  %156 = fmul float %149, %153
  %157 = fadd float %156, 0xBFC19A9FA0000000
  %158 = fmul float %149, %157
  %159 = fadd float %158, 0x3FCEF5B7A0000000
  %160 = fmul float %149, %159
  %161 = fadd float %160, 0xBFD63A40C0000000
  %162 = fmul float %149, %155
  %163 = fadd float %162, 0x3FF7154200000000
  %164 = fmul float %151, %161
  %165 = fmul float %149, %163
  %166 = fadd float %165, %164
  %167 = sitofp i32 %145 to float
  %168 = fadd float %167, %166
  %169 = fmul float %168, 0x3FE62E4300000000
  br label %170

; <label>:170:                                    ; preds = %112, %138
  %171 = phi float [ %169, %138 ], [ 0.000000e+00, %112 ]
  %172 = load float, float* %12, align 4, !tbaa !1
  %173 = fmul float %135, %172
  %174 = fmul float %131, %173
  %175 = fmul float %136, %171
  %176 = load float, float* %57, align 4, !tbaa !1
  %177 = fmul float %175, %176
  %178 = fadd float %174, %177
  %179 = fmul float %133, %173
  %180 = load float, float* %63, align 4, !tbaa !1
  %181 = fmul float %175, %180
  %182 = fadd float %179, %181
  %183 = bitcast i8* %0 to float*
  store float %16, float* %183, align 4, !tbaa !5
  %184 = getelementptr inbounds i8, i8* %0, i64 4
  %185 = bitcast i8* %184 to float*
  store float %78, float* %185, align 4, !tbaa !7
  %186 = getelementptr inbounds i8, i8* %0, i64 8
  %187 = bitcast i8* %186 to float*
  store float %136, float* %187, align 4, !tbaa !8
  %188 = getelementptr inbounds i8, i8* %0, i64 12
  %189 = bitcast i8* %188 to float*
  store float %60, float* %189, align 4, !tbaa !5
  %190 = getelementptr inbounds i8, i8* %0, i64 16
  %191 = bitcast i8* %190 to float*
  store float %120, float* %191, align 4, !tbaa !7
  %192 = getelementptr inbounds i8, i8* %0, i64 20
  %193 = bitcast i8* %192 to float*
  store float %178, float* %193, align 4, !tbaa !8
  %194 = getelementptr inbounds i8, i8* %0, i64 24
  %195 = bitcast i8* %194 to float*
  store float %66, float* %195, align 4, !tbaa !5
  %196 = getelementptr inbounds i8, i8* %0, i64 28
  %197 = bitcast i8* %196 to float*
  store float %124, float* %197, align 4, !tbaa !7
  %198 = getelementptr inbounds i8, i8* %0, i64 32
  %199 = bitcast i8* %198 to float*
  store float %182, float* %199, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvvdf(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = alloca %"class.OSL::Dual2.0", align 4
  %5 = bitcast %"class.OSL::Dual2.0"* %4 to i8*
  call void @llvm.lifetime.start(i64 36, i8* %5) #2
  %6 = bitcast i8* %1 to i32*
  %7 = load i32, i32* %6, align 4, !tbaa !5
  %8 = bitcast %"class.OSL::Dual2.0"* %4 to i32*
  store i32 %7, i32* %8, align 4, !tbaa !5
  %9 = getelementptr inbounds i8, i8* %1, i64 4
  %10 = bitcast i8* %9 to i32*
  %11 = load i32, i32* %10, align 4, !tbaa !7
  %12 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 1
  %13 = bitcast float* %12 to i32*
  store i32 %11, i32* %13, align 4, !tbaa !7
  %14 = getelementptr inbounds i8, i8* %1, i64 8
  %15 = bitcast i8* %14 to i32*
  %16 = load i32, i32* %15, align 4, !tbaa !8
  %17 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 0, i32 2
  %18 = bitcast float* %17 to i32*
  store i32 %16, i32* %18, align 4, !tbaa !8
  %19 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %4, i64 0, i32 1, i32 0
  %20 = bitcast float* %19 to i8*
  call void @llvm.memset.p0i8.i64(i8* %20, i8 0, i64 24, i32 4, i1 false) #2
  call void @osl_pow_dvdvdf(i8* %0, i8* %5, i8* %2)
  call void @llvm.lifetime.end(i64 36, i8* %5) #2
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_pow_dvdvf(i8* nocapture, i8* nocapture readonly, float) local_unnamed_addr #4 {
  %4 = alloca %"class.OSL::Dual2", align 4
  %5 = bitcast %"class.OSL::Dual2"* %4 to i8*
  call void @llvm.lifetime.start(i64 12, i8* %5) #2
  %6 = getelementptr inbounds %"class.OSL::Dual2", %"class.OSL::Dual2"* %4, i64 0, i32 0
  store float %2, float* %6, align 4, !tbaa !9
  %7 = getelementptr inbounds %"class.OSL::Dual2", %"class.OSL::Dual2"* %4, i64 0, i32 1
  store float 0.000000e+00, float* %7, align 4, !tbaa !11
  %8 = getelementptr inbounds %"class.OSL::Dual2", %"class.OSL::Dual2"* %4, i64 0, i32 2
  store float 0.000000e+00, float* %8, align 4, !tbaa !12
  call void @osl_pow_dvdvdf(i8* %0, i8* %1, i8* %5)
  call void @llvm.lifetime.end(i64 12, i8* %5) #2
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_erf_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fsub float 1.000000e+00, %2
  %4 = fsub float 1.000000e+00, %3
  %5 = fmul float %4, 0x3F0693ECE0000000
  %6 = fadd float %5, 0x3F32200720000000
  %7 = fmul float %4, %6
  %8 = fadd float %7, 0x3F23ECC0E0000000
  %9 = fmul float %4, %8
  %10 = fadd float %9, 0x3F82FC6D20000000
  %11 = fmul float %4, %10
  %12 = fadd float %11, 0x3FA5A5FCE0000000
  %13 = fmul float %4, %12
  %14 = fadd float %13, 0x3FB20DCCE0000000
  %15 = fmul float %4, %14
  %16 = fadd float %15, 1.000000e+00
  %17 = fmul float %16, %16
  %18 = fmul float %17, %17
  %19 = fmul float %18, %18
  %20 = fmul float %19, %19
  %21 = fdiv float 1.000000e+00, %20
  %22 = fsub float 1.000000e+00, %21
  %23 = tail call float @copysignf(float %22, float %0) #13
  ret float %23
}

; Function Attrs: nounwind uwtable
define void @osl_erf_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @erff(float %4) #13
  %6 = fmul float %4, %4
  %7 = fsub float -0.000000e+00, %6
  %8 = tail call float @expf(float %7) #13
  %9 = fmul float %8, 0x3FF20DD760000000
  %10 = getelementptr inbounds i8, i8* %1, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = fmul float %9, %12
  %14 = getelementptr inbounds i8, i8* %1, i64 8
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !1
  %17 = fmul float %9, %16
  %18 = insertelement <2 x float> undef, float %5, i32 0
  %19 = insertelement <2 x float> %18, float %13, i32 1
  %20 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %19, <2 x float>* %20, align 4
  %21 = getelementptr inbounds i8, i8* %0, i64 8
  %22 = bitcast i8* %21 to float*
  store float %17, float* %22, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_erf_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fsub float 1.000000e+00, %5
  %7 = fsub float 1.000000e+00, %6
  %8 = fmul float %7, 0x3F0693ECE0000000
  %9 = fadd float %8, 0x3F32200720000000
  %10 = fmul float %7, %9
  %11 = fadd float %10, 0x3F23ECC0E0000000
  %12 = fmul float %7, %11
  %13 = fadd float %12, 0x3F82FC6D20000000
  %14 = fmul float %7, %13
  %15 = fadd float %14, 0x3FA5A5FCE0000000
  %16 = fmul float %7, %15
  %17 = fadd float %16, 0x3FB20DCCE0000000
  %18 = fmul float %7, %17
  %19 = fadd float %18, 1.000000e+00
  %20 = fmul float %19, %19
  %21 = fmul float %20, %20
  %22 = fmul float %21, %21
  %23 = fmul float %22, %22
  %24 = fdiv float 1.000000e+00, %23
  %25 = fsub float 1.000000e+00, %24
  %26 = tail call float @copysignf(float %25, float %4) #13
  %27 = bitcast i8* %0 to float*
  store float %26, float* %27, align 4, !tbaa !1
  %28 = getelementptr inbounds i8, i8* %1, i64 4
  %29 = bitcast i8* %28 to float*
  %30 = load float, float* %29, align 4, !tbaa !1
  %31 = tail call float @fabsf(float %30) #13
  %32 = fsub float 1.000000e+00, %31
  %33 = fsub float 1.000000e+00, %32
  %34 = fmul float %33, 0x3F0693ECE0000000
  %35 = fadd float %34, 0x3F32200720000000
  %36 = fmul float %33, %35
  %37 = fadd float %36, 0x3F23ECC0E0000000
  %38 = fmul float %33, %37
  %39 = fadd float %38, 0x3F82FC6D20000000
  %40 = fmul float %33, %39
  %41 = fadd float %40, 0x3FA5A5FCE0000000
  %42 = fmul float %33, %41
  %43 = fadd float %42, 0x3FB20DCCE0000000
  %44 = fmul float %33, %43
  %45 = fadd float %44, 1.000000e+00
  %46 = fmul float %45, %45
  %47 = fmul float %46, %46
  %48 = fmul float %47, %47
  %49 = fmul float %48, %48
  %50 = fdiv float 1.000000e+00, %49
  %51 = fsub float 1.000000e+00, %50
  %52 = tail call float @copysignf(float %51, float %30) #13
  %53 = getelementptr inbounds i8, i8* %0, i64 4
  %54 = bitcast i8* %53 to float*
  store float %52, float* %54, align 4, !tbaa !1
  %55 = getelementptr inbounds i8, i8* %1, i64 8
  %56 = bitcast i8* %55 to float*
  %57 = load float, float* %56, align 4, !tbaa !1
  %58 = tail call float @fabsf(float %57) #13
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
  %79 = tail call float @copysignf(float %78, float %57) #13
  %80 = getelementptr inbounds i8, i8* %0, i64 8
  %81 = bitcast i8* %80 to float*
  store float %79, float* %81, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_erf_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = tail call float @erff(float %5) #13
  %7 = fmul float %5, %5
  %8 = fsub float -0.000000e+00, %7
  %9 = tail call float @expf(float %8) #13
  %10 = fmul float %9, 0x3FF20DD760000000
  %11 = getelementptr inbounds i8, i8* %1, i64 4
  %12 = getelementptr inbounds i8, i8* %1, i64 28
  %13 = bitcast i8* %11 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = bitcast i8* %12 to float*
  %16 = load float, float* %15, align 4, !tbaa !1
  %17 = tail call float @erff(float %14) #13
  %18 = fmul float %14, %14
  %19 = fsub float -0.000000e+00, %18
  %20 = tail call float @expf(float %19) #13
  %21 = fmul float %20, 0x3FF20DD760000000
  %22 = fmul float %16, %21
  %23 = getelementptr inbounds i8, i8* %1, i64 8
  %24 = getelementptr inbounds i8, i8* %1, i64 32
  %25 = bitcast i8* %23 to float*
  %26 = load float, float* %25, align 4, !tbaa !1
  %27 = bitcast i8* %3 to <4 x float>*
  %28 = load <4 x float>, <4 x float>* %27, align 4, !tbaa !1
  %29 = bitcast i8* %24 to float*
  %30 = load float, float* %29, align 4, !tbaa !1
  %31 = tail call float @erff(float %26) #13
  %32 = fmul float %26, %26
  %33 = fsub float -0.000000e+00, %32
  %34 = tail call float @expf(float %33) #13
  %35 = fmul float %34, 0x3FF20DD760000000
  %36 = insertelement <4 x float> undef, float %10, i32 0
  %37 = insertelement <4 x float> %36, float %21, i32 1
  %38 = insertelement <4 x float> %37, float %35, i32 2
  %39 = insertelement <4 x float> %38, float %10, i32 3
  %40 = fmul <4 x float> %28, %39
  %41 = fmul float %30, %35
  %42 = bitcast i8* %0 to float*
  store float %6, float* %42, align 4, !tbaa !5
  %43 = getelementptr inbounds i8, i8* %0, i64 4
  %44 = bitcast i8* %43 to float*
  store float %17, float* %44, align 4, !tbaa !7
  %45 = getelementptr inbounds i8, i8* %0, i64 8
  %46 = bitcast i8* %45 to float*
  store float %31, float* %46, align 4, !tbaa !8
  %47 = getelementptr inbounds i8, i8* %0, i64 12
  %48 = bitcast i8* %47 to <4 x float>*
  store <4 x float> %40, <4 x float>* %48, align 4, !tbaa !1
  %49 = getelementptr inbounds i8, i8* %0, i64 28
  %50 = bitcast i8* %49 to float*
  store float %22, float* %50, align 4, !tbaa !7
  %51 = getelementptr inbounds i8, i8* %0, i64 32
  %52 = bitcast i8* %51 to float*
  store float %41, float* %52, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_erfc_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fsub float 1.000000e+00, %2
  %4 = fsub float 1.000000e+00, %3
  %5 = fmul float %4, 0x3F0693ECE0000000
  %6 = fadd float %5, 0x3F32200720000000
  %7 = fmul float %4, %6
  %8 = fadd float %7, 0x3F23ECC0E0000000
  %9 = fmul float %4, %8
  %10 = fadd float %9, 0x3F82FC6D20000000
  %11 = fmul float %4, %10
  %12 = fadd float %11, 0x3FA5A5FCE0000000
  %13 = fmul float %4, %12
  %14 = fadd float %13, 0x3FB20DCCE0000000
  %15 = fmul float %4, %14
  %16 = fadd float %15, 1.000000e+00
  %17 = fmul float %16, %16
  %18 = fmul float %17, %17
  %19 = fmul float %18, %18
  %20 = fmul float %19, %19
  %21 = fdiv float 1.000000e+00, %20
  %22 = fsub float 1.000000e+00, %21
  %23 = tail call float @copysignf(float %22, float %0) #13
  %24 = fsub float 1.000000e+00, %23
  ret float %24
}

; Function Attrs: nounwind uwtable
define void @osl_erfc_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @erfcf(float %4) #13
  %6 = fmul float %4, %4
  %7 = fsub float -0.000000e+00, %6
  %8 = tail call float @expf(float %7) #13
  %9 = fmul float %8, 0xBFF20DD760000000
  %10 = getelementptr inbounds i8, i8* %1, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = fmul float %9, %12
  %14 = getelementptr inbounds i8, i8* %1, i64 8
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !1
  %17 = fmul float %9, %16
  %18 = insertelement <2 x float> undef, float %5, i32 0
  %19 = insertelement <2 x float> %18, float %13, i32 1
  %20 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %19, <2 x float>* %20, align 4
  %21 = getelementptr inbounds i8, i8* %0, i64 8
  %22 = bitcast i8* %21 to float*
  store float %17, float* %22, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_erfc_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fsub float 1.000000e+00, %5
  %7 = fsub float 1.000000e+00, %6
  %8 = fmul float %7, 0x3F0693ECE0000000
  %9 = fadd float %8, 0x3F32200720000000
  %10 = fmul float %7, %9
  %11 = fadd float %10, 0x3F23ECC0E0000000
  %12 = fmul float %7, %11
  %13 = fadd float %12, 0x3F82FC6D20000000
  %14 = fmul float %7, %13
  %15 = fadd float %14, 0x3FA5A5FCE0000000
  %16 = fmul float %7, %15
  %17 = fadd float %16, 0x3FB20DCCE0000000
  %18 = fmul float %7, %17
  %19 = fadd float %18, 1.000000e+00
  %20 = fmul float %19, %19
  %21 = fmul float %20, %20
  %22 = fmul float %21, %21
  %23 = fmul float %22, %22
  %24 = fdiv float 1.000000e+00, %23
  %25 = fsub float 1.000000e+00, %24
  %26 = tail call float @copysignf(float %25, float %4) #13
  %27 = fsub float 1.000000e+00, %26
  %28 = bitcast i8* %0 to float*
  store float %27, float* %28, align 4, !tbaa !1
  %29 = getelementptr inbounds i8, i8* %1, i64 4
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !1
  %32 = tail call float @fabsf(float %31) #13
  %33 = fsub float 1.000000e+00, %32
  %34 = fsub float 1.000000e+00, %33
  %35 = fmul float %34, 0x3F0693ECE0000000
  %36 = fadd float %35, 0x3F32200720000000
  %37 = fmul float %34, %36
  %38 = fadd float %37, 0x3F23ECC0E0000000
  %39 = fmul float %34, %38
  %40 = fadd float %39, 0x3F82FC6D20000000
  %41 = fmul float %34, %40
  %42 = fadd float %41, 0x3FA5A5FCE0000000
  %43 = fmul float %34, %42
  %44 = fadd float %43, 0x3FB20DCCE0000000
  %45 = fmul float %34, %44
  %46 = fadd float %45, 1.000000e+00
  %47 = fmul float %46, %46
  %48 = fmul float %47, %47
  %49 = fmul float %48, %48
  %50 = fmul float %49, %49
  %51 = fdiv float 1.000000e+00, %50
  %52 = fsub float 1.000000e+00, %51
  %53 = tail call float @copysignf(float %52, float %31) #13
  %54 = fsub float 1.000000e+00, %53
  %55 = getelementptr inbounds i8, i8* %0, i64 4
  %56 = bitcast i8* %55 to float*
  store float %54, float* %56, align 4, !tbaa !1
  %57 = getelementptr inbounds i8, i8* %1, i64 8
  %58 = bitcast i8* %57 to float*
  %59 = load float, float* %58, align 4, !tbaa !1
  %60 = tail call float @fabsf(float %59) #13
  %61 = fsub float 1.000000e+00, %60
  %62 = fsub float 1.000000e+00, %61
  %63 = fmul float %62, 0x3F0693ECE0000000
  %64 = fadd float %63, 0x3F32200720000000
  %65 = fmul float %62, %64
  %66 = fadd float %65, 0x3F23ECC0E0000000
  %67 = fmul float %62, %66
  %68 = fadd float %67, 0x3F82FC6D20000000
  %69 = fmul float %62, %68
  %70 = fadd float %69, 0x3FA5A5FCE0000000
  %71 = fmul float %62, %70
  %72 = fadd float %71, 0x3FB20DCCE0000000
  %73 = fmul float %62, %72
  %74 = fadd float %73, 1.000000e+00
  %75 = fmul float %74, %74
  %76 = fmul float %75, %75
  %77 = fmul float %76, %76
  %78 = fmul float %77, %77
  %79 = fdiv float 1.000000e+00, %78
  %80 = fsub float 1.000000e+00, %79
  %81 = tail call float @copysignf(float %80, float %59) #13
  %82 = fsub float 1.000000e+00, %81
  %83 = getelementptr inbounds i8, i8* %0, i64 8
  %84 = bitcast i8* %83 to float*
  store float %82, float* %84, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_erfc_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = tail call float @erfcf(float %5) #13
  %7 = fmul float %5, %5
  %8 = fsub float -0.000000e+00, %7
  %9 = tail call float @expf(float %8) #13
  %10 = fmul float %9, 0xBFF20DD760000000
  %11 = getelementptr inbounds i8, i8* %1, i64 4
  %12 = getelementptr inbounds i8, i8* %1, i64 28
  %13 = bitcast i8* %11 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = bitcast i8* %12 to float*
  %16 = load float, float* %15, align 4, !tbaa !1
  %17 = tail call float @erfcf(float %14) #13
  %18 = fmul float %14, %14
  %19 = fsub float -0.000000e+00, %18
  %20 = tail call float @expf(float %19) #13
  %21 = fmul float %20, 0xBFF20DD760000000
  %22 = fmul float %16, %21
  %23 = getelementptr inbounds i8, i8* %1, i64 8
  %24 = getelementptr inbounds i8, i8* %1, i64 32
  %25 = bitcast i8* %23 to float*
  %26 = load float, float* %25, align 4, !tbaa !1
  %27 = bitcast i8* %3 to <4 x float>*
  %28 = load <4 x float>, <4 x float>* %27, align 4, !tbaa !1
  %29 = bitcast i8* %24 to float*
  %30 = load float, float* %29, align 4, !tbaa !1
  %31 = tail call float @erfcf(float %26) #13
  %32 = fmul float %26, %26
  %33 = fsub float -0.000000e+00, %32
  %34 = tail call float @expf(float %33) #13
  %35 = fmul float %34, 0xBFF20DD760000000
  %36 = insertelement <4 x float> undef, float %10, i32 0
  %37 = insertelement <4 x float> %36, float %21, i32 1
  %38 = insertelement <4 x float> %37, float %35, i32 2
  %39 = insertelement <4 x float> %38, float %10, i32 3
  %40 = fmul <4 x float> %28, %39
  %41 = fmul float %30, %35
  %42 = bitcast i8* %0 to float*
  store float %6, float* %42, align 4, !tbaa !5
  %43 = getelementptr inbounds i8, i8* %0, i64 4
  %44 = bitcast i8* %43 to float*
  store float %17, float* %44, align 4, !tbaa !7
  %45 = getelementptr inbounds i8, i8* %0, i64 8
  %46 = bitcast i8* %45 to float*
  store float %31, float* %46, align 4, !tbaa !8
  %47 = getelementptr inbounds i8, i8* %0, i64 12
  %48 = bitcast i8* %47 to <4 x float>*
  store <4 x float> %40, <4 x float>* %48, align 4, !tbaa !1
  %49 = getelementptr inbounds i8, i8* %0, i64 28
  %50 = bitcast i8* %49 to float*
  store float %22, float* %50, align 4, !tbaa !7
  %51 = getelementptr inbounds i8, i8* %0, i64 32
  %52 = bitcast i8* %51 to float*
  store float %41, float* %52, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_sqrt_ff(float) local_unnamed_addr #3 {
  %2 = fcmp ult float %0, 0.000000e+00
  br i1 %2, label %5, label %3

; <label>:3:                                      ; preds = %1
  %4 = tail call float @sqrtf(float %0) #13
  br label %5

; <label>:5:                                      ; preds = %1, %3
  %6 = phi float [ %4, %3 ], [ 0.000000e+00, %1 ]
  ret float %6
}


declare void @llvm.masked.store.v16f32.p0v16f32 (<16 x float>, <16 x float>*, i32,  <16 x i1>)
declare <16 x float> @llvm.sqrt.v16f32(<16 x float>)

; Function Attrs: nounwind readnone uwtable
define <16 x float> @osl_sqrt_w16fw16f(<16 x float>) alwaysinline #3 {
  %2 = fcmp ult <16 x float> %0, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>
  %3 = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %0)
  %"$result" = alloca <16 x float>
  store <16 x float> %3, <16 x float>* %"$result"     
  call void @llvm.masked.store.v16f32.p0v16f32 (<16 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, <16 x float>* %"$result", i32 64,  <16 x i1> %2 )
  %4 = load <16 x float>, <16 x float>* %"$result"
  ret <16 x float> %4
}

; Function Attrs: nounwind uwtable
define void @osl_sqrt_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp ugt float %4, 0.000000e+00
  br i1 %5, label %6, label %20

; <label>:6:                                      ; preds = %2
  %7 = tail call float @sqrtf(float %4) #13
  %8 = fmul float %7, 2.000000e+00
  %9 = fdiv float 1.000000e+00, %8
  %10 = getelementptr inbounds i8, i8* %1, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = fmul float %12, %9
  %14 = getelementptr inbounds i8, i8* %1, i64 8
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !1
  %17 = fmul float %9, %16
  %18 = insertelement <2 x float> undef, float %7, i32 0
  %19 = insertelement <2 x float> %18, float %13, i32 1
  br label %20

; <label>:20:                                     ; preds = %2, %6
  %21 = phi <2 x float> [ %19, %6 ], [ zeroinitializer, %2 ]
  %22 = phi float [ %17, %6 ], [ 0.000000e+00, %2 ]
  %23 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %21, <2 x float>* %23, align 4
  %24 = getelementptr inbounds i8, i8* %0, i64 8
  %25 = bitcast i8* %24 to float*
  store float %22, float* %25, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sqrt_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp ult float %4, 0.000000e+00
  br i1 %5, label %8, label %6

; <label>:6:                                      ; preds = %2
  %7 = tail call float @sqrtf(float %4) #13
  br label %8

; <label>:8:                                      ; preds = %2, %6
  %9 = phi float [ %7, %6 ], [ 0.000000e+00, %2 ]
  %10 = bitcast i8* %0 to float*
  store float %9, float* %10, align 4, !tbaa !1
  %11 = getelementptr inbounds i8, i8* %1, i64 4
  %12 = bitcast i8* %11 to float*
  %13 = load float, float* %12, align 4, !tbaa !1
  %14 = fcmp ult float %13, 0.000000e+00
  br i1 %14, label %17, label %15

; <label>:15:                                     ; preds = %8
  %16 = tail call float @sqrtf(float %13) #13
  br label %17

; <label>:17:                                     ; preds = %8, %15
  %18 = phi float [ %16, %15 ], [ 0.000000e+00, %8 ]
  %19 = getelementptr inbounds i8, i8* %0, i64 4
  %20 = bitcast i8* %19 to float*
  store float %18, float* %20, align 4, !tbaa !1
  %21 = getelementptr inbounds i8, i8* %1, i64 8
  %22 = bitcast i8* %21 to float*
  %23 = load float, float* %22, align 4, !tbaa !1
  %24 = fcmp ult float %23, 0.000000e+00
  br i1 %24, label %27, label %25

; <label>:25:                                     ; preds = %17
  %26 = tail call float @sqrtf(float %23) #13
  br label %27

; <label>:27:                                     ; preds = %17, %25
  %28 = phi float [ %26, %25 ], [ 0.000000e+00, %17 ]
  %29 = getelementptr inbounds i8, i8* %0, i64 8
  %30 = bitcast i8* %29 to float*
  store float %28, float* %30, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_sqrt_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp ugt float %4, 0.000000e+00
  br i1 %5, label %6, label %22

; <label>:6:                                      ; preds = %2
  %7 = getelementptr inbounds i8, i8* %1, i64 24
  %8 = getelementptr inbounds i8, i8* %1, i64 12
  %9 = bitcast i8* %7 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = bitcast i8* %8 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = tail call float @sqrtf(float %4) #13
  %14 = fmul float %13, 2.000000e+00
  %15 = fdiv float 1.000000e+00, %14
  %16 = fmul float %12, %15
  %17 = fmul float %10, %15
  %18 = insertelement <2 x float> undef, float %13, i32 0
  %19 = insertelement <2 x float> %18, float %16, i32 1
  %20 = bitcast <2 x float> %19 to <2 x i32>
  %21 = bitcast float %17 to i32
  br label %22

; <label>:22:                                     ; preds = %2, %6
  %23 = phi <2 x i32> [ %20, %6 ], [ zeroinitializer, %2 ]
  %24 = phi i32 [ %21, %6 ], [ 0, %2 ]
  %25 = getelementptr inbounds i8, i8* %1, i64 4
  %26 = bitcast i8* %25 to float*
  %27 = load float, float* %26, align 4, !tbaa !1
  %28 = fcmp ugt float %27, 0.000000e+00
  br i1 %28, label %29, label %45

; <label>:29:                                     ; preds = %22
  %30 = getelementptr inbounds i8, i8* %1, i64 28
  %31 = bitcast i8* %30 to float*
  %32 = load float, float* %31, align 4, !tbaa !1
  %33 = getelementptr inbounds i8, i8* %1, i64 16
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %34, align 4, !tbaa !1
  %36 = tail call float @sqrtf(float %27) #13
  %37 = fmul float %36, 2.000000e+00
  %38 = fdiv float 1.000000e+00, %37
  %39 = fmul float %35, %38
  %40 = fmul float %32, %38
  %41 = insertelement <2 x float> undef, float %36, i32 0
  %42 = insertelement <2 x float> %41, float %39, i32 1
  %43 = bitcast <2 x float> %42 to <2 x i32>
  %44 = bitcast float %40 to i32
  br label %45

; <label>:45:                                     ; preds = %22, %29
  %46 = phi <2 x i32> [ %43, %29 ], [ zeroinitializer, %22 ]
  %47 = phi i32 [ %44, %29 ], [ 0, %22 ]
  %48 = getelementptr inbounds i8, i8* %1, i64 8
  %49 = bitcast i8* %48 to float*
  %50 = load float, float* %49, align 4, !tbaa !1
  %51 = fcmp ugt float %50, 0.000000e+00
  br i1 %51, label %52, label %68

; <label>:52:                                     ; preds = %45
  %53 = getelementptr inbounds i8, i8* %1, i64 32
  %54 = bitcast i8* %53 to float*
  %55 = load float, float* %54, align 4, !tbaa !1
  %56 = getelementptr inbounds i8, i8* %1, i64 20
  %57 = bitcast i8* %56 to float*
  %58 = load float, float* %57, align 4, !tbaa !1
  %59 = tail call float @sqrtf(float %50) #13
  %60 = fmul float %59, 2.000000e+00
  %61 = fdiv float 1.000000e+00, %60
  %62 = fmul float %58, %61
  %63 = fmul float %55, %61
  %64 = insertelement <2 x float> undef, float %59, i32 0
  %65 = insertelement <2 x float> %64, float %62, i32 1
  %66 = bitcast <2 x float> %65 to <2 x i32>
  %67 = bitcast float %63 to i32
  br label %68

; <label>:68:                                     ; preds = %45, %52
  %69 = phi <2 x i32> [ %66, %52 ], [ zeroinitializer, %45 ]
  %70 = phi i32 [ %67, %52 ], [ 0, %45 ]
  %71 = extractelement <2 x i32> %23, i32 0
  %72 = extractelement <2 x i32> %46, i32 0
  %73 = extractelement <2 x i32> %69, i32 0
  %74 = extractelement <2 x i32> %23, i32 1
  %75 = extractelement <2 x i32> %46, i32 1
  %76 = extractelement <2 x i32> %69, i32 1
  %77 = bitcast i8* %0 to i32*
  store i32 %71, i32* %77, align 4, !tbaa !5
  %78 = getelementptr inbounds i8, i8* %0, i64 4
  %79 = bitcast i8* %78 to i32*
  store i32 %72, i32* %79, align 4, !tbaa !7
  %80 = getelementptr inbounds i8, i8* %0, i64 8
  %81 = bitcast i8* %80 to i32*
  store i32 %73, i32* %81, align 4, !tbaa !8
  %82 = getelementptr inbounds i8, i8* %0, i64 12
  %83 = bitcast i8* %82 to i32*
  store i32 %74, i32* %83, align 4, !tbaa !5
  %84 = getelementptr inbounds i8, i8* %0, i64 16
  %85 = bitcast i8* %84 to i32*
  store i32 %75, i32* %85, align 4, !tbaa !7
  %86 = getelementptr inbounds i8, i8* %0, i64 20
  %87 = bitcast i8* %86 to i32*
  store i32 %76, i32* %87, align 4, !tbaa !8
  %88 = getelementptr inbounds i8, i8* %0, i64 24
  %89 = bitcast i8* %88 to i32*
  store i32 %24, i32* %89, align 4, !tbaa !5
  %90 = getelementptr inbounds i8, i8* %0, i64 28
  %91 = bitcast i8* %90 to i32*
  store i32 %47, i32* %91, align 4, !tbaa !7
  %92 = getelementptr inbounds i8, i8* %0, i64 32
  %93 = bitcast i8* %92 to i32*
  store i32 %70, i32* %93, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_inversesqrt_ff(float) local_unnamed_addr #3 {
  %2 = fcmp ogt float %0, 0.000000e+00
  br i1 %2, label %3, label %6

; <label>:3:                                      ; preds = %1
  %4 = tail call float @sqrtf(float %0) #13
  %5 = fdiv float 1.000000e+00, %4
  br label %6

; <label>:6:                                      ; preds = %1, %3
  %7 = phi float [ %5, %3 ], [ 0.000000e+00, %1 ]
  ret float %7
}

; Function Attrs: nounwind uwtable
define void @osl_inversesqrt_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp ugt float %4, 0.000000e+00
  br i1 %5, label %6, label %22

; <label>:6:                                      ; preds = %2
  %7 = tail call float @sqrtf(float %4) #13
  %8 = fmul float %4, 2.000000e+00
  %9 = fmul float %8, %7
  %10 = fdiv float -1.000000e+00, %9
  %11 = fdiv float 1.000000e+00, %7
  %12 = getelementptr inbounds i8, i8* %1, i64 4
  %13 = bitcast i8* %12 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = fmul float %14, %10
  %16 = getelementptr inbounds i8, i8* %1, i64 8
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4, !tbaa !1
  %19 = fmul float %10, %18
  %20 = insertelement <2 x float> undef, float %11, i32 0
  %21 = insertelement <2 x float> %20, float %15, i32 1
  br label %22

; <label>:22:                                     ; preds = %2, %6
  %23 = phi <2 x float> [ %21, %6 ], [ zeroinitializer, %2 ]
  %24 = phi float [ %19, %6 ], [ 0.000000e+00, %2 ]
  %25 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %23, <2 x float>* %25, align 4
  %26 = getelementptr inbounds i8, i8* %0, i64 8
  %27 = bitcast i8* %26 to float*
  store float %24, float* %27, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_inversesqrt_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp ogt float %4, 0.000000e+00
  br i1 %5, label %6, label %9

; <label>:6:                                      ; preds = %2
  %7 = tail call float @sqrtf(float %4) #13
  %8 = fdiv float 1.000000e+00, %7
  br label %9

; <label>:9:                                      ; preds = %2, %6
  %10 = phi float [ %8, %6 ], [ 0.000000e+00, %2 ]
  %11 = bitcast i8* %0 to float*
  store float %10, float* %11, align 4, !tbaa !1
  %12 = getelementptr inbounds i8, i8* %1, i64 4
  %13 = bitcast i8* %12 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = fcmp ogt float %14, 0.000000e+00
  br i1 %15, label %16, label %19

; <label>:16:                                     ; preds = %9
  %17 = tail call float @sqrtf(float %14) #13
  %18 = fdiv float 1.000000e+00, %17
  br label %19

; <label>:19:                                     ; preds = %9, %16
  %20 = phi float [ %18, %16 ], [ 0.000000e+00, %9 ]
  %21 = getelementptr inbounds i8, i8* %0, i64 4
  %22 = bitcast i8* %21 to float*
  store float %20, float* %22, align 4, !tbaa !1
  %23 = getelementptr inbounds i8, i8* %1, i64 8
  %24 = bitcast i8* %23 to float*
  %25 = load float, float* %24, align 4, !tbaa !1
  %26 = fcmp ogt float %25, 0.000000e+00
  br i1 %26, label %27, label %30

; <label>:27:                                     ; preds = %19
  %28 = tail call float @sqrtf(float %25) #13
  %29 = fdiv float 1.000000e+00, %28
  br label %30

; <label>:30:                                     ; preds = %19, %27
  %31 = phi float [ %29, %27 ], [ 0.000000e+00, %19 ]
  %32 = getelementptr inbounds i8, i8* %0, i64 8
  %33 = bitcast i8* %32 to float*
  store float %31, float* %33, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_inversesqrt_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp ugt float %4, 0.000000e+00
  br i1 %5, label %6, label %24

; <label>:6:                                      ; preds = %2
  %7 = getelementptr inbounds i8, i8* %1, i64 24
  %8 = getelementptr inbounds i8, i8* %1, i64 12
  %9 = bitcast i8* %7 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = bitcast i8* %8 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = tail call float @sqrtf(float %4) #13
  %14 = fmul float %4, 2.000000e+00
  %15 = fmul float %14, %13
  %16 = fdiv float -1.000000e+00, %15
  %17 = fdiv float 1.000000e+00, %13
  %18 = fmul float %12, %16
  %19 = fmul float %10, %16
  %20 = insertelement <2 x float> undef, float %17, i32 0
  %21 = insertelement <2 x float> %20, float %18, i32 1
  %22 = bitcast <2 x float> %21 to <2 x i32>
  %23 = bitcast float %19 to i32
  br label %24

; <label>:24:                                     ; preds = %2, %6
  %25 = phi <2 x i32> [ %22, %6 ], [ zeroinitializer, %2 ]
  %26 = phi i32 [ %23, %6 ], [ 0, %2 ]
  %27 = getelementptr inbounds i8, i8* %1, i64 4
  %28 = bitcast i8* %27 to float*
  %29 = load float, float* %28, align 4, !tbaa !1
  %30 = fcmp ugt float %29, 0.000000e+00
  br i1 %30, label %31, label %49

; <label>:31:                                     ; preds = %24
  %32 = getelementptr inbounds i8, i8* %1, i64 28
  %33 = bitcast i8* %32 to float*
  %34 = load float, float* %33, align 4, !tbaa !1
  %35 = getelementptr inbounds i8, i8* %1, i64 16
  %36 = bitcast i8* %35 to float*
  %37 = load float, float* %36, align 4, !tbaa !1
  %38 = tail call float @sqrtf(float %29) #13
  %39 = fmul float %29, 2.000000e+00
  %40 = fmul float %39, %38
  %41 = fdiv float -1.000000e+00, %40
  %42 = fdiv float 1.000000e+00, %38
  %43 = fmul float %37, %41
  %44 = fmul float %34, %41
  %45 = insertelement <2 x float> undef, float %42, i32 0
  %46 = insertelement <2 x float> %45, float %43, i32 1
  %47 = bitcast <2 x float> %46 to <2 x i32>
  %48 = bitcast float %44 to i32
  br label %49

; <label>:49:                                     ; preds = %24, %31
  %50 = phi <2 x i32> [ %47, %31 ], [ zeroinitializer, %24 ]
  %51 = phi i32 [ %48, %31 ], [ 0, %24 ]
  %52 = getelementptr inbounds i8, i8* %1, i64 8
  %53 = bitcast i8* %52 to float*
  %54 = load float, float* %53, align 4, !tbaa !1
  %55 = fcmp ugt float %54, 0.000000e+00
  br i1 %55, label %56, label %74

; <label>:56:                                     ; preds = %49
  %57 = getelementptr inbounds i8, i8* %1, i64 32
  %58 = bitcast i8* %57 to float*
  %59 = load float, float* %58, align 4, !tbaa !1
  %60 = getelementptr inbounds i8, i8* %1, i64 20
  %61 = bitcast i8* %60 to float*
  %62 = load float, float* %61, align 4, !tbaa !1
  %63 = tail call float @sqrtf(float %54) #13
  %64 = fmul float %54, 2.000000e+00
  %65 = fmul float %64, %63
  %66 = fdiv float -1.000000e+00, %65
  %67 = fdiv float 1.000000e+00, %63
  %68 = fmul float %62, %66
  %69 = fmul float %59, %66
  %70 = insertelement <2 x float> undef, float %67, i32 0
  %71 = insertelement <2 x float> %70, float %68, i32 1
  %72 = bitcast <2 x float> %71 to <2 x i32>
  %73 = bitcast float %69 to i32
  br label %74

; <label>:74:                                     ; preds = %49, %56
  %75 = phi <2 x i32> [ %72, %56 ], [ zeroinitializer, %49 ]
  %76 = phi i32 [ %73, %56 ], [ 0, %49 ]
  %77 = extractelement <2 x i32> %25, i32 0
  %78 = extractelement <2 x i32> %50, i32 0
  %79 = extractelement <2 x i32> %75, i32 0
  %80 = extractelement <2 x i32> %25, i32 1
  %81 = extractelement <2 x i32> %50, i32 1
  %82 = extractelement <2 x i32> %75, i32 1
  %83 = bitcast i8* %0 to i32*
  store i32 %77, i32* %83, align 4, !tbaa !5
  %84 = getelementptr inbounds i8, i8* %0, i64 4
  %85 = bitcast i8* %84 to i32*
  store i32 %78, i32* %85, align 4, !tbaa !7
  %86 = getelementptr inbounds i8, i8* %0, i64 8
  %87 = bitcast i8* %86 to i32*
  store i32 %79, i32* %87, align 4, !tbaa !8
  %88 = getelementptr inbounds i8, i8* %0, i64 12
  %89 = bitcast i8* %88 to i32*
  store i32 %80, i32* %89, align 4, !tbaa !5
  %90 = getelementptr inbounds i8, i8* %0, i64 16
  %91 = bitcast i8* %90 to i32*
  store i32 %81, i32* %91, align 4, !tbaa !7
  %92 = getelementptr inbounds i8, i8* %0, i64 20
  %93 = bitcast i8* %92 to i32*
  store i32 %82, i32* %93, align 4, !tbaa !8
  %94 = getelementptr inbounds i8, i8* %0, i64 24
  %95 = bitcast i8* %94 to i32*
  store i32 %26, i32* %95, align 4, !tbaa !5
  %96 = getelementptr inbounds i8, i8* %0, i64 28
  %97 = bitcast i8* %96 to i32*
  store i32 %51, i32* %97, align 4, !tbaa !7
  %98 = getelementptr inbounds i8, i8* %0, i64 32
  %99 = bitcast i8* %98 to i32*
  store i32 %76, i32* %99, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_logb_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  %3 = fcmp olt float %2, 0x3810000000000000
  br i1 %3, label %7, label %4

; <label>:4:                                      ; preds = %1
  %5 = fcmp ogt float %2, 0x47EFFFFFE0000000
  br i1 %5, label %6, label %7

; <label>:6:                                      ; preds = %4
  br label %7

; <label>:7:                                      ; preds = %1, %4, %6
  %8 = phi float [ 0x47EFFFFFE0000000, %6 ], [ %2, %4 ], [ 0x3810000000000000, %1 ]
  %9 = bitcast float %8 to i32
  %10 = lshr i32 %9, 23
  %11 = add nsw i32 %10, -127
  %12 = sitofp i32 %11 to float
  ret float %12
}

; Function Attrs: nounwind uwtable
define void @osl_logb_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = fcmp olt float %5, 0x3810000000000000
  br i1 %6, label %10, label %7

; <label>:7:                                      ; preds = %2
  %8 = fcmp ogt float %5, 0x47EFFFFFE0000000
  br i1 %8, label %9, label %10

; <label>:9:                                      ; preds = %7
  br label %10

; <label>:10:                                     ; preds = %2, %7, %9
  %11 = phi float [ 0x47EFFFFFE0000000, %9 ], [ %5, %7 ], [ 0x3810000000000000, %2 ]
  %12 = bitcast float %11 to i32
  %13 = lshr i32 %12, 23
  %14 = add nsw i32 %13, -127
  %15 = sitofp i32 %14 to float
  %16 = getelementptr inbounds i8, i8* %1, i64 4
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4, !tbaa !1
  %19 = tail call float @fabsf(float %18) #13
  %20 = fcmp olt float %19, 0x3810000000000000
  br i1 %20, label %24, label %21

; <label>:21:                                     ; preds = %10
  %22 = fcmp ogt float %19, 0x47EFFFFFE0000000
  br i1 %22, label %23, label %24

; <label>:23:                                     ; preds = %21
  br label %24

; <label>:24:                                     ; preds = %10, %21, %23
  %25 = phi float [ 0x47EFFFFFE0000000, %23 ], [ %19, %21 ], [ 0x3810000000000000, %10 ]
  %26 = bitcast float %25 to i32
  %27 = lshr i32 %26, 23
  %28 = add nsw i32 %27, -127
  %29 = sitofp i32 %28 to float
  %30 = getelementptr inbounds i8, i8* %1, i64 8
  %31 = bitcast i8* %30 to float*
  %32 = load float, float* %31, align 4, !tbaa !1
  %33 = tail call float @fabsf(float %32) #13
  %34 = fcmp olt float %33, 0x3810000000000000
  br i1 %34, label %38, label %35

; <label>:35:                                     ; preds = %24
  %36 = fcmp ogt float %33, 0x47EFFFFFE0000000
  br i1 %36, label %37, label %38

; <label>:37:                                     ; preds = %35
  br label %38

; <label>:38:                                     ; preds = %24, %35, %37
  %39 = phi float [ 0x47EFFFFFE0000000, %37 ], [ %33, %35 ], [ 0x3810000000000000, %24 ]
  %40 = bitcast float %39 to i32
  %41 = lshr i32 %40, 23
  %42 = add nsw i32 %41, -127
  %43 = sitofp i32 %42 to float
  %44 = bitcast i8* %0 to float*
  store float %15, float* %44, align 4, !tbaa !5
  %45 = getelementptr inbounds i8, i8* %0, i64 4
  %46 = bitcast i8* %45 to float*
  store float %29, float* %46, align 4, !tbaa !7
  %47 = getelementptr inbounds i8, i8* %0, i64 8
  %48 = bitcast i8* %47 to float*
  store float %43, float* %48, align 4, !tbaa !8
  ret void
}

declare float     @llvm.floor.f32(float  %Val)
declare <16 x float> @llvm.floor.v16f32(<16 x float> %Val)
declare <8 x float> @llvm.floor.v8f32(<8 x float> %Val)
declare <4 x float> @llvm.floor.v4f32(<4 x float> %Val)

; Function Attrs: nounwind readnone uwtable
define float @osl_floor_ff(float) local_unnamed_addr #3 {

  %2 = tail call float @floorf(float %0) #13
  ret float %2
}

; Function Attrs: nounwind readnone uwtable
define <16 x float> @osl_floor_w16fw16f(<16 x float> ) alwaysinline #3 {

  %2 = tail call <16 x float> @llvm.floor.v16f32(<16 x float> %0)
  ret  <16 x float>  %2
}

; Function Attrs: nounwind readnone uwtable
define <8 x float> @osl_floor_w8fw8f(<8 x float> ) alwaysinline #3 {

  %2 = tail call <8 x float> @llvm.floor.v8f32(<8 x float> %0)
  ret  <8 x float>  %2
}

; Function Attrs: nounwind readnone uwtable
define <4 x float> @osl_floor_w4fw4f(<4 x float> ) alwaysinline #3 {

  %2 = tail call <4 x float> @llvm.floor.v4f32(<4 x float> %0)
  ret  <4 x float>  %2
}



; Function Attrs: nounwind readnone
declare float @floorf(float) local_unnamed_addr #9

; Function Attrs: nounwind uwtable
define void @osl_floor_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @floorf(float %4) #13
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = tail call float @floorf(float %8) #13
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = tail call float @floorf(float %12) #13
  %14 = bitcast i8* %0 to float*
  store float %5, float* %14, align 4, !tbaa !5
  %15 = getelementptr inbounds i8, i8* %0, i64 4
  %16 = bitcast i8* %15 to float*
  store float %9, float* %16, align 4, !tbaa !7
  %17 = getelementptr inbounds i8, i8* %0, i64 8
  %18 = bitcast i8* %17 to float*
  store float %13, float* %18, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_ceil_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @ceilf(float %0) #13
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @ceilf(float) local_unnamed_addr #9

; Function Attrs: nounwind uwtable
define void @osl_ceil_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @ceilf(float %4) #13
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = tail call float @ceilf(float %8) #13
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = tail call float @ceilf(float %12) #13
  %14 = bitcast i8* %0 to float*
  store float %5, float* %14, align 4, !tbaa !5
  %15 = getelementptr inbounds i8, i8* %0, i64 4
  %16 = bitcast i8* %15 to float*
  store float %9, float* %16, align 4, !tbaa !7
  %17 = getelementptr inbounds i8, i8* %0, i64 8
  %18 = bitcast i8* %17 to float*
  store float %13, float* %18, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_round_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @roundf(float %0) #13
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @roundf(float) local_unnamed_addr #9

; Function Attrs: nounwind uwtable
define void @osl_round_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @roundf(float %4) #13
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = tail call float @roundf(float %8) #13
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = tail call float @roundf(float %12) #13
  %14 = bitcast i8* %0 to float*
  store float %5, float* %14, align 4, !tbaa !5
  %15 = getelementptr inbounds i8, i8* %0, i64 4
  %16 = bitcast i8* %15 to float*
  store float %9, float* %16, align 4, !tbaa !7
  %17 = getelementptr inbounds i8, i8* %0, i64 8
  %18 = bitcast i8* %17 to float*
  store float %13, float* %18, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_trunc_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @truncf(float %0) #13
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @truncf(float) local_unnamed_addr #9

; Function Attrs: nounwind uwtable
define void @osl_trunc_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @truncf(float %4) #13
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = tail call float @truncf(float %8) #13
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = tail call float @truncf(float %12) #13
  %14 = bitcast i8* %0 to float*
  store float %5, float* %14, align 4, !tbaa !5
  %15 = getelementptr inbounds i8, i8* %0, i64 4
  %16 = bitcast i8* %15 to float*
  store float %9, float* %16, align 4, !tbaa !7
  %17 = getelementptr inbounds i8, i8* %0, i64 8
  %18 = bitcast i8* %17 to float*
  store float %13, float* %18, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_sign_ff(float) local_unnamed_addr #6 {
  %2 = fcmp olt float %0, 0.000000e+00
  %3 = fcmp oeq float %0, 0.000000e+00
  %4 = select i1 %3, float 0.000000e+00, float 1.000000e+00
  %5 = select i1 %2, float -1.000000e+00, float %4
  ret float %5
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_sign_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp olt float %4, 0.000000e+00
  %6 = fcmp oeq float %4, 0.000000e+00
  %7 = select i1 %6, float 0.000000e+00, float 1.000000e+00
  %8 = select i1 %5, float -1.000000e+00, float %7
  %9 = getelementptr inbounds i8, i8* %1, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = fcmp olt float %11, 0.000000e+00
  %13 = fcmp oeq float %11, 0.000000e+00
  %14 = select i1 %13, float 0.000000e+00, float 1.000000e+00
  %15 = select i1 %12, float -1.000000e+00, float %14
  %16 = getelementptr inbounds i8, i8* %1, i64 8
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4, !tbaa !1
  %19 = fcmp olt float %18, 0.000000e+00
  %20 = fcmp oeq float %18, 0.000000e+00
  %21 = select i1 %20, float 0.000000e+00, float 1.000000e+00
  %22 = select i1 %19, float -1.000000e+00, float %21
  %23 = bitcast i8* %0 to float*
  store float %8, float* %23, align 4, !tbaa !5
  %24 = getelementptr inbounds i8, i8* %0, i64 4
  %25 = bitcast i8* %24 to float*
  store float %15, float* %25, align 4, !tbaa !7
  %26 = getelementptr inbounds i8, i8* %0, i64 8
  %27 = bitcast i8* %26 to float*
  store float %22, float* %27, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_step_fff(float, float) local_unnamed_addr #6 {
  %3 = fcmp olt float %1, %0
  %4 = select i1 %3, float 0.000000e+00, float 1.000000e+00
  ret float %4
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_step_vvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #7 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %1 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fcmp olt float %5, %7
  %9 = select i1 %8, float 0.000000e+00, float 1.000000e+00
  %10 = getelementptr inbounds i8, i8* %2, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = getelementptr inbounds i8, i8* %1, i64 4
  %14 = bitcast i8* %13 to float*
  %15 = load float, float* %14, align 4, !tbaa !1
  %16 = fcmp olt float %12, %15
  %17 = select i1 %16, float 0.000000e+00, float 1.000000e+00
  %18 = getelementptr inbounds i8, i8* %2, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = load float, float* %19, align 4, !tbaa !1
  %21 = getelementptr inbounds i8, i8* %1, i64 8
  %22 = bitcast i8* %21 to float*
  %23 = load float, float* %22, align 4, !tbaa !1
  %24 = fcmp olt float %20, %23
  %25 = select i1 %24, float 0.000000e+00, float 1.000000e+00
  %26 = bitcast i8* %0 to float*
  store float %9, float* %26, align 4, !tbaa !5
  %27 = getelementptr inbounds i8, i8* %0, i64 4
  %28 = bitcast i8* %27 to float*
  store float %17, float* %28, align 4, !tbaa !7
  %29 = getelementptr inbounds i8, i8* %0, i64 8
  %30 = bitcast i8* %29 to float*
  store float %25, float* %30, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @osl_isnan_if(float) local_unnamed_addr #6 {
  %2 = fcmp uno float %0, 0.000000e+00
  %3 = zext i1 %2 to i32
  ret i32 %3
}

; Function Attrs: nounwind readnone uwtable
define i32 @osl_isinf_if(float) local_unnamed_addr #3 {
  %2 = tail call float @llvm.fabs.f32(float %0) #13
  %3 = fcmp oeq float %2, 0x7FF0000000000000
  %4 = zext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: nounwind readnone uwtable
define i32 @osl_isfinite_if(float) local_unnamed_addr #3 {
  %2 = tail call float @llvm.fabs.f32(float %0) #13
  %3 = fcmp one float %2, 0x7FF0000000000000
  %4 = zext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @osl_abs_ii(i32) local_unnamed_addr #6 {
  %2 = icmp sgt i32 %0, -1
  %3 = sub i32 0, %0
  %4 = select i1 %2, i32 %0, i32 %3
  ret i32 %4
}

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @osl_fabs_ii(i32) local_unnamed_addr #6 {
  %2 = icmp sgt i32 %0, -1
  %3 = sub i32 0, %0
  %4 = select i1 %2, i32 %0, i32 %3
  ret i32 %4
}

; Function Attrs: nounwind readnone uwtable
define float @osl_abs_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  ret float %2
}


declare <4 x float> @llvm.fabs.v4f32(<4 x float>)

define <4 x float> @osl_abs_w4fw4f(<4 x float> %a) alwaysinline #3 {
  %r = call <4 x float> @llvm.fabs.v4f32(<4 x float> %a)
  ret <4 x float> %r
}

declare <8 x float> @llvm.fabs.v8f32(<8 x float>) 

define <8 x float> @osl_abs_w8fw8f(<8 x float> %a) alwaysinline #3 {
  %r = call <8 x float> @llvm.fabs.v8f32(<8 x float> %a)
  ret <8 x float> %r
}

declare <16 x float> @llvm.fabs.v16f32(<16 x float>)

define <16 x float> @osl_abs_w16fw16f(<16 x float> %a) alwaysinline  #3 {
  %r = call <16 x float> @llvm.fabs.v16f32(<16 x float> %a)
  ret <16 x float> %r
}


; Function Attrs: nounwind readnone
declare float @fabsf(float) local_unnamed_addr #9

; Function Attrs: norecurse nounwind uwtable
define void @osl_abs_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp ult float %4, 0.000000e+00
  %6 = getelementptr inbounds i8, i8* %1, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4
  br i1 %5, label %12, label %9

; <label>:9:                                      ; preds = %2
  %10 = bitcast i8* %1 to <2 x float>*
  %11 = load <2 x float>, <2 x float>* %10, align 4
  br label %20

; <label>:12:                                     ; preds = %2
  %13 = getelementptr inbounds i8, i8* %1, i64 4
  %14 = bitcast i8* %13 to float*
  %15 = load float, float* %14, align 4, !tbaa !1
  %16 = insertelement <2 x float> undef, float %4, i32 0
  %17 = insertelement <2 x float> %16, float %15, i32 1
  %18 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %17
  %19 = fsub float -0.000000e+00, %8
  br label %20

; <label>:20:                                     ; preds = %9, %12
  %21 = phi <2 x float> [ %11, %9 ], [ %18, %12 ]
  %22 = phi float [ %8, %9 ], [ %19, %12 ]
  %23 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %21, <2 x float>* %23, align 4
  %24 = getelementptr inbounds i8, i8* %0, i64 8
  %25 = bitcast i8* %24 to float*
  store float %22, float* %25, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_abs_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = bitcast i8* %0 to float*
  store float %5, float* %6, align 4, !tbaa !1
  %7 = getelementptr inbounds i8, i8* %1, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4, !tbaa !1
  %10 = tail call float @fabsf(float %9) #13
  %11 = getelementptr inbounds i8, i8* %0, i64 4
  %12 = bitcast i8* %11 to float*
  store float %10, float* %12, align 4, !tbaa !1
  %13 = getelementptr inbounds i8, i8* %1, i64 8
  %14 = bitcast i8* %13 to float*
  %15 = load float, float* %14, align 4, !tbaa !1
  %16 = tail call float @fabsf(float %15) #13
  %17 = getelementptr inbounds i8, i8* %0, i64 8
  %18 = bitcast i8* %17 to float*
  store float %16, float* %18, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_abs_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = getelementptr inbounds i8, i8* %1, i64 24
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = insertelement <2 x float> undef, float %6, i32 0
  %8 = bitcast i8* %3 to float*
  %9 = load float, float* %8, align 4, !tbaa !1
  %10 = insertelement <2 x float> %7, float %9, i32 1
  %11 = bitcast i8* %4 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = fcmp ult float %6, 0.000000e+00
  br i1 %13, label %14, label %19

; <label>:14:                                     ; preds = %2
  %15 = insertelement <2 x float> undef, float %6, i32 0
  %16 = insertelement <2 x float> %15, float %9, i32 1
  %17 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %16
  %18 = fsub float -0.000000e+00, %12
  br label %19

; <label>:19:                                     ; preds = %2, %14
  %20 = phi <2 x float> [ %17, %14 ], [ %10, %2 ]
  %21 = phi float [ %18, %14 ], [ %12, %2 ]
  %22 = getelementptr inbounds i8, i8* %1, i64 4
  %23 = getelementptr inbounds i8, i8* %1, i64 16
  %24 = getelementptr inbounds i8, i8* %1, i64 28
  %25 = bitcast i8* %22 to float*
  %26 = load float, float* %25, align 4, !tbaa !1
  %27 = insertelement <2 x float> undef, float %26, i32 0
  %28 = bitcast i8* %23 to float*
  %29 = load float, float* %28, align 4, !tbaa !1
  %30 = insertelement <2 x float> %27, float %29, i32 1
  %31 = bitcast i8* %24 to float*
  %32 = load float, float* %31, align 4, !tbaa !1
  %33 = fcmp ult float %26, 0.000000e+00
  br i1 %33, label %34, label %39

; <label>:34:                                     ; preds = %19
  %35 = insertelement <2 x float> undef, float %26, i32 0
  %36 = insertelement <2 x float> %35, float %29, i32 1
  %37 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %36
  %38 = fsub float -0.000000e+00, %32
  br label %39

; <label>:39:                                     ; preds = %19, %34
  %40 = phi <2 x float> [ %37, %34 ], [ %30, %19 ]
  %41 = phi float [ %38, %34 ], [ %32, %19 ]
  %42 = getelementptr inbounds i8, i8* %1, i64 8
  %43 = getelementptr inbounds i8, i8* %1, i64 20
  %44 = getelementptr inbounds i8, i8* %1, i64 32
  %45 = bitcast i8* %42 to float*
  %46 = load float, float* %45, align 4, !tbaa !1
  %47 = insertelement <2 x float> undef, float %46, i32 0
  %48 = bitcast i8* %43 to float*
  %49 = load float, float* %48, align 4, !tbaa !1
  %50 = insertelement <2 x float> %47, float %49, i32 1
  %51 = bitcast i8* %44 to float*
  %52 = load float, float* %51, align 4, !tbaa !1
  %53 = fcmp ult float %46, 0.000000e+00
  br i1 %53, label %54, label %59

; <label>:54:                                     ; preds = %39
  %55 = insertelement <2 x float> undef, float %46, i32 0
  %56 = insertelement <2 x float> %55, float %49, i32 1
  %57 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %56
  %58 = fsub float -0.000000e+00, %52
  br label %59

; <label>:59:                                     ; preds = %39, %54
  %60 = phi <2 x float> [ %57, %54 ], [ %50, %39 ]
  %61 = phi float [ %58, %54 ], [ %52, %39 ]
  %62 = bitcast <2 x float> %20 to <2 x i32>
  %63 = extractelement <2 x i32> %62, i32 0
  %64 = bitcast <2 x float> %40 to <2 x i32>
  %65 = extractelement <2 x i32> %64, i32 0
  %66 = bitcast <2 x float> %60 to <2 x i32>
  %67 = extractelement <2 x i32> %66, i32 0
  %68 = extractelement <2 x i32> %62, i32 1
  %69 = extractelement <2 x i32> %64, i32 1
  %70 = extractelement <2 x i32> %66, i32 1
  %71 = bitcast i8* %0 to i32*
  store i32 %63, i32* %71, align 4, !tbaa !5
  %72 = getelementptr inbounds i8, i8* %0, i64 4
  %73 = bitcast i8* %72 to i32*
  store i32 %65, i32* %73, align 4, !tbaa !7
  %74 = getelementptr inbounds i8, i8* %0, i64 8
  %75 = bitcast i8* %74 to i32*
  store i32 %67, i32* %75, align 4, !tbaa !8
  %76 = getelementptr inbounds i8, i8* %0, i64 12
  %77 = bitcast i8* %76 to i32*
  store i32 %68, i32* %77, align 4, !tbaa !5
  %78 = getelementptr inbounds i8, i8* %0, i64 16
  %79 = bitcast i8* %78 to i32*
  store i32 %69, i32* %79, align 4, !tbaa !7
  %80 = getelementptr inbounds i8, i8* %0, i64 20
  %81 = bitcast i8* %80 to i32*
  store i32 %70, i32* %81, align 4, !tbaa !8
  %82 = getelementptr inbounds i8, i8* %0, i64 24
  %83 = bitcast i8* %82 to float*
  store float %21, float* %83, align 4, !tbaa !5
  %84 = getelementptr inbounds i8, i8* %0, i64 28
  %85 = bitcast i8* %84 to float*
  store float %41, float* %85, align 4, !tbaa !7
  %86 = getelementptr inbounds i8, i8* %0, i64 32
  %87 = bitcast i8* %86 to float*
  store float %61, float* %87, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readnone uwtable
define float @osl_fabs_ff(float) local_unnamed_addr #3 {
  %2 = tail call float @fabsf(float %0) #13
  ret float %2
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_fabs_dfdf(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #7 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = fcmp ult float %4, 0.000000e+00
  %6 = getelementptr inbounds i8, i8* %1, i64 8
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4
  br i1 %5, label %12, label %9

; <label>:9:                                      ; preds = %2
  %10 = bitcast i8* %1 to <2 x float>*
  %11 = load <2 x float>, <2 x float>* %10, align 4
  br label %20

; <label>:12:                                     ; preds = %2
  %13 = getelementptr inbounds i8, i8* %1, i64 4
  %14 = bitcast i8* %13 to float*
  %15 = load float, float* %14, align 4, !tbaa !1
  %16 = insertelement <2 x float> undef, float %4, i32 0
  %17 = insertelement <2 x float> %16, float %15, i32 1
  %18 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %17
  %19 = fsub float -0.000000e+00, %8
  br label %20

; <label>:20:                                     ; preds = %9, %12
  %21 = phi <2 x float> [ %11, %9 ], [ %18, %12 ]
  %22 = phi float [ %8, %9 ], [ %19, %12 ]
  %23 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %21, <2 x float>* %23, align 4
  %24 = getelementptr inbounds i8, i8* %0, i64 8
  %25 = bitcast i8* %24 to float*
  store float %22, float* %25, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fabs_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = tail call float @fabsf(float %4) #13
  %6 = bitcast i8* %0 to float*
  store float %5, float* %6, align 4, !tbaa !1
  %7 = getelementptr inbounds i8, i8* %1, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4, !tbaa !1
  %10 = tail call float @fabsf(float %9) #13
  %11 = getelementptr inbounds i8, i8* %0, i64 4
  %12 = bitcast i8* %11 to float*
  store float %10, float* %12, align 4, !tbaa !1
  %13 = getelementptr inbounds i8, i8* %1, i64 8
  %14 = bitcast i8* %13 to float*
  %15 = load float, float* %14, align 4, !tbaa !1
  %16 = tail call float @fabsf(float %15) #13
  %17 = getelementptr inbounds i8, i8* %0, i64 8
  %18 = bitcast i8* %17 to float*
  store float %16, float* %18, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fabs_dvdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = getelementptr inbounds i8, i8* %1, i64 24
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = insertelement <2 x float> undef, float %6, i32 0
  %8 = bitcast i8* %3 to float*
  %9 = load float, float* %8, align 4, !tbaa !1
  %10 = insertelement <2 x float> %7, float %9, i32 1
  %11 = bitcast i8* %4 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = fcmp ult float %6, 0.000000e+00
  br i1 %13, label %14, label %19

; <label>:14:                                     ; preds = %2
  %15 = insertelement <2 x float> undef, float %6, i32 0
  %16 = insertelement <2 x float> %15, float %9, i32 1
  %17 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %16
  %18 = fsub float -0.000000e+00, %12
  br label %19

; <label>:19:                                     ; preds = %2, %14
  %20 = phi <2 x float> [ %17, %14 ], [ %10, %2 ]
  %21 = phi float [ %18, %14 ], [ %12, %2 ]
  %22 = getelementptr inbounds i8, i8* %1, i64 4
  %23 = getelementptr inbounds i8, i8* %1, i64 16
  %24 = getelementptr inbounds i8, i8* %1, i64 28
  %25 = bitcast i8* %22 to float*
  %26 = load float, float* %25, align 4, !tbaa !1
  %27 = insertelement <2 x float> undef, float %26, i32 0
  %28 = bitcast i8* %23 to float*
  %29 = load float, float* %28, align 4, !tbaa !1
  %30 = insertelement <2 x float> %27, float %29, i32 1
  %31 = bitcast i8* %24 to float*
  %32 = load float, float* %31, align 4, !tbaa !1
  %33 = fcmp ult float %26, 0.000000e+00
  br i1 %33, label %34, label %39

; <label>:34:                                     ; preds = %19
  %35 = insertelement <2 x float> undef, float %26, i32 0
  %36 = insertelement <2 x float> %35, float %29, i32 1
  %37 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %36
  %38 = fsub float -0.000000e+00, %32
  br label %39

; <label>:39:                                     ; preds = %19, %34
  %40 = phi <2 x float> [ %37, %34 ], [ %30, %19 ]
  %41 = phi float [ %38, %34 ], [ %32, %19 ]
  %42 = getelementptr inbounds i8, i8* %1, i64 8
  %43 = getelementptr inbounds i8, i8* %1, i64 20
  %44 = getelementptr inbounds i8, i8* %1, i64 32
  %45 = bitcast i8* %42 to float*
  %46 = load float, float* %45, align 4, !tbaa !1
  %47 = insertelement <2 x float> undef, float %46, i32 0
  %48 = bitcast i8* %43 to float*
  %49 = load float, float* %48, align 4, !tbaa !1
  %50 = insertelement <2 x float> %47, float %49, i32 1
  %51 = bitcast i8* %44 to float*
  %52 = load float, float* %51, align 4, !tbaa !1
  %53 = fcmp ult float %46, 0.000000e+00
  br i1 %53, label %54, label %59

; <label>:54:                                     ; preds = %39
  %55 = insertelement <2 x float> undef, float %46, i32 0
  %56 = insertelement <2 x float> %55, float %49, i32 1
  %57 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %56
  %58 = fsub float -0.000000e+00, %52
  br label %59

; <label>:59:                                     ; preds = %39, %54
  %60 = phi <2 x float> [ %57, %54 ], [ %50, %39 ]
  %61 = phi float [ %58, %54 ], [ %52, %39 ]
  %62 = bitcast <2 x float> %20 to <2 x i32>
  %63 = extractelement <2 x i32> %62, i32 0
  %64 = bitcast <2 x float> %40 to <2 x i32>
  %65 = extractelement <2 x i32> %64, i32 0
  %66 = bitcast <2 x float> %60 to <2 x i32>
  %67 = extractelement <2 x i32> %66, i32 0
  %68 = extractelement <2 x i32> %62, i32 1
  %69 = extractelement <2 x i32> %64, i32 1
  %70 = extractelement <2 x i32> %66, i32 1
  %71 = bitcast i8* %0 to i32*
  store i32 %63, i32* %71, align 4, !tbaa !5
  %72 = getelementptr inbounds i8, i8* %0, i64 4
  %73 = bitcast i8* %72 to i32*
  store i32 %65, i32* %73, align 4, !tbaa !7
  %74 = getelementptr inbounds i8, i8* %0, i64 8
  %75 = bitcast i8* %74 to i32*
  store i32 %67, i32* %75, align 4, !tbaa !8
  %76 = getelementptr inbounds i8, i8* %0, i64 12
  %77 = bitcast i8* %76 to i32*
  store i32 %68, i32* %77, align 4, !tbaa !5
  %78 = getelementptr inbounds i8, i8* %0, i64 16
  %79 = bitcast i8* %78 to i32*
  store i32 %69, i32* %79, align 4, !tbaa !7
  %80 = getelementptr inbounds i8, i8* %0, i64 20
  %81 = bitcast i8* %80 to i32*
  store i32 %70, i32* %81, align 4, !tbaa !8
  %82 = getelementptr inbounds i8, i8* %0, i64 24
  %83 = bitcast i8* %82 to float*
  store float %21, float* %83, align 4, !tbaa !5
  %84 = getelementptr inbounds i8, i8* %0, i64 28
  %85 = bitcast i8* %84 to float*
  store float %41, float* %85, align 4, !tbaa !7
  %86 = getelementptr inbounds i8, i8* %0, i64 32
  %87 = bitcast i8* %86 to float*
  store float %61, float* %87, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readonly uwtable
define <8 x float> @osl_abs_wfwf(<8 x float>* byval nocapture readonly align 32) local_unnamed_addr #10 {
  %2 = load <8 x float>, <8 x float>* %0, align 32
  %3 = shufflevector <8 x float> %2, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %4 = fcmp oge <4 x float> %3, zeroinitializer
  %5 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %3
  %6 = select <4 x i1> %4, <4 x float> %3, <4 x float> %5
  %7 = shufflevector <4 x float> %6, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x float> %7
}

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @osl_safe_mod_iii(i32, i32) local_unnamed_addr #6 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %6, label %4

; <label>:4:                                      ; preds = %2
  %5 = srem i32 %0, %1
  br label %6

; <label>:6:                                      ; preds = %2, %4
  %7 = phi i32 [ %5, %4 ], [ 0, %2 ]
  ret i32 %7
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_fmod_fff(float, float) local_unnamed_addr #6 {
  %3 = fcmp une float %1, 0.000000e+00
  %4 = frem float %0, %1
  %5 = select i1 %3, float %4, float 0.000000e+00
  ret float %5
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_fmod_dfdfdf(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #7 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %2 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fcmp une float %7, 0.000000e+00
  %9 = frem float %5, %7
  %10 = getelementptr inbounds i8, i8* %1, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = getelementptr inbounds i8, i8* %1, i64 8
  %13 = select i1 %8, float %9, float 0.000000e+00
  %14 = insertelement <2 x float> undef, float %13, i32 0
  %15 = load float, float* %11, align 4, !tbaa !1
  %16 = insertelement <2 x float> %14, float %15, i32 1
  %17 = bitcast i8* %12 to i32*
  %18 = load i32, i32* %17, align 4, !tbaa !1
  %19 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %16, <2 x float>* %19, align 4
  %20 = getelementptr inbounds i8, i8* %0, i64 8
  %21 = bitcast i8* %20 to i32*
  store i32 %18, i32* %21, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dffdf(i8* nocapture, float, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fcmp une float %5, 0.000000e+00
  %7 = frem float %1, %5
  %8 = select i1 %6, float %7, float 0.000000e+00
  %9 = insertelement <2 x float> undef, float %8, i32 0
  %10 = insertelement <2 x float> %9, float 0.000000e+00, i32 1
  %11 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %10, <2 x float>* %11, align 4
  %12 = getelementptr inbounds i8, i8* %0, i64 8
  %13 = bitcast i8* %12 to float*
  store float 0.000000e+00, float* %13, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dfdff(i8* nocapture, i8* nocapture readonly, float) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fcmp une float %2, 0.000000e+00
  %7 = frem float %5, %2
  %8 = getelementptr inbounds i8, i8* %1, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = select i1 %6, float %7, float 0.000000e+00
  %12 = insertelement <2 x float> undef, float %11, i32 0
  %13 = load float, float* %9, align 4, !tbaa !1
  %14 = insertelement <2 x float> %12, float %13, i32 1
  %15 = bitcast i8* %10 to i32*
  %16 = load i32, i32* %15, align 4, !tbaa !1
  %17 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %14, <2 x float>* %17, align 4
  %18 = getelementptr inbounds i8, i8* %0, i64 8
  %19 = bitcast i8* %18 to i32*
  store i32 %16, i32* %19, align 4
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_fmod_vvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #7 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %2 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fcmp une float %7, 0.000000e+00
  %9 = frem float %5, %7
  %10 = select i1 %8, float %9, float 0.000000e+00
  %11 = bitcast i8* %0 to float*
  store float %10, float* %11, align 4, !tbaa !1
  %12 = getelementptr inbounds i8, i8* %1, i64 4
  %13 = bitcast i8* %12 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = getelementptr inbounds i8, i8* %2, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %16, align 4, !tbaa !1
  %18 = fcmp une float %17, 0.000000e+00
  %19 = frem float %14, %17
  %20 = select i1 %18, float %19, float 0.000000e+00
  %21 = getelementptr inbounds i8, i8* %0, i64 4
  %22 = bitcast i8* %21 to float*
  store float %20, float* %22, align 4, !tbaa !1
  %23 = getelementptr inbounds i8, i8* %1, i64 8
  %24 = bitcast i8* %23 to float*
  %25 = load float, float* %24, align 4, !tbaa !1
  %26 = getelementptr inbounds i8, i8* %2, i64 8
  %27 = bitcast i8* %26 to float*
  %28 = load float, float* %27, align 4, !tbaa !1
  %29 = fcmp une float %28, 0.000000e+00
  %30 = frem float %25, %28
  %31 = select i1 %29, float %30, float 0.000000e+00
  %32 = getelementptr inbounds i8, i8* %0, i64 8
  %33 = bitcast i8* %32 to float*
  store float %31, float* %33, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvdvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = getelementptr inbounds i8, i8* %1, i64 12
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = bitcast i8* %2 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = fcmp une float %8, 0.000000e+00
  %10 = frem float %6, %8
  %11 = getelementptr inbounds i8, i8* %1, i64 4
  %12 = getelementptr inbounds i8, i8* %1, i64 28
  %13 = bitcast i8* %11 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = bitcast i8* %12 to i32*
  %16 = load i32, i32* %15, align 4, !tbaa !1
  %17 = getelementptr inbounds i8, i8* %2, i64 4
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = fcmp une float %19, 0.000000e+00
  %21 = frem float %14, %19
  %22 = getelementptr inbounds i8, i8* %1, i64 8
  %23 = getelementptr inbounds i8, i8* %1, i64 32
  %24 = bitcast i8* %22 to float*
  %25 = load float, float* %24, align 4, !tbaa !1
  %26 = bitcast i8* %4 to <4 x i32>*
  %27 = load <4 x i32>, <4 x i32>* %26, align 4, !tbaa !1
  %28 = bitcast i8* %23 to i32*
  %29 = load i32, i32* %28, align 4, !tbaa !1
  %30 = getelementptr inbounds i8, i8* %2, i64 8
  %31 = bitcast i8* %30 to float*
  %32 = load float, float* %31, align 4, !tbaa !1
  %33 = fcmp une float %32, 0.000000e+00
  %34 = frem float %25, %32
  %35 = bitcast float %10 to i32
  %36 = select i1 %9, i32 %35, i32 0
  %37 = bitcast float %21 to i32
  %38 = select i1 %20, i32 %37, i32 0
  %39 = bitcast float %34 to i32
  %40 = select i1 %33, i32 %39, i32 0
  %41 = bitcast i8* %0 to i32*
  store i32 %36, i32* %41, align 4, !tbaa !5
  %42 = getelementptr inbounds i8, i8* %0, i64 4
  %43 = bitcast i8* %42 to i32*
  store i32 %38, i32* %43, align 4, !tbaa !7
  %44 = getelementptr inbounds i8, i8* %0, i64 8
  %45 = bitcast i8* %44 to i32*
  store i32 %40, i32* %45, align 4, !tbaa !8
  %46 = getelementptr inbounds i8, i8* %0, i64 12
  %47 = bitcast i8* %46 to <4 x i32>*
  store <4 x i32> %27, <4 x i32>* %47, align 4, !tbaa !1
  %48 = getelementptr inbounds i8, i8* %0, i64 28
  %49 = bitcast i8* %48 to i32*
  store i32 %16, i32* %49, align 4, !tbaa !7
  %50 = getelementptr inbounds i8, i8* %0, i64 32
  %51 = bitcast i8* %50 to i32*
  store i32 %29, i32* %51, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %1, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = bitcast i8* %2 to float*
  %13 = load float, float* %12, align 4, !tbaa !1
  %14 = fcmp une float %13, 0.000000e+00
  %15 = frem float %5, %13
  %16 = getelementptr inbounds i8, i8* %2, i64 4
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4, !tbaa !1
  %19 = fcmp une float %18, 0.000000e+00
  %20 = frem float %8, %18
  %21 = getelementptr inbounds i8, i8* %2, i64 8
  %22 = bitcast i8* %21 to float*
  %23 = load float, float* %22, align 4, !tbaa !1
  %24 = fcmp une float %23, 0.000000e+00
  %25 = frem float %11, %23
  %26 = bitcast float %15 to i32
  %27 = select i1 %14, i32 %26, i32 0
  %28 = bitcast float %20 to i32
  %29 = select i1 %19, i32 %28, i32 0
  %30 = bitcast float %25 to i32
  %31 = select i1 %24, i32 %30, i32 0
  %32 = bitcast i8* %0 to i32*
  store i32 %27, i32* %32, align 4, !tbaa !5
  %33 = getelementptr inbounds i8, i8* %0, i64 4
  %34 = bitcast i8* %33 to i32*
  store i32 %29, i32* %34, align 4, !tbaa !7
  %35 = getelementptr inbounds i8, i8* %0, i64 8
  %36 = bitcast i8* %35 to i32*
  store i32 %31, i32* %36, align 4, !tbaa !8
  %37 = getelementptr inbounds i8, i8* %0, i64 12
  call void @llvm.memset.p0i8.i64(i8* %37, i8 0, i64 24, i32 4, i1 false)
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvdvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %2, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %2, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = getelementptr inbounds i8, i8* %1, i64 12
  %13 = bitcast i8* %1 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = fcmp une float %5, 0.000000e+00
  %16 = frem float %14, %5
  %17 = getelementptr inbounds i8, i8* %1, i64 4
  %18 = getelementptr inbounds i8, i8* %1, i64 28
  %19 = bitcast i8* %17 to float*
  %20 = load float, float* %19, align 4, !tbaa !1
  %21 = bitcast i8* %18 to i32*
  %22 = load i32, i32* %21, align 4, !tbaa !1
  %23 = fcmp une float %8, 0.000000e+00
  %24 = frem float %20, %8
  %25 = getelementptr inbounds i8, i8* %1, i64 8
  %26 = getelementptr inbounds i8, i8* %1, i64 32
  %27 = bitcast i8* %25 to float*
  %28 = load float, float* %27, align 4, !tbaa !1
  %29 = bitcast i8* %12 to <4 x i32>*
  %30 = load <4 x i32>, <4 x i32>* %29, align 4, !tbaa !1
  %31 = bitcast i8* %26 to i32*
  %32 = load i32, i32* %31, align 4, !tbaa !1
  %33 = fcmp une float %11, 0.000000e+00
  %34 = frem float %28, %11
  %35 = bitcast float %16 to i32
  %36 = select i1 %15, i32 %35, i32 0
  %37 = bitcast float %24 to i32
  %38 = select i1 %23, i32 %37, i32 0
  %39 = bitcast float %34 to i32
  %40 = select i1 %33, i32 %39, i32 0
  %41 = bitcast i8* %0 to i32*
  store i32 %36, i32* %41, align 4, !tbaa !5
  %42 = getelementptr inbounds i8, i8* %0, i64 4
  %43 = bitcast i8* %42 to i32*
  store i32 %38, i32* %43, align 4, !tbaa !7
  %44 = getelementptr inbounds i8, i8* %0, i64 8
  %45 = bitcast i8* %44 to i32*
  store i32 %40, i32* %45, align 4, !tbaa !8
  %46 = getelementptr inbounds i8, i8* %0, i64 12
  %47 = bitcast i8* %46 to <4 x i32>*
  store <4 x i32> %30, <4 x i32>* %47, align 4, !tbaa !1
  %48 = getelementptr inbounds i8, i8* %0, i64 28
  %49 = bitcast i8* %48 to i32*
  store i32 %22, i32* %49, align 4, !tbaa !7
  %50 = getelementptr inbounds i8, i8* %0, i64 32
  %51 = bitcast i8* %50 to i32*
  store i32 %32, i32* %51, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_fmod_vvf(i8* nocapture, i8* nocapture readonly, float) local_unnamed_addr #7 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = fcmp une float %2, 0.000000e+00
  %7 = frem float %5, %2
  %8 = select i1 %6, float %7, float 0.000000e+00
  %9 = bitcast i8* %0 to float*
  store float %8, float* %9, align 4, !tbaa !1
  %10 = getelementptr inbounds i8, i8* %1, i64 4
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = frem float %12, %2
  %14 = select i1 %6, float %13, float 0.000000e+00
  %15 = getelementptr inbounds i8, i8* %0, i64 4
  %16 = bitcast i8* %15 to float*
  store float %14, float* %16, align 4, !tbaa !1
  %17 = getelementptr inbounds i8, i8* %1, i64 8
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = frem float %19, %2
  %21 = select i1 %6, float %20, float 0.000000e+00
  %22 = getelementptr inbounds i8, i8* %0, i64 8
  %23 = bitcast i8* %22 to float*
  store float %21, float* %23, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvdvdf(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = getelementptr inbounds i8, i8* %1, i64 12
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = bitcast i8* %2 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = fcmp une float %8, 0.000000e+00
  %10 = frem float %6, %8
  %11 = getelementptr inbounds i8, i8* %1, i64 4
  %12 = getelementptr inbounds i8, i8* %1, i64 28
  %13 = bitcast i8* %11 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = bitcast i8* %12 to i32*
  %16 = load i32, i32* %15, align 4, !tbaa !1
  %17 = frem float %14, %8
  %18 = getelementptr inbounds i8, i8* %1, i64 8
  %19 = getelementptr inbounds i8, i8* %1, i64 32
  %20 = bitcast i8* %18 to float*
  %21 = load float, float* %20, align 4, !tbaa !1
  %22 = bitcast i8* %4 to <4 x i32>*
  %23 = load <4 x i32>, <4 x i32>* %22, align 4, !tbaa !1
  %24 = bitcast i8* %19 to i32*
  %25 = load i32, i32* %24, align 4, !tbaa !1
  %26 = frem float %21, %8
  %27 = bitcast float %10 to i32
  %28 = select i1 %9, i32 %27, i32 0
  %29 = bitcast float %17 to i32
  %30 = select i1 %9, i32 %29, i32 0
  %31 = bitcast float %26 to i32
  %32 = select i1 %9, i32 %31, i32 0
  %33 = bitcast i8* %0 to i32*
  store i32 %28, i32* %33, align 4, !tbaa !5
  %34 = getelementptr inbounds i8, i8* %0, i64 4
  %35 = bitcast i8* %34 to i32*
  store i32 %30, i32* %35, align 4, !tbaa !7
  %36 = getelementptr inbounds i8, i8* %0, i64 8
  %37 = bitcast i8* %36 to i32*
  store i32 %32, i32* %37, align 4, !tbaa !8
  %38 = getelementptr inbounds i8, i8* %0, i64 12
  %39 = bitcast i8* %38 to <4 x i32>*
  store <4 x i32> %23, <4 x i32>* %39, align 4, !tbaa !1
  %40 = getelementptr inbounds i8, i8* %0, i64 28
  %41 = bitcast i8* %40 to i32*
  store i32 %16, i32* %41, align 4, !tbaa !7
  %42 = getelementptr inbounds i8, i8* %0, i64 32
  %43 = bitcast i8* %42 to i32*
  store i32 %25, i32* %43, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvvdf(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %1, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = bitcast i8* %2 to float*
  %13 = load float, float* %12, align 4, !tbaa !1
  %14 = fcmp une float %13, 0.000000e+00
  %15 = frem float %5, %13
  %16 = frem float %8, %13
  %17 = frem float %11, %13
  %18 = bitcast float %15 to i32
  %19 = select i1 %14, i32 %18, i32 0
  %20 = bitcast float %16 to i32
  %21 = select i1 %14, i32 %20, i32 0
  %22 = bitcast float %17 to i32
  %23 = select i1 %14, i32 %22, i32 0
  %24 = bitcast i8* %0 to i32*
  store i32 %19, i32* %24, align 4, !tbaa !5
  %25 = getelementptr inbounds i8, i8* %0, i64 4
  %26 = bitcast i8* %25 to i32*
  store i32 %21, i32* %26, align 4, !tbaa !7
  %27 = getelementptr inbounds i8, i8* %0, i64 8
  %28 = bitcast i8* %27 to i32*
  store i32 %23, i32* %28, align 4, !tbaa !8
  %29 = getelementptr inbounds i8, i8* %0, i64 12
  call void @llvm.memset.p0i8.i64(i8* %29, i8 0, i64 24, i32 4, i1 false)
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_fmod_dvdvf(i8* nocapture, i8* nocapture readonly, float) local_unnamed_addr #4 {
  %4 = getelementptr inbounds i8, i8* %1, i64 12
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = fcmp une float %2, 0.000000e+00
  %8 = frem float %6, %2
  %9 = getelementptr inbounds i8, i8* %1, i64 4
  %10 = getelementptr inbounds i8, i8* %1, i64 28
  %11 = bitcast i8* %9 to float*
  %12 = load float, float* %11, align 4, !tbaa !1
  %13 = bitcast i8* %10 to i32*
  %14 = load i32, i32* %13, align 4, !tbaa !1
  %15 = frem float %12, %2
  %16 = getelementptr inbounds i8, i8* %1, i64 8
  %17 = getelementptr inbounds i8, i8* %1, i64 32
  %18 = bitcast i8* %16 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = bitcast i8* %4 to <4 x i32>*
  %21 = load <4 x i32>, <4 x i32>* %20, align 4, !tbaa !1
  %22 = bitcast i8* %17 to i32*
  %23 = load i32, i32* %22, align 4, !tbaa !1
  %24 = frem float %19, %2
  %25 = bitcast float %8 to i32
  %26 = select i1 %7, i32 %25, i32 0
  %27 = bitcast float %15 to i32
  %28 = select i1 %7, i32 %27, i32 0
  %29 = bitcast float %24 to i32
  %30 = select i1 %7, i32 %29, i32 0
  %31 = bitcast i8* %0 to i32*
  store i32 %26, i32* %31, align 4, !tbaa !5
  %32 = getelementptr inbounds i8, i8* %0, i64 4
  %33 = bitcast i8* %32 to i32*
  store i32 %28, i32* %33, align 4, !tbaa !7
  %34 = getelementptr inbounds i8, i8* %0, i64 8
  %35 = bitcast i8* %34 to i32*
  store i32 %30, i32* %35, align 4, !tbaa !8
  %36 = getelementptr inbounds i8, i8* %0, i64 12
  %37 = bitcast i8* %36 to <4 x i32>*
  store <4 x i32> %21, <4 x i32>* %37, align 4, !tbaa !1
  %38 = getelementptr inbounds i8, i8* %0, i64 28
  %39 = bitcast i8* %38 to i32*
  store i32 %14, i32* %39, align 4, !tbaa !7
  %40 = getelementptr inbounds i8, i8* %0, i64 32
  %41 = bitcast i8* %40 to i32*
  store i32 %23, i32* %41, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_safe_div_fff(float, float) local_unnamed_addr #6 {
  %3 = fcmp une float %1, 0.000000e+00
  %4 = fdiv float %0, %1
  %5 = select i1 %3, float %4, float 0.000000e+00
  ret float %5
}

; Function Attrs: norecurse nounwind readnone uwtable
define <16 x float> @osl_safe_div_w16fw16fw16f(<16 x float>, <16 x float>) alwaysinline #6 {
  %3 = fcmp une <16 x float> %1, zeroinitializer
  %4 = fdiv <16 x float> %0, %1
  %5 = select <16 x i1> %3, <16 x float> %4, <16 x float> zeroinitializer
  ret <16 x float> %5
}

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @osl_safe_div_iii(i32, i32) local_unnamed_addr #6 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %6, label %4

; <label>:4:                                      ; preds = %2
  %5 = sdiv i32 %0, %1
  br label %6

; <label>:6:                                      ; preds = %2, %4
  %7 = phi i32 [ %5, %4 ], [ 0, %2 ]
  ret i32 %7
}

; Function Attrs: norecurse nounwind readnone uwtable
define float @osl_smoothstep_ffff(float, float, float) local_unnamed_addr #6 {
  %4 = fcmp olt float %2, %0
  br i1 %4, label %15, label %5

; <label>:5:                                      ; preds = %3
  %6 = fcmp ult float %2, %1
  br i1 %6, label %7, label %15

; <label>:7:                                      ; preds = %5
  %8 = fsub float %2, %0
  %9 = fsub float %1, %0
  %10 = fdiv float %8, %9
  %11 = fmul float %10, 2.000000e+00
  %12 = fsub float 3.000000e+00, %11
  %13 = fmul float %10, %10
  %14 = fmul float %13, %12
  br label %15

; <label>:15:                                     ; preds = %3, %5, %7
  %16 = phi float [ %14, %7 ], [ 0.000000e+00, %3 ], [ 1.000000e+00, %5 ]
  ret float %16
}


; Function Attrs: norecurse nounwind readnone uwtable
;define <16 x float> @osl_smoothstep_w16fffw16f(float, float, <16 x float>) alwaysinline #6 {
  ;%.splatinsert = insertelement <16 x float> undef, float %0, i32 0
  ;%e0 = shufflevector <16 x float> %.splatinsert, <16 x float> undef, <16 x i32> zeroinitializer

  ;%.splatinsert1 = insertelement <16 x float> undef, float %1, i32 0
  ;%e1 = shufflevector <16 x float> %.splatinsert1, <16 x float> undef, <16 x i32> zeroinitializer

  ;%11 = fcmp ult <16 x float> %2, %e0
  ;%12 = fcmp ugt <16 x float> %2, %e1
  ;%not. = xor <16 x i1> %11, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, !dbg !24
  ;%13 = and <16 x i1> %12, %not., !dbg !24
  ;%14 = select <16 x i1> %13, <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <16 x float> zeroinitializer, !dbg !25
  ;%15 = or <16 x i1> %11, %12, !dbg !25
  ;%16 = fsub <16 x float> %6, %8, !dbg !26
  ;%17 = fsub <16 x float> %10, %8, !dbg !26
  ;%18 = fcmp une <16 x float> %17, zeroinitializer, !dbg !26
  ;%19 = fdiv <16 x float> %16, %17, !dbg !26
  ;%20 = select <16 x i1> %18, <16 x float> %19, <16 x float> zeroinitializer, !dbg !26
  ;%21 = select <16 x i1> %15, <16 x float> %14, <16 x float> %20, !dbg !26

  ;ret <16 x float> %21
;}

; Function Attrs: norecurse nounwind readnone uwtable
define <16 x float> @osl_smoothstep_w16fw16fw16fw16f(<16 x float>, <16 x float>, <16 x float>) alwaysinline #6 {
  %4 = fcmp ult <16 x float> %2, %0
  %5 = fcmp ugt <16 x float> %2, %1
  %not. = xor <16 x i1> %4, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>
  %6 = and <16 x i1> %5, %not.
  %7 = select <16 x i1> %6, <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <16 x float> zeroinitializer
  %8 = or <16 x i1> %4, %5
  %9 = fsub <16 x float> %2, %0
  %10 = fsub <16 x float> %1, %0
  %11 = fdiv <16 x float> %9, %10
  %12 = fmul <16 x float> %11, <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
  %13 = fsub <16 x float> <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>, %12
  %14 = fmul <16 x float> %11, %11, !dbg !32
  %15 = fmul <16 x float> %14, %13, !dbg !32
  %16 = select <16 x i1> %8, <16 x float> %7, <16 x float> %15
  
  ret <16 x float> %16
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfffdf(i8* nocapture, float, float, i8* nocapture readonly) local_unnamed_addr #4 {
  %5 = bitcast i8* %3 to float*
  %6 = load float, float* %5, align 4
  %7 = getelementptr inbounds i8, i8* %3, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = getelementptr inbounds i8, i8* %3, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = fcmp olt float %6, %1
  br i1 %13, label %46, label %14

; <label>:14:                                     ; preds = %4
  %15 = fcmp ult float %6, %2
  br i1 %15, label %16, label %46

; <label>:16:                                     ; preds = %14
  %17 = fsub float %6, %1
  %18 = fsub float %2, %1
  %19 = fdiv float 1.000000e+00, %18
  %20 = fmul float %19, %17
  %21 = fmul float %20, 0.000000e+00
  %22 = fsub float %9, %21
  %23 = fmul float %19, %22
  %24 = fsub float %12, %21
  %25 = fmul float %19, %24
  %26 = fmul float %20, 2.000000e+00
  %27 = fmul float %23, 2.000000e+00
  %28 = fmul float %25, 2.000000e+00
  %29 = fsub float 3.000000e+00, %26
  %30 = fmul float %20, %29
  %31 = fmul float %29, %23
  %32 = fmul float %20, %27
  %33 = fsub float %31, %32
  %34 = fmul float %29, %25
  %35 = fmul float %20, %28
  %36 = fsub float %34, %35
  %37 = fmul float %20, %30
  %38 = fmul float %30, %23
  %39 = fmul float %20, %33
  %40 = fadd float %38, %39
  %41 = fmul float %30, %25
  %42 = fmul float %20, %36
  %43 = fadd float %41, %42
  %44 = insertelement <2 x float> undef, float %37, i32 0
  %45 = insertelement <2 x float> %44, float %40, i32 1
  br label %46

; <label>:46:                                     ; preds = %4, %14, %16
  %47 = phi <2 x float> [ %45, %16 ], [ zeroinitializer, %4 ], [ <float 1.000000e+00, float 0.000000e+00>, %14 ]
  %48 = phi float [ %43, %16 ], [ 0.000000e+00, %4 ], [ 0.000000e+00, %14 ]
  %49 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %47, <2 x float>* %49, align 4
  %50 = getelementptr inbounds i8, i8* %0, i64 8
  %51 = bitcast i8* %50 to float*
  store float %48, float* %51, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dffdff(i8* nocapture, float, i8* nocapture readonly, float) local_unnamed_addr #4 {
  %5 = bitcast i8* %2 to float*
  %6 = load float, float* %5, align 4
  %7 = getelementptr inbounds i8, i8* %2, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = getelementptr inbounds i8, i8* %2, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = fcmp olt float %3, %1
  br i1 %13, label %47, label %14

; <label>:14:                                     ; preds = %4
  %15 = fcmp ugt float %6, %3
  br i1 %15, label %16, label %47

; <label>:16:                                     ; preds = %14
  %17 = fsub float %3, %1
  %18 = fsub float %6, %1
  %19 = fdiv float 1.000000e+00, %18
  %20 = fmul float %17, %19
  %21 = fmul float %9, %20
  %22 = fsub float 0.000000e+00, %21
  %23 = fmul float %19, %22
  %24 = fmul float %12, %20
  %25 = fsub float 0.000000e+00, %24
  %26 = fmul float %19, %25
  %27 = fmul float %20, 2.000000e+00
  %28 = fmul float %23, 2.000000e+00
  %29 = fmul float %26, 2.000000e+00
  %30 = fsub float 3.000000e+00, %27
  %31 = fmul float %20, %30
  %32 = fmul float %30, %23
  %33 = fmul float %20, %28
  %34 = fsub float %32, %33
  %35 = fmul float %30, %26
  %36 = fmul float %20, %29
  %37 = fsub float %35, %36
  %38 = fmul float %20, %31
  %39 = fmul float %31, %23
  %40 = fmul float %20, %34
  %41 = fadd float %39, %40
  %42 = fmul float %31, %26
  %43 = fmul float %20, %37
  %44 = fadd float %42, %43
  %45 = insertelement <2 x float> undef, float %38, i32 0
  %46 = insertelement <2 x float> %45, float %41, i32 1
  br label %47

; <label>:47:                                     ; preds = %4, %14, %16
  %48 = phi <2 x float> [ %46, %16 ], [ zeroinitializer, %4 ], [ <float 1.000000e+00, float 0.000000e+00>, %14 ]
  %49 = phi float [ %44, %16 ], [ 0.000000e+00, %4 ], [ 0.000000e+00, %14 ]
  %50 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %48, <2 x float>* %50, align 4
  %51 = getelementptr inbounds i8, i8* %0, i64 8
  %52 = bitcast i8* %51 to float*
  store float %49, float* %52, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dffdfdf(i8* nocapture, float, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %5 = bitcast i8* %2 to float*
  %6 = load float, float* %5, align 4
  %7 = getelementptr inbounds i8, i8* %2, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = getelementptr inbounds i8, i8* %2, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = bitcast i8* %3 to float*
  %14 = load float, float* %13, align 4
  %15 = getelementptr inbounds i8, i8* %3, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %16, align 4
  %18 = getelementptr inbounds i8, i8* %3, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = load float, float* %19, align 4
  %21 = fcmp olt float %14, %1
  br i1 %21, label %55, label %22

; <label>:22:                                     ; preds = %4
  %23 = fcmp ult float %14, %6
  br i1 %23, label %24, label %55

; <label>:24:                                     ; preds = %22
  %25 = fsub float %14, %1
  %26 = fsub float %6, %1
  %27 = fdiv float 1.000000e+00, %26
  %28 = fmul float %27, %25
  %29 = fmul float %9, %28
  %30 = fsub float %17, %29
  %31 = fmul float %27, %30
  %32 = fmul float %12, %28
  %33 = fsub float %20, %32
  %34 = fmul float %27, %33
  %35 = fmul float %28, 2.000000e+00
  %36 = fmul float %31, 2.000000e+00
  %37 = fmul float %34, 2.000000e+00
  %38 = fsub float 3.000000e+00, %35
  %39 = fmul float %28, %38
  %40 = fmul float %38, %31
  %41 = fmul float %28, %36
  %42 = fsub float %40, %41
  %43 = fmul float %38, %34
  %44 = fmul float %28, %37
  %45 = fsub float %43, %44
  %46 = fmul float %28, %39
  %47 = fmul float %39, %31
  %48 = fmul float %28, %42
  %49 = fadd float %47, %48
  %50 = fmul float %39, %34
  %51 = fmul float %28, %45
  %52 = fadd float %50, %51
  %53 = insertelement <2 x float> undef, float %46, i32 0
  %54 = insertelement <2 x float> %53, float %49, i32 1
  br label %55

; <label>:55:                                     ; preds = %4, %22, %24
  %56 = phi <2 x float> [ %54, %24 ], [ zeroinitializer, %4 ], [ <float 1.000000e+00, float 0.000000e+00>, %22 ]
  %57 = phi float [ %52, %24 ], [ 0.000000e+00, %4 ], [ 0.000000e+00, %22 ]
  %58 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %56, <2 x float>* %58, align 4
  %59 = getelementptr inbounds i8, i8* %0, i64 8
  %60 = bitcast i8* %59 to float*
  store float %57, float* %60, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfdfff(i8* nocapture, i8* nocapture readonly, float, float) local_unnamed_addr #4 {
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4
  %7 = getelementptr inbounds i8, i8* %1, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = fcmp ogt float %6, %3
  br i1 %13, label %49, label %14

; <label>:14:                                     ; preds = %4
  %15 = fcmp ult float %3, %2
  br i1 %15, label %16, label %49

; <label>:16:                                     ; preds = %14
  %17 = fsub float %3, %6
  %18 = fsub float 0.000000e+00, %9
  %19 = fsub float 0.000000e+00, %12
  %20 = fsub float %2, %6
  %21 = fdiv float 1.000000e+00, %20
  %22 = fmul float %17, %21
  %23 = fmul float %18, %22
  %24 = fsub float %18, %23
  %25 = fmul float %21, %24
  %26 = fmul float %22, %19
  %27 = fsub float %19, %26
  %28 = fmul float %21, %27
  %29 = fmul float %22, 2.000000e+00
  %30 = fmul float %25, 2.000000e+00
  %31 = fmul float %28, 2.000000e+00
  %32 = fsub float 3.000000e+00, %29
  %33 = fmul float %22, %32
  %34 = fmul float %32, %25
  %35 = fmul float %22, %30
  %36 = fsub float %34, %35
  %37 = fmul float %32, %28
  %38 = fmul float %22, %31
  %39 = fsub float %37, %38
  %40 = fmul float %22, %33
  %41 = fmul float %33, %25
  %42 = fmul float %22, %36
  %43 = fadd float %41, %42
  %44 = fmul float %33, %28
  %45 = fmul float %22, %39
  %46 = fadd float %44, %45
  %47 = insertelement <2 x float> undef, float %40, i32 0
  %48 = insertelement <2 x float> %47, float %43, i32 1
  br label %49

; <label>:49:                                     ; preds = %4, %14, %16
  %50 = phi <2 x float> [ %48, %16 ], [ zeroinitializer, %4 ], [ <float 1.000000e+00, float 0.000000e+00>, %14 ]
  %51 = phi float [ %46, %16 ], [ 0.000000e+00, %4 ], [ 0.000000e+00, %14 ]
  %52 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %50, <2 x float>* %52, align 4
  %53 = getelementptr inbounds i8, i8* %0, i64 8
  %54 = bitcast i8* %53 to float*
  store float %51, float* %54, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfdffdf(i8* nocapture, i8* nocapture readonly, float, i8* nocapture readonly) local_unnamed_addr #4 {
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4
  %7 = getelementptr inbounds i8, i8* %1, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = bitcast i8* %3 to float*
  %14 = load float, float* %13, align 4
  %15 = getelementptr inbounds i8, i8* %3, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %16, align 4
  %18 = getelementptr inbounds i8, i8* %3, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = load float, float* %19, align 4
  %21 = fcmp olt float %14, %6
  br i1 %21, label %59, label %22

; <label>:22:                                     ; preds = %4
  %23 = fcmp ult float %14, %2
  br i1 %23, label %24, label %59

; <label>:24:                                     ; preds = %22
  %25 = fsub float %14, %6
  %26 = fsub float %17, %9
  %27 = fsub float %20, %12
  %28 = fsub float %2, %6
  %29 = fsub float 0.000000e+00, %9
  %30 = fsub float 0.000000e+00, %12
  %31 = fdiv float 1.000000e+00, %28
  %32 = fmul float %31, %25
  %33 = fmul float %29, %32
  %34 = fsub float %26, %33
  %35 = fmul float %31, %34
  %36 = fmul float %30, %32
  %37 = fsub float %27, %36
  %38 = fmul float %31, %37
  %39 = fmul float %32, 2.000000e+00
  %40 = fmul float %35, 2.000000e+00
  %41 = fmul float %38, 2.000000e+00
  %42 = fsub float 3.000000e+00, %39
  %43 = fmul float %32, %42
  %44 = fmul float %42, %35
  %45 = fmul float %32, %40
  %46 = fsub float %44, %45
  %47 = fmul float %42, %38
  %48 = fmul float %32, %41
  %49 = fsub float %47, %48
  %50 = fmul float %32, %43
  %51 = fmul float %43, %35
  %52 = fmul float %32, %46
  %53 = fadd float %51, %52
  %54 = fmul float %43, %38
  %55 = fmul float %32, %49
  %56 = fadd float %54, %55
  %57 = insertelement <2 x float> undef, float %50, i32 0
  %58 = insertelement <2 x float> %57, float %53, i32 1
  br label %59

; <label>:59:                                     ; preds = %4, %22, %24
  %60 = phi <2 x float> [ %58, %24 ], [ zeroinitializer, %4 ], [ <float 1.000000e+00, float 0.000000e+00>, %22 ]
  %61 = phi float [ %56, %24 ], [ 0.000000e+00, %4 ], [ 0.000000e+00, %22 ]
  %62 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %60, <2 x float>* %62, align 4
  %63 = getelementptr inbounds i8, i8* %0, i64 8
  %64 = bitcast i8* %63 to float*
  store float %61, float* %64, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfdfdff(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly, float) local_unnamed_addr #4 {
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4
  %7 = getelementptr inbounds i8, i8* %1, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = bitcast i8* %2 to float*
  %14 = load float, float* %13, align 4
  %15 = getelementptr inbounds i8, i8* %2, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %16, align 4
  %18 = getelementptr inbounds i8, i8* %2, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = load float, float* %19, align 4
  %21 = fcmp ogt float %6, %3
  br i1 %21, label %59, label %22

; <label>:22:                                     ; preds = %4
  %23 = fcmp ugt float %14, %3
  br i1 %23, label %24, label %59

; <label>:24:                                     ; preds = %22
  %25 = fsub float %3, %6
  %26 = fsub float 0.000000e+00, %9
  %27 = fsub float 0.000000e+00, %12
  %28 = fsub float %14, %6
  %29 = fsub float %17, %9
  %30 = fsub float %20, %12
  %31 = fdiv float 1.000000e+00, %28
  %32 = fmul float %25, %31
  %33 = fmul float %29, %32
  %34 = fsub float %26, %33
  %35 = fmul float %31, %34
  %36 = fmul float %32, %30
  %37 = fsub float %27, %36
  %38 = fmul float %31, %37
  %39 = fmul float %32, 2.000000e+00
  %40 = fmul float %35, 2.000000e+00
  %41 = fmul float %38, 2.000000e+00
  %42 = fsub float 3.000000e+00, %39
  %43 = fmul float %32, %42
  %44 = fmul float %42, %35
  %45 = fmul float %32, %40
  %46 = fsub float %44, %45
  %47 = fmul float %42, %38
  %48 = fmul float %32, %41
  %49 = fsub float %47, %48
  %50 = fmul float %32, %43
  %51 = fmul float %43, %35
  %52 = fmul float %32, %46
  %53 = fadd float %51, %52
  %54 = fmul float %43, %38
  %55 = fmul float %32, %49
  %56 = fadd float %54, %55
  %57 = insertelement <2 x float> undef, float %50, i32 0
  %58 = insertelement <2 x float> %57, float %53, i32 1
  br label %59

; <label>:59:                                     ; preds = %4, %22, %24
  %60 = phi <2 x float> [ %58, %24 ], [ zeroinitializer, %4 ], [ <float 1.000000e+00, float 0.000000e+00>, %22 ]
  %61 = phi float [ %56, %24 ], [ 0.000000e+00, %4 ], [ 0.000000e+00, %22 ]
  %62 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %60, <2 x float>* %62, align 4
  %63 = getelementptr inbounds i8, i8* %0, i64 8
  %64 = bitcast i8* %63 to float*
  store float %61, float* %64, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_smoothstep_dfdfdfdf(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4
  %7 = getelementptr inbounds i8, i8* %1, i64 4
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = getelementptr inbounds i8, i8* %1, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = bitcast i8* %2 to float*
  %14 = load float, float* %13, align 4
  %15 = getelementptr inbounds i8, i8* %2, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %16, align 4
  %18 = getelementptr inbounds i8, i8* %2, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = load float, float* %19, align 4
  %21 = bitcast i8* %3 to float*
  %22 = load float, float* %21, align 4
  %23 = getelementptr inbounds i8, i8* %3, i64 4
  %24 = bitcast i8* %23 to float*
  %25 = load float, float* %24, align 4
  %26 = getelementptr inbounds i8, i8* %3, i64 8
  %27 = bitcast i8* %26 to float*
  %28 = load float, float* %27, align 4
  %29 = fcmp olt float %22, %6
  br i1 %29, label %67, label %30

; <label>:30:                                     ; preds = %4
  %31 = fcmp ult float %22, %14
  br i1 %31, label %32, label %67

; <label>:32:                                     ; preds = %30
  %33 = fsub float %22, %6
  %34 = fsub float %25, %9
  %35 = fsub float %28, %12
  %36 = fsub float %14, %6
  %37 = fsub float %17, %9
  %38 = fsub float %20, %12
  %39 = fdiv float 1.000000e+00, %36
  %40 = fmul float %39, %33
  %41 = fmul float %37, %40
  %42 = fsub float %34, %41
  %43 = fmul float %39, %42
  %44 = fmul float %38, %40
  %45 = fsub float %35, %44
  %46 = fmul float %39, %45
  %47 = fmul float %40, 2.000000e+00
  %48 = fmul float %43, 2.000000e+00
  %49 = fmul float %46, 2.000000e+00
  %50 = fsub float 3.000000e+00, %47
  %51 = fmul float %40, %50
  %52 = fmul float %50, %43
  %53 = fmul float %40, %48
  %54 = fsub float %52, %53
  %55 = fmul float %50, %46
  %56 = fmul float %40, %49
  %57 = fsub float %55, %56
  %58 = fmul float %40, %51
  %59 = fmul float %51, %43
  %60 = fmul float %40, %54
  %61 = fadd float %59, %60
  %62 = fmul float %51, %46
  %63 = fmul float %40, %57
  %64 = fadd float %62, %63
  %65 = insertelement <2 x float> undef, float %58, i32 0
  %66 = insertelement <2 x float> %65, float %61, i32 1
  br label %67

; <label>:67:                                     ; preds = %4, %30, %32
  %68 = phi <2 x float> [ %66, %32 ], [ zeroinitializer, %4 ], [ <float 1.000000e+00, float 0.000000e+00>, %30 ]
  %69 = phi float [ %64, %32 ], [ 0.000000e+00, %4 ], [ 0.000000e+00, %30 ]
  %70 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %68, <2 x float>* %70, align 4
  %71 = getelementptr inbounds i8, i8* %0, i64 8
  %72 = bitcast i8* %71 to float*
  store float %69, float* %72, align 4
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_transform_vmv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #7 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = getelementptr inbounds i8, i8* %2, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = getelementptr inbounds i8, i8* %2, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = getelementptr inbounds i8, i8* %1, i64 12
  %13 = bitcast i8* %12 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = fmul float %5, %14
  %16 = getelementptr inbounds i8, i8* %1, i64 28
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4, !tbaa !1
  %19 = fmul float %8, %18
  %20 = fadd float %15, %19
  %21 = getelementptr inbounds i8, i8* %1, i64 44
  %22 = bitcast i8* %21 to float*
  %23 = load float, float* %22, align 4, !tbaa !1
  %24 = fmul float %11, %23
  %25 = fadd float %20, %24
  %26 = getelementptr inbounds i8, i8* %1, i64 60
  %27 = bitcast i8* %26 to float*
  %28 = load float, float* %27, align 4, !tbaa !1
  %29 = fadd float %28, %25
  %30 = fcmp une float %29, 0.000000e+00
  br i1 %30, label %31, label %88

; <label>:31:                                     ; preds = %3
  %32 = getelementptr inbounds i8, i8* %1, i64 48
  %33 = bitcast i8* %32 to float*
  %34 = getelementptr inbounds i8, i8* %1, i64 32
  %35 = bitcast i8* %34 to float*
  %36 = getelementptr inbounds i8, i8* %1, i64 16
  %37 = bitcast i8* %36 to float*
  %38 = bitcast i8* %1 to float*
  %39 = getelementptr inbounds i8, i8* %1, i64 56
  %40 = bitcast i8* %39 to float*
  %41 = load float, float* %40, align 4, !tbaa !1
  %42 = getelementptr inbounds i8, i8* %1, i64 40
  %43 = bitcast i8* %42 to float*
  %44 = load float, float* %43, align 4, !tbaa !1
  %45 = getelementptr inbounds i8, i8* %1, i64 24
  %46 = bitcast i8* %45 to float*
  %47 = load float, float* %46, align 4, !tbaa !1
  %48 = getelementptr inbounds i8, i8* %1, i64 8
  %49 = bitcast i8* %48 to float*
  %50 = load float, float* %49, align 4, !tbaa !1
  %51 = getelementptr inbounds i8, i8* %1, i64 52
  %52 = bitcast i8* %51 to float*
  %53 = load float, float* %52, align 4, !tbaa !1
  %54 = getelementptr inbounds i8, i8* %1, i64 36
  %55 = bitcast i8* %54 to float*
  %56 = load float, float* %55, align 4, !tbaa !1
  %57 = getelementptr inbounds i8, i8* %1, i64 20
  %58 = bitcast i8* %57 to float*
  %59 = load float, float* %58, align 4, !tbaa !1
  %60 = getelementptr inbounds i8, i8* %1, i64 4
  %61 = bitcast i8* %60 to float*
  %62 = load float, float* %61, align 4, !tbaa !1
  %63 = load float, float* %33, align 4, !tbaa !1
  %64 = load float, float* %35, align 4, !tbaa !1
  %65 = load float, float* %37, align 4, !tbaa !1
  %66 = load float, float* %38, align 4, !tbaa !1
  %67 = fmul float %5, %50
  %68 = fmul float %8, %47
  %69 = fadd float %68, %67
  %70 = fmul float %11, %44
  %71 = fadd float %70, %69
  %72 = fadd float %41, %71
  %73 = fmul float %5, %62
  %74 = fmul float %8, %59
  %75 = fadd float %74, %73
  %76 = fmul float %11, %56
  %77 = fadd float %76, %75
  %78 = fadd float %53, %77
  %79 = fmul float %5, %66
  %80 = fmul float %8, %65
  %81 = fadd float %80, %79
  %82 = fmul float %11, %64
  %83 = fadd float %82, %81
  %84 = fadd float %63, %83
  %85 = fdiv float %84, %29
  %86 = fdiv float %78, %29
  %87 = fdiv float %72, %29
  br label %88

; <label>:88:                                     ; preds = %3, %31
  %89 = phi float [ %85, %31 ], [ 0.000000e+00, %3 ]
  %90 = phi float [ %86, %31 ], [ 0.000000e+00, %3 ]
  %91 = phi float [ %87, %31 ], [ 0.000000e+00, %3 ]
  %92 = bitcast i8* %0 to float*
  store float %89, float* %92, align 4
  %93 = getelementptr inbounds i8, i8* %0, i64 4
  %94 = bitcast i8* %93 to float*
  store float %90, float* %94, align 4
  %95 = getelementptr inbounds i8, i8* %0, i64 8
  %96 = bitcast i8* %95 to float*
  store float %91, float* %96, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_transform_dvmdv(i8*, i8*, i8*) local_unnamed_addr #4 {
  %4 = bitcast i8* %2 to %"class.OSL::Dual2.0"*
  %5 = bitcast i8* %1 to %"class.Imath_2_2::Matrix44"*
  %6 = bitcast i8* %0 to %"class.OSL::Dual2.0"*
  tail call void @_ZN3OSL20robust_multVecMatrixERKN9Imath_2_28Matrix44IfEERKNS_5Dual2INS0_4Vec3IfEEEERS8_(%"class.Imath_2_2::Matrix44"* dereferenceable(64) %5, %"class.OSL::Dual2.0"* dereferenceable(36) %4, %"class.OSL::Dual2.0"* dereferenceable(36) %6)
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN3OSL20robust_multVecMatrixERKN9Imath_2_28Matrix44IfEERKNS_5Dual2INS0_4Vec3IfEEEERS8_(%"class.Imath_2_2::Matrix44"* dereferenceable(64), %"class.OSL::Dual2.0"* dereferenceable(36), %"class.OSL::Dual2.0"* dereferenceable(36)) local_unnamed_addr #8 comdat {
  %4 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 0, i32 0
  %5 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 1, i32 0
  %6 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 2, i32 0
  %7 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 0, i32 0
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = load float, float* %5, align 4, !tbaa !1
  %10 = load float, float* %6, align 4, !tbaa !1
  %11 = getelementptr inbounds float, float* %4, i64 1
  %12 = getelementptr inbounds float, float* %5, i64 1
  %13 = getelementptr inbounds float, float* %6, i64 1
  %14 = load float, float* %11, align 4, !tbaa !1
  %15 = load float, float* %12, align 4, !tbaa !1
  %16 = load float, float* %13, align 4, !tbaa !1
  %17 = getelementptr inbounds float, float* %4, i64 2
  %18 = getelementptr inbounds float, float* %5, i64 2
  %19 = getelementptr inbounds float, float* %6, i64 2
  %20 = load float, float* %17, align 4, !tbaa !1
  %21 = load float, float* %18, align 4, !tbaa !1
  %22 = load float, float* %19, align 4, !tbaa !1
  %23 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 0
  %24 = load float, float* %23, align 4, !tbaa !1
  %25 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 0
  %26 = load float, float* %25, align 4, !tbaa !1
  %27 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 0
  %28 = load float, float* %27, align 4, !tbaa !1
  %29 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 1
  %30 = load float, float* %29, align 4, !tbaa !1
  %31 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 1
  %32 = load float, float* %31, align 4, !tbaa !1
  %33 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 1
  %34 = load float, float* %33, align 4, !tbaa !1
  %35 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 2
  %36 = load float, float* %35, align 4, !tbaa !1
  %37 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 2
  %38 = load float, float* %37, align 4, !tbaa !1
  %39 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 2
  %40 = load float, float* %39, align 4, !tbaa !1
  %41 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 3
  %42 = load float, float* %41, align 4, !tbaa !1
  %43 = fmul float %8, %42
  %44 = fmul float %9, %42
  %45 = fmul float %10, %42
  %46 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 3
  %47 = load float, float* %46, align 4, !tbaa !1
  %48 = fmul float %14, %47
  %49 = fmul float %15, %47
  %50 = fmul float %16, %47
  %51 = fadd float %43, %48
  %52 = fadd float %44, %49
  %53 = fadd float %45, %50
  %54 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 3
  %55 = load float, float* %54, align 4, !tbaa !1
  %56 = fmul float %20, %55
  %57 = fmul float %21, %55
  %58 = fmul float %22, %55
  %59 = fadd float %51, %56
  %60 = fadd float %52, %57
  %61 = fadd float %53, %58
  %62 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 3
  %63 = load float, float* %62, align 4, !tbaa !1
  %64 = fadd float %63, %59
  %65 = fcmp une float %64, 0.000000e+00
  br i1 %65, label %66, label %155

; <label>:66:                                     ; preds = %3
  %67 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 0
  %68 = fmul float %8, %36
  %69 = fmul float %14, %38
  %70 = fadd float %68, %69
  %71 = fmul float %20, %40
  %72 = fadd float %70, %71
  %73 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 2
  %74 = load float, float* %73, align 4, !tbaa !1
  %75 = fadd float %72, %74
  %76 = fmul float %9, %36
  %77 = fmul float %15, %38
  %78 = fadd float %76, %77
  %79 = fmul float %21, %40
  %80 = fadd float %78, %79
  %81 = fmul float %10, %36
  %82 = fmul float %16, %38
  %83 = fadd float %81, %82
  %84 = fmul float %22, %40
  %85 = fadd float %83, %84
  %86 = fmul float %8, %30
  %87 = fmul float %14, %32
  %88 = fadd float %86, %87
  %89 = fmul float %20, %34
  %90 = fadd float %88, %89
  %91 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 1
  %92 = load float, float* %91, align 4, !tbaa !1
  %93 = fadd float %90, %92
  %94 = fmul float %9, %30
  %95 = fmul float %15, %32
  %96 = fadd float %94, %95
  %97 = fmul float %21, %34
  %98 = fadd float %96, %97
  %99 = fmul float %10, %30
  %100 = fmul float %16, %32
  %101 = fadd float %99, %100
  %102 = fmul float %22, %34
  %103 = fadd float %101, %102
  %104 = fmul float %8, %24
  %105 = fmul float %14, %26
  %106 = fadd float %104, %105
  %107 = fmul float %20, %28
  %108 = fadd float %106, %107
  %109 = load float, float* %67, align 4, !tbaa !1
  %110 = fadd float %108, %109
  %111 = fmul float %24, %9
  %112 = fmul float %26, %15
  %113 = fadd float %111, %112
  %114 = fmul float %28, %21
  %115 = fadd float %113, %114
  %116 = fmul float %24, %10
  %117 = fmul float %26, %16
  %118 = fadd float %116, %117
  %119 = fmul float %28, %22
  %120 = fadd float %118, %119
  %121 = fdiv float 1.000000e+00, %64
  %122 = fmul float %121, %110
  %123 = fmul float %60, %122
  %124 = fsub float %115, %123
  %125 = fmul float %121, %124
  %126 = fmul float %61, %122
  %127 = fsub float %120, %126
  %128 = fmul float %121, %127
  %129 = insertelement <2 x float> undef, float %122, i32 0
  %130 = insertelement <2 x float> %129, float %125, i32 1
  %131 = fmul float %121, %93
  %132 = fmul float %60, %131
  %133 = fsub float %98, %132
  %134 = fmul float %121, %133
  %135 = fmul float %61, %131
  %136 = fsub float %103, %135
  %137 = fmul float %121, %136
  %138 = insertelement <2 x float> undef, float %131, i32 0
  %139 = insertelement <2 x float> %138, float %134, i32 1
  %140 = fmul float %121, %75
  %141 = fmul float %60, %140
  %142 = fsub float %80, %141
  %143 = fmul float %121, %142
  %144 = fmul float %61, %140
  %145 = fsub float %85, %144
  %146 = fmul float %121, %145
  %147 = insertelement <2 x float> undef, float %140, i32 0
  %148 = insertelement <2 x float> %147, float %143, i32 1
  %149 = bitcast <2 x float> %130 to <2 x i32>
  %150 = bitcast <2 x float> %139 to <2 x i32>
  %151 = bitcast <2 x float> %148 to <2 x i32>
  %152 = bitcast float %128 to i32
  %153 = bitcast float %137 to i32
  %154 = bitcast float %146 to i32
  br label %155

; <label>:155:                                    ; preds = %3, %66
  %156 = phi <2 x i32> [ %149, %66 ], [ zeroinitializer, %3 ]
  %157 = phi i32 [ %152, %66 ], [ 0, %3 ]
  %158 = phi <2 x i32> [ %150, %66 ], [ zeroinitializer, %3 ]
  %159 = phi i32 [ %153, %66 ], [ 0, %3 ]
  %160 = phi <2 x i32> [ %151, %66 ], [ zeroinitializer, %3 ]
  %161 = phi i32 [ %154, %66 ], [ 0, %3 ]
  %162 = extractelement <2 x i32> %156, i32 0
  %163 = extractelement <2 x i32> %158, i32 0
  %164 = extractelement <2 x i32> %160, i32 0
  %165 = extractelement <2 x i32> %156, i32 1
  %166 = extractelement <2 x i32> %158, i32 1
  %167 = extractelement <2 x i32> %160, i32 1
  %168 = bitcast %"class.OSL::Dual2.0"* %2 to i32*
  store i32 %162, i32* %168, align 4, !tbaa !5
  %169 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %2, i64 0, i32 0, i32 1
  %170 = bitcast float* %169 to i32*
  store i32 %163, i32* %170, align 4, !tbaa !7
  %171 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %2, i64 0, i32 0, i32 2
  %172 = bitcast float* %171 to i32*
  store i32 %164, i32* %172, align 4, !tbaa !8
  %173 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %2, i64 0, i32 1
  %174 = bitcast %"class.Imath_2_2::Vec3"* %173 to i32*
  store i32 %165, i32* %174, align 4, !tbaa !5
  %175 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %2, i64 0, i32 1, i32 1
  %176 = bitcast float* %175 to i32*
  store i32 %166, i32* %176, align 4, !tbaa !7
  %177 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %2, i64 0, i32 1, i32 2
  %178 = bitcast float* %177 to i32*
  store i32 %167, i32* %178, align 4, !tbaa !8
  %179 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %2, i64 0, i32 2
  %180 = bitcast %"class.Imath_2_2::Vec3"* %179 to i32*
  store i32 %157, i32* %180, align 4, !tbaa !5
  %181 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %2, i64 0, i32 2, i32 1
  %182 = bitcast float* %181 to i32*
  store i32 %159, i32* %182, align 4, !tbaa !7
  %183 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %2, i64 0, i32 2, i32 2
  %184 = bitcast float* %183 to i32*
  store i32 %161, i32* %184, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_transformv_vmv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #7 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %1 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fmul float %5, %7
  %9 = getelementptr inbounds i8, i8* %2, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = getelementptr inbounds i8, i8* %1, i64 16
  %13 = bitcast i8* %12 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = fmul float %11, %14
  %16 = fadd float %8, %15
  %17 = getelementptr inbounds i8, i8* %2, i64 8
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = getelementptr inbounds i8, i8* %1, i64 32
  %21 = bitcast i8* %20 to float*
  %22 = load float, float* %21, align 4, !tbaa !1
  %23 = fmul float %19, %22
  %24 = fadd float %16, %23
  %25 = getelementptr inbounds i8, i8* %1, i64 4
  %26 = bitcast i8* %25 to float*
  %27 = load float, float* %26, align 4, !tbaa !1
  %28 = fmul float %5, %27
  %29 = getelementptr inbounds i8, i8* %1, i64 20
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !1
  %32 = fmul float %11, %31
  %33 = fadd float %28, %32
  %34 = getelementptr inbounds i8, i8* %1, i64 36
  %35 = bitcast i8* %34 to float*
  %36 = load float, float* %35, align 4, !tbaa !1
  %37 = fmul float %19, %36
  %38 = fadd float %33, %37
  %39 = getelementptr inbounds i8, i8* %1, i64 8
  %40 = bitcast i8* %39 to float*
  %41 = load float, float* %40, align 4, !tbaa !1
  %42 = fmul float %5, %41
  %43 = getelementptr inbounds i8, i8* %1, i64 24
  %44 = bitcast i8* %43 to float*
  %45 = load float, float* %44, align 4, !tbaa !1
  %46 = fmul float %11, %45
  %47 = fadd float %42, %46
  %48 = getelementptr inbounds i8, i8* %1, i64 40
  %49 = bitcast i8* %48 to float*
  %50 = load float, float* %49, align 4, !tbaa !1
  %51 = fmul float %19, %50
  %52 = fadd float %47, %51
  %53 = bitcast i8* %0 to float*
  store float %24, float* %53, align 4, !tbaa !5
  %54 = getelementptr inbounds i8, i8* %0, i64 4
  %55 = bitcast i8* %54 to float*
  store float %38, float* %55, align 4, !tbaa !7
  %56 = getelementptr inbounds i8, i8* %0, i64 8
  %57 = bitcast i8* %56 to float*
  store float %52, float* %57, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_transformv_dvmdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #7 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = bitcast i8* %1 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fmul float %5, %7
  %9 = getelementptr inbounds i8, i8* %2, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = getelementptr inbounds i8, i8* %1, i64 16
  %13 = bitcast i8* %12 to float*
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = fmul float %11, %14
  %16 = fadd float %8, %15
  %17 = getelementptr inbounds i8, i8* %2, i64 8
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = getelementptr inbounds i8, i8* %1, i64 32
  %21 = bitcast i8* %20 to float*
  %22 = load float, float* %21, align 4, !tbaa !1
  %23 = fmul float %19, %22
  %24 = fadd float %16, %23
  %25 = getelementptr inbounds i8, i8* %1, i64 4
  %26 = bitcast i8* %25 to float*
  %27 = load float, float* %26, align 4, !tbaa !1
  %28 = fmul float %5, %27
  %29 = getelementptr inbounds i8, i8* %1, i64 20
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !1
  %32 = fmul float %11, %31
  %33 = fadd float %28, %32
  %34 = getelementptr inbounds i8, i8* %1, i64 36
  %35 = bitcast i8* %34 to float*
  %36 = load float, float* %35, align 4, !tbaa !1
  %37 = fmul float %19, %36
  %38 = fadd float %33, %37
  %39 = getelementptr inbounds i8, i8* %1, i64 8
  %40 = bitcast i8* %39 to float*
  %41 = load float, float* %40, align 4, !tbaa !1
  %42 = fmul float %5, %41
  %43 = getelementptr inbounds i8, i8* %1, i64 24
  %44 = bitcast i8* %43 to float*
  %45 = load float, float* %44, align 4, !tbaa !1
  %46 = fmul float %11, %45
  %47 = fadd float %42, %46
  %48 = getelementptr inbounds i8, i8* %1, i64 40
  %49 = bitcast i8* %48 to float*
  %50 = load float, float* %49, align 4, !tbaa !1
  %51 = fmul float %19, %50
  %52 = fadd float %47, %51
  %53 = bitcast i8* %0 to float*
  store float %24, float* %53, align 4, !tbaa !5
  %54 = getelementptr inbounds i8, i8* %0, i64 4
  %55 = bitcast i8* %54 to float*
  store float %38, float* %55, align 4, !tbaa !7
  %56 = getelementptr inbounds i8, i8* %0, i64 8
  %57 = bitcast i8* %56 to float*
  store float %52, float* %57, align 4, !tbaa !8
  %58 = getelementptr inbounds i8, i8* %2, i64 12
  %59 = bitcast i8* %58 to float*
  %60 = load float, float* %59, align 4, !tbaa !1
  %61 = load float, float* %6, align 4, !tbaa !1
  %62 = fmul float %60, %61
  %63 = getelementptr inbounds i8, i8* %2, i64 16
  %64 = bitcast i8* %63 to float*
  %65 = load float, float* %64, align 4, !tbaa !1
  %66 = load float, float* %13, align 4, !tbaa !1
  %67 = fmul float %65, %66
  %68 = fadd float %62, %67
  %69 = getelementptr inbounds i8, i8* %2, i64 20
  %70 = bitcast i8* %69 to float*
  %71 = load float, float* %70, align 4, !tbaa !1
  %72 = load float, float* %21, align 4, !tbaa !1
  %73 = fmul float %71, %72
  %74 = fadd float %68, %73
  %75 = load float, float* %26, align 4, !tbaa !1
  %76 = fmul float %60, %75
  %77 = load float, float* %30, align 4, !tbaa !1
  %78 = fmul float %65, %77
  %79 = fadd float %76, %78
  %80 = load float, float* %35, align 4, !tbaa !1
  %81 = fmul float %71, %80
  %82 = fadd float %79, %81
  %83 = load float, float* %40, align 4, !tbaa !1
  %84 = fmul float %60, %83
  %85 = load float, float* %44, align 4, !tbaa !1
  %86 = fmul float %65, %85
  %87 = fadd float %84, %86
  %88 = load float, float* %49, align 4, !tbaa !1
  %89 = fmul float %71, %88
  %90 = fadd float %87, %89
  %91 = getelementptr inbounds i8, i8* %0, i64 12
  %92 = bitcast i8* %91 to float*
  store float %74, float* %92, align 4, !tbaa !5
  %93 = getelementptr inbounds i8, i8* %0, i64 16
  %94 = bitcast i8* %93 to float*
  store float %82, float* %94, align 4, !tbaa !7
  %95 = getelementptr inbounds i8, i8* %0, i64 20
  %96 = bitcast i8* %95 to float*
  store float %90, float* %96, align 4, !tbaa !8
  %97 = getelementptr inbounds i8, i8* %2, i64 24
  %98 = bitcast i8* %97 to float*
  %99 = load float, float* %98, align 4, !tbaa !1
  %100 = load float, float* %6, align 4, !tbaa !1
  %101 = fmul float %99, %100
  %102 = getelementptr inbounds i8, i8* %2, i64 28
  %103 = bitcast i8* %102 to float*
  %104 = load float, float* %103, align 4, !tbaa !1
  %105 = load float, float* %13, align 4, !tbaa !1
  %106 = fmul float %104, %105
  %107 = fadd float %101, %106
  %108 = getelementptr inbounds i8, i8* %2, i64 32
  %109 = bitcast i8* %108 to float*
  %110 = load float, float* %109, align 4, !tbaa !1
  %111 = load float, float* %21, align 4, !tbaa !1
  %112 = fmul float %110, %111
  %113 = fadd float %107, %112
  %114 = load float, float* %26, align 4, !tbaa !1
  %115 = fmul float %99, %114
  %116 = load float, float* %30, align 4, !tbaa !1
  %117 = fmul float %104, %116
  %118 = fadd float %115, %117
  %119 = load float, float* %35, align 4, !tbaa !1
  %120 = fmul float %110, %119
  %121 = fadd float %118, %120
  %122 = load float, float* %40, align 4, !tbaa !1
  %123 = fmul float %99, %122
  %124 = load float, float* %44, align 4, !tbaa !1
  %125 = fmul float %104, %124
  %126 = fadd float %123, %125
  %127 = load float, float* %49, align 4, !tbaa !1
  %128 = fmul float %110, %127
  %129 = fadd float %126, %128
  %130 = getelementptr inbounds i8, i8* %0, i64 24
  %131 = bitcast i8* %130 to float*
  store float %113, float* %131, align 4, !tbaa !5
  %132 = getelementptr inbounds i8, i8* %0, i64 28
  %133 = bitcast i8* %132 to float*
  store float %121, float* %133, align 4, !tbaa !7
  %134 = getelementptr inbounds i8, i8* %0, i64 32
  %135 = bitcast i8* %134 to float*
  store float %129, float* %135, align 4, !tbaa !8
  ret void
}

; Function Attrs: uwtable
define void @osl_transformn_vmv(i8* nocapture, i8*, i8* nocapture readonly) local_unnamed_addr #11 {
  %4 = alloca %"class.Imath_2_2::Matrix44", align 4
  %5 = bitcast i8* %1 to %"class.Imath_2_2::Matrix44"*
  call void @_ZNK9Imath_2_28Matrix44IfE7inverseEb(%"class.Imath_2_2::Matrix44"* nonnull sret %4, %"class.Imath_2_2::Matrix44"* %5, i1 zeroext false)
  %6 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 0
  %7 = load float, float* %6, align 4, !tbaa !1, !noalias !13
  %8 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 0
  %9 = load float, float* %8, align 4, !tbaa !1, !noalias !13
  %10 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 0
  %11 = load float, float* %10, align 4, !tbaa !1, !noalias !13
  %12 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 1
  %13 = load float, float* %12, align 4, !tbaa !1, !noalias !13
  %14 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 1
  %15 = load float, float* %14, align 4, !tbaa !1, !noalias !13
  %16 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 1
  %17 = load float, float* %16, align 4, !tbaa !1, !noalias !13
  %18 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 2
  %19 = load float, float* %18, align 4, !tbaa !1, !noalias !13
  %20 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 2
  %21 = load float, float* %20, align 4, !tbaa !1, !noalias !13
  %22 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 2
  %23 = load float, float* %22, align 4, !tbaa !1, !noalias !13
  %24 = bitcast i8* %2 to float*
  %25 = load float, float* %24, align 4, !tbaa !1
  %26 = fmul float %7, %25
  %27 = getelementptr inbounds i8, i8* %2, i64 4
  %28 = bitcast i8* %27 to float*
  %29 = load float, float* %28, align 4, !tbaa !1
  %30 = fmul float %13, %29
  %31 = fadd float %26, %30
  %32 = getelementptr inbounds i8, i8* %2, i64 8
  %33 = bitcast i8* %32 to float*
  %34 = load float, float* %33, align 4, !tbaa !1
  %35 = fmul float %19, %34
  %36 = fadd float %31, %35
  %37 = fmul float %9, %25
  %38 = fmul float %15, %29
  %39 = fadd float %37, %38
  %40 = fmul float %21, %34
  %41 = fadd float %39, %40
  %42 = fmul float %11, %25
  %43 = fmul float %17, %29
  %44 = fadd float %42, %43
  %45 = fmul float %23, %34
  %46 = fadd float %44, %45
  %47 = bitcast i8* %0 to float*
  store float %36, float* %47, align 4, !tbaa !5
  %48 = getelementptr inbounds i8, i8* %0, i64 4
  %49 = bitcast i8* %48 to float*
  store float %41, float* %49, align 4, !tbaa !7
  %50 = getelementptr inbounds i8, i8* %0, i64 8
  %51 = bitcast i8* %50 to float*
  store float %46, float* %51, align 4, !tbaa !8
  ret void
}

; Function Attrs: uwtable
define linkonce_odr void @_ZNK9Imath_2_28Matrix44IfE7inverseEb(%"class.Imath_2_2::Matrix44"* noalias sret, %"class.Imath_2_2::Matrix44"*, i1 zeroext) local_unnamed_addr #11 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = alloca %"class.Imath_2_2::Matrix44", align 4
  %5 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 3
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = fcmp une float %6, 0.000000e+00
  br i1 %7, label %20, label %8

; <label>:8:                                      ; preds = %3
  %9 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 3
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = fcmp une float %10, 0.000000e+00
  br i1 %11, label %20, label %12

; <label>:12:                                     ; preds = %8
  %13 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 3
  %14 = load float, float* %13, align 4, !tbaa !1
  %15 = fcmp une float %14, 0.000000e+00
  br i1 %15, label %20, label %16

; <label>:16:                                     ; preds = %12
  %17 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 3, i64 3
  %18 = load float, float* %17, align 4, !tbaa !1
  %19 = fcmp une float %18, 1.000000e+00
  br i1 %19, label %20, label %25

; <label>:20:                                     ; preds = %16, %12, %8, %3
  invoke void @_ZNK9Imath_2_28Matrix44IfE9gjInverseEb(%"class.Imath_2_2::Matrix44"* sret %0, %"class.Imath_2_2::Matrix44"* nonnull %1, i1 zeroext %2)
          to label %197 unwind label %21

; <label>:21:                                     ; preds = %20
  %22 = landingpad { i8*, i32 }
          filter [1 x i8*] [i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN7Iex_2_27MathExcE to i8*)]
  %23 = extractvalue { i8*, i32 } %22, 0
  %24 = extractvalue { i8*, i32 } %22, 1
  br label %192

; <label>:25:                                     ; preds = %16
  %26 = bitcast %"class.Imath_2_2::Matrix44"* %4 to i8*
  call void @llvm.lifetime.start(i64 64, i8* %26) #2
  %27 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 1
  %28 = load float, float* %27, align 4, !tbaa !1
  %29 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 2
  %30 = load float, float* %29, align 4, !tbaa !1
  %31 = fmul float %28, %30
  %32 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 1
  %33 = load float, float* %32, align 4, !tbaa !1
  %34 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 2
  %35 = load float, float* %34, align 4, !tbaa !1
  %36 = fmul float %33, %35
  %37 = fsub float %31, %36
  %38 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 2
  %39 = load float, float* %38, align 4, !tbaa !1
  %40 = fmul float %33, %39
  %41 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 1
  %42 = load float, float* %41, align 4, !tbaa !1
  %43 = fmul float %30, %42
  %44 = fsub float %40, %43
  %45 = fmul float %35, %42
  %46 = fmul float %28, %39
  %47 = fsub float %45, %46
  %48 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2, i64 0
  %49 = load float, float* %48, align 4, !tbaa !1
  %50 = fmul float %35, %49
  %51 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1, i64 0
  %52 = load float, float* %51, align 4, !tbaa !1
  %53 = fmul float %30, %52
  %54 = fsub float %50, %53
  %55 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 0, i64 0
  %56 = load float, float* %55, align 4, !tbaa !1
  %57 = fmul float %30, %56
  %58 = fmul float %39, %49
  %59 = fsub float %57, %58
  %60 = fmul float %39, %52
  %61 = fmul float %35, %56
  %62 = fsub float %60, %61
  %63 = fmul float %33, %52
  %64 = fmul float %28, %49
  %65 = fsub float %63, %64
  %66 = fmul float %42, %49
  %67 = fmul float %33, %56
  %68 = fsub float %66, %67
  %69 = fmul float %28, %56
  %70 = fmul float %42, %52
  %71 = fsub float %69, %70
  %72 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 0
  store float %37, float* %72, align 4, !tbaa !1
  %73 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 1
  store float %44, float* %73, align 4, !tbaa !1
  %74 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 2
  store float %47, float* %74, align 4, !tbaa !1
  %75 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 3
  store float 0.000000e+00, float* %75, align 4, !tbaa !1
  %76 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 0
  store float %54, float* %76, align 4, !tbaa !1
  %77 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 1
  store float %59, float* %77, align 4, !tbaa !1
  %78 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 2
  store float %62, float* %78, align 4, !tbaa !1
  %79 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 3
  store float 0.000000e+00, float* %79, align 4, !tbaa !1
  %80 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 0
  store float %65, float* %80, align 4, !tbaa !1
  %81 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 1
  store float %68, float* %81, align 4, !tbaa !1
  %82 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 2
  store float %71, float* %82, align 4, !tbaa !1
  %83 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 3
  %84 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 3, i64 0
  %85 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 3, i64 1
  %86 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 3, i64 2
  %87 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 3, i64 3
  %88 = bitcast float* %83 to i8*
  call void @llvm.memset.p0i8.i64(i8* %88, i8 0, i64 16, i32 4, i1 false)
  store float 1.000000e+00, float* %87, align 4, !tbaa !1
  %89 = fmul float %56, %37
  %90 = fmul float %42, %54
  %91 = fadd float %89, %90
  %92 = fmul float %39, %65
  %93 = fadd float %91, %92
  %94 = fcmp ogt float %93, 0.000000e+00
  %95 = fsub float -0.000000e+00, %93
  %96 = select i1 %94, float %93, float %95
  %97 = fcmp ult float %96, 1.000000e+00
  br i1 %97, label %107, label %98

; <label>:98:                                     ; preds = %25
  %99 = fdiv float %37, %93
  store float %99, float* %72, align 4, !tbaa !1
  %100 = fdiv float %44, %93
  store float %100, float* %73, align 4, !tbaa !1
  %101 = fdiv float %47, %93
  store float %101, float* %74, align 4, !tbaa !1
  %102 = fdiv float %54, %93
  store float %102, float* %76, align 4, !tbaa !1
  %103 = fdiv float %59, %93
  store float %103, float* %77, align 4, !tbaa !1
  %104 = fdiv float %62, %93
  store float %104, float* %78, align 4, !tbaa !1
  %105 = fdiv float %65, %93
  store float %105, float* %80, align 4, !tbaa !1
  %106 = fdiv float %68, %93
  store float %106, float* %81, align 4, !tbaa !1
  br label %135

; <label>:107:                                    ; preds = %25
  %108 = fmul float %96, 0x47D0000000000000
  %109 = fcmp ogt float %37, 0.000000e+00
  %110 = fsub float -0.000000e+00, %37
  %111 = select i1 %109, float %37, float %110
  %112 = fcmp ogt float %108, %111
  br i1 %112, label %113, label %123

; <label>:113:                                    ; preds = %107
  %114 = fdiv float %37, %93
  store float %114, float* %72, align 4, !tbaa !1
  %115 = fcmp ogt float %44, 0.000000e+00
  %116 = fsub float -0.000000e+00, %44
  %117 = select i1 %115, float %44, float %116
  %118 = fcmp ogt float %108, %117
  br i1 %118, label %202, label %123

; <label>:119:                                    ; preds = %124
  %120 = landingpad { i8*, i32 }
          cleanup
          filter [1 x i8*] [i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN7Iex_2_27MathExcE to i8*)]
  %121 = extractvalue { i8*, i32 } %120, 0
  %122 = extractvalue { i8*, i32 } %120, 1
  call void @llvm.lifetime.end(i64 64, i8* nonnull %26) #2
  br label %192

; <label>:123:                                    ; preds = %238, %232, %226, %220, %214, %208, %202, %113, %107
  br i1 %2, label %124, label %128

; <label>:124:                                    ; preds = %123
  %125 = tail call i8* @__cxa_allocate_exception(i64 24) #2
  %126 = bitcast i8* %125 to %"class.Iex_2_2::BaseExc"*
  tail call void @_ZN7Iex_2_27BaseExcC2EPKc(%"class.Iex_2_2::BaseExc"* %126, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str, i64 0, i64 0)) #2
  %127 = bitcast i8* %125 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @_ZTVN9Imath_2_213SingMatrixExcE, i64 0, i64 2) to i32 (...)**), i32 (...)*** %127, align 8, !tbaa !16
  invoke void @__cxa_throw(i8* %125, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN9Imath_2_213SingMatrixExcE to i8*), i8* bitcast (void (%"class.Iex_2_2::BaseExc"*)* @_ZN7Iex_2_27BaseExcD2Ev to i8*)) #15
          to label %201 unwind label %119

; <label>:128:                                    ; preds = %123
  %129 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 1
  %130 = bitcast float* %129 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %130, i8 0, i64 56, i32 4, i1 false) #2
  %131 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 0
  store float 1.000000e+00, float* %131, align 4, !tbaa !1
  %132 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 1
  store float 1.000000e+00, float* %132, align 4, !tbaa !1
  %133 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 2
  store float 1.000000e+00, float* %133, align 4, !tbaa !1
  %134 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 3
  store float 1.000000e+00, float* %134, align 4, !tbaa !1
  br label %191

; <label>:135:                                    ; preds = %238, %98
  %136 = phi float [ %106, %98 ], [ %239, %238 ]
  %137 = phi float [ %105, %98 ], [ %233, %238 ]
  %138 = phi float [ %104, %98 ], [ %227, %238 ]
  %139 = phi float [ %103, %98 ], [ %221, %238 ]
  %140 = phi float [ %102, %98 ], [ %215, %238 ]
  %141 = phi float [ %101, %98 ], [ %209, %238 ]
  %142 = phi float [ %100, %98 ], [ %203, %238 ]
  %143 = phi float [ %99, %98 ], [ %114, %238 ]
  %144 = fdiv float %71, %93
  store float %144, float* %82, align 4, !tbaa !1
  %145 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 3, i64 0
  %146 = load float, float* %145, align 4, !tbaa !1
  %147 = fmul float %146, %143
  %148 = fsub float -0.000000e+00, %147
  %149 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 3, i64 1
  %150 = load float, float* %149, align 4, !tbaa !1
  %151 = fmul float %150, %140
  %152 = fsub float %148, %151
  %153 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 3, i64 2
  %154 = load float, float* %153, align 4, !tbaa !1
  %155 = fmul float %154, %137
  %156 = fsub float %152, %155
  store float %156, float* %84, align 4, !tbaa !1
  %157 = fmul float %146, %142
  %158 = fsub float -0.000000e+00, %157
  %159 = fmul float %150, %139
  %160 = fsub float %158, %159
  %161 = fmul float %154, %136
  %162 = fsub float %160, %161
  store float %162, float* %85, align 4, !tbaa !1
  %163 = fmul float %146, %141
  %164 = fsub float -0.000000e+00, %163
  %165 = fmul float %150, %138
  %166 = fsub float %164, %165
  %167 = fmul float %154, %144
  %168 = fsub float %166, %167
  store float %168, float* %86, align 4, !tbaa !1
  %169 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 0
  store float %143, float* %169, align 4, !tbaa !1
  %170 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 1
  store float %142, float* %170, align 4, !tbaa !1
  %171 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 2
  store float %141, float* %171, align 4, !tbaa !1
  %172 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 3
  %173 = bitcast float* %172 to i32*
  store i32 0, i32* %173, align 4, !tbaa !1
  %174 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 0
  store float %140, float* %174, align 4, !tbaa !1
  %175 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 1
  store float %139, float* %175, align 4, !tbaa !1
  %176 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 2
  store float %138, float* %176, align 4, !tbaa !1
  %177 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 3
  %178 = bitcast float* %177 to i32*
  store i32 0, i32* %178, align 4, !tbaa !1
  %179 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 0
  store float %137, float* %179, align 4, !tbaa !1
  %180 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 1
  store float %136, float* %180, align 4, !tbaa !1
  %181 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 2
  store float %144, float* %181, align 4, !tbaa !1
  %182 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 3
  %183 = bitcast float* %182 to i32*
  store i32 0, i32* %183, align 4, !tbaa !1
  %184 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 0
  store float %156, float* %184, align 4, !tbaa !1
  %185 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 1
  store float %162, float* %185, align 4, !tbaa !1
  %186 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 2
  store float %168, float* %186, align 4, !tbaa !1
  %187 = bitcast float* %87 to i32*
  %188 = load i32, i32* %187, align 4, !tbaa !1
  %189 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 3
  %190 = bitcast float* %189 to i32*
  store i32 %188, i32* %190, align 4, !tbaa !1
  br label %191

; <label>:191:                                    ; preds = %128, %135
  call void @llvm.lifetime.end(i64 64, i8* nonnull %26) #2
  br label %197

; <label>:192:                                    ; preds = %119, %21
  %193 = phi i32 [ %24, %21 ], [ %122, %119 ]
  %194 = phi i8* [ %23, %21 ], [ %121, %119 ]
  %195 = icmp slt i32 %193, 0
  br i1 %195, label %196, label %198

; <label>:196:                                    ; preds = %192
  tail call void @__cxa_call_unexpected(i8* %194) #15
  unreachable

; <label>:197:                                    ; preds = %20, %191
  ret void

; <label>:198:                                    ; preds = %192
  %199 = insertvalue { i8*, i32 } undef, i8* %194, 0
  %200 = insertvalue { i8*, i32 } %199, i32 %193, 1
  resume { i8*, i32 } %200

; <label>:201:                                    ; preds = %124
  unreachable

; <label>:202:                                    ; preds = %113
  %203 = fdiv float %44, %93
  store float %203, float* %73, align 4, !tbaa !1
  %204 = fcmp ogt float %47, 0.000000e+00
  %205 = fsub float -0.000000e+00, %47
  %206 = select i1 %204, float %47, float %205
  %207 = fcmp ogt float %108, %206
  br i1 %207, label %208, label %123

; <label>:208:                                    ; preds = %202
  %209 = fdiv float %47, %93
  store float %209, float* %74, align 4, !tbaa !1
  %210 = fcmp ogt float %54, 0.000000e+00
  %211 = fsub float -0.000000e+00, %54
  %212 = select i1 %210, float %54, float %211
  %213 = fcmp ogt float %108, %212
  br i1 %213, label %214, label %123

; <label>:214:                                    ; preds = %208
  %215 = fdiv float %54, %93
  store float %215, float* %76, align 4, !tbaa !1
  %216 = fcmp ogt float %59, 0.000000e+00
  %217 = fsub float -0.000000e+00, %59
  %218 = select i1 %216, float %59, float %217
  %219 = fcmp ogt float %108, %218
  br i1 %219, label %220, label %123

; <label>:220:                                    ; preds = %214
  %221 = fdiv float %59, %93
  store float %221, float* %77, align 4, !tbaa !1
  %222 = fcmp ogt float %62, 0.000000e+00
  %223 = fsub float -0.000000e+00, %62
  %224 = select i1 %222, float %62, float %223
  %225 = fcmp ogt float %108, %224
  br i1 %225, label %226, label %123

; <label>:226:                                    ; preds = %220
  %227 = fdiv float %62, %93
  store float %227, float* %78, align 4, !tbaa !1
  %228 = fcmp ogt float %65, 0.000000e+00
  %229 = fsub float -0.000000e+00, %65
  %230 = select i1 %228, float %65, float %229
  %231 = fcmp ogt float %108, %230
  br i1 %231, label %232, label %123

; <label>:232:                                    ; preds = %226
  %233 = fdiv float %65, %93
  store float %233, float* %80, align 4, !tbaa !1
  %234 = fcmp ogt float %68, 0.000000e+00
  %235 = fsub float -0.000000e+00, %68
  %236 = select i1 %234, float %68, float %235
  %237 = fcmp ogt float %108, %236
  br i1 %237, label %238, label %123

; <label>:238:                                    ; preds = %232
  %239 = fdiv float %68, %93
  store float %239, float* %81, align 4, !tbaa !1
  %240 = fcmp ogt float %71, 0.000000e+00
  %241 = fsub float -0.000000e+00, %71
  %242 = select i1 %240, float %71, float %241
  %243 = fcmp ogt float %108, %242
  br i1 %243, label %135, label %123
}

; Function Attrs: uwtable
define void @osl_transformn_dvmdv(i8* nocapture, i8*, i8* nocapture readonly) local_unnamed_addr #11 {
  %4 = alloca %"class.Imath_2_2::Matrix44", align 4
  %5 = bitcast i8* %1 to %"class.Imath_2_2::Matrix44"*
  call void @_ZNK9Imath_2_28Matrix44IfE7inverseEb(%"class.Imath_2_2::Matrix44"* nonnull sret %4, %"class.Imath_2_2::Matrix44"* %5, i1 zeroext false)
  %6 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 0
  %7 = load float, float* %6, align 4, !tbaa !1, !noalias !18
  %8 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 0
  %9 = load float, float* %8, align 4, !tbaa !1, !noalias !18
  %10 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 0
  %11 = load float, float* %10, align 4, !tbaa !1, !noalias !18
  %12 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 1
  %13 = load float, float* %12, align 4, !tbaa !1, !noalias !18
  %14 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 1
  %15 = load float, float* %14, align 4, !tbaa !1, !noalias !18
  %16 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 1
  %17 = load float, float* %16, align 4, !tbaa !1, !noalias !18
  %18 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 2
  %19 = load float, float* %18, align 4, !tbaa !1, !noalias !18
  %20 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 2
  %21 = load float, float* %20, align 4, !tbaa !1, !noalias !18
  %22 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 2
  %23 = load float, float* %22, align 4, !tbaa !1, !noalias !18
  %24 = bitcast i8* %2 to float*
  %25 = load float, float* %24, align 4, !tbaa !1
  %26 = fmul float %7, %25
  %27 = getelementptr inbounds i8, i8* %2, i64 4
  %28 = bitcast i8* %27 to float*
  %29 = load float, float* %28, align 4, !tbaa !1
  %30 = fmul float %13, %29
  %31 = fadd float %26, %30
  %32 = getelementptr inbounds i8, i8* %2, i64 8
  %33 = bitcast i8* %32 to float*
  %34 = load float, float* %33, align 4, !tbaa !1
  %35 = fmul float %19, %34
  %36 = fadd float %31, %35
  %37 = fmul float %9, %25
  %38 = fmul float %15, %29
  %39 = fadd float %37, %38
  %40 = fmul float %21, %34
  %41 = fadd float %39, %40
  %42 = fmul float %11, %25
  %43 = fmul float %17, %29
  %44 = fadd float %42, %43
  %45 = fmul float %23, %34
  %46 = fadd float %44, %45
  %47 = bitcast i8* %0 to float*
  store float %36, float* %47, align 4, !tbaa !5
  %48 = getelementptr inbounds i8, i8* %0, i64 4
  %49 = bitcast i8* %48 to float*
  store float %41, float* %49, align 4, !tbaa !7
  %50 = getelementptr inbounds i8, i8* %0, i64 8
  %51 = bitcast i8* %50 to float*
  store float %46, float* %51, align 4, !tbaa !8
  %52 = getelementptr inbounds i8, i8* %2, i64 12
  %53 = bitcast i8* %52 to float*
  %54 = load float, float* %53, align 4, !tbaa !1
  %55 = fmul float %7, %54
  %56 = getelementptr inbounds i8, i8* %2, i64 16
  %57 = bitcast i8* %56 to float*
  %58 = load float, float* %57, align 4, !tbaa !1
  %59 = fmul float %13, %58
  %60 = fadd float %55, %59
  %61 = getelementptr inbounds i8, i8* %2, i64 20
  %62 = bitcast i8* %61 to float*
  %63 = load float, float* %62, align 4, !tbaa !1
  %64 = fmul float %19, %63
  %65 = fadd float %60, %64
  %66 = fmul float %9, %54
  %67 = fmul float %15, %58
  %68 = fadd float %66, %67
  %69 = fmul float %21, %63
  %70 = fadd float %68, %69
  %71 = fmul float %11, %54
  %72 = fmul float %17, %58
  %73 = fadd float %71, %72
  %74 = fmul float %23, %63
  %75 = fadd float %73, %74
  %76 = getelementptr inbounds i8, i8* %0, i64 12
  %77 = bitcast i8* %76 to float*
  store float %65, float* %77, align 4, !tbaa !5
  %78 = getelementptr inbounds i8, i8* %0, i64 16
  %79 = bitcast i8* %78 to float*
  store float %70, float* %79, align 4, !tbaa !7
  %80 = getelementptr inbounds i8, i8* %0, i64 20
  %81 = bitcast i8* %80 to float*
  store float %75, float* %81, align 4, !tbaa !8
  %82 = getelementptr inbounds i8, i8* %2, i64 24
  %83 = bitcast i8* %82 to float*
  %84 = load float, float* %83, align 4, !tbaa !1
  %85 = fmul float %7, %84
  %86 = getelementptr inbounds i8, i8* %2, i64 28
  %87 = bitcast i8* %86 to float*
  %88 = load float, float* %87, align 4, !tbaa !1
  %89 = fmul float %13, %88
  %90 = fadd float %85, %89
  %91 = getelementptr inbounds i8, i8* %2, i64 32
  %92 = bitcast i8* %91 to float*
  %93 = load float, float* %92, align 4, !tbaa !1
  %94 = fmul float %19, %93
  %95 = fadd float %90, %94
  %96 = fmul float %9, %84
  %97 = fmul float %15, %88
  %98 = fadd float %96, %97
  %99 = fmul float %21, %93
  %100 = fadd float %98, %99
  %101 = fmul float %11, %84
  %102 = fmul float %17, %88
  %103 = fadd float %101, %102
  %104 = fmul float %23, %93
  %105 = fadd float %103, %104
  %106 = getelementptr inbounds i8, i8* %0, i64 24
  %107 = bitcast i8* %106 to float*
  store float %95, float* %107, align 4, !tbaa !5
  %108 = getelementptr inbounds i8, i8* %0, i64 28
  %109 = bitcast i8* %108 to float*
  store float %100, float* %109, align 4, !tbaa !7
  %110 = getelementptr inbounds i8, i8* %0, i64 32
  %111 = bitcast i8* %110 to float*
  store float %105, float* %111, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readonly uwtable
define float @osl_dot_fvv(i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #10 {
  %3 = bitcast i8* %0 to float*
  %4 = load float, float* %3, align 4, !tbaa !5
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !5
  %7 = fmul float %4, %6
  %8 = getelementptr inbounds i8, i8* %0, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float, float* %9, align 4, !tbaa !7
  %11 = getelementptr inbounds i8, i8* %1, i64 4
  %12 = bitcast i8* %11 to float*
  %13 = load float, float* %12, align 4, !tbaa !7
  %14 = fmul float %10, %13
  %15 = fadd float %7, %14
  %16 = getelementptr inbounds i8, i8* %0, i64 8
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4, !tbaa !8
  %19 = getelementptr inbounds i8, i8* %1, i64 8
  %20 = bitcast i8* %19 to float*
  %21 = load float, float* %20, align 4, !tbaa !8
  %22 = fmul float %18, %21
  %23 = fadd float %15, %22
  ret float %23
}

; Function Attrs: norecurse nounwind uwtable
define void @osl_dot_dfdvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #7 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1
  %6 = getelementptr inbounds i8, i8* %1, i64 12
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !1
  %9 = getelementptr inbounds i8, i8* %1, i64 24
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !1
  %12 = getelementptr inbounds i8, i8* %1, i64 4
  %13 = bitcast i8* %12 to float*
  %14 = getelementptr inbounds i8, i8* %1, i64 16
  %15 = bitcast i8* %14 to float*
  %16 = getelementptr inbounds i8, i8* %1, i64 28
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %13, align 4, !tbaa !1
  %19 = load float, float* %15, align 4, !tbaa !1
  %20 = load float, float* %17, align 4, !tbaa !1
  %21 = getelementptr inbounds i8, i8* %1, i64 8
  %22 = bitcast i8* %21 to float*
  %23 = getelementptr inbounds i8, i8* %1, i64 20
  %24 = bitcast i8* %23 to float*
  %25 = getelementptr inbounds i8, i8* %1, i64 32
  %26 = bitcast i8* %25 to float*
  %27 = load float, float* %22, align 4, !tbaa !1
  %28 = load float, float* %24, align 4, !tbaa !1
  %29 = load float, float* %26, align 4, !tbaa !1
  %30 = bitcast i8* %2 to float*
  %31 = load float, float* %30, align 4, !tbaa !1
  %32 = getelementptr inbounds i8, i8* %2, i64 12
  %33 = bitcast i8* %32 to float*
  %34 = load float, float* %33, align 4, !tbaa !1
  %35 = getelementptr inbounds i8, i8* %2, i64 24
  %36 = bitcast i8* %35 to float*
  %37 = load float, float* %36, align 4, !tbaa !1
  %38 = getelementptr inbounds i8, i8* %2, i64 4
  %39 = bitcast i8* %38 to float*
  %40 = getelementptr inbounds i8, i8* %2, i64 16
  %41 = bitcast i8* %40 to float*
  %42 = getelementptr inbounds i8, i8* %2, i64 28
  %43 = bitcast i8* %42 to float*
  %44 = load float, float* %39, align 4, !tbaa !1
  %45 = load float, float* %41, align 4, !tbaa !1
  %46 = load float, float* %43, align 4, !tbaa !1
  %47 = getelementptr inbounds i8, i8* %2, i64 8
  %48 = bitcast i8* %47 to float*
  %49 = getelementptr inbounds i8, i8* %2, i64 20
  %50 = bitcast i8* %49 to float*
  %51 = getelementptr inbounds i8, i8* %2, i64 32
  %52 = bitcast i8* %51 to float*
  %53 = load float, float* %48, align 4, !tbaa !1
  %54 = load float, float* %50, align 4, !tbaa !1
  %55 = load float, float* %52, align 4, !tbaa !1
  %56 = fmul float %5, %31
  %57 = fmul float %5, %34
  %58 = fmul float %8, %31
  %59 = fadd float %58, %57
  %60 = fmul float %5, %37
  %61 = fmul float %11, %31
  %62 = fadd float %61, %60
  %63 = fmul float %18, %44
  %64 = fmul float %18, %45
  %65 = fmul float %19, %44
  %66 = fadd float %65, %64
  %67 = fmul float %18, %46
  %68 = fmul float %20, %44
  %69 = fadd float %68, %67
  %70 = insertelement <2 x float> undef, float %56, i32 0
  %71 = insertelement <2 x float> %70, float %59, i32 1
  %72 = insertelement <2 x float> undef, float %63, i32 0
  %73 = insertelement <2 x float> %72, float %66, i32 1
  %74 = fadd <2 x float> %71, %73
  %75 = fadd float %62, %69
  %76 = fmul float %27, %53
  %77 = fmul float %27, %54
  %78 = fmul float %28, %53
  %79 = fadd float %78, %77
  %80 = fmul float %27, %55
  %81 = fmul float %29, %53
  %82 = fadd float %81, %80
  %83 = insertelement <2 x float> undef, float %76, i32 0
  %84 = insertelement <2 x float> %83, float %79, i32 1
  %85 = fadd <2 x float> %74, %84
  %86 = fadd float %75, %82
  %87 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %85, <2 x float>* %87, align 4
  %88 = getelementptr inbounds i8, i8* %0, i64 8
  %89 = bitcast i8* %88 to float*
  store float %86, float* %89, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_dot_dfdvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %2, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %2, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = bitcast i8* %1 to float*
  %13 = load float, float* %12, align 4, !tbaa !1
  %14 = getelementptr inbounds i8, i8* %1, i64 12
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !1
  %17 = getelementptr inbounds i8, i8* %1, i64 24
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = getelementptr inbounds i8, i8* %1, i64 4
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8, i8* %1, i64 16
  %23 = bitcast i8* %22 to float*
  %24 = getelementptr inbounds i8, i8* %1, i64 28
  %25 = bitcast i8* %24 to float*
  %26 = load float, float* %21, align 4, !tbaa !1
  %27 = load float, float* %23, align 4, !tbaa !1
  %28 = load float, float* %25, align 4, !tbaa !1
  %29 = getelementptr inbounds i8, i8* %1, i64 8
  %30 = bitcast i8* %29 to float*
  %31 = getelementptr inbounds i8, i8* %1, i64 20
  %32 = bitcast i8* %31 to float*
  %33 = getelementptr inbounds i8, i8* %1, i64 32
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %30, align 4, !tbaa !1
  %36 = load float, float* %32, align 4, !tbaa !1
  %37 = load float, float* %34, align 4, !tbaa !1
  %38 = fmul float %5, %13
  %39 = fmul float %13, 0.000000e+00
  %40 = fmul float %5, %16
  %41 = fadd float %39, %40
  %42 = fmul float %5, %19
  %43 = fadd float %39, %42
  %44 = fmul float %8, %26
  %45 = fmul float %26, 0.000000e+00
  %46 = fmul float %8, %27
  %47 = fadd float %45, %46
  %48 = fmul float %8, %28
  %49 = fadd float %45, %48
  %50 = insertelement <2 x float> undef, float %38, i32 0
  %51 = insertelement <2 x float> %50, float %41, i32 1
  %52 = insertelement <2 x float> undef, float %44, i32 0
  %53 = insertelement <2 x float> %52, float %47, i32 1
  %54 = fadd <2 x float> %51, %53
  %55 = fadd float %43, %49
  %56 = fmul float %11, %35
  %57 = fmul float %35, 0.000000e+00
  %58 = fmul float %11, %36
  %59 = fadd float %57, %58
  %60 = fmul float %11, %37
  %61 = fadd float %57, %60
  %62 = insertelement <2 x float> undef, float %56, i32 0
  %63 = insertelement <2 x float> %62, float %59, i32 1
  %64 = fadd <2 x float> %54, %63
  %65 = fadd float %55, %61
  %66 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %64, <2 x float>* %66, align 4
  %67 = getelementptr inbounds i8, i8* %0, i64 8
  %68 = bitcast i8* %67 to float*
  store float %65, float* %68, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_dot_dfvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %1, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = bitcast i8* %2 to float*
  %13 = load float, float* %12, align 4, !tbaa !1
  %14 = getelementptr inbounds i8, i8* %2, i64 12
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !1
  %17 = getelementptr inbounds i8, i8* %2, i64 24
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = getelementptr inbounds i8, i8* %2, i64 4
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8, i8* %2, i64 16
  %23 = bitcast i8* %22 to float*
  %24 = getelementptr inbounds i8, i8* %2, i64 28
  %25 = bitcast i8* %24 to float*
  %26 = load float, float* %21, align 4, !tbaa !1
  %27 = load float, float* %23, align 4, !tbaa !1
  %28 = load float, float* %25, align 4, !tbaa !1
  %29 = getelementptr inbounds i8, i8* %2, i64 8
  %30 = bitcast i8* %29 to float*
  %31 = getelementptr inbounds i8, i8* %2, i64 20
  %32 = bitcast i8* %31 to float*
  %33 = getelementptr inbounds i8, i8* %2, i64 32
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %30, align 4, !tbaa !1
  %36 = load float, float* %32, align 4, !tbaa !1
  %37 = load float, float* %34, align 4, !tbaa !1
  %38 = fmul float %5, %13
  %39 = fmul float %5, %16
  %40 = fmul float %13, 0.000000e+00
  %41 = fadd float %40, %39
  %42 = fmul float %5, %19
  %43 = fadd float %40, %42
  %44 = fmul float %8, %26
  %45 = fmul float %8, %27
  %46 = fmul float %26, 0.000000e+00
  %47 = fadd float %46, %45
  %48 = fmul float %8, %28
  %49 = fadd float %46, %48
  %50 = insertelement <2 x float> undef, float %38, i32 0
  %51 = insertelement <2 x float> %50, float %41, i32 1
  %52 = insertelement <2 x float> undef, float %44, i32 0
  %53 = insertelement <2 x float> %52, float %47, i32 1
  %54 = fadd <2 x float> %51, %53
  %55 = fadd float %43, %49
  %56 = fmul float %11, %35
  %57 = fmul float %11, %36
  %58 = fmul float %35, 0.000000e+00
  %59 = fadd float %58, %57
  %60 = fmul float %11, %37
  %61 = fadd float %58, %60
  %62 = insertelement <2 x float> undef, float %56, i32 0
  %63 = insertelement <2 x float> %62, float %59, i32 1
  %64 = fadd <2 x float> %54, %63
  %65 = fadd float %55, %61
  %66 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %64, <2 x float>* %66, align 4
  %67 = getelementptr inbounds i8, i8* %0, i64 8
  %68 = bitcast i8* %67 to float*
  store float %65, float* %68, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cross_vvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = getelementptr inbounds i8, i8* %1, i64 4
  %5 = bitcast i8* %4 to float*
  %6 = load float, float* %5, align 4, !tbaa !7, !noalias !21
  %7 = getelementptr inbounds i8, i8* %2, i64 8
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4, !tbaa !8, !noalias !21
  %10 = fmul float %6, %9
  %11 = getelementptr inbounds i8, i8* %1, i64 8
  %12 = bitcast i8* %11 to float*
  %13 = load float, float* %12, align 4, !tbaa !8, !noalias !21
  %14 = getelementptr inbounds i8, i8* %2, i64 4
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !7, !noalias !21
  %17 = fmul float %13, %16
  %18 = fsub float %10, %17
  %19 = bitcast i8* %2 to float*
  %20 = load float, float* %19, align 4, !tbaa !5, !noalias !21
  %21 = fmul float %13, %20
  %22 = bitcast i8* %1 to float*
  %23 = load float, float* %22, align 4, !tbaa !5, !noalias !21
  %24 = fmul float %9, %23
  %25 = fsub float %21, %24
  %26 = fmul float %16, %23
  %27 = fmul float %6, %20
  %28 = fsub float %26, %27
  %29 = bitcast i8* %0 to float*
  store float %18, float* %29, align 4, !tbaa !5
  %30 = getelementptr inbounds i8, i8* %0, i64 4
  %31 = bitcast i8* %30 to float*
  store float %25, float* %31, align 4, !tbaa !7
  %32 = getelementptr inbounds i8, i8* %0, i64 8
  %33 = bitcast i8* %32 to float*
  store float %28, float* %33, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cross_dvdvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !1, !noalias !24
  %6 = getelementptr inbounds i8, i8* %1, i64 12
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !1, !noalias !24
  %9 = getelementptr inbounds i8, i8* %1, i64 24
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !1, !noalias !24
  %12 = getelementptr inbounds i8, i8* %1, i64 4
  %13 = bitcast i8* %12 to float*
  %14 = getelementptr inbounds i8, i8* %1, i64 16
  %15 = bitcast i8* %14 to float*
  %16 = getelementptr inbounds i8, i8* %1, i64 28
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %13, align 4, !tbaa !1, !noalias !24
  %19 = load float, float* %15, align 4, !tbaa !1, !noalias !24
  %20 = load float, float* %17, align 4, !tbaa !1, !noalias !24
  %21 = getelementptr inbounds i8, i8* %1, i64 8
  %22 = bitcast i8* %21 to float*
  %23 = getelementptr inbounds i8, i8* %1, i64 20
  %24 = bitcast i8* %23 to float*
  %25 = getelementptr inbounds i8, i8* %1, i64 32
  %26 = bitcast i8* %25 to float*
  %27 = load float, float* %22, align 4, !tbaa !1, !noalias !24
  %28 = load float, float* %24, align 4, !tbaa !1, !noalias !24
  %29 = load float, float* %26, align 4, !tbaa !1, !noalias !24
  %30 = bitcast i8* %2 to float*
  %31 = load float, float* %30, align 4, !tbaa !1, !noalias !24
  %32 = getelementptr inbounds i8, i8* %2, i64 12
  %33 = bitcast i8* %32 to float*
  %34 = load float, float* %33, align 4, !tbaa !1, !noalias !24
  %35 = getelementptr inbounds i8, i8* %2, i64 24
  %36 = bitcast i8* %35 to float*
  %37 = load float, float* %36, align 4, !tbaa !1, !noalias !24
  %38 = getelementptr inbounds i8, i8* %2, i64 4
  %39 = bitcast i8* %38 to float*
  %40 = getelementptr inbounds i8, i8* %2, i64 16
  %41 = bitcast i8* %40 to float*
  %42 = getelementptr inbounds i8, i8* %2, i64 28
  %43 = bitcast i8* %42 to float*
  %44 = load float, float* %39, align 4, !tbaa !1, !noalias !24
  %45 = load float, float* %41, align 4, !tbaa !1, !noalias !24
  %46 = load float, float* %43, align 4, !tbaa !1, !noalias !24
  %47 = getelementptr inbounds i8, i8* %2, i64 8
  %48 = bitcast i8* %47 to float*
  %49 = getelementptr inbounds i8, i8* %2, i64 20
  %50 = bitcast i8* %49 to float*
  %51 = getelementptr inbounds i8, i8* %2, i64 32
  %52 = bitcast i8* %51 to float*
  %53 = load float, float* %48, align 4, !tbaa !1, !noalias !24
  %54 = load float, float* %50, align 4, !tbaa !1, !noalias !24
  %55 = load float, float* %52, align 4, !tbaa !1, !noalias !24
  %56 = fmul float %18, %53
  %57 = fmul float %18, %54
  %58 = fmul float %19, %53
  %59 = fadd float %58, %57
  %60 = fmul float %27, %44
  %61 = fmul float %27, %45
  %62 = fmul float %28, %44
  %63 = fadd float %62, %61
  %64 = fmul float %27, %31
  %65 = fmul float %5, %53
  %66 = fmul float %5, %44
  %67 = insertelement <4 x float> undef, float %27, i32 0
  %68 = insertelement <4 x float> %67, float %5, i32 1
  %69 = insertelement <4 x float> %68, float %18, i32 2
  %70 = insertelement <4 x float> %69, float %27, i32 3
  %71 = insertelement <4 x float> undef, float %34, i32 0
  %72 = insertelement <4 x float> %71, float %45, i32 1
  %73 = insertelement <4 x float> %72, float %55, i32 2
  %74 = insertelement <4 x float> %73, float %37, i32 3
  %75 = fmul <4 x float> %70, %74
  %76 = insertelement <4 x float> undef, float %28, i32 0
  %77 = insertelement <4 x float> %76, float %8, i32 1
  %78 = insertelement <4 x float> %77, float %20, i32 2
  %79 = insertelement <4 x float> %78, float %29, i32 3
  %80 = insertelement <4 x float> undef, float %31, i32 0
  %81 = insertelement <4 x float> %80, float %44, i32 1
  %82 = insertelement <4 x float> %81, float %53, i32 2
  %83 = insertelement <4 x float> %82, float %31, i32 3
  %84 = fmul <4 x float> %79, %83
  %85 = fadd <4 x float> %84, %75
  %86 = fmul float %5, %46
  %87 = fmul float %11, %44
  %88 = fadd float %87, %86
  %89 = fmul float %18, %31
  %90 = insertelement <4 x float> undef, float %5, i32 0
  %91 = insertelement <4 x float> %90, float %18, i32 1
  %92 = insertelement <4 x float> %91, float %27, i32 2
  %93 = insertelement <4 x float> %92, float %5, i32 3
  %94 = insertelement <4 x float> undef, float %54, i32 0
  %95 = insertelement <4 x float> %94, float %34, i32 1
  %96 = insertelement <4 x float> %95, float %46, i32 2
  %97 = insertelement <4 x float> %96, float %55, i32 3
  %98 = fmul <4 x float> %93, %97
  %99 = insertelement <4 x float> undef, float %8, i32 0
  %100 = insertelement <4 x float> %99, float %19, i32 1
  %101 = insertelement <4 x float> %100, float %29, i32 2
  %102 = insertelement <4 x float> %101, float %11, i32 3
  %103 = insertelement <4 x float> undef, float %53, i32 0
  %104 = insertelement <4 x float> %103, float %31, i32 1
  %105 = insertelement <4 x float> %104, float %44, i32 2
  %106 = insertelement <4 x float> %105, float %53, i32 3
  %107 = fmul <4 x float> %102, %106
  %108 = fadd <4 x float> %107, %98
  %109 = fmul float %18, %37
  %110 = fmul float %20, %31
  %111 = fadd float %110, %109
  %112 = insertelement <4 x float> undef, float %56, i32 0
  %113 = insertelement <4 x float> %112, float %64, i32 1
  %114 = insertelement <4 x float> %113, float %66, i32 2
  %115 = insertelement <4 x float> %114, float %59, i32 3
  %116 = insertelement <4 x float> undef, float %60, i32 0
  %117 = insertelement <4 x float> %116, float %65, i32 1
  %118 = insertelement <4 x float> %117, float %89, i32 2
  %119 = insertelement <4 x float> %118, float %63, i32 3
  %120 = fsub <4 x float> %115, %119
  %121 = fsub <4 x float> %85, %108
  %122 = fsub float %88, %111
  %123 = bitcast i8* %0 to <4 x float>*
  store <4 x float> %120, <4 x float>* %123, align 4, !tbaa !1
  %124 = getelementptr inbounds i8, i8* %0, i64 16
  %125 = bitcast i8* %124 to <4 x float>*
  store <4 x float> %121, <4 x float>* %125, align 4, !tbaa !1
  %126 = getelementptr inbounds i8, i8* %0, i64 32
  %127 = bitcast i8* %126 to float*
  store float %122, float* %127, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cross_dvdvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %2, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %2, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = bitcast i8* %1 to float*
  %13 = load float, float* %12, align 4, !tbaa !1, !noalias !27
  %14 = getelementptr inbounds i8, i8* %1, i64 12
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !1, !noalias !27
  %17 = getelementptr inbounds i8, i8* %1, i64 24
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !1, !noalias !27
  %20 = getelementptr inbounds i8, i8* %1, i64 4
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8, i8* %1, i64 16
  %23 = bitcast i8* %22 to float*
  %24 = getelementptr inbounds i8, i8* %1, i64 28
  %25 = bitcast i8* %24 to float*
  %26 = load float, float* %21, align 4, !tbaa !1, !noalias !27
  %27 = load float, float* %23, align 4, !tbaa !1, !noalias !27
  %28 = load float, float* %25, align 4, !tbaa !1, !noalias !27
  %29 = getelementptr inbounds i8, i8* %1, i64 8
  %30 = bitcast i8* %29 to float*
  %31 = getelementptr inbounds i8, i8* %1, i64 20
  %32 = bitcast i8* %31 to float*
  %33 = getelementptr inbounds i8, i8* %1, i64 32
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %30, align 4, !tbaa !1, !noalias !27
  %36 = load float, float* %32, align 4, !tbaa !1, !noalias !27
  %37 = load float, float* %34, align 4, !tbaa !1, !noalias !27
  %38 = fmul float %11, %26
  %39 = fmul float %26, 0.000000e+00
  %40 = fmul float %11, %27
  %41 = fadd float %39, %40
  %42 = fmul float %8, %35
  %43 = fmul float %35, 0.000000e+00
  %44 = fmul float %8, %36
  %45 = fadd float %43, %44
  %46 = fmul float %5, %35
  %47 = fmul float %11, %13
  %48 = fmul float %13, 0.000000e+00
  %49 = fmul float %8, %13
  %50 = insertelement <4 x float> undef, float %5, i32 0
  %51 = insertelement <4 x float> %50, float %8, i32 1
  %52 = insertelement <4 x float> %51, float %11, i32 2
  %53 = insertelement <4 x float> %52, float %5, i32 3
  %54 = insertelement <4 x float> undef, float %36, i32 0
  %55 = insertelement <4 x float> %54, float %16, i32 1
  %56 = insertelement <4 x float> %55, float %28, i32 2
  %57 = insertelement <4 x float> %56, float %37, i32 3
  %58 = fmul <4 x float> %53, %57
  %59 = insertelement <4 x float> undef, float %43, i32 0
  %60 = insertelement <4 x float> %59, float %48, i32 1
  %61 = insertelement <4 x float> %60, float %39, i32 2
  %62 = insertelement <4 x float> %61, float %43, i32 3
  %63 = fadd <4 x float> %62, %58
  %64 = fmul float %8, %19
  %65 = fadd float %48, %64
  %66 = fmul float %5, %26
  %67 = insertelement <4 x float> undef, float %11, i32 0
  %68 = insertelement <4 x float> %67, float %5, i32 1
  %69 = insertelement <4 x float> %68, float %8, i32 2
  %70 = insertelement <4 x float> %69, float %11, i32 3
  %71 = insertelement <4 x float> undef, float %16, i32 0
  %72 = insertelement <4 x float> %71, float %27, i32 1
  %73 = insertelement <4 x float> %72, float %37, i32 2
  %74 = insertelement <4 x float> %73, float %19, i32 3
  %75 = fmul <4 x float> %70, %74
  %76 = insertelement <4 x float> undef, float %48, i32 0
  %77 = insertelement <4 x float> %76, float %39, i32 1
  %78 = insertelement <4 x float> %77, float %43, i32 2
  %79 = insertelement <4 x float> %78, float %48, i32 3
  %80 = fadd <4 x float> %79, %75
  %81 = fmul float %5, %28
  %82 = fadd float %39, %81
  %83 = insertelement <4 x float> undef, float %38, i32 0
  %84 = insertelement <4 x float> %83, float %46, i32 1
  %85 = insertelement <4 x float> %84, float %49, i32 2
  %86 = insertelement <4 x float> %85, float %41, i32 3
  %87 = insertelement <4 x float> undef, float %42, i32 0
  %88 = insertelement <4 x float> %87, float %47, i32 1
  %89 = insertelement <4 x float> %88, float %66, i32 2
  %90 = insertelement <4 x float> %89, float %45, i32 3
  %91 = fsub <4 x float> %86, %90
  %92 = fsub <4 x float> %63, %80
  %93 = fsub float %65, %82
  %94 = bitcast i8* %0 to <4 x float>*
  store <4 x float> %91, <4 x float>* %94, align 4, !tbaa !1
  %95 = getelementptr inbounds i8, i8* %0, i64 16
  %96 = bitcast i8* %95 to <4 x float>*
  store <4 x float> %92, <4 x float>* %96, align 4, !tbaa !1
  %97 = getelementptr inbounds i8, i8* %0, i64 32
  %98 = bitcast i8* %97 to float*
  store float %93, float* %98, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_cross_dvvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %1, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = bitcast i8* %2 to float*
  %13 = load float, float* %12, align 4, !tbaa !1, !noalias !30
  %14 = getelementptr inbounds i8, i8* %2, i64 12
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !1, !noalias !30
  %17 = getelementptr inbounds i8, i8* %2, i64 24
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !1, !noalias !30
  %20 = getelementptr inbounds i8, i8* %2, i64 4
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8, i8* %2, i64 16
  %23 = bitcast i8* %22 to float*
  %24 = getelementptr inbounds i8, i8* %2, i64 28
  %25 = bitcast i8* %24 to float*
  %26 = load float, float* %21, align 4, !tbaa !1, !noalias !30
  %27 = load float, float* %23, align 4, !tbaa !1, !noalias !30
  %28 = load float, float* %25, align 4, !tbaa !1, !noalias !30
  %29 = getelementptr inbounds i8, i8* %2, i64 8
  %30 = bitcast i8* %29 to float*
  %31 = getelementptr inbounds i8, i8* %2, i64 20
  %32 = bitcast i8* %31 to float*
  %33 = getelementptr inbounds i8, i8* %2, i64 32
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %30, align 4, !tbaa !1, !noalias !30
  %36 = load float, float* %32, align 4, !tbaa !1, !noalias !30
  %37 = load float, float* %34, align 4, !tbaa !1, !noalias !30
  %38 = fmul float %8, %35
  %39 = fmul float %8, %36
  %40 = fmul float %35, 0.000000e+00
  %41 = fadd float %40, %39
  %42 = fmul float %11, %26
  %43 = fmul float %11, %27
  %44 = fmul float %26, 0.000000e+00
  %45 = fadd float %44, %43
  %46 = fmul float %11, %13
  %47 = fmul float %13, 0.000000e+00
  %48 = fmul float %5, %35
  %49 = fmul float %5, %26
  %50 = insertelement <4 x float> undef, float %11, i32 0
  %51 = insertelement <4 x float> %50, float %5, i32 1
  %52 = insertelement <4 x float> %51, float %8, i32 2
  %53 = insertelement <4 x float> %52, float %11, i32 3
  %54 = insertelement <4 x float> undef, float %16, i32 0
  %55 = insertelement <4 x float> %54, float %27, i32 1
  %56 = insertelement <4 x float> %55, float %37, i32 2
  %57 = insertelement <4 x float> %56, float %19, i32 3
  %58 = fmul <4 x float> %53, %57
  %59 = insertelement <4 x float> undef, float %47, i32 0
  %60 = insertelement <4 x float> %59, float %44, i32 1
  %61 = insertelement <4 x float> %60, float %40, i32 2
  %62 = insertelement <4 x float> %61, float %47, i32 3
  %63 = fadd <4 x float> %62, %58
  %64 = fmul float %5, %28
  %65 = fadd float %44, %64
  %66 = fmul float %8, %13
  %67 = insertelement <4 x float> undef, float %5, i32 0
  %68 = insertelement <4 x float> %67, float %8, i32 1
  %69 = insertelement <4 x float> %68, float %11, i32 2
  %70 = insertelement <4 x float> %69, float %5, i32 3
  %71 = insertelement <4 x float> undef, float %36, i32 0
  %72 = insertelement <4 x float> %71, float %16, i32 1
  %73 = insertelement <4 x float> %72, float %28, i32 2
  %74 = insertelement <4 x float> %73, float %37, i32 3
  %75 = fmul <4 x float> %70, %74
  %76 = insertelement <4 x float> undef, float %40, i32 0
  %77 = insertelement <4 x float> %76, float %47, i32 1
  %78 = insertelement <4 x float> %77, float %44, i32 2
  %79 = insertelement <4 x float> %78, float %40, i32 3
  %80 = fadd <4 x float> %79, %75
  %81 = fmul float %8, %19
  %82 = fadd float %47, %81
  %83 = insertelement <4 x float> undef, float %38, i32 0
  %84 = insertelement <4 x float> %83, float %46, i32 1
  %85 = insertelement <4 x float> %84, float %49, i32 2
  %86 = insertelement <4 x float> %85, float %41, i32 3
  %87 = insertelement <4 x float> undef, float %42, i32 0
  %88 = insertelement <4 x float> %87, float %48, i32 1
  %89 = insertelement <4 x float> %88, float %66, i32 2
  %90 = insertelement <4 x float> %89, float %45, i32 3
  %91 = fsub <4 x float> %86, %90
  %92 = fsub <4 x float> %63, %80
  %93 = fsub float %65, %82
  %94 = bitcast i8* %0 to <4 x float>*
  store <4 x float> %91, <4 x float>* %94, align 4, !tbaa !1
  %95 = getelementptr inbounds i8, i8* %0, i64 16
  %96 = bitcast i8* %95 to <4 x float>*
  store <4 x float> %92, <4 x float>* %96, align 4, !tbaa !1
  %97 = getelementptr inbounds i8, i8* %0, i64 32
  %98 = bitcast i8* %97 to float*
  store float %93, float* %98, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readonly uwtable
define float @osl_length_fv(i8* nocapture readonly) local_unnamed_addr #12 {
  %2 = bitcast i8* %0 to float*
  %3 = load float, float* %2, align 4, !tbaa !5
  %4 = fmul float %3, %3
  %5 = getelementptr inbounds i8, i8* %0, i64 4
  %6 = bitcast i8* %5 to float*
  %7 = load float, float* %6, align 4, !tbaa !7
  %8 = fmul float %7, %7
  %9 = fadd float %4, %8
  %10 = getelementptr inbounds i8, i8* %0, i64 8
  %11 = bitcast i8* %10 to float*
  %12 = load float, float* %11, align 4, !tbaa !8
  %13 = fmul float %12, %12
  %14 = fadd float %9, %13
  %15 = fcmp olt float %14, 0x3820000000000000
  br i1 %15, label %16, label %42

; <label>:16:                                     ; preds = %1
  %17 = fcmp oge float %3, 0.000000e+00
  %18 = fsub float -0.000000e+00, %3
  %19 = select i1 %17, float %3, float %18
  %20 = fcmp oge float %7, 0.000000e+00
  %21 = fsub float -0.000000e+00, %7
  %22 = select i1 %20, float %7, float %21
  %23 = fcmp oge float %12, 0.000000e+00
  %24 = fsub float -0.000000e+00, %12
  %25 = select i1 %23, float %12, float %24
  %26 = fcmp olt float %19, %22
  %27 = select i1 %26, float %22, float %19
  %28 = fcmp olt float %27, %25
  %29 = select i1 %28, float %25, float %27
  %30 = fcmp oeq float %29, 0.000000e+00
  br i1 %30, label %44, label %31

; <label>:31:                                     ; preds = %16
  %32 = fdiv float %19, %29
  %33 = fdiv float %22, %29
  %34 = fdiv float %25, %29
  %35 = fmul float %32, %32
  %36 = fmul float %33, %33
  %37 = fadd float %35, %36
  %38 = fmul float %34, %34
  %39 = fadd float %38, %37
  %40 = tail call float @sqrtf(float %39) #13
  %41 = fmul float %29, %40
  br label %44

; <label>:42:                                     ; preds = %1
  %43 = tail call float @sqrtf(float %14) #13
  br label %44

; <label>:44:                                     ; preds = %16, %31, %42
  %45 = phi float [ %43, %42 ], [ %41, %31 ], [ 0.000000e+00, %16 ]
  ret float %45
}

; Function Attrs: nounwind uwtable
define void @osl_length_dfdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = getelementptr inbounds i8, i8* %1, i64 12
  %6 = bitcast i8* %5 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = getelementptr inbounds i8, i8* %1, i64 24
  %9 = bitcast i8* %8 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = getelementptr inbounds i8, i8* %1, i64 4
  %12 = bitcast i8* %11 to float*
  %13 = getelementptr inbounds i8, i8* %1, i64 16
  %14 = bitcast i8* %13 to float*
  %15 = getelementptr inbounds i8, i8* %1, i64 28
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %12, align 4, !tbaa !1
  %18 = load float, float* %14, align 4, !tbaa !1
  %19 = load float, float* %16, align 4, !tbaa !1
  %20 = getelementptr inbounds i8, i8* %1, i64 8
  %21 = bitcast i8* %20 to float*
  %22 = getelementptr inbounds i8, i8* %1, i64 20
  %23 = bitcast i8* %22 to float*
  %24 = getelementptr inbounds i8, i8* %1, i64 32
  %25 = bitcast i8* %24 to float*
  %26 = load float, float* %21, align 4, !tbaa !1
  %27 = load float, float* %23, align 4, !tbaa !1
  %28 = load float, float* %25, align 4, !tbaa !1
  %29 = fmul float %4, %4
  %30 = fmul float %4, %7
  %31 = fmul float %4, %10
  %32 = fmul float %17, %17
  %33 = fmul float %17, %18
  %34 = fmul float %17, %19
  %35 = fadd float %29, %32
  %36 = fmul float %26, %26
  %37 = fmul float %26, %27
  %38 = fmul float %26, %28
  %39 = fadd float %35, %36
  %40 = fcmp ugt float %39, 0.000000e+00
  br i1 %40, label %41, label %59

; <label>:41:                                     ; preds = %2
  %42 = fadd float %30, %30
  %43 = fadd float %33, %33
  %44 = fadd float %42, %43
  %45 = fadd float %37, %37
  %46 = fadd float %44, %45
  %47 = fadd float %31, %31
  %48 = fadd float %34, %34
  %49 = fadd float %47, %48
  %50 = fadd float %38, %38
  %51 = fadd float %49, %50
  %52 = tail call float @sqrtf(float %39) #13
  %53 = fmul float %52, 2.000000e+00
  %54 = fdiv float 1.000000e+00, %53
  %55 = fmul float %46, %54
  %56 = fmul float %51, %54
  %57 = insertelement <2 x float> undef, float %52, i32 0
  %58 = insertelement <2 x float> %57, float %55, i32 1
  br label %59

; <label>:59:                                     ; preds = %2, %41
  %60 = phi <2 x float> [ %58, %41 ], [ zeroinitializer, %2 ]
  %61 = phi float [ %56, %41 ], [ 0.000000e+00, %2 ]
  %62 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %60, <2 x float>* %62, align 4
  %63 = getelementptr inbounds i8, i8* %0, i64 8
  %64 = bitcast i8* %63 to float*
  store float %61, float* %64, align 4
  ret void
}

; Function Attrs: nounwind readonly uwtable
define float @osl_distance_fvv(i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #12 {
  %3 = bitcast i8* %0 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = bitcast i8* %1 to float*
  %6 = load float, float* %5, align 4, !tbaa !1
  %7 = fsub float %4, %6
  %8 = getelementptr inbounds i8, i8* %0, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float, float* %9, align 4, !tbaa !1
  %11 = getelementptr inbounds i8, i8* %1, i64 4
  %12 = bitcast i8* %11 to float*
  %13 = load float, float* %12, align 4, !tbaa !1
  %14 = fsub float %10, %13
  %15 = getelementptr inbounds i8, i8* %0, i64 8
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %16, align 4, !tbaa !1
  %18 = getelementptr inbounds i8, i8* %1, i64 8
  %19 = bitcast i8* %18 to float*
  %20 = load float, float* %19, align 4, !tbaa !1
  %21 = fsub float %17, %20
  %22 = fmul float %7, %7
  %23 = fmul float %14, %14
  %24 = fadd float %22, %23
  %25 = fmul float %21, %21
  %26 = fadd float %24, %25
  %27 = tail call float @sqrtf(float %26) #13
  ret float %27
}

; Function Attrs: nounwind readnone
declare float @sqrtf(float) local_unnamed_addr #9

; Function Attrs: nounwind uwtable
define void @osl_distance_dfdvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !5, !noalias !33
  %6 = bitcast i8* %2 to float*
  %7 = load float, float* %6, align 4, !tbaa !5, !noalias !33
  %8 = fsub float %5, %7
  %9 = getelementptr inbounds i8, i8* %1, i64 4
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !7, !noalias !33
  %12 = getelementptr inbounds i8, i8* %2, i64 4
  %13 = bitcast i8* %12 to float*
  %14 = load float, float* %13, align 4, !tbaa !7, !noalias !33
  %15 = fsub float %11, %14
  %16 = getelementptr inbounds i8, i8* %1, i64 8
  %17 = bitcast i8* %16 to float*
  %18 = load float, float* %17, align 4, !tbaa !8, !noalias !33
  %19 = getelementptr inbounds i8, i8* %2, i64 8
  %20 = bitcast i8* %19 to float*
  %21 = load float, float* %20, align 4, !tbaa !8, !noalias !33
  %22 = fsub float %18, %21
  %23 = getelementptr inbounds i8, i8* %1, i64 12
  %24 = bitcast i8* %23 to float*
  %25 = load float, float* %24, align 4, !tbaa !5, !noalias !38
  %26 = getelementptr inbounds i8, i8* %2, i64 12
  %27 = bitcast i8* %26 to float*
  %28 = load float, float* %27, align 4, !tbaa !5, !noalias !38
  %29 = fsub float %25, %28
  %30 = getelementptr inbounds i8, i8* %1, i64 16
  %31 = bitcast i8* %30 to float*
  %32 = load float, float* %31, align 4, !tbaa !7, !noalias !38
  %33 = getelementptr inbounds i8, i8* %2, i64 16
  %34 = bitcast i8* %33 to float*
  %35 = load float, float* %34, align 4, !tbaa !7, !noalias !38
  %36 = fsub float %32, %35
  %37 = getelementptr inbounds i8, i8* %1, i64 20
  %38 = bitcast i8* %37 to float*
  %39 = load float, float* %38, align 4, !tbaa !8, !noalias !38
  %40 = getelementptr inbounds i8, i8* %2, i64 20
  %41 = bitcast i8* %40 to float*
  %42 = load float, float* %41, align 4, !tbaa !8, !noalias !38
  %43 = fsub float %39, %42
  %44 = getelementptr inbounds i8, i8* %1, i64 24
  %45 = bitcast i8* %44 to float*
  %46 = load float, float* %45, align 4, !tbaa !5, !noalias !41
  %47 = getelementptr inbounds i8, i8* %2, i64 24
  %48 = bitcast i8* %47 to float*
  %49 = load float, float* %48, align 4, !tbaa !5, !noalias !41
  %50 = fsub float %46, %49
  %51 = getelementptr inbounds i8, i8* %1, i64 28
  %52 = bitcast i8* %51 to float*
  %53 = load float, float* %52, align 4, !tbaa !7, !noalias !41
  %54 = getelementptr inbounds i8, i8* %2, i64 28
  %55 = bitcast i8* %54 to float*
  %56 = load float, float* %55, align 4, !tbaa !7, !noalias !41
  %57 = fsub float %53, %56
  %58 = getelementptr inbounds i8, i8* %1, i64 32
  %59 = bitcast i8* %58 to float*
  %60 = load float, float* %59, align 4, !tbaa !8, !noalias !41
  %61 = getelementptr inbounds i8, i8* %2, i64 32
  %62 = bitcast i8* %61 to float*
  %63 = load float, float* %62, align 4, !tbaa !8, !noalias !41
  %64 = fsub float %60, %63
  %65 = fmul float %8, %8
  %66 = fmul float %8, %29
  %67 = fmul float %8, %50
  %68 = fmul float %15, %15
  %69 = fmul float %15, %36
  %70 = fmul float %15, %57
  %71 = fadd float %65, %68
  %72 = fmul float %22, %22
  %73 = fmul float %22, %43
  %74 = fmul float %22, %64
  %75 = fadd float %71, %72
  %76 = fcmp ugt float %75, 0.000000e+00
  br i1 %76, label %77, label %95

; <label>:77:                                     ; preds = %3
  %78 = fadd float %66, %66
  %79 = fadd float %69, %69
  %80 = fadd float %78, %79
  %81 = fadd float %73, %73
  %82 = fadd float %80, %81
  %83 = fadd float %67, %67
  %84 = fadd float %70, %70
  %85 = fadd float %83, %84
  %86 = fadd float %74, %74
  %87 = fadd float %85, %86
  %88 = tail call float @sqrtf(float %75) #13
  %89 = fmul float %88, 2.000000e+00
  %90 = fdiv float 1.000000e+00, %89
  %91 = fmul float %82, %90
  %92 = fmul float %87, %90
  %93 = insertelement <2 x float> undef, float %88, i32 0
  %94 = insertelement <2 x float> %93, float %91, i32 1
  br label %95

; <label>:95:                                     ; preds = %3, %77
  %96 = phi <2 x float> [ %94, %77 ], [ zeroinitializer, %3 ]
  %97 = phi float [ %92, %77 ], [ 0.000000e+00, %3 ]
  %98 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %96, <2 x float>* %98, align 4
  %99 = getelementptr inbounds i8, i8* %0, i64 8
  %100 = bitcast i8* %99 to float*
  store float %97, float* %100, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_distance_dfdvv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %2 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %2, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %2, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = bitcast i8* %1 to float*
  %13 = load float, float* %12, align 4, !tbaa !5, !noalias !44
  %14 = fsub float %13, %5
  %15 = getelementptr inbounds i8, i8* %1, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %16, align 4, !tbaa !7, !noalias !44
  %18 = fsub float %17, %8
  %19 = getelementptr inbounds i8, i8* %1, i64 8
  %20 = bitcast i8* %19 to float*
  %21 = load float, float* %20, align 4, !tbaa !8, !noalias !44
  %22 = fsub float %21, %11
  %23 = getelementptr inbounds i8, i8* %1, i64 12
  %24 = bitcast i8* %23 to float*
  %25 = load float, float* %24, align 4, !tbaa !5, !noalias !49
  %26 = getelementptr inbounds i8, i8* %1, i64 16
  %27 = bitcast i8* %26 to float*
  %28 = load float, float* %27, align 4, !tbaa !7, !noalias !49
  %29 = getelementptr inbounds i8, i8* %1, i64 20
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !8, !noalias !49
  %32 = getelementptr inbounds i8, i8* %1, i64 24
  %33 = bitcast i8* %32 to float*
  %34 = load float, float* %33, align 4, !tbaa !5, !noalias !52
  %35 = getelementptr inbounds i8, i8* %1, i64 28
  %36 = bitcast i8* %35 to float*
  %37 = load float, float* %36, align 4, !tbaa !7, !noalias !52
  %38 = getelementptr inbounds i8, i8* %1, i64 32
  %39 = bitcast i8* %38 to float*
  %40 = load float, float* %39, align 4, !tbaa !8, !noalias !52
  %41 = fmul float %14, %14
  %42 = fmul float %14, %25
  %43 = fmul float %14, %34
  %44 = fmul float %18, %18
  %45 = fmul float %18, %28
  %46 = fmul float %18, %37
  %47 = fadd float %41, %44
  %48 = fmul float %22, %22
  %49 = fmul float %22, %31
  %50 = fmul float %22, %40
  %51 = fadd float %47, %48
  %52 = fcmp ugt float %51, 0.000000e+00
  br i1 %52, label %53, label %71

; <label>:53:                                     ; preds = %3
  %54 = fadd float %42, %42
  %55 = fadd float %45, %45
  %56 = fadd float %54, %55
  %57 = fadd float %49, %49
  %58 = fadd float %56, %57
  %59 = fadd float %43, %43
  %60 = fadd float %46, %46
  %61 = fadd float %59, %60
  %62 = fadd float %50, %50
  %63 = fadd float %61, %62
  %64 = tail call float @sqrtf(float %51) #13
  %65 = fmul float %64, 2.000000e+00
  %66 = fdiv float 1.000000e+00, %65
  %67 = fmul float %58, %66
  %68 = fmul float %63, %66
  %69 = insertelement <2 x float> undef, float %64, i32 0
  %70 = insertelement <2 x float> %69, float %67, i32 1
  br label %71

; <label>:71:                                     ; preds = %3, %53
  %72 = phi <2 x float> [ %70, %53 ], [ zeroinitializer, %3 ]
  %73 = phi float [ %68, %53 ], [ 0.000000e+00, %3 ]
  %74 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %72, <2 x float>* %74, align 4
  %75 = getelementptr inbounds i8, i8* %0, i64 8
  %76 = bitcast i8* %75 to float*
  store float %73, float* %76, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_distance_dfvdv(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = bitcast i8* %1 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7
  %9 = getelementptr inbounds i8, i8* %1, i64 8
  %10 = bitcast i8* %9 to float*
  %11 = load float, float* %10, align 4, !tbaa !8
  %12 = bitcast i8* %2 to float*
  %13 = load float, float* %12, align 4, !tbaa !5, !noalias !55
  %14 = fsub float %5, %13
  %15 = getelementptr inbounds i8, i8* %2, i64 4
  %16 = bitcast i8* %15 to float*
  %17 = load float, float* %16, align 4, !tbaa !7, !noalias !55
  %18 = fsub float %8, %17
  %19 = getelementptr inbounds i8, i8* %2, i64 8
  %20 = bitcast i8* %19 to float*
  %21 = load float, float* %20, align 4, !tbaa !8, !noalias !55
  %22 = fsub float %11, %21
  %23 = getelementptr inbounds i8, i8* %2, i64 12
  %24 = bitcast i8* %23 to float*
  %25 = load float, float* %24, align 4, !tbaa !5, !noalias !60
  %26 = fsub float 0.000000e+00, %25
  %27 = getelementptr inbounds i8, i8* %2, i64 16
  %28 = bitcast i8* %27 to float*
  %29 = load float, float* %28, align 4, !tbaa !7, !noalias !60
  %30 = fsub float 0.000000e+00, %29
  %31 = getelementptr inbounds i8, i8* %2, i64 20
  %32 = bitcast i8* %31 to float*
  %33 = load float, float* %32, align 4, !tbaa !8, !noalias !60
  %34 = fsub float 0.000000e+00, %33
  %35 = getelementptr inbounds i8, i8* %2, i64 24
  %36 = bitcast i8* %35 to float*
  %37 = load float, float* %36, align 4, !tbaa !5, !noalias !63
  %38 = fsub float 0.000000e+00, %37
  %39 = getelementptr inbounds i8, i8* %2, i64 28
  %40 = bitcast i8* %39 to float*
  %41 = load float, float* %40, align 4, !tbaa !7, !noalias !63
  %42 = fsub float 0.000000e+00, %41
  %43 = getelementptr inbounds i8, i8* %2, i64 32
  %44 = bitcast i8* %43 to float*
  %45 = load float, float* %44, align 4, !tbaa !8, !noalias !63
  %46 = fsub float 0.000000e+00, %45
  %47 = fmul float %14, %14
  %48 = fmul float %14, %26
  %49 = fmul float %14, %38
  %50 = fmul float %18, %18
  %51 = fmul float %18, %30
  %52 = fmul float %18, %42
  %53 = fadd float %47, %50
  %54 = fmul float %22, %22
  %55 = fmul float %22, %34
  %56 = fmul float %22, %46
  %57 = fadd float %53, %54
  %58 = fcmp ugt float %57, 0.000000e+00
  br i1 %58, label %59, label %77

; <label>:59:                                     ; preds = %3
  %60 = fadd float %48, %48
  %61 = fadd float %51, %51
  %62 = fadd float %60, %61
  %63 = fadd float %55, %55
  %64 = fadd float %62, %63
  %65 = fadd float %49, %49
  %66 = fadd float %52, %52
  %67 = fadd float %65, %66
  %68 = fadd float %56, %56
  %69 = fadd float %67, %68
  %70 = tail call float @sqrtf(float %57) #13
  %71 = fmul float %70, 2.000000e+00
  %72 = fdiv float 1.000000e+00, %71
  %73 = fmul float %64, %72
  %74 = fmul float %69, %72
  %75 = insertelement <2 x float> undef, float %70, i32 0
  %76 = insertelement <2 x float> %75, float %73, i32 1
  br label %77

; <label>:77:                                     ; preds = %3, %59
  %78 = phi <2 x float> [ %76, %59 ], [ zeroinitializer, %3 ]
  %79 = phi float [ %74, %59 ], [ 0.000000e+00, %3 ]
  %80 = bitcast i8* %0 to <2 x float>*
  store <2 x float> %78, <2 x float>* %80, align 4
  %81 = getelementptr inbounds i8, i8* %0, i64 8
  %82 = bitcast i8* %81 to float*
  store float %79, float* %82, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_normalize_vv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = bitcast i8* %1 to float*
  %4 = load float, float* %3, align 4, !tbaa !5, !noalias !66
  %5 = fmul float %4, %4
  %6 = getelementptr inbounds i8, i8* %1, i64 4
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !7, !noalias !66
  %9 = fmul float %8, %8
  %10 = fadd float %5, %9
  %11 = getelementptr inbounds i8, i8* %1, i64 8
  %12 = bitcast i8* %11 to float*
  %13 = load float, float* %12, align 4, !tbaa !8, !noalias !66
  %14 = fmul float %13, %13
  %15 = fadd float %10, %14
  %16 = fcmp olt float %15, 0x3820000000000000
  br i1 %16, label %17, label %43

; <label>:17:                                     ; preds = %2
  %18 = fcmp oge float %4, 0.000000e+00
  %19 = fsub float -0.000000e+00, %4
  %20 = select i1 %18, float %4, float %19
  %21 = fcmp oge float %8, 0.000000e+00
  %22 = fsub float -0.000000e+00, %8
  %23 = select i1 %21, float %8, float %22
  %24 = fcmp oge float %13, 0.000000e+00
  %25 = fsub float -0.000000e+00, %13
  %26 = select i1 %24, float %13, float %25
  %27 = fcmp olt float %20, %23
  %28 = select i1 %27, float %23, float %20
  %29 = fcmp olt float %28, %26
  %30 = select i1 %29, float %26, float %28
  %31 = fcmp oeq float %30, 0.000000e+00
  br i1 %31, label %55, label %32

; <label>:32:                                     ; preds = %17
  %33 = fdiv float %20, %30
  %34 = fdiv float %23, %30
  %35 = fdiv float %26, %30
  %36 = fmul float %33, %33
  %37 = fmul float %34, %34
  %38 = fadd float %36, %37
  %39 = fmul float %35, %35
  %40 = fadd float %39, %38
  %41 = tail call float @sqrtf(float %40) #13
  %42 = fmul float %30, %41
  br label %45

; <label>:43:                                     ; preds = %2
  %44 = tail call float @sqrtf(float %15) #13
  br label %45

; <label>:45:                                     ; preds = %43, %32
  %46 = phi float [ %44, %43 ], [ %42, %32 ]
  %47 = fcmp oeq float %46, 0.000000e+00
  br i1 %47, label %55, label %48

; <label>:48:                                     ; preds = %45
  %49 = fdiv float %4, %46
  %50 = fdiv float %8, %46
  %51 = fdiv float %13, %46
  %52 = bitcast float %49 to i32
  %53 = bitcast float %50 to i32
  %54 = bitcast float %51 to i32
  br label %55

; <label>:55:                                     ; preds = %17, %45, %48
  %56 = phi i32 [ %54, %48 ], [ 0, %45 ], [ 0, %17 ]
  %57 = phi i32 [ %53, %48 ], [ 0, %45 ], [ 0, %17 ]
  %58 = phi i32 [ %52, %48 ], [ 0, %45 ], [ 0, %17 ]
  %59 = bitcast i8* %0 to i32*
  store i32 %58, i32* %59, align 4, !tbaa !5
  %60 = getelementptr inbounds i8, i8* %0, i64 4
  %61 = bitcast i8* %60 to i32*
  store i32 %57, i32* %61, align 4, !tbaa !7
  %62 = getelementptr inbounds i8, i8* %0, i64 8
  %63 = bitcast i8* %62 to i32*
  store i32 %56, i32* %63, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_normalize_dvdv(i8* nocapture, i8*) local_unnamed_addr #4 {
  %3 = alloca %"class.OSL::Dual2.0", align 16
  %4 = bitcast %"class.OSL::Dual2.0"* %3 to i8*
  call void @llvm.lifetime.start(i64 36, i8* %4) #2
  %5 = bitcast i8* %1 to %"class.OSL::Dual2.0"*
  call void @_ZN3OSL9normalizeERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE(%"class.OSL::Dual2.0"* nonnull sret %3, %"class.OSL::Dual2.0"* dereferenceable(36) %5)
  %6 = bitcast %"class.OSL::Dual2.0"* %3 to <4 x i32>*
  %7 = load <4 x i32>, <4 x i32>* %6, align 16, !tbaa !1
  %8 = bitcast i8* %0 to <4 x i32>*
  store <4 x i32> %7, <4 x i32>* %8, align 4, !tbaa !1
  %9 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %3, i64 0, i32 1, i32 1
  %10 = getelementptr inbounds i8, i8* %0, i64 16
  %11 = bitcast float* %9 to <4 x i32>*
  %12 = load <4 x i32>, <4 x i32>* %11, align 4, !tbaa !1
  %13 = bitcast i8* %10 to <4 x i32>*
  store <4 x i32> %12, <4 x i32>* %13, align 4, !tbaa !1
  %14 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %3, i64 0, i32 2, i32 2
  %15 = bitcast float* %14 to i32*
  %16 = load i32, i32* %15, align 8, !tbaa !8
  %17 = getelementptr inbounds i8, i8* %0, i64 32
  %18 = bitcast i8* %17 to i32*
  store i32 %16, i32* %18, align 4, !tbaa !8
  call void @llvm.lifetime.end(i64 36, i8* %4) #2
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN3OSL9normalizeERKNS_5Dual2IN9Imath_2_24Vec3IfEEEE(%"class.OSL::Dual2.0"* noalias sret, %"class.OSL::Dual2.0"* dereferenceable(36)) local_unnamed_addr #8 comdat {
  %3 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 0, i32 0
  %4 = load float, float* %3, align 4, !tbaa !5
  %5 = fcmp oeq float %4, 0.000000e+00
  %6 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 0, i32 1
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fcmp oeq float %7, 0.000000e+00
  %9 = and i1 %5, %8
  br i1 %9, label %10, label %16

; <label>:10:                                     ; preds = %2
  %11 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 0, i32 2
  %12 = load float, float* %11, align 4, !tbaa !8
  %13 = fcmp oeq float %12, 0.000000e+00
  br i1 %13, label %14, label %16

; <label>:14:                                     ; preds = %10
  %15 = bitcast %"class.OSL::Dual2.0"* %0 to i8*
  call void @llvm.memset.p0i8.i64(i8* %15, i8 0, i64 36, i32 4, i1 false)
  br label %103

; <label>:16:                                     ; preds = %2, %10
  %17 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 1, i32 0
  %18 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 2, i32 1
  %19 = load float, float* %18, align 4, !tbaa !1
  %20 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 0, i32 2
  %21 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %1, i64 0, i32 2, i32 2
  %22 = load float, float* %20, align 4, !tbaa !1
  %23 = bitcast float* %17 to <4 x float>*
  %24 = load <4 x float>, <4 x float>* %23, align 4, !tbaa !1
  %25 = load float, float* %21, align 4, !tbaa !1
  %26 = fmul float %4, %4
  %27 = extractelement <4 x float> %24, i32 0
  %28 = fmul float %4, %27
  %29 = extractelement <4 x float> %24, i32 3
  %30 = fmul float %4, %29
  %31 = fmul float %7, %7
  %32 = extractelement <4 x float> %24, i32 1
  %33 = fmul float %7, %32
  %34 = fmul float %7, %19
  %35 = fadd float %26, %31
  %36 = fmul float %22, %22
  %37 = extractelement <4 x float> %24, i32 2
  %38 = fmul float %22, %37
  %39 = fmul float %22, %25
  %40 = fadd float %35, %36
  %41 = fcmp ugt float %40, 0.000000e+00
  br i1 %41, label %42, label %60

; <label>:42:                                     ; preds = %16
  %43 = fadd float %28, %28
  %44 = fadd float %33, %33
  %45 = fadd float %43, %44
  %46 = fadd float %38, %38
  %47 = fadd float %45, %46
  %48 = fadd float %30, %30
  %49 = fadd float %34, %34
  %50 = fadd float %48, %49
  %51 = fadd float %39, %39
  %52 = fadd float %50, %51
  %53 = tail call float @sqrtf(float %40) #13
  %54 = fmul float %53, 2.000000e+00
  %55 = fdiv float 1.000000e+00, %54
  %56 = fmul float %47, %55
  %57 = fmul float %52, %55
  %58 = insertelement <2 x float> undef, float %53, i32 0
  %59 = insertelement <2 x float> %58, float %56, i32 1
  br label %60

; <label>:60:                                     ; preds = %16, %42
  %61 = phi <2 x float> [ %59, %42 ], [ zeroinitializer, %16 ]
  %62 = phi float [ %57, %42 ], [ 0.000000e+00, %16 ]
  %63 = extractelement <2 x float> %61, i32 0
  %64 = fdiv float 1.000000e+00, %63
  %65 = extractelement <2 x float> %61, i32 1
  %66 = fmul float %65, %64
  %67 = fmul float %64, %66
  %68 = fsub float -0.000000e+00, %67
  %69 = fmul float %62, %64
  %70 = fmul float %64, %69
  %71 = fsub float -0.000000e+00, %70
  %72 = fmul float %4, %64
  %73 = fmul float %7, %64
  %74 = fmul float %7, %71
  %75 = fmul float %19, %64
  %76 = fadd float %75, %74
  %77 = fmul float %22, %64
  %78 = insertelement <4 x float> undef, float %4, i32 0
  %79 = insertelement <4 x float> %78, float %7, i32 1
  %80 = insertelement <4 x float> %79, float %22, i32 2
  %81 = insertelement <4 x float> %80, float %4, i32 3
  %82 = insertelement <4 x float> undef, float %68, i32 0
  %83 = insertelement <4 x float> %82, float %68, i32 1
  %84 = insertelement <4 x float> %83, float %68, i32 2
  %85 = insertelement <4 x float> %84, float %71, i32 3
  %86 = fmul <4 x float> %81, %85
  %87 = insertelement <4 x float> undef, float %64, i32 0
  %88 = insertelement <4 x float> %87, float %64, i32 1
  %89 = insertelement <4 x float> %88, float %64, i32 2
  %90 = insertelement <4 x float> %89, float %64, i32 3
  %91 = fmul <4 x float> %24, %90
  %92 = fadd <4 x float> %91, %86
  %93 = fmul float %22, %71
  %94 = fmul float %25, %64
  %95 = fadd float %94, %93
  %96 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %0, i64 0, i32 0, i32 0
  store float %72, float* %96, align 4, !tbaa !5
  %97 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %0, i64 0, i32 0, i32 1
  store float %73, float* %97, align 4, !tbaa !7
  %98 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %0, i64 0, i32 0, i32 2
  store float %77, float* %98, align 4, !tbaa !8
  %99 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %0, i64 0, i32 1, i32 0
  %100 = bitcast float* %99 to <4 x float>*
  store <4 x float> %92, <4 x float>* %100, align 4, !tbaa !1
  %101 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %0, i64 0, i32 2, i32 1
  store float %76, float* %101, align 4, !tbaa !7
  %102 = getelementptr inbounds %"class.OSL::Dual2.0", %"class.OSL::Dual2.0"* %0, i64 0, i32 2, i32 2
  store float %95, float* %102, align 4, !tbaa !8
  br label %103

; <label>:103:                                    ; preds = %60, %14
  ret void
}

; Function Attrs: nounwind uwtable
define void @osl_calculatenormal(i8* nocapture, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #4 {
  %4 = getelementptr inbounds i8, i8* %1, i64 288
  %5 = bitcast i8* %4 to i32*
  %6 = load i32, i32* %5, align 8, !tbaa !69
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %36, label %8

; <label>:8:                                      ; preds = %3
  %9 = getelementptr inbounds i8, i8* %2, i64 24
  %10 = getelementptr inbounds i8, i8* %2, i64 12
  %11 = getelementptr inbounds i8, i8* %2, i64 28
  %12 = bitcast i8* %11 to float*
  %13 = load float, float* %12, align 4, !tbaa !7, !noalias !73
  %14 = getelementptr inbounds i8, i8* %2, i64 20
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !8, !noalias !73
  %17 = fmul float %13, %16
  %18 = getelementptr inbounds i8, i8* %2, i64 32
  %19 = bitcast i8* %18 to float*
  %20 = load float, float* %19, align 4, !tbaa !8, !noalias !73
  %21 = getelementptr inbounds i8, i8* %2, i64 16
  %22 = bitcast i8* %21 to float*
  %23 = load float, float* %22, align 4, !tbaa !7, !noalias !73
  %24 = fmul float %20, %23
  %25 = fsub float %17, %24
  %26 = bitcast i8* %10 to float*
  %27 = load float, float* %26, align 4, !tbaa !5, !noalias !73
  %28 = fmul float %20, %27
  %29 = bitcast i8* %9 to float*
  %30 = load float, float* %29, align 4, !tbaa !5, !noalias !73
  %31 = fmul float %16, %30
  %32 = fsub float %28, %31
  %33 = fmul float %23, %30
  %34 = fmul float %13, %27
  %35 = fsub float %33, %34
  br label %64

; <label>:36:                                     ; preds = %3
  %37 = getelementptr inbounds i8, i8* %2, i64 12
  %38 = getelementptr inbounds i8, i8* %2, i64 24
  %39 = getelementptr inbounds i8, i8* %2, i64 16
  %40 = bitcast i8* %39 to float*
  %41 = load float, float* %40, align 4, !tbaa !7, !noalias !78
  %42 = getelementptr inbounds i8, i8* %2, i64 32
  %43 = bitcast i8* %42 to float*
  %44 = load float, float* %43, align 4, !tbaa !8, !noalias !78
  %45 = fmul float %41, %44
  %46 = getelementptr inbounds i8, i8* %2, i64 20
  %47 = bitcast i8* %46 to float*
  %48 = load float, float* %47, align 4, !tbaa !8, !noalias !78
  %49 = getelementptr inbounds i8, i8* %2, i64 28
  %50 = bitcast i8* %49 to float*
  %51 = load float, float* %50, align 4, !tbaa !7, !noalias !78
  %52 = fmul float %48, %51
  %53 = fsub float %45, %52
  %54 = bitcast i8* %38 to float*
  %55 = load float, float* %54, align 4, !tbaa !5, !noalias !78
  %56 = fmul float %48, %55
  %57 = bitcast i8* %37 to float*
  %58 = load float, float* %57, align 4, !tbaa !5, !noalias !78
  %59 = fmul float %44, %58
  %60 = fsub float %56, %59
  %61 = fmul float %51, %58
  %62 = fmul float %41, %55
  %63 = fsub float %61, %62
  br label %64

; <label>:64:                                     ; preds = %8, %36
  %65 = phi float [ %53, %36 ], [ %25, %8 ]
  %66 = phi float [ %60, %36 ], [ %32, %8 ]
  %67 = phi float [ %63, %36 ], [ %35, %8 ]
  %68 = bitcast i8* %0 to float*
  store float %65, float* %68, align 4, !tbaa !5
  %69 = getelementptr inbounds i8, i8* %0, i64 4
  %70 = bitcast i8* %69 to float*
  store float %66, float* %70, align 4, !tbaa !7
  %71 = getelementptr inbounds i8, i8* %0, i64 8
  %72 = bitcast i8* %71 to float*
  store float %67, float* %72, align 4, !tbaa !8
  ret void
}

; Function Attrs: nounwind readonly uwtable
define float @osl_area(i8* nocapture readonly) local_unnamed_addr #12 {
  %2 = getelementptr inbounds i8, i8* %0, i64 12
  %3 = getelementptr inbounds i8, i8* %0, i64 24
  %4 = getelementptr inbounds i8, i8* %0, i64 16
  %5 = bitcast i8* %4 to float*
  %6 = load float, float* %5, align 4, !tbaa !7, !noalias !81
  %7 = getelementptr inbounds i8, i8* %0, i64 32
  %8 = bitcast i8* %7 to float*
  %9 = load float, float* %8, align 4, !tbaa !8, !noalias !81
  %10 = fmul float %6, %9
  %11 = getelementptr inbounds i8, i8* %0, i64 20
  %12 = bitcast i8* %11 to float*
  %13 = load float, float* %12, align 4, !tbaa !8, !noalias !81
  %14 = getelementptr inbounds i8, i8* %0, i64 28
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !7, !noalias !81
  %17 = fmul float %13, %16
  %18 = fsub float %10, %17
  %19 = bitcast i8* %3 to float*
  %20 = load float, float* %19, align 4, !tbaa !5, !noalias !81
  %21 = fmul float %13, %20
  %22 = bitcast i8* %2 to float*
  %23 = load float, float* %22, align 4, !tbaa !5, !noalias !81
  %24 = fmul float %9, %23
  %25 = fsub float %21, %24
  %26 = fmul float %16, %23
  %27 = fmul float %6, %20
  %28 = fsub float %26, %27
  %29 = fmul float %18, %18
  %30 = fmul float %25, %25
  %31 = fadd float %29, %30
  %32 = fmul float %28, %28
  %33 = fadd float %32, %31
  %34 = fcmp olt float %33, 0x3820000000000000
  br i1 %34, label %35, label %61

; <label>:35:                                     ; preds = %1
  %36 = fcmp oge float %18, 0.000000e+00
  %37 = fsub float -0.000000e+00, %18
  %38 = select i1 %36, float %18, float %37
  %39 = fcmp oge float %25, 0.000000e+00
  %40 = fsub float -0.000000e+00, %25
  %41 = select i1 %39, float %25, float %40
  %42 = fcmp oge float %28, 0.000000e+00
  %43 = fsub float -0.000000e+00, %28
  %44 = select i1 %42, float %28, float %43
  %45 = fcmp olt float %38, %41
  %46 = select i1 %45, float %41, float %38
  %47 = fcmp olt float %46, %44
  %48 = select i1 %47, float %44, float %46
  %49 = fcmp oeq float %48, 0.000000e+00
  br i1 %49, label %63, label %50

; <label>:50:                                     ; preds = %35
  %51 = fdiv float %38, %48
  %52 = fdiv float %41, %48
  %53 = fdiv float %44, %48
  %54 = fmul float %51, %51
  %55 = fmul float %52, %52
  %56 = fadd float %54, %55
  %57 = fmul float %53, %53
  %58 = fadd float %57, %56
  %59 = tail call float @sqrtf(float %58) #13
  %60 = fmul float %48, %59
  br label %63

; <label>:61:                                     ; preds = %1
  %62 = tail call float @sqrtf(float %33) #13
  br label %63

; <label>:63:                                     ; preds = %35, %50, %61
  %64 = phi float [ %62, %61 ], [ %60, %50 ], [ 0.000000e+00, %35 ]
  ret float %64
}

; Function Attrs: nounwind readonly uwtable
define float @osl_filterwidth_fdf(i8* nocapture readonly) local_unnamed_addr #12 {
  %2 = getelementptr inbounds i8, i8* %0, i64 4
  %3 = bitcast i8* %2 to float*
  %4 = load float, float* %3, align 4, !tbaa !1
  %5 = getelementptr inbounds i8, i8* %0, i64 8
  %6 = bitcast i8* %5 to float*
  %7 = load float, float* %6, align 4, !tbaa !1
  %8 = fmul float %4, %4
  %9 = fmul float %7, %7
  %10 = fadd float %8, %9
  %11 = tail call float @sqrtf(float %10) #13
  ret float %11
}

; Function Attrs: nounwind uwtable
define void @osl_filterwidth_vdv(i8* nocapture, i8* nocapture readonly) local_unnamed_addr #4 {
  %3 = getelementptr inbounds i8, i8* %1, i64 12
  %4 = bitcast i8* %3 to float*
  %5 = load float, float* %4, align 4, !tbaa !5
  %6 = getelementptr inbounds i8, i8* %1, i64 24
  %7 = bitcast i8* %6 to float*
  %8 = load float, float* %7, align 4, !tbaa !5
  %9 = fmul float %5, %5
  %10 = fmul float %8, %8
  %11 = fadd float %9, %10
  %12 = tail call float @sqrtf(float %11) #13
  %13 = bitcast i8* %0 to float*
  store float %12, float* %13, align 4, !tbaa !5
  %14 = getelementptr inbounds i8, i8* %1, i64 16
  %15 = bitcast i8* %14 to float*
  %16 = load float, float* %15, align 4, !tbaa !7
  %17 = getelementptr inbounds i8, i8* %1, i64 28
  %18 = bitcast i8* %17 to float*
  %19 = load float, float* %18, align 4, !tbaa !7
  %20 = fmul float %16, %16
  %21 = fmul float %19, %19
  %22 = fadd float %20, %21
  %23 = tail call float @sqrtf(float %22) #13
  %24 = getelementptr inbounds i8, i8* %0, i64 4
  %25 = bitcast i8* %24 to float*
  store float %23, float* %25, align 4, !tbaa !7
  %26 = getelementptr inbounds i8, i8* %1, i64 20
  %27 = bitcast i8* %26 to float*
  %28 = load float, float* %27, align 4, !tbaa !8
  %29 = getelementptr inbounds i8, i8* %1, i64 32
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4, !tbaa !8
  %32 = fmul float %28, %28
  %33 = fmul float %31, %31
  %34 = fadd float %32, %33
  %35 = tail call float @sqrtf(float %34) #13
  %36 = getelementptr inbounds i8, i8* %0, i64 8
  %37 = bitcast i8* %36 to float*
  store float %35, float* %37, align 4, !tbaa !8
  ret void
}

; Function Attrs: norecurse nounwind readonly uwtable
define i32 @osl_raytype_bit(i8* nocapture readonly, i32) local_unnamed_addr #10 {
  %3 = getelementptr inbounds i8, i8* %0, i64 284
  %4 = bitcast i8* %3 to i32*
  %5 = load i32, i32* %4, align 4, !tbaa !86
  %6 = and i32 %5, %1
  %7 = icmp ne i32 %6, 0
  %8 = zext i1 %7 to i32
  ret i32 %8
}

; Function Attrs: nounwind readnone
declare float @copysignf(float, float) local_unnamed_addr #9

; Function Attrs: nounwind readnone
declare float @llvm.fabs.f32(float) #13

; Function Attrs: nounwind readnone
declare float @erff(float) local_unnamed_addr #9

; Function Attrs: nounwind readnone
declare float @expf(float) local_unnamed_addr #9

; Function Attrs: nounwind readnone
declare float @erfcf(float) local_unnamed_addr #9

; Function Attrs: uwtable
define linkonce_odr void @_ZNK9Imath_2_28Matrix44IfE9gjInverseEb(%"class.Imath_2_2::Matrix44"* noalias sret, %"class.Imath_2_2::Matrix44"*, i1 zeroext) local_unnamed_addr #11 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = alloca %"class.Imath_2_2::Matrix44", align 16
  %5 = alloca %"class.Imath_2_2::Matrix44", align 16
  %6 = bitcast %"class.Imath_2_2::Matrix44"* %4 to i8*
  call void @llvm.lifetime.start(i64 64, i8* %6) #2
  %7 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 1
  %8 = bitcast float* %7 to i8*
  call void @llvm.memset.p0i8.i64(i8* %8, i8 0, i64 56, i32 4, i1 false) #2
  %9 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 0, i64 0
  store float 1.000000e+00, float* %9, align 16, !tbaa !1
  %10 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1, i64 1
  store float 1.000000e+00, float* %10, align 4, !tbaa !1
  %11 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2, i64 2
  store float 1.000000e+00, float* %11, align 8, !tbaa !1
  %12 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 3, i64 3
  store float 1.000000e+00, float* %12, align 4, !tbaa !1
  %13 = bitcast %"class.Imath_2_2::Matrix44"* %5 to i8*
  call void @llvm.lifetime.start(i64 64, i8* %13) #2
  %14 = bitcast %"class.Imath_2_2::Matrix44"* %1 to <4 x i32>*
  %15 = load <4 x i32>, <4 x i32>* %14, align 4, !tbaa !1
  %16 = bitcast %"class.Imath_2_2::Matrix44"* %5 to <4 x i32>*
  store <4 x i32> %15, <4 x i32>* %16, align 16, !tbaa !1
  %17 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 1
  %18 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 1
  %19 = bitcast [4 x float]* %17 to <4 x i32>*
  %20 = load <4 x i32>, <4 x i32>* %19, align 4, !tbaa !1
  %21 = bitcast [4 x float]* %18 to <4 x i32>*
  store <4 x i32> %20, <4 x i32>* %21, align 16, !tbaa !1
  %22 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 2
  %23 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 2
  %24 = bitcast [4 x float]* %22 to <4 x i32>*
  %25 = load <4 x i32>, <4 x i32>* %24, align 4, !tbaa !1
  %26 = bitcast [4 x float]* %23 to <4 x i32>*
  store <4 x i32> %25, <4 x i32>* %26, align 16, !tbaa !1
  %27 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %1, i64 0, i32 0, i64 3
  %28 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 3
  %29 = bitcast [4 x float]* %27 to <4 x i32>*
  %30 = load <4 x i32>, <4 x i32>* %29, align 4, !tbaa !1
  %31 = bitcast [4 x float]* %28 to <4 x i32>*
  store <4 x i32> %30, <4 x i32>* %31, align 16, !tbaa !1
  br label %36

; <label>:32:                                     ; preds = %145
  %33 = icmp slt i64 %45, 3
  %34 = add nuw nsw i64 %38, 1
  br i1 %33, label %36, label %35

; <label>:35:                                     ; preds = %32
  br label %172

; <label>:36:                                     ; preds = %32, %3
  %37 = phi i64 [ 0, %3 ], [ %45, %32 ]
  %38 = phi i64 [ 1, %3 ], [ %34, %32 ]
  %39 = sub i64 3, %37
  %40 = trunc i64 %39 to i32
  %41 = sub i64 2, %37
  %42 = trunc i64 %41 to i32
  %43 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %37, i64 %37
  %44 = load float, float* %43, align 4, !tbaa !1
  %45 = add nuw nsw i64 %37, 1
  %46 = trunc i64 %37 to i32
  %47 = fcmp olt float %44, 0.000000e+00
  %48 = fsub float -0.000000e+00, %44
  %49 = select i1 %47, float %48, float %44
  %50 = and i32 %40, 1
  %51 = icmp eq i32 %42, 0
  br i1 %51, label %84, label %52

; <label>:52:                                     ; preds = %36
  %53 = sub i32 %40, %50
  br label %54

; <label>:54:                                     ; preds = %54, %52
  %55 = phi i64 [ %38, %52 ], [ %78, %54 ]
  %56 = phi float [ %49, %52 ], [ %77, %54 ]
  %57 = phi i32 [ %46, %52 ], [ %76, %54 ]
  %58 = phi i32 [ %53, %52 ], [ %79, %54 ]
  %59 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %55, i64 %37
  %60 = load float, float* %59, align 4, !tbaa !1
  %61 = fcmp olt float %60, 0.000000e+00
  %62 = fsub float -0.000000e+00, %60
  %63 = select i1 %61, float %62, float %60
  %64 = fcmp ogt float %63, %56
  %65 = trunc i64 %55 to i32
  %66 = select i1 %64, i32 %65, i32 %57
  %67 = select i1 %64, float %63, float %56
  %68 = add nuw nsw i64 %55, 1
  %69 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %68, i64 %37
  %70 = load float, float* %69, align 4, !tbaa !1
  %71 = fcmp olt float %70, 0.000000e+00
  %72 = fsub float -0.000000e+00, %70
  %73 = select i1 %71, float %72, float %70
  %74 = fcmp ogt float %73, %67
  %75 = trunc i64 %68 to i32
  %76 = select i1 %74, i32 %75, i32 %66
  %77 = select i1 %74, float %73, float %67
  %78 = add nsw i64 %55, 2
  %79 = add i32 %58, -2
  %80 = icmp eq i32 %79, 0
  br i1 %80, label %83, label %54

; <label>:81:                                     ; preds = %108
  %82 = landingpad { i8*, i32 }
          cleanup
          filter [1 x i8*] [i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN7Iex_2_27MathExcE to i8*)]
  br label %262

; <label>:83:                                     ; preds = %54
  br label %84

; <label>:84:                                     ; preds = %83, %36
  %85 = phi i32 [ undef, %36 ], [ %76, %83 ]
  %86 = phi float [ undef, %36 ], [ %77, %83 ]
  %87 = phi i64 [ %38, %36 ], [ %78, %83 ]
  %88 = phi float [ %49, %36 ], [ %77, %83 ]
  %89 = phi i32 [ %46, %36 ], [ %76, %83 ]
  %90 = icmp eq i32 %50, 0
  br i1 %90, label %103, label %91

; <label>:91:                                     ; preds = %84
  br label %92

; <label>:92:                                     ; preds = %91
  %93 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %87, i64 %37
  %94 = load float, float* %93, align 4, !tbaa !1
  %95 = fcmp olt float %94, 0.000000e+00
  %96 = fsub float -0.000000e+00, %94
  %97 = select i1 %95, float %96, float %94
  %98 = fcmp ogt float %97, %88
  br label %99

; <label>:99:                                     ; preds = %92
  %100 = select i1 %98, float %97, float %88
  %101 = trunc i64 %87 to i32
  %102 = select i1 %98, i32 %101, i32 %89
  br label %103

; <label>:103:                                    ; preds = %84, %99
  %104 = phi i32 [ %85, %84 ], [ %102, %99 ]
  %105 = phi float [ %86, %84 ], [ %100, %99 ]
  %106 = fcmp oeq float %105, 0.000000e+00
  br i1 %106, label %107, label %119

; <label>:107:                                    ; preds = %103
  br i1 %2, label %108, label %112

; <label>:108:                                    ; preds = %107
  %109 = tail call i8* @__cxa_allocate_exception(i64 24) #2
  %110 = bitcast i8* %109 to %"class.Iex_2_2::BaseExc"*
  tail call void @_ZN7Iex_2_27BaseExcC2EPKc(%"class.Iex_2_2::BaseExc"* %110, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str, i64 0, i64 0)) #2
  %111 = bitcast i8* %109 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @_ZTVN9Imath_2_213SingMatrixExcE, i64 0, i64 2) to i32 (...)**), i32 (...)*** %111, align 8, !tbaa !16
  invoke void @__cxa_throw(i8* %109, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN9Imath_2_213SingMatrixExcE to i8*), i8* bitcast (void (%"class.Iex_2_2::BaseExc"*)* @_ZN7Iex_2_27BaseExcD2Ev to i8*)) #15
          to label %269 unwind label %81

; <label>:112:                                    ; preds = %107
  %113 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 1
  %114 = bitcast float* %113 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %114, i8 0, i64 56, i32 4, i1 false) #2
  %115 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 0
  store float 1.000000e+00, float* %115, align 4, !tbaa !1
  %116 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 1
  store float 1.000000e+00, float* %116, align 4, !tbaa !1
  %117 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 2
  store float 1.000000e+00, float* %117, align 4, !tbaa !1
  %118 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 3
  store float 1.000000e+00, float* %118, align 4, !tbaa !1
  br label %261

; <label>:119:                                    ; preds = %103
  %120 = zext i32 %104 to i64
  %121 = icmp eq i64 %120, %37
  br i1 %121, label %140, label %122

; <label>:122:                                    ; preds = %119
  %123 = sext i32 %104 to i64
  %124 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %37, i64 0
  %125 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %123, i64 0
  %126 = bitcast float* %124 to <4 x i32>*
  %127 = load <4 x i32>, <4 x i32>* %126, align 16, !tbaa !1
  %128 = bitcast float* %125 to <4 x i32>*
  %129 = load <4 x i32>, <4 x i32>* %128, align 16, !tbaa !1
  %130 = bitcast float* %124 to <4 x i32>*
  store <4 x i32> %129, <4 x i32>* %130, align 16, !tbaa !1
  %131 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 %37, i64 0
  %132 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 %123, i64 0
  %133 = bitcast float* %131 to <4 x i32>*
  %134 = load <4 x i32>, <4 x i32>* %133, align 16, !tbaa !1
  %135 = bitcast float* %132 to <4 x i32>*
  %136 = load <4 x i32>, <4 x i32>* %135, align 16, !tbaa !1
  %137 = bitcast float* %131 to <4 x i32>*
  store <4 x i32> %136, <4 x i32>* %137, align 16, !tbaa !1
  %138 = bitcast float* %125 to <4 x i32>*
  store <4 x i32> %127, <4 x i32>* %138, align 16, !tbaa !1
  %139 = bitcast float* %132 to <4 x i32>*
  store <4 x i32> %134, <4 x i32>* %139, align 16, !tbaa !1
  br label %140

; <label>:140:                                    ; preds = %119, %122
  %141 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %37, i64 0
  %142 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 %37, i64 0
  %143 = bitcast float* %141 to <4 x float>*
  %144 = bitcast float* %142 to <4 x float>*
  br label %145

; <label>:145:                                    ; preds = %145, %140
  %146 = phi i64 [ %38, %140 ], [ %169, %145 ]
  %147 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %146, i64 %37
  %148 = load float, float* %147, align 4, !tbaa !1
  %149 = load float, float* %43, align 4, !tbaa !1
  %150 = fdiv float %148, %149
  %151 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %146, i64 0
  %152 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 %146, i64 0
  %153 = load <4 x float>, <4 x float>* %143, align 16, !tbaa !1
  %154 = insertelement <4 x float> undef, float %150, i32 0
  %155 = insertelement <4 x float> %154, float %150, i32 1
  %156 = insertelement <4 x float> %155, float %150, i32 2
  %157 = insertelement <4 x float> %156, float %150, i32 3
  %158 = fmul <4 x float> %157, %153
  %159 = bitcast float* %151 to <4 x float>*
  %160 = load <4 x float>, <4 x float>* %159, align 16, !tbaa !1
  %161 = fsub <4 x float> %160, %158
  %162 = bitcast float* %151 to <4 x float>*
  store <4 x float> %161, <4 x float>* %162, align 16, !tbaa !1
  %163 = load <4 x float>, <4 x float>* %144, align 16, !tbaa !1
  %164 = fmul <4 x float> %157, %163
  %165 = bitcast float* %152 to <4 x float>*
  %166 = load <4 x float>, <4 x float>* %165, align 16, !tbaa !1
  %167 = fsub <4 x float> %166, %164
  %168 = bitcast float* %152 to <4 x float>*
  store <4 x float> %167, <4 x float>* %168, align 16, !tbaa !1
  %169 = add nuw nsw i64 %146, 1
  %170 = trunc i64 %169 to i32
  %171 = icmp eq i32 %170, 4
  br i1 %171, label %32, label %145

; <label>:172:                                    ; preds = %35, %239
  %173 = phi i64 [ %240, %239 ], [ 3, %35 ]
  %174 = phi i32 [ %241, %239 ], [ 3, %35 ]
  %175 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %173, i64 %173
  %176 = load float, float* %175, align 4, !tbaa !1
  %177 = fcmp oeq float %176, 0.000000e+00
  br i1 %177, label %197, label %178

; <label>:178:                                    ; preds = %172
  %179 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %173, i64 0
  %180 = bitcast float* %179 to <4 x float>*
  %181 = load <4 x float>, <4 x float>* %180, align 16, !tbaa !1
  %182 = insertelement <4 x float> undef, float %176, i32 0
  %183 = insertelement <4 x float> %182, float %176, i32 1
  %184 = insertelement <4 x float> %183, float %176, i32 2
  %185 = insertelement <4 x float> %184, float %176, i32 3
  %186 = fdiv <4 x float> %181, %185
  %187 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 %173, i64 0
  %188 = bitcast float* %187 to <4 x float>*
  %189 = load <4 x float>, <4 x float>* %188, align 16, !tbaa !1
  %190 = fdiv <4 x float> %189, %185
  %191 = bitcast float* %179 to <4 x float>*
  store <4 x float> %186, <4 x float>* %191, align 16, !tbaa !1
  %192 = bitcast float* %187 to <4 x float>*
  store <4 x float> %190, <4 x float>* %192, align 16, !tbaa !1
  %193 = icmp sgt i64 %173, 0
  br i1 %193, label %194, label %242

; <label>:194:                                    ; preds = %178
  %195 = bitcast float* %179 to <4 x float>*
  %196 = bitcast float* %187 to <4 x float>*
  br label %211

; <label>:197:                                    ; preds = %172
  br i1 %2, label %198, label %204

; <label>:198:                                    ; preds = %197
  %199 = tail call i8* @__cxa_allocate_exception(i64 24) #2
  %200 = bitcast i8* %199 to %"class.Iex_2_2::BaseExc"*
  tail call void @_ZN7Iex_2_27BaseExcC2EPKc(%"class.Iex_2_2::BaseExc"* %200, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str, i64 0, i64 0)) #2
  %201 = bitcast i8* %199 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @_ZTVN9Imath_2_213SingMatrixExcE, i64 0, i64 2) to i32 (...)**), i32 (...)*** %201, align 8, !tbaa !16
  invoke void @__cxa_throw(i8* %199, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN9Imath_2_213SingMatrixExcE to i8*), i8* bitcast (void (%"class.Iex_2_2::BaseExc"*)* @_ZN7Iex_2_27BaseExcD2Ev to i8*)) #15
          to label %269 unwind label %202

; <label>:202:                                    ; preds = %198
  %203 = landingpad { i8*, i32 }
          cleanup
          filter [1 x i8*] [i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN7Iex_2_27MathExcE to i8*)]
  br label %262

; <label>:204:                                    ; preds = %197
  %205 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 1
  %206 = bitcast float* %205 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %206, i8 0, i64 56, i32 4, i1 false) #2
  %207 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 0, i64 0
  store float 1.000000e+00, float* %207, align 4, !tbaa !1
  %208 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1, i64 1
  store float 1.000000e+00, float* %208, align 4, !tbaa !1
  %209 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2, i64 2
  store float 1.000000e+00, float* %209, align 4, !tbaa !1
  %210 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3, i64 3
  store float 1.000000e+00, float* %210, align 4, !tbaa !1
  br label %261

; <label>:211:                                    ; preds = %194, %236
  %212 = phi i64 [ %233, %236 ], [ 0, %194 ]
  %213 = phi <4 x float> [ %237, %236 ], [ %186, %194 ]
  %214 = phi <4 x float> [ %238, %236 ], [ %190, %194 ]
  %215 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %212, i64 %173
  %216 = load float, float* %215, align 4, !tbaa !1
  %217 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %5, i64 0, i32 0, i64 %212, i64 0
  %218 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 %212, i64 0
  %219 = insertelement <4 x float> undef, float %216, i32 0
  %220 = insertelement <4 x float> %219, float %216, i32 1
  %221 = insertelement <4 x float> %220, float %216, i32 2
  %222 = insertelement <4 x float> %221, float %216, i32 3
  %223 = fmul <4 x float> %222, %213
  %224 = bitcast float* %217 to <4 x float>*
  %225 = load <4 x float>, <4 x float>* %224, align 16, !tbaa !1
  %226 = fsub <4 x float> %225, %223
  %227 = bitcast float* %217 to <4 x float>*
  store <4 x float> %226, <4 x float>* %227, align 16, !tbaa !1
  %228 = fmul <4 x float> %222, %214
  %229 = bitcast float* %218 to <4 x float>*
  %230 = load <4 x float>, <4 x float>* %229, align 16, !tbaa !1
  %231 = fsub <4 x float> %230, %228
  %232 = bitcast float* %218 to <4 x float>*
  store <4 x float> %231, <4 x float>* %232, align 16, !tbaa !1
  %233 = add nuw nsw i64 %212, 1
  %234 = trunc i64 %233 to i32
  %235 = icmp eq i32 %234, %174
  br i1 %235, label %239, label %236

; <label>:236:                                    ; preds = %211
  %237 = load <4 x float>, <4 x float>* %195, align 16, !tbaa !1
  %238 = load <4 x float>, <4 x float>* %196, align 16, !tbaa !1
  br label %211

; <label>:239:                                    ; preds = %211
  %240 = add nsw i64 %173, -1
  %241 = add nsw i32 %174, -1
  br i1 %193, label %172, label %242

; <label>:242:                                    ; preds = %178, %239
  %243 = bitcast %"class.Imath_2_2::Matrix44"* %4 to <4 x i32>*
  %244 = load <4 x i32>, <4 x i32>* %243, align 16, !tbaa !1
  %245 = bitcast %"class.Imath_2_2::Matrix44"* %0 to <4 x i32>*
  store <4 x i32> %244, <4 x i32>* %245, align 4, !tbaa !1
  %246 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 1
  %247 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 1
  %248 = bitcast [4 x float]* %246 to <4 x i32>*
  %249 = load <4 x i32>, <4 x i32>* %248, align 16, !tbaa !1
  %250 = bitcast [4 x float]* %247 to <4 x i32>*
  store <4 x i32> %249, <4 x i32>* %250, align 4, !tbaa !1
  %251 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 2
  %252 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 2
  %253 = bitcast [4 x float]* %251 to <4 x i32>*
  %254 = load <4 x i32>, <4 x i32>* %253, align 16, !tbaa !1
  %255 = bitcast [4 x float]* %252 to <4 x i32>*
  store <4 x i32> %254, <4 x i32>* %255, align 4, !tbaa !1
  %256 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %4, i64 0, i32 0, i64 3
  %257 = getelementptr inbounds %"class.Imath_2_2::Matrix44", %"class.Imath_2_2::Matrix44"* %0, i64 0, i32 0, i64 3
  %258 = bitcast [4 x float]* %256 to <4 x i32>*
  %259 = load <4 x i32>, <4 x i32>* %258, align 16, !tbaa !1
  %260 = bitcast [4 x float]* %257 to <4 x i32>*
  store <4 x i32> %259, <4 x i32>* %260, align 4, !tbaa !1
  br label %261

; <label>:261:                                    ; preds = %204, %112, %242
  call void @llvm.lifetime.end(i64 64, i8* nonnull %13) #2
  call void @llvm.lifetime.end(i64 64, i8* %6) #2
  ret void

; <label>:262:                                    ; preds = %202, %81
  %263 = phi { i8*, i32 } [ %82, %81 ], [ %203, %202 ]
  %264 = extractvalue { i8*, i32 } %263, 1
  call void @llvm.lifetime.end(i64 64, i8* nonnull %13) #2
  call void @llvm.lifetime.end(i64 64, i8* %6) #2
  %265 = icmp slt i32 %264, 0
  br i1 %265, label %266, label %268

; <label>:266:                                    ; preds = %262
  %267 = extractvalue { i8*, i32 } %263, 0
  tail call void @__cxa_call_unexpected(i8* %267) #15
  unreachable

; <label>:268:                                    ; preds = %262
  resume { i8*, i32 } %263

; <label>:269:                                    ; preds = %198, %108
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_allocate_exception(i64) local_unnamed_addr

declare void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr

declare void @__cxa_call_unexpected(i8*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN9Imath_2_213SingMatrixExcD0Ev(%"class.Imath_2_2::SingMatrixExc"*) unnamed_addr #4 comdat align 2 {
  %2 = getelementptr inbounds %"class.Imath_2_2::SingMatrixExc", %"class.Imath_2_2::SingMatrixExc"* %0, i64 0, i32 0, i32 0
  tail call void @_ZN7Iex_2_27BaseExcD2Ev(%"class.Iex_2_2::BaseExc"* %2) #2
  %3 = bitcast %"class.Imath_2_2::SingMatrixExc"* %0 to i8*
  tail call void @_ZdlPv(i8* %3) #16
  ret void
}

; Function Attrs: nounwind
declare i8* @_ZNK7Iex_2_27BaseExc4whatEv(%"class.Iex_2_2::BaseExc"*) unnamed_addr #1

; Function Attrs: nounwind
declare void @_ZN7Iex_2_27BaseExcC2EPKc(%"class.Iex_2_2::BaseExc"*, i8*) unnamed_addr #1

; Function Attrs: nounwind
declare void @_ZN7Iex_2_27BaseExcD2Ev(%"class.Iex_2_2::BaseExc"*) unnamed_addr #1

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) local_unnamed_addr #14

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #5

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_llvm_ops.cpp() #11 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*)) #2
  ret void
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { norecurse nounwind readnone uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { norecurse nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { inlinehint nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { norecurse nounwind readonly uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { nounwind readonly uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { nounwind readnone }
attributes #14 = { nobuiltin nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #15 = { noreturn }
attributes #16 = { builtin nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.1 (tags/RELEASE_391/final)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !2, i64 0}
!6 = !{!"_ZTSN9Imath_2_24Vec3IfEE", !2, i64 0, !2, i64 4, !2, i64 8}
!7 = !{!6, !2, i64 4}
!8 = !{!6, !2, i64 8}
!9 = !{!10, !2, i64 0}
!10 = !{!"_ZTSN3OSL5Dual2IfEE", !2, i64 0, !2, i64 4, !2, i64 8}
!11 = !{!10, !2, i64 4}
!12 = !{!10, !2, i64 8}
!13 = !{!14}
!14 = distinct !{!14, !15, !"_ZNK9Imath_2_28Matrix44IfE10transposedEv: argument 0"}
!15 = distinct !{!15, !"_ZNK9Imath_2_28Matrix44IfE10transposedEv"}
!16 = !{!17, !17, i64 0}
!17 = !{!"vtable pointer", !4, i64 0}
!18 = !{!19}
!19 = distinct !{!19, !20, !"_ZNK9Imath_2_28Matrix44IfE10transposedEv: argument 0"}
!20 = distinct !{!20, !"_ZNK9Imath_2_28Matrix44IfE10transposedEv"}
!21 = !{!22}
!22 = distinct !{!22, !23, !"_ZNK9Imath_2_24Vec3IfE5crossERKS1_: argument 0"}
!23 = distinct !{!23, !"_ZNK9Imath_2_24Vec3IfE5crossERKS1_"}
!24 = !{!25}
!25 = distinct !{!25, !26, !"_ZN3OSL5crossERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_: argument 0"}
!26 = distinct !{!26, !"_ZN3OSL5crossERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"_ZN3OSL5crossERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_: argument 0"}
!29 = distinct !{!29, !"_ZN3OSL5crossERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"_ZN3OSL5crossERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_: argument 0"}
!32 = distinct !{!32, !"_ZN3OSL5crossERKNS_5Dual2IN9Imath_2_24Vec3IfEEEES6_"}
!33 = !{!34, !36}
!34 = distinct !{!34, !35, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!35 = distinct !{!35, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!36 = distinct !{!36, !37, !"_ZN3OSLmiIN9Imath_2_24Vec3IfEEEENS_5Dual2IT_EERKS6_S8_: argument 0"}
!37 = distinct !{!37, !"_ZN3OSLmiIN9Imath_2_24Vec3IfEEEENS_5Dual2IT_EERKS6_S8_"}
!38 = !{!39, !36}
!39 = distinct !{!39, !40, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!40 = distinct !{!40, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!41 = !{!42, !36}
!42 = distinct !{!42, !43, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!43 = distinct !{!43, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!44 = !{!45, !47}
!45 = distinct !{!45, !46, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!46 = distinct !{!46, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!47 = distinct !{!47, !48, !"_ZN3OSLmiIN9Imath_2_24Vec3IfEEEENS_5Dual2IT_EERKS6_S8_: argument 0"}
!48 = distinct !{!48, !"_ZN3OSLmiIN9Imath_2_24Vec3IfEEEENS_5Dual2IT_EERKS6_S8_"}
!49 = !{!50, !47}
!50 = distinct !{!50, !51, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!51 = distinct !{!51, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!52 = !{!53, !47}
!53 = distinct !{!53, !54, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!54 = distinct !{!54, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!55 = !{!56, !58}
!56 = distinct !{!56, !57, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!57 = distinct !{!57, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!58 = distinct !{!58, !59, !"_ZN3OSLmiIN9Imath_2_24Vec3IfEEEENS_5Dual2IT_EERKS6_S8_: argument 0"}
!59 = distinct !{!59, !"_ZN3OSLmiIN9Imath_2_24Vec3IfEEEENS_5Dual2IT_EERKS6_S8_"}
!60 = !{!61, !58}
!61 = distinct !{!61, !62, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!62 = distinct !{!62, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!63 = !{!64, !58}
!64 = distinct !{!64, !65, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_: argument 0"}
!65 = distinct !{!65, !"_ZNK9Imath_2_24Vec3IfEmiERKS1_"}
!66 = !{!67}
!67 = distinct !{!67, !68, !"_ZNK9Imath_2_24Vec3IfE10normalizedEv: argument 0"}
!68 = distinct !{!68, !"_ZNK9Imath_2_24Vec3IfE10normalizedEv"}
!69 = !{!70, !72, i64 288}
!70 = !{!"_ZTSN3OSL13ShaderGlobalsE", !6, i64 0, !6, i64 12, !6, i64 24, !6, i64 36, !6, i64 48, !6, i64 60, !6, i64 72, !6, i64 84, !6, i64 96, !2, i64 108, !2, i64 112, !2, i64 116, !2, i64 120, !2, i64 124, !2, i64 128, !6, i64 132, !6, i64 144, !2, i64 156, !2, i64 160, !6, i64 164, !6, i64 176, !6, i64 188, !6, i64 200, !71, i64 216, !71, i64 224, !71, i64 232, !71, i64 240, !71, i64 248, !71, i64 256, !71, i64 264, !71, i64 272, !2, i64 280, !72, i64 284, !72, i64 288, !72, i64 292}
!71 = !{!"any pointer", !3, i64 0}
!72 = !{!"int", !3, i64 0}
!73 = !{!74, !76}
!74 = distinct !{!74, !75, !"_ZNK9Imath_2_24Vec3IfE5crossERKS1_: argument 0"}
!75 = distinct !{!75, !"_ZNK9Imath_2_24Vec3IfE5crossERKS1_"}
!76 = distinct !{!76, !77, !"_Z15calculatenormalPvb: argument 0"}
!77 = distinct !{!77, !"_Z15calculatenormalPvb"}
!78 = !{!79, !76}
!79 = distinct !{!79, !80, !"_ZNK9Imath_2_24Vec3IfE5crossERKS1_: argument 0"}
!80 = distinct !{!80, !"_ZNK9Imath_2_24Vec3IfE5crossERKS1_"}
!81 = !{!82, !84}
!82 = distinct !{!82, !83, !"_ZNK9Imath_2_24Vec3IfE5crossERKS1_: argument 0"}
!83 = distinct !{!83, !"_ZNK9Imath_2_24Vec3IfE5crossERKS1_"}
!84 = distinct !{!84, !85, !"_Z15calculatenormalPvb: argument 0"}
!85 = distinct !{!85, !"_Z15calculatenormalPvb"}
!86 = !{!70, !72, i64 284}
