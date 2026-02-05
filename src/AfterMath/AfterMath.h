// Description: Unified include file for Math Library - comprehensive mathematics
//              library for games and scientific computing with SSE optimization
// Author: DeepSeek, NS_Deathman
#pragma once

// ============================================================================
// Configuration
// ============================================================================
#include "math_config.h"

// ============================================================================
// Core Constants and Functions
// ============================================================================
#include "math_constants.h"
#include "math_functions.h"
#include "math_fast_functions.h"

// ============================================================================
// Matrix Types
// ============================================================================
#include "math_float2x2.h"
#include "math_float3x3.h"
#include "math_float4x4.h"

// ============================================================================
// Advanced Types
// ============================================================================
#include "math_quaternion.h"
#include "math_aabb.h"

// ============================================================================
// Vector Types
// ============================================================================
#include "math_float2.h"
#include "math_float3.h" 
#include "math_float4.h"

#include "math_half.h"
#include "math_half2.h"
#include "math_half3.h" 
#include "math_half4.h"

// ============================================================================
// Global Using Declarations for Convenience
// ============================================================================
namespace AfterMath
{
    // Constants
    using Constants::PI;
    using Constants::TWO_PI;
    using Constants::HALF_PI;
    using Constants::DEG_TO_RAD;
    using Constants::RAD_TO_DEG;
    using Constants::EPSILON;
    using Constants::INFINITY;
    using Constants::NAN;

    // Math Functions
    using AfterMathFunctions::approximately;
    using AfterMathFunctions::approximately_zero;
    using AfterMathFunctions::approximately_angle;
    using AfterMathFunctions::is_finite;
    using AfterMathFunctions::clamp;
    using AfterMathFunctions::lerp;

    // Fast Math Functions
    using FastMath::fast_sin;
    using FastMath::fast_cos;
    using FastMath::fast_tan;
    using FastMath::fast_asin;
    using FastMath::fast_acos;
    using FastMath::fast_atan;
    using FastMath::fast_atan2;
    using FastMath::fast_sqrt;
    using FastMath::fast_inv_sqrt;
    using FastMath::fast_exp;
    using FastMath::fast_log;
    using FastMath::fast_pow;

    // Common Type Aliases
    using half = AfterMath::half;
    using half2 = AfterMath::half2;
    using half3 = AfterMath::half3;
    using half4 = AfterMath::half4;
    using float2 = AfterMath::float2;
    using float3 = AfterMath::float3;
    using float4 = AfterMath::float4;
    using float2x2 = AfterMath::float2x2;
    using float3x3 = AfterMath::float3x3;
    using float4x4 = AfterMath::float4x4;
    using quaternion = AfterMath::quaternion;

} // namespace AfterMath

// ============================================================================
// Common Global Constants
// ============================================================================

// Half-precision constants
extern const AfterMath::half half_Zero;
extern const AfterMath::half half_One;
extern const AfterMath::half half_Max;
extern const AfterMath::half half_Min;
extern const AfterMath::half half_Epsilon;
extern const AfterMath::half half_PI;
extern const AfterMath::half half_TwoPI;
extern const AfterMath::half half_HalfPI;
extern const AfterMath::half half_QuarterPI;
extern const AfterMath::half half_InvPI;
extern const AfterMath::half half_InvTwoPI;
extern const AfterMath::half half_DegToRad;
extern const AfterMath::half half_RadToDeg;
extern const AfterMath::half half_Sqrt2;
extern const AfterMath::half half_E;
extern const AfterMath::half half_GoldenRatio;

// 2D vector constants
extern const AfterMath::float2 float2_Zero;
extern const AfterMath::float2 float2_One;
extern const AfterMath::float2 float2_UnitX;
extern const AfterMath::float2 float2_UnitY;
extern const AfterMath::float2 float2_Right;
extern const AfterMath::float2 float2_Left;
extern const AfterMath::float2 float2_Up;
extern const AfterMath::float2 float2_Down;

// 3D vector constants
extern const AfterMath::float3 float3_Zero;
extern const AfterMath::float3 float3_One;
extern const AfterMath::float3 float3_UnitX;
extern const AfterMath::float3 float3_UnitY;
extern const AfterMath::float3 float3_UnitZ;
extern const AfterMath::float3 float3_Forward;
extern const AfterMath::float3 float3_Up;
extern const AfterMath::float3 float3_Right;

// 4D vector constants
extern const AfterMath::float4 float4_Zero;
extern const AfterMath::float4 float4_One;

// Matrix constants
extern const AfterMath::float2x2 float2x2_Identity;
extern const AfterMath::float2x2 float2x2_Zero;
extern const AfterMath::float3x3 float3x3_Identity;
extern const AfterMath::float3x3 float3x3_Zero;
extern const AfterMath::float4x4 float4x4_Identity;
extern const AfterMath::float4x4 float4x4_Zero;
