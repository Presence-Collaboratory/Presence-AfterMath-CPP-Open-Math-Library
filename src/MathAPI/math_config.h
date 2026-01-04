#pragma once

//#define MATH_SUPPORT_D3DX
#define MATH_API

//#ifdef MATH_EXPORTS
//#define MATH_API __declspec(dllexport)
//#else
//#define MATH_API __declspec(dllimport)
//#endif

#ifndef MATH_ASSERT
#include <cassert>
#define MATH_ASSERT(x) assert(x)
#endif