#include "autotests_float2.h"
#include "autotests_float3.h"
#include "autotests_float4.h"

#include "autotests_float2x2.h"
#include "autotests_float3x3.h"
#include "autotests_float4x4.h"

#include "autotests_half.h"
#include "autotests_half2.h"
#include "autotests_half3.h"
#include "autotests_half4.h"

#include "autotests_quaternion.h"

#include "autotests_rect.h"

#include "autotests_ray.h"

int main()
{
    bool all_passed = true;
    bool verbose = false;

#define RUN_TEST_SUITE(name, verbose_flag, func)                       \
    do {                                                               \
        AfterMathTests::TestSuite suite(name, verbose_flag);           \
        suite.header();                                                \
        func(suite);                                                   \
        suite.footer();                                                \
        if (suite.get_failed_count() > 0) all_passed = false;          \
    } while(0)

    RUN_TEST_SUITE("float2", verbose, AfterMathTests::RunFloat2Tests);
    RUN_TEST_SUITE("float3", verbose, AfterMathTests::RunFloat3Tests);
    RUN_TEST_SUITE("float4", verbose, AfterMathTests::RunFloat4Tests);
    RUN_TEST_SUITE("float2x2", verbose, AfterMathTests::RunFloat2x2Tests);
    RUN_TEST_SUITE("float3x3", verbose, AfterMathTests::RunFloat3x3Tests);
    RUN_TEST_SUITE("float4x4", verbose, AfterMathTests::RunFloat4x4Tests);
    RUN_TEST_SUITE("half", verbose, AfterMathTests::RunHalfTests);
    RUN_TEST_SUITE("half2", verbose, AfterMathTests::RunHalf2Tests);
    RUN_TEST_SUITE("half3", verbose, AfterMathTests::RunHalf3Tests);
    RUN_TEST_SUITE("half4", verbose, AfterMathTests::RunHalf4Tests);
    RUN_TEST_SUITE("quaternion", verbose, AfterMathTests::RunQuaternionTests);
    RUN_TEST_SUITE("Rect", verbose, AfterMathTests::RunRectTests);
    RUN_TEST_SUITE("Ray", verbose, AfterMathTests::RunRayTests);

    return all_passed ? 0 : 1;
}