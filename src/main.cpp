#include <iostream>
#include <windows.h>
#include "Autotests/AutotestCore.h"
#include "Autotests/autotests_float2.h"
#include "Autotests/autotests_float3x3.h"
#include "Autotests/autotests_float4x4.h"
#include "Autotests/autotests_quaternion.h"
#include "Autotests/autotests_aabb.h"

int main()
{
    AfterMathTests::RunFloat2Tests();

    //AfterMathTests::RunFloat3x3Tests();
    //AfterMathTests::RunFloat4x4Tests();
    
    //AfterMathTests::RunQuaternionTests();
    
    //AfterMathTests::RunAABBTests();

    // Для ожидания перед закрытием (если консольное приложение)
    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    return 0;
}