#include <iostream>
#include <windows.h>
#include "Autotests/AutotestCore.h"
#include "Autotests/autotests_float3x3.h"
#include "Autotests/autotests_float4x4.h"

int main()
{
    //MathTests::RunFloat3x3Tests();
    MathTests::RunFloat4x4Tests();

    // Для ожидания перед закрытием (если консольное приложение)
    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    return 0;
}