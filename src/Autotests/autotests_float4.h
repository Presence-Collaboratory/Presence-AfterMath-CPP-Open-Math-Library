// Author: DeepSeek
// Test suite for AfterMath::float4 class

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunFloat4Tests()
    {
        TestSuite suite("float4 Tests", true);
        suite.header();

        using namespace AfterMath;

        // ============================================================================
        // 1. Конструкторы
        // ============================================================================
        suite.section("Конструкторы");

        // Тест конструктора по умолчанию
        {
            float4 v;
            suite.assert_approximately_equal(v.x, 0.0f, "Default constructor x");
            suite.assert_approximately_equal(v.y, 0.0f, "Default constructor y");
            suite.assert_approximately_equal(v.z, 0.0f, "Default constructor z");
            suite.assert_approximately_equal(v.w, 0.0f, "Default constructor w");
        }

        // Тест конструктора с компонентами
        {
            float4 v(1.5f, 2.5f, 3.5f, 4.5f);
            suite.assert_approximately_equal(v.x, 1.5f, "Component constructor x");
            suite.assert_approximately_equal(v.y, 2.5f, "Component constructor y");
            suite.assert_approximately_equal(v.z, 3.5f, "Component constructor z");
            suite.assert_approximately_equal(v.w, 4.5f, "Component constructor w");
        }

        // Тест конструктора со скаляром
        {
            float4 v(3.0f);
            suite.assert_approximately_equal(v.x, 3.0f, "Scalar constructor x");
            suite.assert_approximately_equal(v.y, 3.0f, "Scalar constructor y");
            suite.assert_approximately_equal(v.z, 3.0f, "Scalar constructor z");
            suite.assert_approximately_equal(v.w, 3.0f, "Scalar constructor w");
        }

        // Тест конструктора из float2
        {
            float2 vec(1.0f, 2.0f);
            float4 v(vec, 3.0f, 4.0f);
            suite.assert_approximately_equal(v.x, 1.0f, "float2 constructor x");
            suite.assert_approximately_equal(v.y, 2.0f, "float2 constructor y");
            suite.assert_approximately_equal(v.z, 3.0f, "float2 constructor z");
            suite.assert_approximately_equal(v.w, 4.0f, "float2 constructor w");
        }

        // Тест конструктора из float3
        {
            float3 vec(1.0f, 2.0f, 3.0f);
            float4 v(vec, 4.0f);
            suite.assert_approximately_equal(v.x, 1.0f, "float3 constructor x");
            suite.assert_approximately_equal(v.y, 2.0f, "float3 constructor y");
            suite.assert_approximately_equal(v.z, 3.0f, "float3 constructor z");
            suite.assert_approximately_equal(v.w, 4.0f, "float3 constructor w");
        }

        // Тест конструктора из массива
        {
            float data[4] = { 4.0f, 5.0f, 6.0f, 7.0f };
            float4 v(data);
            suite.assert_approximately_equal(v.x, 4.0f, "Array constructor x");
            suite.assert_approximately_equal(v.y, 5.0f, "Array constructor y");
            suite.assert_approximately_equal(v.z, 6.0f, "Array constructor z");
            suite.assert_approximately_equal(v.w, 7.0f, "Array constructor w");
        }

        // Тест копирующего конструктора
        {
            float4 original(6.0f, 7.0f, 8.0f, 9.0f);
            float4 copy(original);
            suite.assert_approximately_equal(copy.x, 6.0f, "Copy constructor x");
            suite.assert_approximately_equal(copy.y, 7.0f, "Copy constructor y");
            suite.assert_approximately_equal(copy.z, 8.0f, "Copy constructor z");
            suite.assert_approximately_equal(copy.w, 9.0f, "Copy constructor w");
        }

        // Тест статических конструкторов
        {
            suite.assert_approximately_equal(float4::zero(), float4(0.0f, 0.0f, 0.0f, 0.0f), "zero()");
            suite.assert_approximately_equal(float4::one(), float4(1.0f, 1.0f, 1.0f, 1.0f), "one()");
            suite.assert_approximately_equal(float4::unit_x(), float4(1.0f, 0.0f, 0.0f, 0.0f), "unit_x()");
            suite.assert_approximately_equal(float4::unit_y(), float4(0.0f, 1.0f, 0.0f, 0.0f), "unit_y()");
            suite.assert_approximately_equal(float4::unit_z(), float4(0.0f, 0.0f, 1.0f, 0.0f), "unit_z()");
            suite.assert_approximately_equal(float4::unit_w(), float4(0.0f, 0.0f, 0.0f, 1.0f), "unit_w()");
        }

        // Тест статических конструкторов цветов
        {
            // Тест from_rgba
            float4 rgba_color = float4::from_rgba(255, 128, 64, 200);
            suite.assert_approximately_equal(rgba_color.x, 1.0f, "from_rgba() red", 0.01f);
            suite.assert_approximately_equal(rgba_color.y, 128.0f / 255.0f, "from_rgba() green", 0.01f);
            suite.assert_approximately_equal(rgba_color.z, 64.0f / 255.0f, "from_rgba() blue", 0.01f);
            suite.assert_approximately_equal(rgba_color.w, 200.0f / 255.0f, "from_rgba() alpha", 0.01f);

            // Тест from_color
            float4 float_color = float4::from_color(0.2f, 0.4f, 0.6f, 0.8f);
            suite.assert_approximately_equal(float_color, float4(0.2f, 0.4f, 0.6f, 0.8f), "from_color()");
        }

        // ============================================================================
        // 2. Операторы присваивания
        // ============================================================================
        suite.section("Операторы присваивания");

        // Тест присваивания скаляра
        {
            float4 v;
            v = 2.5f;
            suite.assert_approximately_equal(v.x, 2.5f, "Scalar assignment x");
            suite.assert_approximately_equal(v.y, 2.5f, "Scalar assignment y");
            suite.assert_approximately_equal(v.z, 2.5f, "Scalar assignment z");
            suite.assert_approximately_equal(v.w, 2.5f, "Scalar assignment w");
        }

        // Тест присваивания из float3
        {
            float4 v(1.0f, 2.0f, 3.0f, 4.0f);
            float3 xyz(5.0f, 6.0f, 7.0f);
            v = xyz;
            suite.assert_approximately_equal(v.x, 5.0f, "float3 assignment x");
            suite.assert_approximately_equal(v.y, 6.0f, "float3 assignment y");
            suite.assert_approximately_equal(v.z, 7.0f, "float3 assignment z");
            suite.assert_approximately_equal(v.w, 4.0f, "float3 assignment preserves w");
        }

        // Тест составных операторов присваивания
        {
            float4 v(1.0f, 2.0f, 3.0f, 4.0f);

            v += float4(3.0f, 4.0f, 5.0f, 6.0f);
            suite.assert_approximately_equal(v, float4(4.0f, 6.0f, 8.0f, 10.0f), "Operator +=");

            v -= float4(1.0f, 2.0f, 3.0f, 4.0f);
            suite.assert_approximately_equal(v, float4(3.0f, 4.0f, 5.0f, 6.0f), "Operator -=");

            v *= float4(2.0f, 3.0f, 4.0f, 5.0f);
            suite.assert_approximately_equal(v, float4(6.0f, 12.0f, 20.0f, 30.0f), "Operator *=");

            v /= float4(2.0f, 3.0f, 4.0f, 5.0f);
            suite.assert_approximately_equal(v, float4(3.0f, 4.0f, 5.0f, 6.0f), "Operator /=");

            v *= 2.0f;
            suite.assert_approximately_equal(v, float4(6.0f, 8.0f, 10.0f, 12.0f), "Operator *= scalar");

            v /= 2.0f;
            suite.assert_approximately_equal(v, float4(3.0f, 4.0f, 5.0f, 6.0f), "Operator /= scalar");
        }

        // ============================================================================
        // 3. Бинарные и унарные операторы
        // ============================================================================
        suite.section("Бинарные и унарные операторы");

        // Тест сложения
        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 b(4.0f, 5.0f, 6.0f, 7.0f);
            float4 result = a + b;
            suite.assert_approximately_equal(result, float4(5.0f, 7.0f, 9.0f, 11.0f), "Operator +");
        }

        // Тест вычитания
        {
            float4 a(5.0f, 6.0f, 7.0f, 8.0f);
            float4 b(2.0f, 3.0f, 4.0f, 5.0f);
            float4 result = a - b;
            suite.assert_approximately_equal(result, float4(3.0f, 3.0f, 3.0f, 3.0f), "Operator -");
        }

        // Тест умножения компонентного
        {
            float4 a(2.0f, 3.0f, 4.0f, 5.0f);
            float4 b(1.0f, 2.0f, 3.0f, 4.0f);
            float4 result = a * b;
            suite.assert_approximately_equal(result, float4(2.0f, 6.0f, 12.0f, 20.0f), "Operator * (component-wise)");
        }

        // Тест деления компонентного
        {
            float4 a(6.0f, 8.0f, 10.0f, 12.0f);
            float4 b(2.0f, 4.0f, 5.0f, 6.0f);
            float4 result = a / b;
            suite.assert_approximately_equal(result, float4(3.0f, 2.0f, 2.0f, 2.0f), "Operator / (component-wise)");
        }

        // Тест унарных операторов
        {
            float4 a(1.0f, -2.0f, 3.0f, -4.0f);
            suite.assert_approximately_equal(+a, float4(1.0f, -2.0f, 3.0f, -4.0f), "Unary +");
            suite.assert_approximately_equal(-a, float4(-1.0f, 2.0f, -3.0f, 4.0f), "Unary -");
        }

        // Тест скалярных операций
        {
            float4 v(2.0f, 3.0f, 4.0f, 5.0f);

            float4 result1 = v * 2.0f;
            suite.assert_approximately_equal(result1, float4(4.0f, 6.0f, 8.0f, 10.0f), "Vector * scalar");

            float4 result2 = 2.0f * v;
            suite.assert_approximately_equal(result2, float4(4.0f, 6.0f, 8.0f, 10.0f), "Scalar * vector");

            float4 result3 = v / 2.0f;
            suite.assert_approximately_equal(result3, float4(1.0f, 1.5f, 2.0f, 2.5f), "Vector / scalar");
        }

        // ============================================================================
        // 4. Операторы доступа и преобразования
        // ============================================================================
        suite.section("Операторы доступа и преобразования");

        // Тест оператора индексации
        {
            float4 v(7.0f, 8.0f, 9.0f, 10.0f);
            suite.assert_approximately_equal(v[0], 7.0f, "Operator [] index 0");
            suite.assert_approximately_equal(v[1], 8.0f, "Operator [] index 1");
            suite.assert_approximately_equal(v[2], 9.0f, "Operator [] index 2");
            suite.assert_approximately_equal(v[3], 10.0f, "Operator [] index 3");

            v[0] = 11.0f;
            v[1] = 12.0f;
            v[2] = 13.0f;
            v[3] = 14.0f;
            suite.assert_approximately_equal(v.x, 11.0f, "Operator [] mutable x");
            suite.assert_approximately_equal(v.y, 12.0f, "Operator [] mutable y");
            suite.assert_approximately_equal(v.z, 13.0f, "Operator [] mutable z");
            suite.assert_approximately_equal(v.w, 14.0f, "Operator [] mutable w");
        }

        // Тест преобразования в указатель
        {
            float4 v(1.0f, 2.0f, 3.0f, 4.0f);
            const float* ptr = v;
            suite.assert_approximately_equal(ptr[0], 1.0f, "Conversion to const float* index 0");
            suite.assert_approximately_equal(ptr[1], 2.0f, "Conversion to const float* index 1");
            suite.assert_approximately_equal(ptr[2], 3.0f, "Conversion to const float* index 2");
            suite.assert_approximately_equal(ptr[3], 4.0f, "Conversion to const float* index 3");

            float* mutable_ptr = v;
            mutable_ptr[0] = 5.0f;
            suite.assert_approximately_equal(v.x, 5.0f, "Conversion to float* mutable");
        }

        // Тест преобразования в __m128
        {
            float4 v(1.0f, 2.0f, 3.0f, 4.0f);
            __m128 simd = v;
            alignas(16) float temp[4];
            _mm_store_ps(temp, simd);
            suite.assert_approximately_equal(temp[0], 1.0f, "Conversion to __m128 x", 1e-6f);
            suite.assert_approximately_equal(temp[1], 2.0f, "Conversion to __m128 y", 1e-6f);
            suite.assert_approximately_equal(temp[2], 3.0f, "Conversion to __m128 z", 1e-6f);
            suite.assert_approximately_equal(temp[3], 4.0f, "Conversion to __m128 w", 1e-6f);
        }

        // ============================================================================
        // 5. Математические функции
        // ============================================================================
        suite.section("Математические функции");

        // Тест длины
        {
            float4 v(2.0f, 3.0f, 6.0f, 8.0f);
            // √(4 + 9 + 36 + 64) = √113 ≈ 10.6301
            float expected_length = std::sqrt(113.0f);
            suite.assert_approximately_equal(v.length(), expected_length, "length()", 1e-6f);
            suite.assert_approximately_equal(v.length_sq(), 113.0f, "length_sq()");

            float4 zero(0.0f, 0.0f, 0.0f, 0.0f);
            suite.assert_approximately_equal(zero.length(), 0.0f, "length() of zero vector");
        }

        // Тест нормализации
        {
            float4 v(2.0f, 3.0f, 6.0f, 8.0f);
            float4 normalized = v.normalize();
            float expected_len = 1.0f;
            suite.assert_approximately_equal(normalized.length(), expected_len, "normalize() length", 1e-6f);

            float len = v.length();
            suite.assert_approximately_equal(normalized.x, 2.0f / len, "normalize() x", 1e-6f);
            suite.assert_approximately_equal(normalized.y, 3.0f / len, "normalize() y", 1e-6f);
            suite.assert_approximately_equal(normalized.z, 6.0f / len, "normalize() z", 1e-6f);
            suite.assert_approximately_equal(normalized.w, 8.0f / len, "normalize() w", 1e-6f);

            // Тест нормализации нулевого вектора
            float4 zero(0.0f, 0.0f, 0.0f, 0.0f);
            float4 zero_norm = zero.normalize();
            suite.assert_approximately_equal(zero_norm, float4::zero(), "normalize() zero vector");
        }

        // Тест скалярного произведения
        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 b(5.0f, 6.0f, 7.0f, 8.0f);
            float dot_result = a.dot(b);
            // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
            suite.assert_approximately_equal(dot_result, 70.0f, "dot()");

            // Ортогональные векторы
            float4 orth1(1.0f, 0.0f, 0.0f, 0.0f);
            float4 orth2(0.0f, 1.0f, 0.0f, 0.0f);
            suite.assert_approximately_equal(orth1.dot(orth2), 0.0f, "dot() orthogonal vectors");
        }

        // Тест 3D скалярного произведения (игнорирует w)
        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 b(5.0f, 6.0f, 7.0f, 8.0f);
            float dot3_result = a.dot3(b);
            // 1*5 + 2*6 + 3*7 = 5 + 12 + 21 = 38 (игнорирует w компоненты)
            suite.assert_approximately_equal(dot3_result, 38.0f, "dot3()");
        }

        // Тест векторного произведения (3D, игнорирует w)
        {
            float4 a(1.0f, 0.0f, 0.0f, 0.0f);
            float4 b(0.0f, 1.0f, 0.0f, 0.0f);
            float4 cross_result = a.cross(b);
            suite.assert_approximately_equal(cross_result, float4(0.0f, 0.0f, 1.0f, 0.0f), "cross() i × j = k");

            float4 c(2.0f, 3.0f, 4.0f, 5.0f);
            float4 d(5.0f, 6.0f, 7.0f, 8.0f);
            float4 cross2 = c.cross(d);
            // (3*7 - 4*6, 4*5 - 2*7, 2*6 - 3*5, 0) = (21-24, 20-14, 12-15, 0) = (-3, 6, -3, 0)
            suite.assert_approximately_equal(cross2, float4(-3.0f, 6.0f, -3.0f, 0.0f), "cross() arbitrary vectors");

            // Векторное произведение вектора с самим собой равно нулю
            suite.assert_approximately_equal(a.cross(a), float4(0.0f, 0.0f, 0.0f, 0.0f), "cross() vector with itself");
        }

        // Тест расстояния
        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 b(5.0f, 6.0f, 7.0f, 8.0f);
            float distance = a.distance(b);
            float distance_sq = a.distance_sq(b);

            // (5-1)² + (6-2)² + (7-3)² + (8-4)² = 16 + 16 + 16 + 16 = 64
            suite.assert_approximately_equal(distance, std::sqrt(64.0f), "distance()");
            suite.assert_approximately_equal(distance_sq, 64.0f, "distance_sq()");
        }

        // ============================================================================
        // 6. HLSL-подобные функции
        // ============================================================================
        suite.section("HLSL-подобные функции");

        // Тест abs
        {
            float4 v(-1.5f, 2.5f, -3.5f, 4.5f);
            float4 result = v.abs();
            suite.assert_approximately_equal(result, float4(1.5f, 2.5f, 3.5f, 4.5f), "abs()");
        }

        // Тест sign
        {
            float4 v(-2.0f, 0.0f, 3.0f, -0.0f);
            float4 result = v.sign();
            suite.assert_approximately_equal(result, float4(-1.0f, 0.0f, 1.0f, 0.0f), "sign()");
        }

        // Тест floor
        {
            float4 v(1.7f, -2.3f, 3.9f, -4.1f);
            float4 result = v.floor();
            suite.assert_approximately_equal(result, float4(1.0f, -3.0f, 3.0f, -5.0f), "floor()");
        }

        // Тест ceil
        {
            float4 v(1.2f, -2.7f, 3.1f, -4.9f);
            float4 result = v.ceil();
            suite.assert_approximately_equal(result, float4(2.0f, -2.0f, 4.0f, -4.0f), "ceil()");
        }

        // Тест round
        {
            float4 v(1.4f, 1.6f, -1.5f, -2.5f);
            float4 result = v.round();
            suite.assert_approximately_equal(result, float4(1.0f, 2.0f, -2.0f, -2.0f), "round()");
        }

        // Тест frac
        {
            float4 v(1.7f, -2.3f, 3.0f, -4.9f);
            float4 result = v.frac();
            suite.assert_approximately_equal(result.x, 0.7f, "frac() x", 1e-6f);
            suite.assert_approximately_equal(result.y, 0.7f, "frac() y", 1e-6f); // -2.3 - (-3) = 0.7
            suite.assert_approximately_equal(result.z, 0.0f, "frac() z", 1e-6f);
            suite.assert_approximately_equal(result.w, 0.1f, "frac() w", 1e-6f); // -4.9 - (-5) = 0.1
        }

        // Тест saturate
        {
            float4 v(-0.5f, 0.5f, 1.5f, 0.0f);
            float4 result = v.saturate();
            suite.assert_approximately_equal(result, float4(0.0f, 0.5f, 1.0f, 0.0f), "saturate()");
        }

        // Тест step
        {
            float4 v(0.5f, 1.0f, 1.5f, 2.0f);
            float4 result = v.step(1.0f);
            suite.assert_approximately_equal(result, float4(0.0f, 1.0f, 1.0f, 1.0f), "step()");
        }

        // Тест min/max
        {
            float4 a(1.0f, 3.0f, 5.0f, 7.0f);
            float4 b(2.0f, 2.0f, 6.0f, 8.0f);

            float4 min_result = float4::min(a, b);
            suite.assert_approximately_equal(min_result, float4(1.0f, 2.0f, 5.0f, 7.0f), "min()");

            float4 max_result = float4::max(a, b);
            suite.assert_approximately_equal(max_result, float4(2.0f, 3.0f, 6.0f, 8.0f), "max()");
        }

        // ============================================================================
        // 7. Цветовые операции
        // ============================================================================
        suite.section("Цветовые операции");

        // Тест luminance
        {
            // Белый цвет: (1, 1, 1, 1)
            // Luminance = 0.2126*1 + 0.7152*1 + 0.0722*1 = 1.0
            float4 white(1.0f, 1.0f, 1.0f, 1.0f);
            suite.assert_approximately_equal(white.luminance(), 1.0f, "luminance() white", 1e-6f);

            // Красный цвет: (1, 0, 0, 1)
            // Luminance = 0.2126*1 + 0.7152*0 + 0.0722*0 = 0.2126
            float4 red(1.0f, 0.0f, 0.0f, 1.0f);
            suite.assert_approximately_equal(red.luminance(), 0.2126f, "luminance() red", 1e-6f);

            // Зеленый цвет: (0, 1, 0, 1)
            // Luminance = 0.2126*0 + 0.7152*1 + 0.0722*0 = 0.7152
            float4 green(0.0f, 1.0f, 0.0f, 1.0f);
            suite.assert_approximately_equal(green.luminance(), 0.7152f, "luminance() green", 1e-6f);

            // Синий цвет: (0, 0, 1, 1)
            // Luminance = 0.2126*0 + 0.7152*0 + 0.0722*1 = 0.0722
            float4 blue(0.0f, 0.0f, 1.0f, 1.0f);
            suite.assert_approximately_equal(blue.luminance(), 0.0722f, "luminance() blue", 1e-6f);
        }

        // Тест brightness
        {
            // Белый цвет: (1, 1, 1, 1)
            // Brightness = (1 + 1 + 1) / 3 = 1.0
            float4 white(1.0f, 1.0f, 1.0f, 1.0f);
            suite.assert_approximately_equal(white.brightness(), 1.0f, "brightness() white");

            // Серый цвет: (0.5, 0.5, 0.5, 1.0)
            // Brightness = (0.5 + 0.5 + 0.5) / 3 = 0.5
            float4 gray(0.5f, 0.5f, 0.5f, 1.0f);
            suite.assert_approximately_equal(gray.brightness(), 0.5f, "brightness() gray");
        }

        // Тест premultiply_alpha
        {
            float4 color(0.5f, 0.6f, 0.7f, 0.8f);
            float4 premultiplied = color.premultiply_alpha();
            // RGB компоненты умножаются на alpha, alpha остается прежней
            suite.assert_approximately_equal(premultiplied.x, 0.5f * 0.8f, "premultiply_alpha() red", 1e-6f);
            suite.assert_approximately_equal(premultiplied.y, 0.6f * 0.8f, "premultiply_alpha() green", 1e-6f);
            suite.assert_approximately_equal(premultiplied.z, 0.7f * 0.8f, "premultiply_alpha() blue", 1e-6f);
            suite.assert_approximately_equal(premultiplied.w, 0.8f, "premultiply_alpha() alpha unchanged");

            // Тест с alpha = 0
            float4 transparent(0.5f, 0.6f, 0.7f, 0.0f);
            float4 premult_transparent = transparent.premultiply_alpha();
            suite.assert_approximately_equal(premult_transparent, float4(0.0f, 0.0f, 0.0f, 0.0f), "premultiply_alpha() with alpha=0");
        }

        // Тест unpremultiply_alpha
        {
            // Создаем premultiplied цвет
            float4 premultiplied(0.4f, 0.48f, 0.56f, 0.8f); // (0.5*0.8, 0.6*0.8, 0.7*0.8, 0.8)
            float4 unpremultiplied = premultiplied.unpremultiply_alpha();
            // Должны получить оригинальный цвет (0.5, 0.6, 0.7, 0.8)
            suite.assert_approximately_equal(unpremultiplied.x, 0.5f, "unpremultiply_alpha() red", 1e-6f);
            suite.assert_approximately_equal(unpremultiplied.y, 0.6f, "unpremultiply_alpha() green", 1e-6f);
            suite.assert_approximately_equal(unpremultiplied.z, 0.7f, "unpremultiply_alpha() blue", 1e-6f);
            suite.assert_approximately_equal(unpremultiplied.w, 0.8f, "unpremultiply_alpha() alpha unchanged");

            // Тест с alpha = 0 (должен вернуть оригинальный вектор)
            float4 transparent(0.5f, 0.6f, 0.7f, 0.0f);
            float4 unpremult_transparent = transparent.unpremultiply_alpha();
            suite.assert_approximately_equal(unpremult_transparent, transparent, "unpremultiply_alpha() with alpha=0");
        }

        // Тест grayscale
        {
            float4 color(0.5f, 0.6f, 0.7f, 0.8f);
            float lum = color.luminance();
            float4 grayscale_result = color.grayscale();

            suite.assert_approximately_equal(grayscale_result.x, lum, "grayscale() red = luminance", 1e-6f);
            suite.assert_approximately_equal(grayscale_result.y, lum, "grayscale() green = luminance", 1e-6f);
            suite.assert_approximately_equal(grayscale_result.z, lum, "grayscale() blue = luminance", 1e-6f);
            suite.assert_approximately_equal(grayscale_result.w, 0.8f, "grayscale() alpha unchanged");
        }

        // ============================================================================
        // 8. Геометрические операции
        // ============================================================================
        suite.section("Геометрические операции");

        // Тест project (проекция однородных координат)
        {
            // Однородные координаты точки (2, 3, 4, 1)
            float4 homogeneous(2.0f, 3.0f, 4.0f, 1.0f);
            float3 projected = homogeneous.project();
            // Проекция: (x/w, y/w, z/w) = (2/1, 3/1, 4/1) = (2, 3, 4)
            suite.assert_approximately_equal(projected, float3(2.0f, 3.0f, 4.0f), "project() with w=1");

            // Однородные координаты точки (4, 6, 8, 2)
            float4 homogeneous2(4.0f, 6.0f, 8.0f, 2.0f);
            float3 projected2 = homogeneous2.project();
            // Проекция: (4/2, 6/2, 8/2) = (2, 3, 4)
            suite.assert_approximately_equal(projected2, float3(2.0f, 3.0f, 4.0f), "project() with w=2");

            // Тест с w=0 (точка на бесконечности)
            float4 infinite(2.0f, 3.0f, 4.0f, 0.0f);
            float3 infinite_projected = infinite.project();
            // Должен вернуть нулевой вектор
            suite.assert_approximately_equal(infinite_projected, float3::zero(), "project() with w=0");
        }

        // Тест to_homogeneous
        {
            float4 point(2.0f, 3.0f, 4.0f, 0.5f);
            float4 homogeneous = point.to_homogeneous();
            // Должен установить w=1, остальные компоненты сохранить
            suite.assert_approximately_equal(homogeneous, float4(2.0f, 3.0f, 4.0f, 1.0f), "to_homogeneous()");
        }

        // ============================================================================
        // 9. Swizzle операции
        // ============================================================================
        suite.section("Swizzle операции");

        {
            float4 v(2.0f, 3.0f, 4.0f, 5.0f);

            // 2-компонентные swizzles
            suite.assert_approximately_equal(v.xy(), float2(2.0f, 3.0f), "xy()");
            suite.assert_approximately_equal(v.xz(), float2(2.0f, 4.0f), "xz()");
            suite.assert_approximately_equal(v.xw(), float2(2.0f, 5.0f), "xw()");
            suite.assert_approximately_equal(v.yz(), float2(3.0f, 4.0f), "yz()");
            suite.assert_approximately_equal(v.yw(), float2(3.0f, 5.0f), "yw()");
            suite.assert_approximately_equal(v.zw(), float2(4.0f, 5.0f), "zw()");

            // 3-компонентные swizzles
            suite.assert_approximately_equal(v.xyz(), float3(2.0f, 3.0f, 4.0f), "xyz()");
            suite.assert_approximately_equal(v.xyw(), float3(2.0f, 3.0f, 5.0f), "xyw()");
            suite.assert_approximately_equal(v.xzw(), float3(2.0f, 4.0f, 5.0f), "xzw()");
            suite.assert_approximately_equal(v.yzw(), float3(3.0f, 4.0f, 5.0f), "yzw()");

            // 4-компонентные swizzles
            suite.assert_approximately_equal(v.yxzw(), float4(3.0f, 2.0f, 4.0f, 5.0f), "yxzw()");
            suite.assert_approximately_equal(v.zxyw(), float4(4.0f, 2.0f, 3.0f, 5.0f), "zxyw()");
            suite.assert_approximately_equal(v.zyxw(), float4(4.0f, 3.0f, 2.0f, 5.0f), "zyxw()");
            suite.assert_approximately_equal(v.wzyx(), float4(5.0f, 4.0f, 3.0f, 2.0f), "wzyx()");

            // Цветовые swizzles
            suite.assert_approximately_equal(v.r(), 2.0f, "r()");
            suite.assert_approximately_equal(v.g(), 3.0f, "g()");
            suite.assert_approximately_equal(v.b(), 4.0f, "b()");
            suite.assert_approximately_equal(v.a(), 5.0f, "a()");
            suite.assert_approximately_equal(v.rg(), float2(2.0f, 3.0f), "rg()");
            suite.assert_approximately_equal(v.rb(), float2(2.0f, 4.0f), "rb()");
            suite.assert_approximately_equal(v.ra(), float2(2.0f, 5.0f), "ra()");
            suite.assert_approximately_equal(v.gb(), float2(3.0f, 4.0f), "gb()");
            suite.assert_approximately_equal(v.ga(), float2(3.0f, 5.0f), "ga()");
            suite.assert_approximately_equal(v.ba(), float2(4.0f, 5.0f), "ba()");
            suite.assert_approximately_equal(v.rgb(), float3(2.0f, 3.0f, 4.0f), "rgb()");
            suite.assert_approximately_equal(v.rga(), float3(2.0f, 3.0f, 5.0f), "rga()");
            suite.assert_approximately_equal(v.rba(), float3(2.0f, 4.0f, 5.0f), "rba()");
            suite.assert_approximately_equal(v.gba(), float3(3.0f, 4.0f, 5.0f), "gba()");
            suite.assert_approximately_equal(v.grba(), float4(3.0f, 2.0f, 4.0f, 5.0f), "grba()");
            suite.assert_approximately_equal(v.brga(), float4(4.0f, 2.0f, 3.0f, 5.0f), "brga()");
            suite.assert_approximately_equal(v.bgra(), float4(4.0f, 3.0f, 2.0f, 5.0f), "bgra()");
            suite.assert_approximately_equal(v.abgr(), float4(5.0f, 4.0f, 3.0f, 2.0f), "abgr()");
        }

        // ============================================================================
        // 10. Утилитарные методы
        // ============================================================================
        suite.section("Утилитарные методы");

        // Тест isValid
        {
            float4 valid(1.0f, 2.0f, 3.0f, 4.0f);
            suite.assert_true(valid.isValid(), "isValid() for valid vector");
            suite.skip_test("isValid() with NaN/INF", "Requires special NaN/INF construction");
        }

        // Тест approximately
        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 b(1.000001f, 2.000001f, 3.000001f, 4.000001f);
            float4 c(1.1f, 2.1f, 3.1f, 4.1f);

            suite.assert_true(a.approximately(b, 1e-5f), "approximately() within epsilon");
            suite.assert_false(a.approximately(c, 1e-5f), "approximately() outside epsilon");
        }

        // Тест approximately_zero
        {
            float4 zero(0.0f, 0.0f, 0.0f, 0.0f);
            float4 near_zero(0.000001f, 0.000001f, 0.000001f, 0.000001f);
            float4 not_zero(0.1f, 0.1f, 0.1f, 0.1f);

            suite.assert_true(zero.approximately_zero(1e-5f), "approximately_zero() for zero");
            suite.assert_true(near_zero.approximately_zero(1e-4f), "approximately_zero() for near zero");
            suite.assert_false(not_zero.approximately_zero(1e-5f), "approximately_zero() for non-zero");
        }

        // Тест is_normalized
        {
            float4 normalized(0.182574f, 0.365148f, 0.547723f, 0.730297f); // (1,2,3,4) normalized
            float4 not_normalized(1.0f, 2.0f, 3.0f, 4.0f);

            suite.assert_true(normalized.is_normalized(1e-5f), "is_normalized() for normalized vector");
            suite.assert_false(not_normalized.is_normalized(1e-5f), "is_normalized() for non-normalized");
        }

        // Тест to_string
        {
            float4 v(1.5f, 2.5f, 3.5f, 4.5f);
            std::string str = v.to_string();

            suite.assert_true(str.find("1.5") != std::string::npos || str.find("1.500") != std::string::npos,
                "to_string() contains x value");
            suite.assert_true(str.find("2.5") != std::string::npos || str.find("2.500") != std::string::npos,
                "to_string() contains y value");
            suite.assert_true(str.find("3.5") != std::string::npos || str.find("3.500") != std::string::npos,
                "to_string() contains z value");
            suite.assert_true(str.find("4.5") != std::string::npos || str.find("4.500") != std::string::npos,
                "to_string() contains w value");
        }

        // Тест data()
        {
            float4 v(7.0f, 8.0f, 9.0f, 10.0f);
            const float* cdata = v.data();
            float* data = v.data();

            suite.assert_approximately_equal(cdata[0], 7.0f, "data() const access x");
            suite.assert_approximately_equal(cdata[1], 8.0f, "data() const access y");
            suite.assert_approximately_equal(cdata[2], 9.0f, "data() const access z");
            suite.assert_approximately_equal(cdata[3], 10.0f, "data() const access w");

            data[0] = 11.0f;
            suite.assert_approximately_equal(v.x, 11.0f, "data() mutable modification");
        }

        // Тест set_xyz
        {
            float4 v(1.0f, 2.0f, 3.0f, 4.0f);
            float3 xyz(5.0f, 6.0f, 7.0f);
            v.set_xyz(xyz);
            suite.assert_approximately_equal(v.x, 5.0f, "set_xyz() x");
            suite.assert_approximately_equal(v.y, 6.0f, "set_xyz() y");
            suite.assert_approximately_equal(v.z, 7.0f, "set_xyz() z");
            suite.assert_approximately_equal(v.w, 4.0f, "set_xyz() w unchanged");
        }

        // Тест set_xy
        {
            float4 v(1.0f, 2.0f, 3.0f, 4.0f);
            float2 xy(5.0f, 6.0f);
            v.set_xy(xy);
            suite.assert_approximately_equal(v.x, 5.0f, "set_xy() x");
            suite.assert_approximately_equal(v.y, 6.0f, "set_xy() y");
            suite.assert_approximately_equal(v.z, 3.0f, "set_xy() z unchanged");
            suite.assert_approximately_equal(v.w, 4.0f, "set_xy() w unchanged");
        }

        // Тест set_zw
        {
            float4 v(1.0f, 2.0f, 3.0f, 4.0f);
            float2 zw(5.0f, 6.0f);
            v.set_zw(zw);
            suite.assert_approximately_equal(v.x, 1.0f, "set_zw() x unchanged");
            suite.assert_approximately_equal(v.y, 2.0f, "set_zw() y unchanged");
            suite.assert_approximately_equal(v.z, 5.0f, "set_zw() z");
            suite.assert_approximately_equal(v.w, 6.0f, "set_zw() w");
        }

        // Тест SSE методов
        {
            float data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };

            // Тест load_unaligned
            float4 v1 = float4::load_unaligned(data);
            suite.assert_approximately_equal(v1, float4(1.0f, 2.0f, 3.0f, 4.0f), "load_unaligned()");

            // Тест store_unaligned
            float data_out[4];
            v1.store_unaligned(data_out);
            suite.assert_approximately_equal(data_out[0], 1.0f, "store_unaligned() [0]");
            suite.assert_approximately_equal(data_out[1], 2.0f, "store_unaligned() [1]");
            suite.assert_approximately_equal(data_out[2], 3.0f, "store_unaligned() [2]");
            suite.assert_approximately_equal(data_out[3], 4.0f, "store_unaligned() [3]");

            // Тест get_simd и set_simd
            __m128 simd = v1.get_simd();
            float4 v2;
            v2.set_simd(simd);
            suite.assert_approximately_equal(v2, v1, "get_simd()/set_simd()");
        }

        // ============================================================================
        // 11. Операторы сравнения
        // ============================================================================
        suite.section("Операторы сравнения");

        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 b(1.0f, 2.0f, 3.0f, 4.0f);
            float4 c(1.1f, 2.1f, 3.1f, 4.1f);

            suite.assert_true(a == b, "Operator == for equal vectors");
            suite.assert_false(a == c, "Operator == for different vectors");
            suite.assert_false(a != b, "Operator != for equal vectors");
            suite.assert_true(a != c, "Operator != for different vectors");
        }

        // ============================================================================
        // 12. Глобальные операторы и функции
        // ============================================================================
        suite.section("Глобальные операторы и функции");

        // Тест глобальных операторов
        {
            float4 a(2.0f, 3.0f, 4.0f, 5.0f);
            float4 b(1.0f, 2.0f, 3.0f, 4.0f);

            float4 add_result = a + b;
            suite.assert_approximately_equal(add_result, float4(3.0f, 5.0f, 7.0f, 9.0f), "Global operator +");

            float4 sub_result = a - b;
            suite.assert_approximately_equal(sub_result, float4(1.0f, 1.0f, 1.0f, 1.0f), "Global operator -");

            float4 mul_result = a * b;
            suite.assert_approximately_equal(mul_result, float4(2.0f, 6.0f, 12.0f, 20.0f), "Global operator *");

            float4 div_result = a / b;
            suite.assert_approximately_equal(div_result, float4(2.0f, 1.5f, 4.0f / 3.0f, 1.25f), "Global operator /", 1e-6f);
        }

        // Тест глобальных математических функций
        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 b(5.0f, 6.0f, 7.0f, 8.0f);

            suite.assert_approximately_equal(distance(a, b), std::sqrt(64.0f), "Global distance()");
            suite.assert_approximately_equal(distance_sq(a, b), 64.0f, "Global distance_sq()");
            suite.assert_approximately_equal(dot(a, b), 70.0f, "Global dot()");
            suite.assert_approximately_equal(dot3(a, b), 38.0f, "Global dot3()");

            float4 cross_result = cross(a, b);
            // (2*7 - 3*6, 3*5 - 1*7, 1*6 - 2*5, 0) = (14-18, 15-7, 6-10, 0) = (-4, 8, -4, 0)
            suite.assert_approximately_equal(cross_result, float4(-4.0f, 8.0f, -4.0f, 0.0f), "Global cross()");

            float4 norm = normalize(a);
            suite.assert_approximately_equal(norm.length(), 1.0f, "Global normalize() length", 1e-6f);

            float4 lerp_result = lerp(a, b, 0.5f);
            suite.assert_approximately_equal(lerp_result, float4(3.0f, 4.0f, 5.0f, 6.0f), "Global lerp()");
        }

        // Тест глобальных approximately
        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 b(1.000001f, 2.000001f, 3.000001f, 4.000001f);

            suite.assert_true(approximately(a, b, 1e-5f), "Global approximately()");
            suite.assert_true(is_normalized(float4(1.0f, 0.0f, 0.0f, 0.0f), 1e-5f), "Global is_normalized()");
            suite.assert_true(isValid(a), "Global isValid()");
        }

        // Тест глобальных HLSL-функций
        {
            float4 v(-1.5f, 2.5f, -3.5f, 4.5f);

            suite.assert_approximately_equal(abs(v), float4(1.5f, 2.5f, 3.5f, 4.5f), "Global abs()");
            suite.assert_approximately_equal(sign(v), float4(-1.0f, 1.0f, -1.0f, 1.0f), "Global sign()");
            suite.assert_approximately_equal(floor(v), float4(-2.0f, 2.0f, -4.0f, 4.0f), "Global floor()");
            suite.assert_approximately_equal(ceil(v), float4(-1.0f, 3.0f, -3.0f, 5.0f), "Global ceil()");
            suite.assert_approximately_equal(round(float4(1.4f, 1.6f, -1.5f, 2.5f)),
                float4(1.0f, 2.0f, -2.0f, 2.0f), "Global round()");
            suite.assert_approximately_equal(saturate(float4(-0.5f, 0.5f, 1.5f, 0.0f)),
                float4(0.0f, 0.5f, 1.0f, 0.0f), "Global saturate()");
            suite.assert_approximately_equal(frac(float4(1.7f, -2.3f, 3.0f, 4.8f)),
                float4(0.7f, 0.7f, 0.0f, 0.8f), "Global frac()", 1e-6f);
            suite.assert_approximately_equal(step(1.0f, float4(0.5f, 1.0f, 1.5f, 2.0f)),
                float4(0.0f, 1.0f, 1.0f, 1.0f), "Global step()");
        }

        // Тест глобальных clamp, min, max
        {
            float4 v(0.5f, 1.5f, -0.5f, 2.0f);
            float4 min_val(0.0f, 0.0f, 0.0f, 0.0f);
            float4 max_val(1.0f, 1.0f, 1.0f, 1.0f);

            float4 clamped = clamp(v, min_val, max_val);
            suite.assert_approximately_equal(clamped, float4(0.5f, 1.0f, 0.0f, 1.0f), "Global clamp()");

            float4 a(1.0f, 3.0f, 5.0f, 7.0f);
            float4 b(2.0f, 2.0f, 6.0f, 8.0f);

            float4 min_result = min(a, b);
            suite.assert_approximately_equal(min_result, float4(1.0f, 2.0f, 5.0f, 7.0f), "Global min()");

            float4 max_result = max(a, b);
            suite.assert_approximately_equal(max_result, float4(2.0f, 3.0f, 6.0f, 8.0f), "Global max()");
        }

        // Тест глобальных цветовых операций
        {
            float4 color(0.5f, 0.6f, 0.7f, 0.8f);

            suite.assert_approximately_equal(luminance(color), color.luminance(), "Global luminance()");
            suite.assert_approximately_equal(brightness(color), color.brightness(), "Global brightness()");

            float4 premult = premultiply_alpha(color);
            suite.assert_approximately_equal(premult, color.premultiply_alpha(), "Global premultiply_alpha()");

            float4 unpremult = unpremultiply_alpha(premult);
            suite.assert_approximately_equal(unpremult, premult.unpremultiply_alpha(), "Global unpremultiply_alpha()");

            float4 gray = grayscale(color);
            suite.assert_approximately_equal(gray, color.grayscale(), "Global grayscale()");
        }

        // Тест глобальных геометрических операций
        {
            float4 homogeneous(2.0f, 3.0f, 4.0f, 1.0f);

            float3 projected = project(homogeneous);
            suite.assert_approximately_equal(projected, homogeneous.project(), "Global project()");

            float4 to_homo = to_homogeneous(homogeneous);
            suite.assert_approximately_equal(to_homo, homogeneous.to_homogeneous(), "Global to_homogeneous()");
        }

        // ============================================================================
        // 13. Граничные случаи
        // ============================================================================
        suite.section("Граничные случаи");

        // Тест с очень маленькими значениями
        {
            float epsilon = 1e-30f;
            float4 tiny(epsilon, epsilon, epsilon, epsilon);
            suite.assert_true(tiny.approximately_zero(1e-20f), "Tiny values approximately_zero");
        }

        // Тест с очень большими значениями
        {
            float large = 1e10f;
            float4 huge(large, large, large, large);
            suite.assert_false(huge.approximately_zero(), "Huge values not approximately_zero");

            float4 normalized_huge = huge.normalize();
            float expected_length = 1.0f;
            suite.assert_approximately_equal(normalized_huge.length(), expected_length,
                "Normalize huge values", 1e-6f);
        }

        // Тест деления на ноль (векторное)
        {
            float4 a(1.0f, 2.0f, 3.0f, 4.0f);
            float4 zero_vec(0.0f, 0.0f, 0.0f, 0.0f);

            // Должно вызвать деление на ноль
            float4 result = a / zero_vec;

            // Проверяем что результат содержит бесконечности или NaN
            suite.assert_true(!std::isfinite(result.x) || !std::isfinite(result.y) ||
                !std::isfinite(result.z) || !std::isfinite(result.w),
                "Division by zero vector produces non-finite values");
        }

        // Тест деления на ноль (скалярное)
        {
            float4 v(1.0f, 2.0f, 3.0f, 4.0f);
            float4 result = v / 0.0f;

            suite.assert_true(!std::isfinite(result.x) && !std::isfinite(result.y) &&
                !std::isfinite(result.z) && !std::isfinite(result.w),
                "Division by zero scalar produces non-finite values");
        }

        // Тест нормализации очень маленького вектора
        {
            float4 tiny(1e-20f, 1e-20f, 1e-20f, 1e-20f);
            float4 normalized = tiny.normalize();

            // Должен вернуть нулевой вектор, так как длина меньше epsilon
            suite.assert_approximately_equal(normalized, float4::zero(),
                "Normalize tiny vector returns zero");
        }

        // Тест linear interpolation
        {
            float4 a(0.0f, 0.0f, 0.0f, 0.0f);
            float4 b(10.0f, 20.0f, 30.0f, 40.0f);

            // Тест t за пределами [0,1]
            float4 before = lerp(a, b, -0.5f);
            suite.assert_approximately_equal(before, a, "lerp() with t < 0");

            float4 after = lerp(a, b, 1.5f);
            suite.assert_approximately_equal(after, b, "lerp() with t > 1");
        }

        // Тест скалярного произведения с самим собой
        {
            float4 v(3.0f, 4.0f, 5.0f, 6.0f);
            float dot_self = v.dot(v);
            float expected = v.length_sq();
            suite.assert_approximately_equal(dot_self, expected, "Dot product with self equals length squared");
        }

        // Тест 3D скалярного произведения с самим собой
        {
            float4 v(3.0f, 4.0f, 5.0f, 6.0f);
            float dot3_self = v.dot3(v);
            float3 xyz = v.xyz();
            float expected = xyz.length_sq();
            suite.assert_approximately_equal(dot3_self, expected, "Dot3 product with self equals 3D length squared");
        }

        // Тест векторного произведения с самим собой
        {
            float4 v(2.0f, 3.0f, 4.0f, 5.0f);
            float4 cross_self = v.cross(v);
            suite.assert_approximately_equal(cross_self, float4(0.0f, 0.0f, 0.0f, 0.0f), "Cross product with self equals zero");
        }

        // Тест is_normalized для нулевого вектора
        {
            float4 zero(0.0f, 0.0f, 0.0f, 0.0f);
            suite.assert_false(zero.is_normalized(), "Zero vector is not normalized");
        }

        // Тест premultiply/unpremultiply обратные операции
        {
            float4 color(0.5f, 0.6f, 0.7f, 0.8f);
            float4 premult = color.premultiply_alpha();
            float4 unpremult = premult.unpremultiply_alpha();

            // После premultiply и unpremultiply должны вернуться к исходному цвету
            suite.assert_approximately_equal(unpremult, color, "premultiply/unpremultiply roundtrip", 1e-6f);
        }

        // Тест project с разными значениями w
        {
            // w = 0.5
            float4 v1(2.0f, 3.0f, 4.0f, 0.5f);
            float3 p1 = project(v1);
            suite.assert_approximately_equal(p1, float3(4.0f, 6.0f, 8.0f), "project() with w=0.5");

            // w = -1
            float4 v2(2.0f, 3.0f, 4.0f, -1.0f);
            float3 p2 = project(v2);
            suite.assert_approximately_equal(p2, float3(-2.0f, -3.0f, -4.0f), "project() with w=-1");

            // w очень маленькое
            float4 v3(2.0f, 3.0f, 4.0f, 1e-20f);
            float3 p3 = project(v3);
            // Должен вернуть нулевой вектор, так как деление на очень маленькое w
            suite.assert_approximately_equal(p3, float3::zero(), "project() with very small w");
        }

        suite.footer();
    }
}
