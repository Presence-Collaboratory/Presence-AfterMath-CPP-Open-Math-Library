// Author: DeepSeek, NSDeathman
// Test suite for Math::half2 class

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunHalf2Tests()
    {
        TestSuite suite("half2 Tests", true);
        suite.header();

        using namespace AfterMath;

        // Константы для half-тестов
        constexpr float HALF_EPSILON = 0.002f;        // Общая точность half
        constexpr float HALF_MATH_EPSILON = 2e-3f;   // Математические функции
        constexpr float HALF_ANGLE_EPSILON = 0.2f;   // Тригонометрия
        constexpr float HALF_LARGE_EPSILON = 0.5f;   // Большие значения

        // ============================================================================
        // 1. Конструкторы
        // ============================================================================
        suite.section("Конструкторы");

        // Тест конструктора по умолчанию
        {
            half2 v;
            suite.assert_approximately_equal(float(v.x), 0.0f, "Default constructor x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 0.0f, "Default constructor y", HALF_EPSILON);
        }

        // Тест конструктора с half компонентами
        {
            half2 v(half(1.5f), half(2.5f));
            suite.assert_approximately_equal(float(v.x), 1.5f, "Half component constructor x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 2.5f, "Half component constructor y", HALF_EPSILON);
        }

        // Тест конструктора с float компонентами
        {
            half2 v(1.5f, 2.5f);
            suite.assert_approximately_equal(float(v.x), 1.5f, "Float component constructor x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 2.5f, "Float component constructor y", HALF_EPSILON);
        }

        // Тест конструктора со скаляром (half)
        {
            half2 v(half(3.0f));
            suite.assert_approximately_equal(float(v.x), 3.0f, "Half scalar constructor x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 3.0f, "Half scalar constructor y", HALF_EPSILON);
        }

        // Тест конструктора со скаляром (float)
        {
            half2 v(3.0f);
            suite.assert_approximately_equal(float(v.x), 3.0f, "Float scalar constructor x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 3.0f, "Float scalar constructor y", HALF_EPSILON);
        }

        // Тест конструктора из float2
        {
            float2 fv(4.0f, 5.0f);
            half2 v(fv);
            suite.assert_approximately_equal(float(v.x), 4.0f, "float2 constructor x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 5.0f, "float2 constructor y", HALF_EPSILON);
        }

        // Тест копирующего конструктора
        {
            half2 original(6.0f, 7.0f);
            half2 copy(original);
            suite.assert_approximately_equal(float(copy.x), 6.0f, "Copy constructor x", HALF_EPSILON);
            suite.assert_approximately_equal(float(copy.y), 7.0f, "Copy constructor y", HALF_EPSILON);
        }

        // Тест статических конструкторов
        {
            half2 zero = half2::zero();
            suite.assert_approximately_equal(float(zero.x), 0.0f, "zero() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(zero.y), 0.0f, "zero() y", HALF_EPSILON);

            half2 one = half2::one();
            suite.assert_approximately_equal(float(one.x), 1.0f, "one() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(one.y), 1.0f, "one() y", HALF_EPSILON);

            half2 unit_x = half2::unit_x();
            suite.assert_approximately_equal(float(unit_x.x), 1.0f, "unit_x() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(unit_x.y), 0.0f, "unit_x() y", HALF_EPSILON);

            half2 unit_y = half2::unit_y();
            suite.assert_approximately_equal(float(unit_y.x), 0.0f, "unit_y() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(unit_y.y), 1.0f, "unit_y() y", HALF_EPSILON);

            half2 uv = half2::uv(half(0.3f), half(0.7f));
            suite.assert_approximately_equal(float(uv.x), 0.3f, "uv() u", HALF_EPSILON);
            suite.assert_approximately_equal(float(uv.y), 0.7f, "uv() v", HALF_EPSILON);
        }

        // ============================================================================
        // 2. Операторы присваивания
        // ============================================================================
        suite.section("Операторы присваивания");

        // Тест присваивания half скаляра
        {
            half2 v;
            v = half(2.5f);
            suite.assert_approximately_equal(float(v.x), 2.5f, "Half scalar assignment x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 2.5f, "Half scalar assignment y", HALF_EPSILON);
        }

        // Тест присваивания float скаляра
        {
            half2 v;
            v = 2.5f;
            suite.assert_approximately_equal(float(v.x), 2.5f, "Float scalar assignment x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 2.5f, "Float scalar assignment y", HALF_EPSILON);
        }

        // Тест присваивания float2
        {
            half2 v;
            float2 fv(3.5f, 4.5f);
            v = fv;
            suite.assert_approximately_equal(float(v.x), 3.5f, "float2 assignment x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 4.5f, "float2 assignment y", HALF_EPSILON);
        }

        // Тест составных операторов присваивания (half2)
        {
            half2 v(1.0f, 2.0f);

            v += half2(3.0f, 4.0f);
            suite.assert_approximately_equal(float(v.x), 4.0f, "Operator += x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 6.0f, "Operator += y", HALF_EPSILON);

            v -= half2(1.0f, 2.0f);
            suite.assert_approximately_equal(float(v.x), 3.0f, "Operator -= x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 4.0f, "Operator -= y", HALF_EPSILON);

            v *= half2(2.0f, 3.0f);
            suite.assert_approximately_equal(float(v.x), 6.0f, "Operator *= x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 12.0f, "Operator *= y", HALF_EPSILON);

            v /= half2(2.0f, 3.0f);
            suite.assert_approximately_equal(float(v.x), 3.0f, "Operator /= x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 4.0f, "Operator /= y", HALF_EPSILON);
        }

        // Тест составных операторов присваивания (скаляры)
        {
            half2 v(2.0f, 3.0f);

            v *= half(2.0f);
            suite.assert_approximately_equal(float(v.x), 4.0f, "Operator *= half scalar x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 6.0f, "Operator *= half scalar y", HALF_EPSILON);

            v /= half(2.0f);
            suite.assert_approximately_equal(float(v.x), 2.0f, "Operator /= half scalar x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 3.0f, "Operator /= half scalar y", HALF_EPSILON);

            v *= 2.0f;
            suite.assert_approximately_equal(float(v.x), 4.0f, "Operator *= float scalar x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 6.0f, "Operator *= float scalar y", HALF_EPSILON);

            v /= 2.0f;
            suite.assert_approximately_equal(float(v.x), 2.0f, "Operator /= float scalar x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 3.0f, "Operator /= float scalar y", HALF_EPSILON);
        }

        // ============================================================================
        // 3. Бинарные операторы
        // ============================================================================
        suite.section("Бинарные операторы");

        // Тест сложения
        {
            half2 a(1.0f, 2.0f);
            half2 b(3.0f, 4.0f);
            half2 result = a + b;
            suite.assert_approximately_equal(float(result.x), 4.0f, "Operator + x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 6.0f, "Operator + y", HALF_EPSILON);
        }

        // Тест вычитания
        {
            half2 a(5.0f, 6.0f);
            half2 b(2.0f, 3.0f);
            half2 result = a - b;
            suite.assert_approximately_equal(float(result.x), 3.0f, "Operator - x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 3.0f, "Operator - y", HALF_EPSILON);
        }

        // Тест унарных операторов
        {
            half2 a(1.0f, 2.0f);
            half2 pos = +a;
            suite.assert_approximately_equal(float(pos.x), 1.0f, "Unary + x", HALF_EPSILON);
            suite.assert_approximately_equal(float(pos.y), 2.0f, "Unary + y", HALF_EPSILON);

            half2 neg = -a;
            suite.assert_approximately_equal(float(neg.x), -1.0f, "Unary - x", HALF_EPSILON);
            suite.assert_approximately_equal(float(neg.y), -2.0f, "Unary - y", HALF_EPSILON);
        }

        // Тест скалярных операций (half)
        {
            half2 v(2.0f, 3.0f);

            half2 result1 = v + half(1.0f);
            suite.assert_approximately_equal(float(result1.x), 3.0f, "Vector + half x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result1.y), 4.0f, "Vector + half y", HALF_EPSILON);

            half2 result2 = half(1.0f) + v;
            suite.assert_approximately_equal(float(result2.x), 3.0f, "half + vector x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result2.y), 4.0f, "half + vector y", HALF_EPSILON);

            half2 result3 = v * half(2.0f);
            suite.assert_approximately_equal(float(result3.x), 4.0f, "Vector * half x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result3.y), 6.0f, "Vector * half y", HALF_EPSILON);

            half2 result4 = half(2.0f) * v;
            suite.assert_approximately_equal(float(result4.x), 4.0f, "half * vector x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result4.y), 6.0f, "half * vector y", HALF_EPSILON);

            half2 result5 = v / half(2.0f);
            suite.assert_approximately_equal(float(result5.x), 1.0f, "Vector / half x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result5.y), 1.5f, "Vector / half y", HALF_EPSILON);
        }

        // Тест скалярных операций (float)
        {
            half2 v(2.0f, 3.0f);

            half2 result1 = v + 1.0f;
            suite.assert_approximately_equal(float(result1.x), 3.0f, "Vector + float x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result1.y), 4.0f, "Vector + float y", HALF_EPSILON);

            half2 result2 = 1.0f + v;
            suite.assert_approximately_equal(float(result2.x), 3.0f, "float + vector x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result2.y), 4.0f, "float + vector y", HALF_EPSILON);

            half2 result3 = v * 2.0f;
            suite.assert_approximately_equal(float(result3.x), 4.0f, "Vector * float x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result3.y), 6.0f, "Vector * float y", HALF_EPSILON);

            half2 result4 = 2.0f * v;
            suite.assert_approximately_equal(float(result4.x), 4.0f, "float * vector x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result4.y), 6.0f, "float * vector y", HALF_EPSILON);

            half2 result5 = v / 2.0f;
            suite.assert_approximately_equal(float(result5.x), 1.0f, "Vector / float x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result5.y), 1.5f, "Vector / float y", HALF_EPSILON);
        }

        // Тест операций с float2
        {
            half2 hv(1.0f, 2.0f);
            float2 fv(3.0f, 4.0f);

            half2 add = hv + fv;
            suite.assert_approximately_equal(float(add.x), 4.0f, "half2 + float2 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(add.y), 6.0f, "half2 + float2 y", HALF_EPSILON);

            half2 add2 = fv + hv;
            suite.assert_approximately_equal(float(add2.x), 4.0f, "float2 + half2 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(add2.y), 6.0f, "float2 + half2 y", HALF_EPSILON);
        }

        // ============================================================================
        // 4. Операторы доступа и преобразования
        // ============================================================================
        suite.section("Операторы доступа и преобразования");

        // Тест оператора индексации
        {
            half2 v(7.0f, 8.0f);
            suite.assert_approximately_equal(float(v[0]), 7.0f, "Operator [] index 0", HALF_EPSILON);
            suite.assert_approximately_equal(float(v[1]), 8.0f, "Operator [] index 1", HALF_EPSILON);

            v[0] = half(9.0f);
            v[1] = half(10.0f);
            suite.assert_approximately_equal(float(v.x), 9.0f, "Operator [] mutable x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 10.0f, "Operator [] mutable y", HALF_EPSILON);
        }

        // Тест преобразования в float2
        {
            half2 v(1.5f, 2.5f);
            float2 fv = float2(v);
            suite.assert_approximately_equal(fv.x, 1.5f, "Conversion to float2 x", HALF_EPSILON);
            suite.assert_approximately_equal(fv.y, 2.5f, "Conversion to float2 y", HALF_EPSILON);
        }

        // Тест to_float2 функции
        {
            half2 v(3.0f, 4.0f);
            float2 fv = to_float2(v);
            suite.assert_approximately_equal(fv.x, 3.0f, "to_float2 x", HALF_EPSILON);
            suite.assert_approximately_equal(fv.y, 4.0f, "to_float2 y", HALF_EPSILON);
        }

        // Тест to_half2 функции
        {
            float2 fv(3.0f, 4.0f);
            half2 v = to_half2(fv);
            suite.assert_approximately_equal(float(v.x), 3.0f, "to_half2 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(v.y), 4.0f, "to_half2 y", HALF_EPSILON);
        }

        // ============================================================================
        // 5. Математические функции
        // ============================================================================
        suite.section("Математические функции");

        // Тест длины
        {
            half2 v(3.0f, 4.0f);
            half len = v.length();
            half len_sq = v.length_sq();

            suite.assert_approximately_equal(float(len), 5.0f, "length()", HALF_EPSILON);
            suite.assert_approximately_equal(float(len_sq), 25.0f, "length_sq()", HALF_EPSILON);

            half2 zero(0.0f, 0.0f);
            suite.assert_approximately_equal(float(zero.length()), 0.0f, "length() of zero vector", HALF_EPSILON);
        }

        // Тест нормализации
        {
            half2 v(3.0f, 4.0f);
            half2 normalized = v.normalize();
            float normalized_len = float(normalized.length());

            suite.assert_approximately_equal(normalized_len, 1.0f, "normalize() length", HALF_EPSILON);
            suite.assert_approximately_equal(float(normalized.x), 0.6f, "normalize() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(normalized.y), 0.8f, "normalize() y", HALF_EPSILON);

            // Тест нормализации нулевого вектора
            half2 zero(0.0f, 0.0f);
            half2 zero_norm = zero.normalize();
            suite.assert_approximately_equal(float(zero_norm.x), 0.0f, "normalize() zero vector x", HALF_EPSILON);
            suite.assert_approximately_equal(float(zero_norm.y), 0.0f, "normalize() zero vector y", HALF_EPSILON);
        }

        // Тест скалярного произведения
        {
            half2 a(1.0f, 2.0f);
            half2 b(3.0f, 4.0f);
            half dot_result = a.dot(b);
            suite.assert_approximately_equal(float(dot_result), 11.0f, "dot()", HALF_EPSILON);

            // Статический метод dot
            half static_dot = half2::dot(a, b);
            suite.assert_approximately_equal(float(static_dot), 11.0f, "half2::dot()", HALF_EPSILON);

            // Глобальная функция dot
            half global_dot = dot(a, b);
            suite.assert_approximately_equal(float(global_dot), 11.0f, "global dot()", HALF_EPSILON);

            // Ортогональные векторы
            half2 orth1(1.0f, 0.0f);
            half2 orth2(0.0f, 1.0f);
            half orth_dot = orth1.dot(orth2);
            suite.assert_approximately_equal(float(orth_dot), 0.0f, "dot() orthogonal vectors", HALF_EPSILON);
        }

        // Тест векторного произведения (2D)
        {
            half2 a(1.0f, 2.0f);
            half2 b(3.0f, 4.0f);
            half cross_result = cross(a, b);
            suite.assert_approximately_equal(float(cross_result), -2.0f, "cross()", HALF_EPSILON);
        }

        // Тест расстояния
        {
            half2 a(1.0f, 2.0f);
            half2 b(4.0f, 6.0f);
            half dist = a.distance(b);
            half dist_sq = a.distance_sq(b);

            // (4-1)² + (6-2)² = 9 + 16 = 25
            suite.assert_approximately_equal(float(dist), 5.0f, "distance()", HALF_EPSILON);
            suite.assert_approximately_equal(float(dist_sq), 25.0f, "distance_sq()", HALF_EPSILON);

            // Глобальные функции distance и distance_sq
            half global_distance = distance(a, b);
            half global_distance_sq = distance_sq(a, b);
            suite.assert_approximately_equal(float(global_distance), 5.0f, "global distance()", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_distance_sq), 25.0f, "global distance_sq()", HALF_EPSILON);
        }

        // Тест перпендикуляра
        {
            half2 v(2.0f, 3.0f);
            half2 perp = v.perpendicular();
            half2 global_perp = perpendicular(v);

            suite.assert_approximately_equal(float(perp.x), -3.0f, "perpendicular() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(perp.y), 2.0f, "perpendicular() y", HALF_EPSILON);

            suite.assert_approximately_equal(float(global_perp.x), -3.0f, "global perpendicular() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_perp.y), 2.0f, "global perpendicular() y", HALF_EPSILON);
        }

        // Тест угла
        {
            half2 v(1.0f, 0.0f);
            half angle_result = angle(v);
            suite.assert_approximately_equal(float(angle_result), 0.0f, "angle() for (1,0)", HALF_ANGLE_EPSILON);

            half2 v2(0.0f, 1.0f);
            half angle2 = angle(v2);
            suite.assert_approximately_equal(float(angle2), Constants::Constants<float>::Pi / 2.0f,
                "angle() for (0,1)", HALF_ANGLE_EPSILON);
        }

        // Тест угла между векторами
        {
            half2 a(1.0f, 0.0f);
            half2 b(0.0f, 1.0f);
            half angle_ab = angle_between(a, b);
            suite.assert_approximately_equal(float(angle_ab), Constants::Constants<float>::Pi / 2.0f,
                "angle_between() 90 degrees", HALF_ANGLE_EPSILON);
        }

        // ============================================================================
        // 6. HLSL-подобные функции
        // ============================================================================
        suite.section("HLSL-подобные функции");

        // Тест abs
        {
            half2 v(-1.5f, 2.5f);
            half2 result = v.abs();
            half2 global_result = abs(v);

            suite.assert_approximately_equal(float(result.x), 1.5f, "abs() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 2.5f, "abs() y", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_result.x), 1.5f, "global abs() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_result.y), 2.5f, "global abs() y", HALF_EPSILON);
        }

        // Тест sign
        {
            half2 v(-2.0f, 3.0f);
            half2 result = v.sign();
            suite.assert_approximately_equal(float(result.x), -1.0f, "sign() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 1.0f, "sign() y", HALF_EPSILON);
        }

        // Тест floor
        {
            half2 v(1.7f, -2.3f);
            half2 result = v.floor();
            suite.assert_approximately_equal(float(result.x), 1.0f, "floor() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), -3.0f, "floor() y", HALF_EPSILON);
        }

        // Тест ceil
        {
            half2 v(1.2f, -2.7f);
            half2 result = v.ceil();
            suite.assert_approximately_equal(float(result.x), 2.0f, "ceil() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), -2.0f, "ceil() y", HALF_EPSILON);
        }

        // Тест round
        {
            half2 v(1.4f, 1.6f);
            half2 result = v.round();
            suite.assert_approximately_equal(float(result.x), 1.0f, "round() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 2.0f, "round() y", HALF_EPSILON);
        }

        // Тест frac (HLSL-семантика)
        {
            half2 v(1.7f, -2.3f);
            half2 result = v.frac();
            // HLSL: frac(x) = x - floor(x)
            // frac(1.7) = 1.7 - 1.0 = 0.7
            // frac(-2.3) = -2.3 - (-3.0) = 0.7
            suite.assert_approximately_equal(float(result.x), 0.7f, "frac() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 0.7f, "frac() y", HALF_EPSILON);
        }

        // Тест saturate
        {
            half2 v(-0.5f, 1.5f);
            half2 result = v.saturate();
            half2 static_result = half2::saturate(v);
            half2 global_result = saturate(v);

            suite.assert_approximately_equal(float(result.x), 0.0f, "saturate() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 1.0f, "saturate() y", HALF_EPSILON);
            suite.assert_approximately_equal(float(static_result.x), 0.0f, "half2::saturate() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(static_result.y), 1.0f, "half2::saturate() y", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_result.x), 0.0f, "global saturate() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_result.y), 1.0f, "global saturate() y", HALF_EPSILON);
        }

        // Тест step
        {
            half2 v(0.5f, 1.5f);
            half2 result = v.step(half(1.0f));
            half2 global_result = step(half(1.0f), v);

            suite.assert_approximately_equal(float(result.x), 0.0f, "step() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 1.0f, "step() y", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_result.x), 0.0f, "global step() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_result.y), 1.0f, "global step() y", HALF_EPSILON);
        }

        // Тест smoothstep
        {
            half2 v(0.5f, 1.5f);
            half2 result = smoothstep(half(0.0f), half(2.0f), v);
            // Для t=0.25: 3t² - 2t³ = 3*0.0625 - 2*0.015625 = 0.1875 - 0.03125 = 0.15625
            // Для t=0.75: 3*0.5625 - 2*0.421875 = 1.6875 - 0.84375 = 0.84375
            suite.assert_approximately_equal(float(result.x), 0.15625f, "smoothstep() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), 0.84375f, "smoothstep() y", HALF_EPSILON);
        }

        // Тест min/max
        {
            half2 a(1.0f, 3.0f);
            half2 b(2.0f, 2.0f);

            half2 min_result = half2::min(a, b);
            half2 max_result = half2::max(a, b);
            half2 global_min = min(a, b);
            half2 global_max = max(a, b);

            suite.assert_approximately_equal(float(min_result.x), 1.0f, "half2::min() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(min_result.y), 2.0f, "half2::min() y", HALF_EPSILON);
            suite.assert_approximately_equal(float(max_result.x), 2.0f, "half2::max() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(max_result.y), 3.0f, "half2::max() y", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_min.x), 1.0f, "global min() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_min.y), 2.0f, "global min() y", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_max.x), 2.0f, "global max() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_max.y), 3.0f, "global max() y", HALF_EPSILON);
        }

        // Тест clamp
        {
            half2 v(2.5f, -0.5f);
            half2 min_val(1.0f, 0.0f);
            half2 max_val(2.0f, 1.0f);

            half2 clamped = clamp(v, min_val, max_val);
            suite.assert_approximately_equal(float(clamped.x), 2.0f, "clamp() vector x", HALF_EPSILON);
            suite.assert_approximately_equal(float(clamped.y), 0.0f, "clamp() vector y", HALF_EPSILON);

            half2 clamped_scalar = clamp(v, 0.0f, 1.0f);
            suite.assert_approximately_equal(float(clamped_scalar.x), 1.0f, "clamp() scalar x", HALF_EPSILON);
            suite.assert_approximately_equal(float(clamped_scalar.y), 0.0f, "clamp() scalar y", HALF_EPSILON);
        }

        // ============================================================================
        // 7. Swizzle операции
        // ============================================================================
        suite.section("Swizzle операции");

        {
            half2 v(2.0f, 3.0f);

            half2 yx = v.yx();
            suite.assert_approximately_equal(float(yx.x), 3.0f, "yx() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(yx.y), 2.0f, "yx() y", HALF_EPSILON);

            half2 xx = v.xx();
            suite.assert_approximately_equal(float(xx.x), 2.0f, "xx() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(xx.y), 2.0f, "xx() y", HALF_EPSILON);

            half2 yy = v.yy();
            suite.assert_approximately_equal(float(yy.x), 3.0f, "yy() x", HALF_EPSILON);
            suite.assert_approximately_equal(float(yy.y), 3.0f, "yy() y", HALF_EPSILON);
        }

        // ============================================================================
        // 8. Текстурные координаты
        // ============================================================================
        suite.section("Текстурные координаты");

        {
            half2 uv(0.3f, 0.7f);

            half u = uv.u();
            half v = uv.v();
            suite.assert_approximately_equal(float(u), 0.3f, "u()", HALF_EPSILON);
            suite.assert_approximately_equal(float(v), 0.7f, "v()", HALF_EPSILON);

            uv.set_u(half(0.4f));
            uv.set_v(half(0.8f));
            suite.assert_approximately_equal(float(uv.x), 0.4f, "set_u()", HALF_EPSILON);
            suite.assert_approximately_equal(float(uv.y), 0.8f, "set_v()", HALF_EPSILON);
        }

        // ============================================================================
        // 9. Утилитарные методы
        // ============================================================================
        suite.section("Утилитарные методы");

        // Тест isValid
        {
            half2 valid(1.0f, 2.0f);
            suite.assert_true(valid.is_valid(), "is_valid() for valid vector");

            suite.assert_true(is_valid(valid), "global is_valid() for valid vector");
        }

        // Тест approximately
        {
            half2 a(1.0f, 2.0f);
            half2 b(1.001f, 2.001f);
            half2 c(1.1f, 2.1f);

            suite.assert_true(a.approximately(b, 0.01f), "approximately() within epsilon");
            suite.assert_false(a.approximately(c, 0.01f), "approximately() outside epsilon");

            bool global_approx = approximately(a, b, 0.01f);
            suite.assert_true(global_approx, "global approximately() within epsilon");
        }

        // Тест approximately_zero
        {
            half2 zero(0.0f, 0.0f);
            half2 near_zero(0.001f, 0.001f);
            half2 not_zero(0.1f, 0.1f);

            suite.assert_true(zero.approximately_zero(0.01f), "approximately_zero() for zero");
            suite.assert_true(near_zero.approximately_zero(0.01f), "approximately_zero() for near zero");
            suite.assert_false(not_zero.approximately_zero(0.01f), "approximately_zero() for non-zero");
        }

        // Тест is_normalized
        {
            half2 normalized(0.6f, 0.8f);
            half2 not_normalized(1.0f, 2.0f);

            suite.assert_true(normalized.is_normalized(0.01f), "is_normalized() for normalized vector");
            suite.assert_false(not_normalized.is_normalized(0.01f), "is_normalized() for non-normalized");

            bool global_normalized = is_normalized(normalized, 0.01f);
            suite.assert_true(global_normalized, "global is_normalized() for normalized vector");
        }

        // Тест to_string
        {
            half2 v(1.5f, 2.5f);
            std::string str = v.to_string();

            // Проверяем наличие ожидаемых значений в строке
            suite.assert_true(str.find("1.5") != std::string::npos || str.find("1.500") != std::string::npos,
                "to_string() contains x value");
            suite.assert_true(str.find("2.5") != std::string::npos || str.find("2.500") != std::string::npos,
                "to_string() contains y value");
        }

        // Тест data()
        {
            half2 v(7.0f, 8.0f);
            const half* cdata = v.data();
            half* data = v.data();

            suite.assert_approximately_equal(float(cdata[0]), 7.0f, "data() const access x", HALF_EPSILON);
            suite.assert_approximately_equal(float(cdata[1]), 8.0f, "data() const access y", HALF_EPSILON);

            data[0] = half(9.0f);
            suite.assert_approximately_equal(float(v.x), 9.0f, "data() mutable modification x", HALF_EPSILON);
        }

        // ============================================================================
        // 10. Операторы сравнения
        // ============================================================================
        suite.section("Операторы сравнения");

        {
            half2 a(1.0f, 2.0f);
            half2 b(1.0f, 2.0f);
            half2 c(1.1f, 2.1f);

            suite.assert_true(a == b, "Operator == for equal vectors");
            suite.assert_false(a == c, "Operator == for different vectors");
            suite.assert_false(a != b, "Operator != for equal vectors");
            suite.assert_true(a != c, "Operator != for different vectors");
        }

        // ============================================================================
        // 11. Специальные значения
        // ============================================================================
        suite.section("Специальные значения");

        // Тест с нулями
        {
            half2 pos_zero(0.0f, 0.0f);
            half2 neg_zero(-0.0f, -0.0f);

            suite.assert_true(pos_zero.is_zero(), "is_zero() for positive zero");
            suite.assert_true(neg_zero.is_zero(), "is_zero() for negative zero");
            suite.assert_true(pos_zero.is_all_zero(), "is_all_zero() for positive zero");
            suite.assert_true(neg_zero.is_all_zero(), "is_all_zero() for negative zero");

            bool global_zero = is_zero(pos_zero);
            bool global_all_zero = is_all_zero(pos_zero);
            suite.assert_true(global_zero, "global is_zero()");
            suite.assert_true(global_all_zero, "global is_all_zero()");
        }

        // Тест с бесконечностями
        {
            half2 pos_inf(std::numeric_limits<float>::infinity(), 1.0f);
            half2 neg_inf(-std::numeric_limits<float>::infinity(), 1.0f);

            suite.assert_true(pos_inf.is_inf(), "is_inf() for vector with positive infinity");
            suite.assert_true(neg_inf.is_inf(), "is_inf() for vector with negative infinity");

            bool global_inf = is_inf(pos_inf);
            suite.assert_true(global_inf, "global is_inf()");
        }

        // Тест с NaN
        {
            half2 nan_vec(std::numeric_limits<float>::quiet_NaN(), 1.0f);

            suite.assert_true(nan_vec.is_nan(), "is_nan() for vector with NaN");

            bool global_nan = is_nan(nan_vec);
            suite.assert_true(global_nan, "global is_nan()");
        }

        // Тест с конечными значениями
        {
            half2 finite(1.0f, 2.0f);

            suite.assert_true(finite.is_finite(), "is_finite() for finite vector");
            suite.assert_true(finite.is_all_finite(), "is_all_finite() for finite vector");

            bool global_finite = is_finite(finite);
            bool global_all_finite = is_all_finite(finite);
            suite.assert_true(global_finite, "global is_finite()");
            suite.assert_true(global_all_finite, "global is_all_finite()");
        }

        // Тест с положительными/отрицательными значениями
        {
            half2 pos(1.0f, 2.0f);
            half2 neg(-1.0f, -2.0f);
            half2 mixed(-1.0f, 2.0f);

            suite.assert_true(pos.is_positive(), "is_positive() for positive vector");
            suite.assert_true(pos.is_all_positive(), "is_all_positive() for positive vector");
            suite.assert_true(neg.is_negative(), "is_negative() for negative vector");
            suite.assert_true(neg.is_all_negative(), "is_all_negative() for negative vector");
            suite.assert_true(mixed.is_positive(), "is_positive() for mixed vector");
            suite.assert_true(mixed.is_negative(), "is_negative() for mixed vector");
            suite.assert_false(mixed.is_all_positive(), "is_all_positive() for mixed vector");
            suite.assert_false(mixed.is_all_negative(), "is_all_negative() for mixed vector");

            bool global_positive = is_positive(pos);
            bool global_all_positive = is_all_positive(pos);
            suite.assert_true(global_positive, "global is_positive()");
            suite.assert_true(global_all_positive, "global is_all_positive()");
        }

        // ============================================================================
        // 12. Линейная интерполяция
        // ============================================================================
        suite.section("Линейная интерполяция");

        {
            half2 a(0.0f, 0.0f);
            half2 b(10.0f, 20.0f);

            half2 lerp_result = half2::lerp(a, b, half(0.5f));
            suite.assert_approximately_equal(float(lerp_result.x), 5.0f, "half2::lerp() at 0.5 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(lerp_result.y), 10.0f, "half2::lerp() at 0.5 y", HALF_EPSILON);

            half2 lerp_float = half2::lerp(a, b, 0.5f);
            suite.assert_approximately_equal(float(lerp_float.x), 5.0f, "half2::lerp() with float at 0.5 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(lerp_float.y), 10.0f, "half2::lerp() with float at 0.5 y", HALF_EPSILON);

            half2 global_lerp = lerp(a, b, half(0.5f));
            suite.assert_approximately_equal(float(global_lerp.x), 5.0f, "global lerp() at 0.5 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_lerp.y), 10.0f, "global lerp() at 0.5 y", HALF_EPSILON);

            half2 global_lerp_float = lerp(a, b, 0.5f);
            suite.assert_approximately_equal(float(global_lerp_float.x), 5.0f, "global lerp() with float at 0.5 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(global_lerp_float.y), 10.0f, "global lerp() with float at 0.5 y", HALF_EPSILON);

            half2 lerp_start = lerp(a, b, half(0.0f));
            suite.assert_approximately_equal(float(lerp_start.x), 0.0f, "lerp() at 0.0 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(lerp_start.y), 0.0f, "lerp() at 0.0 y", HALF_EPSILON);

            half2 lerp_end = lerp(a, b, half(1.0f));
            suite.assert_approximately_equal(float(lerp_end.x), 10.0f, "lerp() at 1.0 x", HALF_EPSILON);
            suite.assert_approximately_equal(float(lerp_end.y), 20.0f, "lerp() at 1.0 y", HALF_EPSILON);
        }

        // ============================================================================
        // 13. Глобальные константы
        // ============================================================================
        suite.section("Глобальные константы");

        {
            suite.assert_approximately_equal(float(half2_Zero.x), 0.0f, "half2_Zero x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_Zero.y), 0.0f, "half2_Zero y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_One.x), 1.0f, "half2_One x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_One.y), 1.0f, "half2_One y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_UnitX.x), 1.0f, "half2_UnitX x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_UnitX.y), 0.0f, "half2_UnitX y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_UnitY.x), 0.0f, "half2_UnitY x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_UnitY.y), 1.0f, "half2_UnitY y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_UV_Zero.x), 0.0f, "half2_UV_Zero x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_UV_Zero.y), 0.0f, "half2_UV_Zero y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_UV_One.x), 1.0f, "half2_UV_One x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_UV_One.y), 1.0f, "half2_UV_One y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_UV_Half.x), 0.5f, "half2_UV_Half x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_UV_Half.y), 0.5f, "half2_UV_Half y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_Right.x), 1.0f, "half2_Right x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_Right.y), 0.0f, "half2_Right y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_Left.x), -1.0f, "half2_Left x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_Left.y), 0.0f, "half2_Left y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_Up.x), 0.0f, "half2_Up x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_Up.y), 1.0f, "half2_Up y", HALF_EPSILON);

            suite.assert_approximately_equal(float(half2_Down.x), 0.0f, "half2_Down x", HALF_EPSILON);
            suite.assert_approximately_equal(float(half2_Down.y), -1.0f, "half2_Down y", HALF_EPSILON);
        }

        // ============================================================================
        // 14. Граничные случаи
        // ============================================================================
        suite.section("Граничные случаи");

        // Тест с очень маленькими значениями
        {
            half2 tiny(1e-6f, 1e-6f);
            suite.assert_true(tiny.approximately_zero(1e-3f), "Tiny values approximately_zero");
        }

        // Тест с очень большими значениями
        {
            half2 huge(50000.0f, 50000.0f);
            suite.assert_false(huge.approximately_zero(), "Huge values not approximately_zero");

            // Нормализация больших значений
            half2 normalized_huge = huge.normalize();
            float normalized_len = float(normalized_huge.length());
            suite.assert_approximately_equal(normalized_len, 1.0f, "Normalize huge values length", HALF_EPSILON);
        }

        // Тест деления на ноль (векторное)
        {
            half2 a(1.0f, 2.0f);
            half2 zero_vec(0.0f, 0.0f);

            half2 result = a / zero_vec;
            suite.assert_true(result.x.is_inf() || result.x.is_nan(), "Division by zero vector x produces non-finite");
            suite.assert_true(result.y.is_inf() || result.y.is_nan(), "Division by zero vector y produces non-finite");
        }

        // Тест деления на ноль (скалярное)
        {
            half2 v(1.0f, 2.0f);
            half2 result = v / 0.0f;

            suite.assert_true(result.x.is_inf() || result.x.is_nan(), "Division by zero scalar x produces non-finite");
            suite.assert_true(result.y.is_inf() || result.y.is_nan(), "Division by zero scalar y produces non-finite");
        }

        // Тест нормализации очень маленького вектора
        {
            half2 tiny(1e-20f, 1e-20f);
            half2 normalized = tiny.normalize();

            // Должен вернуть нулевой вектор, так как длина меньше epsilon
            suite.assert_approximately_equal(float(normalized.x), 0.0f, "Normalize tiny vector x", HALF_EPSILON);
            suite.assert_approximately_equal(float(normalized.y), 0.0f, "Normalize tiny vector y", HALF_EPSILON);
        }

        // Тест smoothstep с edge0 = edge1
        {
            half2 v(0.5f, 1.5f);
            half2 result = smoothstep(half(1.0f), half(1.0f), v);
            // При edge0 = edge1 используется step(edge0)
            half2 step_result = step(half(1.0f), v);
            suite.assert_approximately_equal(float(result.x), float(step_result.x), "smoothstep() with equal edges x", HALF_EPSILON);
            suite.assert_approximately_equal(float(result.y), float(step_result.y), "smoothstep() with equal edges y", HALF_EPSILON);
        }

        // Тест скалярного произведения с самим собой
        {
            half2 v(3.0f, 4.0f);
            half dot_self = v.dot(v);
            half expected = v.length_sq();
            suite.assert_approximately_equal(float(dot_self), float(expected), "Dot product with self equals length squared", HALF_EPSILON);
        }

        // Тест векторного произведения с самим собой
        {
            half2 v(2.0f, 3.0f);
            half cross_self = cross(v, v);
            suite.assert_approximately_equal(float(cross_self), 0.0f, "Cross product with self equals zero", HALF_EPSILON);
        }

        // Тест угла нулевого вектора
        {
            half2 zero(0.0f, 0.0f);
            half angle = AfterMath::angle(zero);
            // Угол нулевого вектора не определен, но atan2(0,0) обычно возвращает 0
            suite.assert_approximately_equal(float(angle), 0.0f, "Angle of zero vector", HALF_ANGLE_EPSILON);
        }

        // Тест is_normalized для нулевого вектора
        {
            half2 zero(0.0f, 0.0f);
            suite.assert_false(zero.is_normalized(), "Zero vector is not normalized");
        }

        // Тест с числами, близкими к максимальному значению half
        {
            half max_half = half::max_value();
            half2 max_vec(max_half, max_half);

            // Убедимся, что значения не стали NaN или Inf
            suite.assert_true(max_vec.is_finite(), "Vector with max half values is finite");

            // Нормализация вектора с максимальными значениями
            half2 normalized = max_vec.normalize();
            float normalized_len = float(normalized.length());
            suite.assert_approximately_equal(normalized_len, 1.0f, "Normalize max values length", HALF_EPSILON);
        }

        // Тест с денормализованными числами
        {
            half min_denorm = half::min_denormal_value();
            half2 denorm_vec(min_denorm, min_denorm);

            suite.assert_true(denorm_vec.is_finite(), "Vector with denormalized values is finite");

            // Умножение денормализованных чисел
            half2 multiplied = denorm_vec * half(2.0f);
            suite.assert_true(multiplied.is_finite(), "Multiplication of denormalized values is finite");
        }

        // Тест насыщения с отрицательными числами
        {
            half2 v(-2.0f, 0.5f);
            half2 saturated = v.saturate();

            suite.assert_approximately_equal(float(saturated.x), 0.0f, "saturate() negative value", HALF_EPSILON);
            suite.assert_approximately_equal(float(saturated.y), 0.5f, "saturate() positive value < 1", HALF_EPSILON);

            half2 v2(1.5f, -0.5f);
            half2 saturated2 = v2.saturate();

            suite.assert_approximately_equal(float(saturated2.x), 1.0f, "saturate() positive value > 1", HALF_EPSILON);
            suite.assert_approximately_equal(float(saturated2.y), 0.0f, "saturate() negative value", HALF_EPSILON);
        }

        // Тест округления граничных значений
        {
            half2 v1(1.4999f, 1.5001f);
            half2 rounded1 = v1.round();
            suite.assert_approximately_equal(float(rounded1.x), 1.0f, "round() 1.4999", HALF_EPSILON);
            suite.assert_approximately_equal(float(rounded1.y), 2.0f, "round() 1.5001", HALF_EPSILON);

            half2 v2(-1.4999f, -1.5001f);
            half2 rounded2 = v2.round();
            suite.assert_approximately_equal(float(rounded2.x), -1.0f, "round() -1.4999", HALF_EPSILON);
            suite.assert_approximately_equal(float(rounded2.y), -2.0f, "round() -1.5001", HALF_EPSILON);
        }

        // Тест floor/ceil граничных значений (замените весь блок)
        {
            // Используем значения, которые сохраняют дробную часть в half
            half2 v(1.5f, -1.5f);

            half2 floored = v.floor();
            suite.assert_approximately_equal(float(floored.x), 1.0f, "floor() 1.5", HALF_EPSILON);
            suite.assert_approximately_equal(float(floored.y), -2.0f, "floor() -1.5", HALF_EPSILON);

            half2 ceiled = v.ceil();
            suite.assert_approximately_equal(float(ceiled.x), 2.0f, "ceil() 1.5", HALF_EPSILON);
            suite.assert_approximately_equal(float(ceiled.y), -1.0f, "ceil() -1.5", HALF_EPSILON);
        }

        suite.footer();
    }
}
