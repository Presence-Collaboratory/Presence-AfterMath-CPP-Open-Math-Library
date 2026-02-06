// Author: DeepSeek
// Test suite for AfterMath::float3 class

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunFloat3Tests()
    {
        TestSuite suite("float3 Tests", true);
        suite.header();

        using namespace AfterMath;

        // ============================================================================
        // 1. Конструкторы
        // ============================================================================
        suite.section("Конструкторы");

        // Тест конструктора по умолчанию
        {
            float3 v;
            suite.assert_approximately_equal(v.x, 0.0f, "Default constructor x");
            suite.assert_approximately_equal(v.y, 0.0f, "Default constructor y");
            suite.assert_approximately_equal(v.z, 0.0f, "Default constructor z");
        }

        // Тест конструктора с компонентами
        {
            float3 v(1.5f, 2.5f, 3.5f);
            suite.assert_approximately_equal(v.x, 1.5f, "Component constructor x");
            suite.assert_approximately_equal(v.y, 2.5f, "Component constructor y");
            suite.assert_approximately_equal(v.z, 3.5f, "Component constructor z");
        }

        // Тест конструктора со скаляром
        {
            float3 v(3.0f);
            suite.assert_approximately_equal(v.x, 3.0f, "Scalar constructor x");
            suite.assert_approximately_equal(v.y, 3.0f, "Scalar constructor y");
            suite.assert_approximately_equal(v.z, 3.0f, "Scalar constructor z");
        }

        // Тест конструктора из float2
        {
            float2 vec(1.0f, 2.0f);
            float3 v(vec, 3.0f);
            suite.assert_approximately_equal(v.x, 1.0f, "float2 constructor x");
            suite.assert_approximately_equal(v.y, 2.0f, "float2 constructor y");
            suite.assert_approximately_equal(v.z, 3.0f, "float2 constructor z");
        }

        // Тест конструктора из массива
        {
            float data[3] = { 4.0f, 5.0f, 6.0f };
            float3 v(data);
            suite.assert_approximately_equal(v.x, 4.0f, "Array constructor x");
            suite.assert_approximately_equal(v.y, 5.0f, "Array constructor y");
            suite.assert_approximately_equal(v.z, 6.0f, "Array constructor z");
        }

        // Тест копирующего конструктора
        {
            float3 original(6.0f, 7.0f, 8.0f);
            float3 copy(original);
            suite.assert_approximately_equal(copy.x, 6.0f, "Copy constructor x");
            suite.assert_approximately_equal(copy.y, 7.0f, "Copy constructor y");
            suite.assert_approximately_equal(copy.z, 8.0f, "Copy constructor z");
        }

        // Тест статических конструкторов
        {
            suite.assert_approximately_equal(float3::zero(), float3(0.0f, 0.0f, 0.0f), "zero()");
            suite.assert_approximately_equal(float3::one(), float3(1.0f, 1.0f, 1.0f), "one()");
            suite.assert_approximately_equal(float3::unit_x(), float3(1.0f, 0.0f, 0.0f), "unit_x()");
            suite.assert_approximately_equal(float3::unit_y(), float3(0.0f, 1.0f, 0.0f), "unit_y()");
            suite.assert_approximately_equal(float3::unit_z(), float3(0.0f, 0.0f, 1.0f), "unit_z()");
            suite.assert_approximately_equal(float3::forward(), float3(0.0f, 0.0f, 1.0f), "forward()");
            suite.assert_approximately_equal(float3::up(), float3(0.0f, 1.0f, 0.0f), "up()");
            suite.assert_approximately_equal(float3::right(), float3(1.0f, 0.0f, 0.0f), "right()");
        }

        // ============================================================================
        // 2. Операторы присваивания
        // ============================================================================
        suite.section("Операторы присваивания");

        // Тест присваивания скаляра
        {
            float3 v;
            v = 2.5f;
            suite.assert_approximately_equal(v.x, 2.5f, "Scalar assignment x");
            suite.assert_approximately_equal(v.y, 2.5f, "Scalar assignment y");
            suite.assert_approximately_equal(v.z, 2.5f, "Scalar assignment z");
        }

        // Тест составных операторов присваивания
        {
            float3 v(1.0f, 2.0f, 3.0f);

            v += float3(3.0f, 4.0f, 5.0f);
            suite.assert_approximately_equal(v, float3(4.0f, 6.0f, 8.0f), "Operator +=");

            v -= float3(1.0f, 2.0f, 3.0f);
            suite.assert_approximately_equal(v, float3(3.0f, 4.0f, 5.0f), "Operator -=");

            v *= float3(2.0f, 3.0f, 4.0f);
            suite.assert_approximately_equal(v, float3(6.0f, 12.0f, 20.0f), "Operator *=");

            v /= float3(2.0f, 3.0f, 4.0f);
            suite.assert_approximately_equal(v, float3(3.0f, 4.0f, 5.0f), "Operator /=");

            v *= 2.0f;
            suite.assert_approximately_equal(v, float3(6.0f, 8.0f, 10.0f), "Operator *= scalar");

            v /= 2.0f;
            suite.assert_approximately_equal(v, float3(3.0f, 4.0f, 5.0f), "Operator /= scalar");
        }

        // ============================================================================
        // 3. Бинарные и унарные операторы
        // ============================================================================
        suite.section("Бинарные и унарные операторы");

        // Тест сложения
        {
            float3 a(1.0f, 2.0f, 3.0f);
            float3 b(4.0f, 5.0f, 6.0f);
            float3 result = a + b;
            suite.assert_approximately_equal(result, float3(5.0f, 7.0f, 9.0f), "Operator +");
        }

        // Тест вычитания
        {
            float3 a(5.0f, 6.0f, 7.0f);
            float3 b(2.0f, 3.0f, 4.0f);
            float3 result = a - b;
            suite.assert_approximately_equal(result, float3(3.0f, 3.0f, 3.0f), "Operator -");
        }

        // Тест умножения компонентного
        {
            float3 a(2.0f, 3.0f, 4.0f);
            float3 b(1.0f, 2.0f, 3.0f);
            float3 result = a * b;
            suite.assert_approximately_equal(result, float3(2.0f, 6.0f, 12.0f), "Operator * (component-wise)");
        }

        // Тест деления компонентного
        {
            float3 a(6.0f, 8.0f, 10.0f);
            float3 b(2.0f, 4.0f, 5.0f);
            float3 result = a / b;
            suite.assert_approximately_equal(result, float3(3.0f, 2.0f, 2.0f), "Operator / (component-wise)");
        }

        // Тест унарных операторов
        {
            float3 a(1.0f, -2.0f, 3.0f);
            suite.assert_approximately_equal(+a, float3(1.0f, -2.0f, 3.0f), "Unary +");
            suite.assert_approximately_equal(-a, float3(-1.0f, 2.0f, -3.0f), "Unary -");
        }

        // Тест скалярных операций
        {
            float3 v(2.0f, 3.0f, 4.0f);

            float3 result1 = v * 2.0f;
            suite.assert_approximately_equal(result1, float3(4.0f, 6.0f, 8.0f), "Vector * scalar");

            float3 result2 = 2.0f * v;
            suite.assert_approximately_equal(result2, float3(4.0f, 6.0f, 8.0f), "Scalar * vector");

            float3 result3 = v / 2.0f;
            suite.assert_approximately_equal(result3, float3(1.0f, 1.5f, 2.0f), "Vector / scalar");

            float3 result4 = 12.0f / v;
            suite.assert_approximately_equal(result4, float3(6.0f, 4.0f, 3.0f), "Scalar / vector");
        }

        // ============================================================================
        // 4. Операторы доступа и преобразования
        // ============================================================================
        suite.section("Операторы доступа и преобразования");

        // Тест оператора индексации
        {
            float3 v(7.0f, 8.0f, 9.0f);
            suite.assert_approximately_equal(v[0], 7.0f, "Operator [] index 0");
            suite.assert_approximately_equal(v[1], 8.0f, "Operator [] index 1");
            suite.assert_approximately_equal(v[2], 9.0f, "Operator [] index 2");

            v[0] = 10.0f;
            v[1] = 11.0f;
            v[2] = 12.0f;
            suite.assert_approximately_equal(v.x, 10.0f, "Operator [] mutable x");
            suite.assert_approximately_equal(v.y, 11.0f, "Operator [] mutable y");
            suite.assert_approximately_equal(v.z, 12.0f, "Operator [] mutable z");
        }

        // Тест преобразования в указатель
        {
            float3 v(1.0f, 2.0f, 3.0f);
            const float* ptr = v;
            suite.assert_approximately_equal(ptr[0], 1.0f, "Conversion to const float* index 0");
            suite.assert_approximately_equal(ptr[1], 2.0f, "Conversion to const float* index 1");
            suite.assert_approximately_equal(ptr[2], 3.0f, "Conversion to const float* index 2");

            float* mutable_ptr = v;
            mutable_ptr[0] = 4.0f;
            suite.assert_approximately_equal(v.x, 4.0f, "Conversion to float* mutable");
        }

        // Тест преобразования в __m128
        {
            float3 v(1.0f, 2.0f, 3.0f);
            __m128 simd = v;
            alignas(16) float temp[4];
            _mm_store_ps(temp, simd);
            suite.assert_approximately_equal(temp[0], 1.0f, "Conversion to __m128 x", 1e-6f);
            suite.assert_approximately_equal(temp[1], 2.0f, "Conversion to __m128 y", 1e-6f);
            suite.assert_approximately_equal(temp[2], 3.0f, "Conversion to __m128 z", 1e-6f);
        }

        // ============================================================================
        // 5. Математические функции
        // ============================================================================
        suite.section("Математические функции");

        // Тест длины
        {
            float3 v(2.0f, 3.0f, 6.0f);
            // √(4 + 9 + 36) = √49 = 7
            suite.assert_approximately_equal(v.length(), 7.0f, "length()");
            suite.assert_approximately_equal(v.length_sq(), 49.0f, "length_sq()");

            float3 zero(0.0f, 0.0f, 0.0f);
            suite.assert_approximately_equal(zero.length(), 0.0f, "length() of zero vector");
        }

        // Тест нормализации
        {
            float3 v(2.0f, 3.0f, 6.0f);
            float3 normalized = v.normalize();
            float expected_len = 1.0f;
            suite.assert_approximately_equal(normalized.length(), expected_len, "normalize() length", 1e-6f);
            suite.assert_approximately_equal(normalized.x, 2.0f / 7.0f, "normalize() x", 1e-6f);
            suite.assert_approximately_equal(normalized.y, 3.0f / 7.0f, "normalize() y", 1e-6f);
            suite.assert_approximately_equal(normalized.z, 6.0f / 7.0f, "normalize() z", 1e-6f);

            // Тест нормализации нулевого вектора
            float3 zero(0.0f, 0.0f, 0.0f);
            float3 zero_norm = zero.normalize();
            suite.assert_approximately_equal(zero_norm, float3::zero(), "normalize() zero vector");
        }

        // Тест скалярного произведения
        {
            float3 a(1.0f, 2.0f, 3.0f);
            float3 b(4.0f, 5.0f, 6.0f);
            float dot_result = a.dot(b);
            suite.assert_approximately_equal(dot_result, 32.0f, "dot()"); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

            // Ортогональные векторы
            float3 orth1(1.0f, 0.0f, 0.0f);
            float3 orth2(0.0f, 1.0f, 0.0f);
            suite.assert_approximately_equal(orth1.dot(orth2), 0.0f, "dot() orthogonal vectors");
        }

        // Тест векторного произведения
        {
            float3 a(1.0f, 0.0f, 0.0f);
            float3 b(0.0f, 1.0f, 0.0f);
            float3 cross_result = a.cross(b);
            suite.assert_approximately_equal(cross_result, float3(0.0f, 0.0f, 1.0f), "cross() i * j = k");

            float3 c(2.0f, 3.0f, 4.0f);
            float3 d(5.0f, 6.0f, 7.0f);
            float3 cross2 = c.cross(d);
            // (3*7 - 4*6, 4*5 - 2*7, 2*6 - 3*5) = (21-24, 20-14, 12-15) = (-3, 6, -3)
            suite.assert_approximately_equal(cross2, float3(-3.0f, 6.0f, -3.0f), "cross() arbitrary vectors");

            // Векторное произведение вектора с самим собой равно нулю
            suite.assert_approximately_equal(a.cross(a), float3::zero(), "cross() vector with itself");
        }

        // Тест расстояния
        {
            float3 a(1.0f, 2.0f, 3.0f);
            float3 b(4.0f, 6.0f, 8.0f);
            float distance = a.distance(b);
            float distance_sq = a.distance_sq(b);

            // (4-1)² + (6-2)² + (8-3)² = 9 + 16 + 25 = 50
            suite.assert_approximately_equal(distance, std::sqrt(50.0f), "distance()");
            suite.assert_approximately_equal(distance_sq, 50.0f, "distance_sq()");
        }

        // ============================================================================
        // 6. HLSL-подобные функции
        // ============================================================================
        suite.section("HLSL-подобные функции");

        // Тест abs
        {
            float3 v(-1.5f, 2.5f, -3.5f);
            float3 result = v.abs();
            suite.assert_approximately_equal(result, float3(1.5f, 2.5f, 3.5f), "abs()");
        }

        // Тест sign
        {
            float3 v(-2.0f, 0.0f, 3.0f);
            float3 result = v.sign();
            suite.assert_approximately_equal(result, float3(-1.0f, 0.0f, 1.0f), "sign()");
        }

        // Тест floor
        {
            float3 v(1.7f, -2.3f, 3.9f);
            float3 result = v.floor();
            suite.assert_approximately_equal(result, float3(1.0f, -3.0f, 3.0f), "floor()");
        }

        // Тест ceil
        {
            float3 v(1.2f, -2.7f, 3.1f);
            float3 result = v.ceil();
            suite.assert_approximately_equal(result, float3(2.0f, -2.0f, 4.0f), "ceil()");
        }

        // Тест round
        {
            float3 v(1.4f, 1.6f, -1.5f);
            float3 result = v.round();
            suite.assert_approximately_equal(result, float3(1.0f, 2.0f, -2.0f), "round()");
        }

        // Тест frac
        {
            float3 v(1.7f, -2.3f, 3.0f);
            float3 result = v.frac();
            suite.assert_approximately_equal(result.x, 0.7f, "frac() x", 1e-6f);
            suite.assert_approximately_equal(result.y, 0.7f, "frac() y", 1e-6f); // -2.3 - (-3) = 0.7
            suite.assert_approximately_equal(result.z, 0.0f, "frac() z", 1e-6f);
        }

        // Тест saturate
        {
            float3 v(-0.5f, 0.5f, 1.5f);
            float3 result = v.saturate();
            suite.assert_approximately_equal(result, float3(0.0f, 0.5f, 1.0f), "saturate()");
        }

        // Тест step
        {
            float3 v(0.5f, 1.0f, 1.5f);
            float3 result = v.step(1.0f);
            suite.assert_approximately_equal(result, float3(0.0f, 1.0f, 1.0f), "step()");
        }

        // Тест clamp (компонентный)
        {
            float3 v(0.5f, 1.5f, -0.5f);
            float3 min_val(0.0f, 0.0f, 0.0f);
            float3 max_val(1.0f, 1.0f, 1.0f);
            float3 result = float3::clamp(v, min_val, max_val);
            suite.assert_approximately_equal(result, float3(0.5f, 1.0f, 0.0f), "clamp() component-wise");
        }

        // Тест clamp (скалярный)
        {
            float3 v(0.5f, 1.5f, -0.5f);
            float3 result = float3::clamp(v, 0.0f, 1.0f);
            suite.assert_approximately_equal(result, float3(0.5f, 1.0f, 0.0f), "clamp() scalar");
        }

        // Тест min/max
        {
            float3 a(1.0f, 3.0f, 5.0f);
            float3 b(2.0f, 2.0f, 6.0f);

            float3 min_result = float3::min(a, b);
            suite.assert_approximately_equal(min_result, float3(1.0f, 2.0f, 5.0f), "min()");

            float3 max_result = float3::max(a, b);
            suite.assert_approximately_equal(max_result, float3(2.0f, 3.0f, 6.0f), "max()");
        }

        // ============================================================================
        // 7. Геометрические операции
        // ============================================================================
        suite.section("Геометрические операции");

        // Тест отражения
        {
            float3 incident(1.0f, -1.0f, 0.0f);
            float3 normal(0.0f, 1.0f, 0.0f); // Нормаль вверх
            normal = normal.normalize();
            float3 reflected = incident.reflect(normal);
            // R = I - 2*(I·N)*N = (1,-1,0) - 2*(-1)*(0,1,0) = (1,-1,0) + (0,2,0) = (1,1,0)
            suite.assert_approximately_equal(reflected, float3(1.0f, 1.0f, 0.0f), "reflect()");

            // Статическая версия
            float3 reflected2 = float3::reflect(incident, normal);
            suite.assert_approximately_equal(reflected2, float3(1.0f, 1.0f, 0.0f), "static reflect()");
        }

        // Тест отражения
        {
            // Для перехода из ВОДЫ (n=1.33) в ВОЗДУХ (n=1.0)
            float eta_water_to_air = 1.33f; // вода → воздух

            // Проверим условие полного внутреннего отражения:
            // Критический угол: θ_c = arcsin(1/eta) = arcsin(1/1.33) approx 48.8°
            // Угол 60° > 48.8° → должно быть полное внутреннее отражение

            float3 incident_large(0.866f, -0.5f, 0.0f); // 60° угол
            float3 normal(0.0f, 1.0f, 0.0f);
            normal = normal.normalize();

            float3 total_reflection = incident_large.refract(normal, eta_water_to_air);

            // Должен вернуть нулевой вектор
            suite.assert_approximately_equal(total_reflection, float3::zero(),
                "refract() total internal reflection", 1e-6f);
        }

        // Тест проекции
        {
            float3 v(2.0f, 3.0f, 4.0f);
            float3 onto(1.0f, 0.0f, 0.0f); // Ось X

            float3 projected = v.project(onto);
            suite.assert_approximately_equal(projected, float3(2.0f, 0.0f, 0.0f), "project() onto X axis");

            // Проекция на себя должна дать себя
            float3 self_projected = v.project(v);
            suite.assert_approximately_equal(self_projected, v, "project() onto itself");
        }

        // Тест отклонения
        {
            float3 v(2.0f, 3.0f, 4.0f);
            float3 onto(1.0f, 0.0f, 0.0f); // Ось X

            float3 rejected = v.reject(onto);
            // v = (2,3,4), проекция на X = (2,0,0), отклонение = (0,3,4)
            suite.assert_approximately_equal(rejected, float3(0.0f, 3.0f, 4.0f), "reject() from X axis");

            // Отклонение от себя должно дать 0
            float3 self_rejected = v.reject(v);
            suite.assert_approximately_equal(self_rejected, float3::zero(), "reject() from itself");
        }

        // Тест линейной интерполяции
        {
            float3 a(0.0f, 0.0f, 0.0f);
            float3 b(10.0f, 20.0f, 30.0f);

            float3 lerp_result = float3::lerp(a, b, 0.5f);
            suite.assert_approximately_equal(lerp_result, float3(5.0f, 10.0f, 15.0f), "lerp() at 0.5");

            float3 lerp_start = float3::lerp(a, b, 0.0f);
            suite.assert_approximately_equal(lerp_start, a, "lerp() at 0.0");

            float3 lerp_end = float3::lerp(a, b, 1.0f);
            suite.assert_approximately_equal(lerp_end, b, "lerp() at 1.0");
        }

        // Тест сферической линейной интерполяции
        {
            float3 a(1.0f, 0.0f, 0.0f);
            float3 b(0.0f, 1.0f, 0.0f);

            a = a.normalize();
            b = b.normalize();

            float3 slerp_result = float3::slerp(a, b, 0.5f);
            float expected_length = 1.0f;
            suite.assert_approximately_equal(slerp_result.length(), expected_length, "slerp() length", 1e-6f);

            // В середине между (1,0,0) и (0,1,0) должно быть примерно (√2/2, √2/2, 0)
            float expected_val = std::sqrt(2.0f) / 2.0f;
            suite.assert_approximately_equal(slerp_result.x, expected_val, "slerp() x at 0.5", 1e-6f);
            suite.assert_approximately_equal(slerp_result.y, expected_val, "slerp() y at 0.5", 1e-6f);
            suite.assert_approximately_equal(slerp_result.z, 0.0f, "slerp() z at 0.5", 1e-6f);
        }

        // ============================================================================
        // 8. Swizzle операции
        // ============================================================================
        suite.section("Swizzle операции");

        {
            float3 v(2.0f, 3.0f, 4.0f);

            // 2-компонентные swizzles
            suite.assert_approximately_equal(v.xy(), float2(2.0f, 3.0f), "xy()");
            suite.assert_approximately_equal(v.xz(), float2(2.0f, 4.0f), "xz()");
            suite.assert_approximately_equal(v.yz(), float2(3.0f, 4.0f), "yz()");
            suite.assert_approximately_equal(v.yx(), float2(3.0f, 2.0f), "yx()");
            suite.assert_approximately_equal(v.zx(), float2(4.0f, 2.0f), "zx()");
            suite.assert_approximately_equal(v.zy(), float2(4.0f, 3.0f), "zy()");

            // 3-компонентные swizzles
            suite.assert_approximately_equal(v.yxz(), float3(3.0f, 2.0f, 4.0f), "yxz()");
            suite.assert_approximately_equal(v.zxy(), float3(4.0f, 2.0f, 3.0f), "zxy()");
            suite.assert_approximately_equal(v.zyx(), float3(4.0f, 3.0f, 2.0f), "zyx()");
            suite.assert_approximately_equal(v.xzy(), float3(2.0f, 4.0f, 3.0f), "xzy()");
            suite.assert_approximately_equal(v.xyx(), float3(2.0f, 3.0f, 2.0f), "xyx()");
            suite.assert_approximately_equal(v.xyz(), float3(2.0f, 3.0f, 4.0f), "xyz()");
            suite.assert_approximately_equal(v.xzx(), float3(2.0f, 4.0f, 2.0f), "xzx()");
            suite.assert_approximately_equal(v.yxy(), float3(3.0f, 2.0f, 3.0f), "yxy()");
            suite.assert_approximately_equal(v.yzy(), float3(3.0f, 4.0f, 3.0f), "yzy()");
            suite.assert_approximately_equal(v.zxz(), float3(4.0f, 2.0f, 4.0f), "zxz()");
            suite.assert_approximately_equal(v.zyz(), float3(4.0f, 3.0f, 4.0f), "zyz()");

            // Цветовые swizzles
            suite.assert_approximately_equal(v.r(), 2.0f, "r()");
            suite.assert_approximately_equal(v.g(), 3.0f, "g()");
            suite.assert_approximately_equal(v.b(), 4.0f, "b()");
            suite.assert_approximately_equal(v.rg(), float2(2.0f, 3.0f), "rg()");
            suite.assert_approximately_equal(v.rb(), float2(2.0f, 4.0f), "rb()");
            suite.assert_approximately_equal(v.gb(), float2(3.0f, 4.0f), "gb()");
            suite.assert_approximately_equal(v.rgb(), float3(2.0f, 3.0f, 4.0f), "rgb()");
            suite.assert_approximately_equal(v.bgr(), float3(4.0f, 3.0f, 2.0f), "bgr()");
            suite.assert_approximately_equal(v.gbr(), float3(3.0f, 4.0f, 2.0f), "gbr()");
        }

        // ============================================================================
        // 9. Утилитарные методы
        // ============================================================================
        suite.section("Утилитарные методы");

        // Тест isValid
        {
            float3 valid(1.0f, 2.0f, 3.0f);
            suite.assert_true(valid.isValid(), "isValid() for valid vector");

            // Note: Для тестов с NaN и INF нужно специальное создание
            suite.skip_test("isValid() with NaN/INF", "Requires special NaN/INF construction");
        }

        // Тест approximately
        {
            float3 a(1.0f, 2.0f, 3.0f);
            float3 b(1.000001f, 2.000001f, 3.000001f);
            float3 c(1.1f, 2.1f, 3.1f);

            suite.assert_true(a.approximately(b, 1e-5f), "approximately() within epsilon");
            suite.assert_false(a.approximately(c, 1e-5f), "approximately() outside epsilon");
        }

        // Тест approximately_zero
        {
            float3 zero(0.0f, 0.0f, 0.0f);
            float3 near_zero(0.000001f, 0.000001f, 0.000001f);
            float3 not_zero(0.1f, 0.1f, 0.1f);

            suite.assert_true(zero.approximately_zero(1e-5f), "approximately_zero() for zero");
            suite.assert_true(near_zero.approximately_zero(1e-4f), "approximately_zero() for near zero");
            suite.assert_false(not_zero.approximately_zero(1e-5f), "approximately_zero() for non-zero");
        }

        // Тест is_normalized
        {
            float3 normalized(0.267261f, 0.534522f, 0.801784f); // (1,2,3) normalized
            float3 not_normalized(1.0f, 2.0f, 3.0f);

            suite.assert_true(normalized.is_normalized(1e-5f), "is_normalized() for normalized vector");
            suite.assert_false(not_normalized.is_normalized(1e-5f), "is_normalized() for non-normalized");
        }

        // Тест to_string
        {
            float3 v(1.5f, 2.5f, 3.5f);
            std::string str = v.to_string();

            // Проверяем наличие ожидаемых значений в строке
            suite.assert_true(str.find("1.5") != std::string::npos || str.find("1.500") != std::string::npos,
                "to_string() contains x value");
            suite.assert_true(str.find("2.5") != std::string::npos || str.find("2.500") != std::string::npos,
                "to_string() contains y value");
            suite.assert_true(str.find("3.5") != std::string::npos || str.find("3.500") != std::string::npos,
                "to_string() contains z value");
        }

        // Тест data()
        {
            float3 v(7.0f, 8.0f, 9.0f);
            const float* cdata = v.data();
            float* data = v.data();

            suite.assert_approximately_equal(cdata[0], 7.0f, "data() const access x");
            suite.assert_approximately_equal(cdata[1], 8.0f, "data() const access y");
            suite.assert_approximately_equal(cdata[2], 9.0f, "data() const access z");

            data[0] = 10.0f;
            suite.assert_approximately_equal(v.x, 10.0f, "data() mutable modification");
        }

        // Тест set_xy
        {
            float3 v(1.0f, 2.0f, 3.0f);
            float2 xy(4.0f, 5.0f);
            v.set_xy(xy);
            suite.assert_approximately_equal(v.x, 4.0f, "set_xy() x");
            suite.assert_approximately_equal(v.y, 5.0f, "set_xy() y");
            suite.assert_approximately_equal(v.z, 3.0f, "set_xy() z unchanged");
        }

        // Тест операций с компонентами
        {
            float3 v(2.0f, 3.0f, 4.0f);

            suite.assert_approximately_equal(v.min_component(), 2.0f, "min_component()");
            suite.assert_approximately_equal(v.max_component(), 4.0f, "max_component()");
            suite.assert_equal(v.min_component_index(), 0, "min_component_index()");
            suite.assert_equal(v.max_component_index(), 2, "max_component_index()");
            suite.assert_approximately_equal(v.sum_components(), 9.0f, "sum_components()");
            suite.assert_approximately_equal(v.product_components(), 24.0f, "product_components()");
            suite.assert_approximately_equal(v.average(), 3.0f, "average()");
        }

        // Тест has_nan, has_infinite, all_finite
        {
            float3 v(1.0f, 2.0f, 3.0f);
            suite.assert_false(v.has_nan(), "has_nan() for normal vector");
            suite.assert_false(v.has_infinite(), "has_infinite() for normal vector");
            suite.assert_true(v.all_finite(), "all_finite() for normal vector");

            suite.skip_test("has_nan() with NaN", "Requires NaN construction");
            suite.skip_test("has_infinite() with INF", "Requires INF construction");
        }

        // ============================================================================
        // 10. Операторы сравнения
        // ============================================================================
        suite.section("Операторы сравнения");

        {
            float3 a(1.0f, 2.0f, 3.0f);
            float3 b(1.0f, 2.0f, 3.0f);
            float3 c(1.1f, 2.1f, 3.1f);

            suite.assert_true(a == b, "Operator == for equal vectors");
            suite.assert_false(a == c, "Operator == for different vectors");
            suite.assert_false(a != b, "Operator != for equal vectors");
            suite.assert_true(a != c, "Operator != for different vectors");
        }

        // ============================================================================
        // 11. Глобальные операторы и функции
        // ============================================================================
        suite.section("Глобальные операторы и функции");

        // Тест глобальных операторов
        {
            float3 a(2.0f, 3.0f, 4.0f);
            float3 b(1.0f, 2.0f, 3.0f);

            float3 add_result = a + b;
            suite.assert_approximately_equal(add_result, float3(3.0f, 5.0f, 7.0f), "Global operator +");

            float3 sub_result = a - b;
            suite.assert_approximately_equal(sub_result, float3(1.0f, 1.0f, 1.0f), "Global operator -");

            float3 mul_result = a * b;
            suite.assert_approximately_equal(mul_result, float3(2.0f, 6.0f, 12.0f), "Global operator *");

            float3 div_result = a / b;
            suite.assert_approximately_equal(div_result, float3(2.0f, 1.5f, 4.0f / 3.0f), "Global operator /", 1e-6f);
        }

        // Тест глобальных математических функций
        {
            float3 a(1.0f, 2.0f, 3.0f);
            float3 b(4.0f, 6.0f, 8.0f);

            suite.assert_approximately_equal(distance(a, b), std::sqrt(50.0f), "Global distance()");
            suite.assert_approximately_equal(distance_sq(a, b), 50.0f, "Global distance_sq()");
            suite.assert_approximately_equal(dot(a, b), 40.0f, "Global dot()");

            float3 cross_result = cross(a, b);
            // (2*8 - 3*6, 3*4 - 1*8, 1*6 - 2*4) = (16-18, 12-8, 6-8) = (-2, 4, -2)
            suite.assert_approximately_equal(cross_result, float3(-2.0f, 4.0f, -2.0f), "Global cross()");

            float3 norm = normalize(a);
            suite.assert_approximately_equal(norm.length(), 1.0f, "Global normalize() length", 1e-6f);

            float3 lerp_result = lerp(a, b, 0.5f);
            suite.assert_approximately_equal(lerp_result, float3(2.5f, 4.0f, 5.5f), "Global lerp()");
        }

        // Тест глобальных approximately
        {
            float3 a(1.0f, 2.0f, 3.0f);
            float3 b(1.000001f, 2.000001f, 3.000001f);

            suite.assert_true(approximately(a, b, 1e-5f), "Global approximately()");
            suite.assert_true(is_normalized(float3(1.0f, 0.0f, 0.0f), 1e-5f), "Global is_normalized()");
            suite.assert_true(isValid(a), "Global isValid()");
        }

        // Тест глобальных геометрических функций
        {
            float3 incident(1.0f, -1.0f, 0.0f);
            float3 normal(0.0f, 1.0f, 0.0f);
            normal = normalize(normal);

            float3 reflected = reflect(incident, normal);
            suite.assert_approximately_equal(reflected, float3(1.0f, 1.0f, 0.0f), "Global reflect()");

            float3 v(2.0f, 3.0f, 4.0f);
            float3 onto(1.0f, 0.0f, 0.0f);

            float3 projected = project(v, onto);
            suite.assert_approximately_equal(projected, float3(2.0f, 0.0f, 0.0f), "Global project()");

            float3 rejected = reject(v, onto);
            suite.assert_approximately_equal(rejected, float3(0.0f, 3.0f, 4.0f), "Global reject()");

            float3 a(1.0f, 0.0f, 0.0f);
            float3 b(0.0f, 1.0f, 0.0f);
            float angle = angle_between(a, b);
            suite.assert_approximately_equal(angle, Constants::Constants<float>::Pi / 2.0f,
                "Global angle_between() 90 degrees", 1e-6f);
        }

        // Тест глобальных HLSL-функций
        {
            float3 v(-1.5f, 2.5f, -3.5f);

            suite.assert_approximately_equal(abs(v), float3(1.5f, 2.5f, 3.5f), "Global abs()");
            suite.assert_approximately_equal(sign(v), float3(-1.0f, 1.0f, -1.0f), "Global sign()");
            suite.assert_approximately_equal(floor(v), float3(-2.0f, 2.0f, -4.0f), "Global floor()");
            suite.assert_approximately_equal(ceil(v), float3(-1.0f, 3.0f, -3.0f), "Global ceil()");
            suite.assert_approximately_equal(saturate(float3(-0.5f, 0.5f, 1.5f)),
                float3(0.0f, 0.5f, 1.0f), "Global saturate()");
        }

        // Тест глобальных clamp, min, max
        {
            float3 v(0.5f, 1.5f, -0.5f);
            float3 min_val(0.0f, 0.0f, 0.0f);
            float3 max_val(1.0f, 1.0f, 1.0f);

            float3 clamped = clamp(v, min_val, max_val);
            suite.assert_approximately_equal(clamped, float3(0.5f, 1.0f, 0.0f), "Global clamp()");

            float3 a(1.0f, 3.0f, 5.0f);
            float3 b(2.0f, 2.0f, 6.0f);

            float3 min_result = min(a, b);
            suite.assert_approximately_equal(min_result, float3(1.0f, 2.0f, 5.0f), "Global min()");

            float3 max_result = max(a, b);
            suite.assert_approximately_equal(max_result, float3(2.0f, 3.0f, 6.0f), "Global max()");
        }

        // Тест глобальных операций с компонентами
        {
            float3 v(2.0f, 3.0f, 4.0f);

            suite.assert_approximately_equal(min_component(v), 2.0f, "Global min_component()");
            suite.assert_approximately_equal(max_component(v), 4.0f, "Global max_component()");
            suite.assert_approximately_equal(sum_components(v), 9.0f, "Global sum_components()");
            suite.assert_approximately_equal(product_components(v), 24.0f, "Global product_components()");
            suite.assert_approximately_equal(average(v), 3.0f, "Global average()");
        }

        // ============================================================================
        // 12. Граничные случаи
        // ============================================================================
        suite.section("Граничные случаи");

        // Тест с очень маленькими значениями
        {
            float epsilon = 1e-30f;
            float3 tiny(epsilon, epsilon, epsilon);
            suite.assert_true(tiny.approximately_zero(1e-20f), "Tiny values approximately_zero");
        }

        // Тест с очень большими значениями
        {
            float large = 1e10f;
            float3 huge(large, large, large);
            suite.assert_false(huge.approximately_zero(), "Huge values not approximately_zero");

            float3 normalized_huge = huge.normalize();
            float expected_length = 1.0f;
            suite.assert_approximately_equal(normalized_huge.length(), expected_length,
                "Normalize huge values", 1e-6f);
        }

        // Тест деления на ноль (векторное)
        {
            float3 a(1.0f, 2.0f, 3.0f);
            float3 zero_vec(0.0f, 0.0f, 0.0f);

            // Должно вызвать деление на ноль, но мы проверим что происходит
            float3 result = a / zero_vec;

            // Проверяем что результат содержит бесконечности или NaN
            suite.assert_true(!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z),
                "Division by zero vector produces non-finite values");
        }

        // Тест деления на ноль (скалярное)
        {
            float3 v(1.0f, 2.0f, 3.0f);
            float3 result = v / 0.0f;

            suite.assert_true(!std::isfinite(result.x) && !std::isfinite(result.y) && !std::isfinite(result.z),
                "Division by zero scalar produces non-finite values");
        }

        // Тест нормализации очень маленького вектора
        {
            float3 tiny(1e-20f, 1e-20f, 1e-20f);
            float3 normalized = tiny.normalize();

            // Должен вернуть нулевой вектор, так как длина меньше epsilon
            suite.assert_approximately_equal(normalized, float3::zero(),
                "Normalize tiny vector returns zero");
        }

        // Тест slerp с параллельными векторами
        {
            float3 a(1.0f, 0.0f, 0.0f);
            float3 b(2.0f, 0.0f, 0.0f); // Коллинеарный, но не нормализованный

            a = a.normalize();
            b = b.normalize();

            float3 slerp_result = slerp(a, b, 0.5f);
            suite.assert_approximately_equal(slerp_result.length(), 1.0f,
                "slerp() with parallel vectors length", 1e-6f);
            suite.assert_approximately_equal(slerp_result, a,
                "slerp() with parallel vectors returns first", 1e-6f);
        }

        // Тест slerp с противоположными векторами
        {
            float3 a(1.0f, 0.0f, 0.0f);
            float3 b(-1.0f, 0.0f, 0.0f);

            a = a.normalize();
            b = b.normalize();

            float3 slerp_result = slerp(a, b, 0.5f);
            // При противоположных векторах slerp должен работать корректно
            suite.assert_approximately_equal(slerp_result.length(), 1.0f,
                "slerp() with opposite vectors length", 1e-6f);
        }

        // Тест отражения от нулевой нормали
        {
            float3 v(1.0f, 2.0f, 3.0f);
            float3 zero_normal(0.0f, 0.0f, 0.0f);

            float3 reflected = v.reflect(zero_normal);
            // R = I - 2*(I·N)*N = I - 0 = I
            suite.assert_approximately_equal(reflected, v, "Reflect with zero normal returns original");
        }

        // Тест проекции на нулевой вектор
        {
            float3 v(1.0f, 2.0f, 3.0f);
            float3 zero(0.0f, 0.0f, 0.0f);

            float3 projected = v.project(zero);
            suite.assert_approximately_equal(projected, float3::zero(), "Project onto zero vector returns zero");
        }

        // Тест отклонения от нулевого вектора
        {
            float3 v(1.0f, 2.0f, 3.0f);
            float3 zero(0.0f, 0.0f, 0.0f);

            float3 rejected = v.reject(zero);
            suite.assert_approximately_equal(rejected, v, "Reject from zero vector returns original");
        }

        // Тест is_normalized для нулевого вектора
        {
            float3 zero(0.0f, 0.0f, 0.0f);
            suite.assert_false(zero.is_normalized(), "Zero vector is not normalized");
        }

        // Тест are_orthogonal и is_orthonormal_basis
        {
            float3 x(1.0f, 0.0f, 0.0f);
            float3 y(0.0f, 1.0f, 0.0f);
            float3 z(0.0f, 0.0f, 1.0f);

            suite.assert_true(are_orthogonal(x, y, 1e-6f), "are_orthogonal() for orthogonal vectors");
            suite.assert_true(is_orthonormal_basis(x, y, z, 1e-6f), "is_orthonormal_basis() for standard basis");

            float3 not_orth(1.0f, 1.0f, 0.0f);
            suite.assert_false(are_orthogonal(x, not_orth, 1e-6f), "are_orthogonal() for non-orthogonal vectors");
        }

        suite.footer();
    }
}
