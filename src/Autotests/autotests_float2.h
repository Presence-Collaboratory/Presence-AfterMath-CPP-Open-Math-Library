// Author: DeepSeek, NSDeathman
// Test suite for Math::float2 class

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunFloat2Tests()
    {
        TestSuite suite("float2 Tests", true);
        suite.header();

        using namespace AfterMath;

        // ============================================================================
        // 1. Конструкторы
        // ============================================================================
        suite.section("Конструкторы");

        // Тест конструктора по умолчанию
        {
            float2 v;
            suite.assert_approximately_equal(v.x, 0.0f, "Default constructor x");
            suite.assert_approximately_equal(v.y, 0.0f, "Default constructor y");
        }

        // Тест конструктора с компонентами
        {
            float2 v(1.5f, 2.5f);
            suite.assert_approximately_equal(v.x, 1.5f, "Component constructor x");
            suite.assert_approximately_equal(v.y, 2.5f, "Component constructor y");
        }

        // Тест конструктора со скаляром
        {
            float2 v(3.0f);
            suite.assert_approximately_equal(v.x, 3.0f, "Scalar constructor x");
            suite.assert_approximately_equal(v.y, 3.0f, "Scalar constructor y");
        }

        // Тест конструктора из массива
        {
            float data[2] = { 4.0f, 5.0f };
            float2 v(data);
            suite.assert_approximately_equal(v.x, 4.0f, "Array constructor x");
            suite.assert_approximately_equal(v.y, 5.0f, "Array constructor y");
        }

        // Тест копирующего конструктора
        {
            float2 original(6.0f, 7.0f);
            float2 copy(original);
            suite.assert_approximately_equal(copy.x, 6.0f, "Copy constructor x");
            suite.assert_approximately_equal(copy.y, 7.0f, "Copy constructor y");
        }

        // Тест статических конструкторов
        {
            suite.assert_approximately_equal(float2::zero(), float2(0.0f, 0.0f), "zero()");
            suite.assert_approximately_equal(float2::one(), float2(1.0f, 1.0f), "one()");
            suite.assert_approximately_equal(float2::unit_x(), float2(1.0f, 0.0f), "unit_x()");
            suite.assert_approximately_equal(float2::unit_y(), float2(0.0f, 1.0f), "unit_y()");
        }

        // ============================================================================
        // 2. Операторы присваивания
        // ============================================================================
        suite.section("Операторы присваивания");

        // Тест присваивания скаляра
        {
            float2 v;
            v = 2.5f;
            suite.assert_approximately_equal(v.x, 2.5f, "Scalar assignment x");
            suite.assert_approximately_equal(v.y, 2.5f, "Scalar assignment y");
        }

        // Тест составных операторов присваивания
        {
            float2 v(1.0f, 2.0f);

            v += float2(3.0f, 4.0f);
            suite.assert_approximately_equal(v, float2(4.0f, 6.0f), "Operator +=");

            v -= float2(1.0f, 2.0f);
            suite.assert_approximately_equal(v, float2(3.0f, 4.0f), "Operator -=");

            v *= float2(2.0f, 3.0f);
            suite.assert_approximately_equal(v, float2(6.0f, 12.0f), "Operator *=");

            v /= float2(2.0f, 3.0f);
            suite.assert_approximately_equal(v, float2(3.0f, 4.0f), "Operator /=");

            v *= 2.0f;
            suite.assert_approximately_equal(v, float2(6.0f, 8.0f), "Operator *= scalar");

            v /= 2.0f;
            suite.assert_approximately_equal(v, float2(3.0f, 4.0f), "Operator /= scalar");
        }

        // ============================================================================
        // 3. Бинарные операторы
        // ============================================================================
        suite.section("Бинарные операторы");

        // Тест сложения
        {
            float2 a(1.0f, 2.0f);
            float2 b(3.0f, 4.0f);
            float2 result = a + b;
            suite.assert_approximately_equal(result, float2(4.0f, 6.0f), "Operator +");
        }

        // Тест вычитания
        {
            float2 a(5.0f, 6.0f);
            float2 b(2.0f, 3.0f);
            float2 result = a - b;
            suite.assert_approximately_equal(result, float2(3.0f, 3.0f), "Operator -");
        }

        // Тест унарных операторов
        {
            float2 a(1.0f, 2.0f);
            suite.assert_approximately_equal(+a, float2(1.0f, 2.0f), "Unary +");
            suite.assert_approximately_equal(-a, float2(-1.0f, -2.0f), "Unary -");
        }

        // Тест скалярных операций
        {
            float2 v(2.0f, 3.0f);

            float2 result1 = v + 1.0f;
            suite.assert_approximately_equal(result1, float2(3.0f, 4.0f), "Vector + scalar");

            float2 result2 = v - 1.0f;
            suite.assert_approximately_equal(result2, float2(1.0f, 2.0f), "Vector - scalar");

            float2 result3 = v * 2.0f;
            suite.assert_approximately_equal(result3, float2(4.0f, 6.0f), "Vector * scalar");

            float2 result4 = 2.0f * v;
            suite.assert_approximately_equal(result4, float2(4.0f, 6.0f), "Scalar * vector");

            float2 result5 = v / 2.0f;
            suite.assert_approximately_equal(result5, float2(1.0f, 1.5f), "Vector / scalar");

            float2 result6 = 2.0f + v;
            suite.assert_approximately_equal(result6, float2(4.0f, 5.0f), "Scalar + vector");
        }

        // ============================================================================
        // 4. Операторы доступа и преобразования
        // ============================================================================
        suite.section("Операторы доступа и преобразования");

        // Тест оператора индексации
        {
            float2 v(7.0f, 8.0f);
            suite.assert_approximately_equal(v[0], 7.0f, "Operator [] index 0");
            suite.assert_approximately_equal(v[1], 8.0f, "Operator [] index 1");

            v[0] = 9.0f;
            v[1] = 10.0f;
            suite.assert_approximately_equal(v.x, 9.0f, "Operator [] mutable x");
            suite.assert_approximately_equal(v.y, 10.0f, "Operator [] mutable y");
        }

        // Тест преобразования в указатель
        {
            float2 v(1.0f, 2.0f);
            const float* ptr = v;
            suite.assert_approximately_equal(ptr[0], 1.0f, "Conversion to const float* index 0");
            suite.assert_approximately_equal(ptr[1], 2.0f, "Conversion to const float* index 1");

            float* mutable_ptr = v;
            mutable_ptr[0] = 3.0f;
            suite.assert_approximately_equal(v.x, 3.0f, "Conversion to float* mutable");
        }

        // ============================================================================
        // 5. Математические функции
        // ============================================================================
        suite.section("Математические функции");

        // Тест длины
        {
            float2 v(3.0f, 4.0f);
            suite.assert_approximately_equal(v.length(), 5.0f, "length()");
            suite.assert_approximately_equal(v.length_sq(), 25.0f, "length_sq()");

            float2 zero(0.0f, 0.0f);
            suite.assert_approximately_equal(zero.length(), 0.0f, "length() of zero vector");
        }

        // Тест нормализации
        {
            float2 v(3.0f, 4.0f);
            float2 normalized = v.normalize();
            float expected_len = 1.0f;
            suite.assert_approximately_equal(normalized.length(), expected_len, "normalize() length");
            suite.assert_approximately_equal(normalized.x, 0.6f, "normalize() x");
            suite.assert_approximately_equal(normalized.y, 0.8f, "normalize() y");

            // Тест нормализации нулевого вектора
            float2 zero(0.0f, 0.0f);
            float2 zero_norm = zero.normalize();
            suite.assert_approximately_equal(zero_norm, float2::zero(), "normalize() zero vector");
        }

        // Тест скалярного произведения
        {
            float2 a(1.0f, 2.0f);
            float2 b(3.0f, 4.0f);
            float dot_result = a.dot(b);
            suite.assert_approximately_equal(dot_result, 11.0f, "dot()");

            // Ортогональные векторы
            float2 orth1(1.0f, 0.0f);
            float2 orth2(0.0f, 1.0f);
            suite.assert_approximately_equal(orth1.dot(orth2), 0.0f, "dot() orthogonal vectors");
        }

        // Тест векторного произведения (2D)
        {
            float2 a(1.0f, 2.0f);
            float2 b(3.0f, 4.0f);
            float cross_result = a.cross(b);
            suite.assert_approximately_equal(cross_result, -2.0f, "cross()");
        }

        // Тест расстояния
        {
            float2 a(1.0f, 2.0f);
            float2 b(4.0f, 6.0f);
            float distance = a.distance(b);
            float distance_sq = a.distance_sq(b);

            // (4-1)² + (6-2)² = 9 + 16 = 25
            suite.assert_approximately_equal(distance, 5.0f, "distance()");
            suite.assert_approximately_equal(distance_sq, 25.0f, "distance_sq()");
        }

        // ============================================================================
        // 6. HLSL-подобные функции
        // ============================================================================
        suite.section("HLSL-подобные функции");

        // Тест abs
        {
            float2 v(-1.5f, 2.5f);
            float2 result = v.abs();
            suite.assert_approximately_equal(result, float2(1.5f, 2.5f), "abs()");
        }

        // Тест sign
        {
            float2 v(-2.0f, 3.0f);
            float2 result = v.sign();
            suite.assert_approximately_equal(result, float2(-1.0f, 1.0f), "sign() positive/negative");

            float2 zero_mixed(0.0f, -0.0f);
            float2 sign_zero = zero_mixed.sign();
            suite.assert_approximately_equal(sign_zero, float2(0.0f, 0.0f), "sign() zero");
        }

        // Тест floor
        {
            float2 v(1.7f, -2.3f);
            float2 result = v.floor();
            suite.assert_approximately_equal(result, float2(1.0f, -3.0f), "floor()");
        }

        // Тест ceil
        {
            float2 v(1.2f, -2.7f);
            float2 result = v.ceil();
            suite.assert_approximately_equal(result, float2(2.0f, -2.0f), "ceil()");
        }

        // Тест round
        {
            float2 v(1.4f, 1.6f);
            float2 result = v.round();
            suite.assert_approximately_equal(result, float2(1.0f, 2.0f), "round()");
        }

        // Тест frac
        {
            float2 v(1.7f, -2.3f);
            float2 result = v.frac();
            suite.assert_approximately_equal(result.x, 0.7f, "frac() x", 1e-6f);
            suite.assert_approximately_equal(result.y, 0.7f, "frac() y", 1e-6f); // -2.3 - (-3) = 0.7
        }

        // Тест saturate
        {
            float2 v(-0.5f, 1.5f);
            float2 result = v.saturate();
            suite.assert_approximately_equal(result, float2(0.0f, 1.0f), "saturate()");
        }

        // Тест step
        {
            float2 v(0.5f, 1.5f);
            float2 result = v.step(1.0f);
            suite.assert_approximately_equal(result, float2(0.0f, 1.0f), "step()");
        }

        // Тест smoothstep
        {
            float2 v(0.5f, 1.5f);
            float2 result = v.smoothstep(0.0f, 2.0f);
            // Для t=0.25: 3t² - 2t³ = 3*0.0625 - 2*0.015625 = 0.1875 - 0.03125 = 0.15625
            // Для t=0.75: 3*0.5625 - 2*0.421875 = 1.6875 - 0.84375 = 0.84375
            suite.assert_approximately_equal(result.x, 0.15625f, "smoothstep() x", 1e-6f);
            suite.assert_approximately_equal(result.y, 0.84375f, "smoothstep() y", 1e-6f);
        }

        // ============================================================================
        // 7. Геометрические операции
        // ============================================================================
        suite.section("Геометрические операции");

        // Тест перпендикуляра
        {
            float2 v(2.0f, 3.0f);
            float2 perp = v.perpendicular();
            suite.assert_approximately_equal(perp, float2(-3.0f, 2.0f), "perpendicular()");
        }

        // Тест отражения
        {
            float2 incident(1.0f, -1.0f);
            float2 normal(0.0f, 1.0f); // Нормаль вверх
            float2 reflected = incident.reflect(normal.normalize());
            // R = I - 2*(I·N)*N = (1,-1) - 2*(-1)*(0,1) = (1,-1) + (0,2) = (1,1)
            suite.assert_approximately_equal(reflected, float2(1.0f, 1.0f), "reflect()");
        }

        // Тест преломления
        {
            float2 incident(1.0f, -1.0f);
            float2 normal(0.0f, 1.0f);
            normal = normal.normalize();

            // Для eta = 1.0 (та же среда) - вектор должен остаться прежним
            float2 refracted = incident.refract(normal, 1.0f);
            suite.assert_approximately_equal(refracted, incident, "refract() with eta=1.0");

            // Для eta = 0.5 - вычисляем ожидаемый результат
            // incident = (1, -1), normal = (0, 1), eta = 0.5
            // d = dot(incident, normal) = -1
            // k = 1 - eta² * (1 - d²) = 1 - 0.25 * (1 - 1) = 1
            // refracted = eta * incident - (eta * d + sqrt(k)) * normal
            // = 0.5 * (1, -1) - (0.5 * (-1) + 1) * (0, 1)
            // = (0.5, -0.5) - (0.5) * (0, 1)
            // = (0.5, -1.0)
            float2 refracted2 = incident.refract(normal, 0.5f);
            float2 expected2(0.5f, -1.0f);
            suite.assert_approximately_equal(refracted2, expected2, "refract() with eta=0.5");

            // Тест полного внутреннего отражения: ВОДА→ВОЗДУХ
            // n1 = 1.33 (вода), n2 = 1.0 (воздух), eta = n2/n1 = 1.0/1.33 ≈ 0.7519
            // Критический угол: θ_c = arcsin(eta) = arcsin(0.7519) ≈ 48.8°
            // Берем угол падения 60° > 48.8° - должно быть полное внутреннее отражение

            float angle = Constants::Constants<float>::Pi / 3.0f; // 60°
            float2 incident_large_angle(std::sin(angle), -std::cos(angle));

            float eta_water_to_air = 1.33f;

            float2 total_reflection = incident_large_angle.refract(normal, eta_water_to_air);

            // Должен вернуть нулевой вектор при полном отражении
            suite.assert_approximately_equal(total_reflection, float2::zero(),
                "refract() total internal reflection for 60° angle, water->air");
        }

        // Тест вращения
        {
            float2 v(1.0f, 0.0f);
            float angle = Constants::Constants<float>::Pi / 2.0f; // 90 градусов
            float2 rotated = v.rotate(angle);
            suite.assert_approximately_equal(rotated.x, 0.0f, "rotate() 90° x", 1e-6f);
            suite.assert_approximately_equal(rotated.y, 1.0f, "rotate() 90° y", 1e-6f);

            // 180 градусов
            float2 rotated180 = v.rotate(Constants::Constants<float>::Pi);
            suite.assert_approximately_equal(rotated180.x, -1.0f, "rotate() 180° x", 1e-6f);
            suite.assert_approximately_equal(rotated180.y, 0.0f, "rotate() 180° y", 1e-6f);
        }

        // Тест угла вектора
        {
            float2 v(1.0f, 0.0f);
            float angle = v.angle();
            suite.assert_approximately_equal(angle, 0.0f, "angle() for (1,0)");

            float2 v2(0.0f, 1.0f);
            float angle2 = v2.angle();
            suite.assert_approximately_equal(angle2, Constants::Constants<float>::Pi / 2.0f, "angle() for (0,1)");

            float2 v3(-1.0f, 0.0f);
            float angle3 = v3.angle();
            suite.assert_approximately_equal(angle3, Constants::Constants<float>::Pi, "angle() for (-1,0)", 1e-6f);
        }

        // ============================================================================
        // 8. Swizzle операции
        // ============================================================================
        suite.section("Swizzle операции");

        {
            float2 v(2.0f, 3.0f);

            suite.assert_approximately_equal(v.yx(), float2(3.0f, 2.0f), "yx()");
            suite.assert_approximately_equal(v.xx(), float2(2.0f, 2.0f), "xx()");
            suite.assert_approximately_equal(v.yy(), float2(3.0f, 3.0f), "yy()");
        }

        // ============================================================================
        // 9. Утилитарные методы
        // ============================================================================
        suite.section("Утилитарные методы");

        // Тест isValid
        {
            float2 valid(1.0f, 2.0f);
            suite.assert_true(valid.isValid(), "isValid() for valid vector");

            // Note: Для тестов с NaN и INF нужно специальное создание
            suite.skip_test("isValid() with NaN/INF", "Requires special NaN/INF construction");
        }

        // Тест approximately
        {
            float2 a(1.0f, 2.0f);
            float2 b(1.000001f, 2.000001f);
            float2 c(1.1f, 2.1f);

            suite.assert_true(a.approximately(b, 1e-5f), "approximately() within epsilon");
            suite.assert_false(a.approximately(c, 1e-5f), "approximately() outside epsilon");
        }

        // Тест approximately_zero
        {
            float2 zero(0.0f, 0.0f);
            float2 near_zero(0.000001f, 0.000001f);
            float2 not_zero(0.1f, 0.1f);

            suite.assert_true(zero.approximately_zero(1e-5f), "approximately_zero() for zero");
            suite.assert_true(near_zero.approximately_zero(1e-4f), "approximately_zero() for near zero");
            suite.assert_false(not_zero.approximately_zero(1e-5f), "approximately_zero() for non-zero");
        }

        // Тест is_normalized
        {
            float2 normalized(0.6f, 0.8f);
            float2 not_normalized(1.0f, 2.0f);

            suite.assert_true(normalized.is_normalized(1e-5f), "is_normalized() for normalized vector");
            suite.assert_false(not_normalized.is_normalized(1e-5f), "is_normalized() for non-normalized");
        }

        // Тест to_string
        {
            float2 v(1.5f, 2.5f);
            std::string str = v.to_string();

            // Проверяем наличие ожидаемых значений в строке
            suite.assert_true(str.find("1.5") != std::string::npos || str.find("1.500") != std::string::npos,
                "to_string() contains x value");
            suite.assert_true(str.find("2.5") != std::string::npos || str.find("2.500") != std::string::npos,
                "to_string() contains y value");
        }

        // Тест data()
        {
            float2 v(7.0f, 8.0f);
            const float* cdata = v.data();
            float* data = v.data();

            suite.assert_approximately_equal(cdata[0], 7.0f, "data() const access x");
            suite.assert_approximately_equal(cdata[1], 8.0f, "data() const access y");

            data[0] = 9.0f;
            suite.assert_approximately_equal(v.x, 9.0f, "data() mutable modification");
        }

        // ============================================================================
        // 10. Операторы сравнения
        // ============================================================================
        suite.section("Операторы сравнения");

        {
            float2 a(1.0f, 2.0f);
            float2 b(1.0f, 2.0f);
            float2 c(1.1f, 2.1f);

            suite.assert_true(a == b, "Operator == for equal vectors");
            suite.assert_false(a == c, "Operator == for different vectors");
            suite.assert_false(a != b, "Operator != for equal vectors");
            suite.assert_true(a != c, "Operator != for different vectors");
        }

        // ============================================================================
        // 11. Глобальные операторы
        // ============================================================================
        suite.section("Глобальные операторы");

        // Тест умножения и деления векторов
        {
            float2 a(2.0f, 3.0f);
            float2 b(4.0f, 5.0f);

            float2 mul_result = a * b;
            suite.assert_approximately_equal(mul_result, float2(8.0f, 15.0f), "Global operator *");

            float2 div_result = a / b;
            suite.assert_approximately_equal(div_result, float2(0.5f, 0.6f), "Global operator /");
        }

        // ============================================================================
        // 12. Глобальные математические функции
        // ============================================================================
        suite.section("Глобальные математические функции");

        // Тест distance и distance_sq
        {
            float2 a(1.0f, 2.0f);
            float2 b(4.0f, 6.0f);

            suite.assert_approximately_equal(distance(a, b), 5.0f, "Global distance()");
            suite.assert_approximately_equal(distance_sq(a, b), 25.0f, "Global distance_sq()");
        }

        // Тест dot и cross
        {
            float2 a(1.0f, 2.0f);
            float2 b(3.0f, 4.0f);

            suite.assert_approximately_equal(dot(a, b), 11.0f, "Global dot()");
            suite.assert_approximately_equal(cross(a, b), -2.0f, "Global cross()");
        }

        // Тест approximately глобальная
        {
            float2 a(1.0f, 2.0f);
            float2 b(1.000001f, 2.000001f);

            suite.assert_true(approximately(a, b, 1e-5f), "Global approximately()");
        }

        // Тест isValid глобальная
        {
            float2 v(1.0f, 2.0f);
            suite.assert_true(isValid(v), "Global isValid()");
        }

        // Тест линейной интерполяции
        {
            float2 a(0.0f, 0.0f);
            float2 b(10.0f, 20.0f);

            float2 lerp_result = lerp(a, b, 0.5f);
            suite.assert_approximately_equal(lerp_result, float2(5.0f, 10.0f), "Global lerp() at 0.5");

            float2 lerp_start = lerp(a, b, 0.0f);
            suite.assert_approximately_equal(lerp_start, a, "Global lerp() at 0.0");

            float2 lerp_end = lerp(a, b, 1.0f);
            suite.assert_approximately_equal(lerp_end, b, "Global lerp() at 1.0");
        }

        // Тест сферической линейной интерполяции
        {
            float2 a(1.0f, 0.0f);
            float2 b(0.0f, 1.0f);

            a = a.normalize();
            b = b.normalize();

            float2 slerp_result = slerp(a, b, 0.5f);
            float expected_length = 1.0f;
            suite.assert_approximately_equal(slerp_result.length(), expected_length, "Global slerp() length");

            // В середине между (1,0) и (0,1) должно быть примерно (√2/2, √2/2)
            float expected_val = std::sqrt(2.0f) / 2.0f;
            suite.assert_approximately_equal(slerp_result.x, expected_val, "Global slerp() x at 0.5", 1e-6f);
            suite.assert_approximately_equal(slerp_result.y, expected_val, "Global slerp() y at 0.5", 1e-6f);
        }

        // Тест angle_between
        {
            float2 a(1.0f, 0.0f);
            float2 b(0.0f, 1.0f);
            float2 c(-1.0f, 0.0f);

            float angle_ab = angle_between(a, b);
            suite.assert_approximately_equal(angle_ab, Constants::Constants<float>::Pi / 2.0f,
                "Global angle_between() 90 degrees");

            float angle_ac = angle_between(a, c);
            suite.assert_approximately_equal(angle_ac, Constants::Constants<float>::Pi,
                "Global angle_between() 180 degrees", 1e-6f);
        }

        // Тест signed_angle_between
        {
            float2 a(1.0f, 0.0f);
            float2 b(0.0f, 1.0f);
            float2 c(0.0f, -1.0f);

            float angle_ab = signed_angle_between(a, b);
            float angle_ac = signed_angle_between(a, c);

            suite.assert_approximately_equal(angle_ab, Constants::Constants<float>::Pi / 2.0f,
                "Global signed_angle_between() positive 90");
            suite.assert_approximately_equal(angle_ac, -Constants::Constants<float>::Pi / 2.0f,
                "Global signed_angle_between() negative 90", 1e-6f);
        }

        // Тест проекции
        {
            float2 v(2.0f, 3.0f);
            float2 onto(1.0f, 0.0f); // Ось X

            float2 projected = project(v, onto);
            suite.assert_approximately_equal(projected, float2(2.0f, 0.0f), "Global project() onto X axis");

            // Проекция на себя должна дать себя
            float2 self_projected = project(v, v);
            suite.assert_approximately_equal(self_projected, v, "Global project() onto itself");
        }

        // Тест отклонения
        {
            float2 v(2.0f, 3.0f);
            float2 onto(1.0f, 0.0f); // Ось X

            float2 rejected = reject(v, onto);
            // v = (2,3), проекция на X = (2,0), отклонение = (0,3)
            suite.assert_approximately_equal(rejected, float2(0.0f, 3.0f), "Global reject() from X axis");

            // Отклонение от себя должно дать 0
            float2 self_rejected = reject(v, v);
            suite.assert_approximately_equal(self_rejected, float2::zero(), "Global reject() from itself");
        }

        // ============================================================================
        // 13. Глобальные HLSL-функции
        // ============================================================================
        suite.section("Глобальные HLSL-функции");

        {
            float2 v(-1.5f, 2.5f);

            suite.assert_approximately_equal(abs(v), float2(1.5f, 2.5f), "Global abs()");
            suite.assert_approximately_equal(sign(v), float2(-1.0f, 1.0f), "Global sign()");
            suite.assert_approximately_equal(floor(v), float2(-2.0f, 2.0f), "Global floor()");
            suite.assert_approximately_equal(ceil(v), float2(-1.0f, 3.0f), "Global ceil()");
            suite.assert_approximately_equal(round(float2(1.4f, 1.6f)), float2(1.0f, 2.0f), "Global round()");
            suite.assert_approximately_equal(saturate(float2(-0.5f, 1.5f)), float2(0.0f, 1.0f), "Global saturate()");
        }

        // Тест clamp, min, max
        {
            float2 v(0.5f, 1.5f);
            float2 min_val(0.0f, 0.0f);
            float2 max_val(1.0f, 1.0f);

            float2 clamped = clamp(v, min_val, max_val);
            suite.assert_approximately_equal(clamped, float2(0.5f, 1.0f), "Global clamp()");

            float2 a(1.0f, 3.0f);
            float2 b(2.0f, 2.0f);

            float2 min_result = min(a, b);
            suite.assert_approximately_equal(min_result, float2(1.0f, 2.0f), "Global min()");

            float2 max_result = max(a, b);
            suite.assert_approximately_equal(max_result, float2(2.0f, 3.0f), "Global max()");
        }

        // ============================================================================
        // 14. Утилитарные глобальные функции
        // ============================================================================
        suite.section("Утилитарные глобальные функции");

        // Тест distance_to_line_segment
        {
            float2 point(0.0f, 0.0f);
            float2 line_start(1.0f, 0.0f);
            float2 line_end(3.0f, 0.0f);

            float distance = distance_to_line_segment(point, line_start, line_end);
            // Расстояние от (0,0) до отрезка [(1,0),(3,0)] = 1.0
            suite.assert_approximately_equal(distance, 1.0f, "distance_to_line_segment() to horizontal line");

            // Точка внутри проекции отрезка
            float2 point2(2.0f, 2.0f);
            float distance2 = distance_to_line_segment(point2, line_start, line_end);
            // Расстояние от (2,2) до горизонтальной линии на y=0 = 2.0
            suite.assert_approximately_equal(distance2, 2.0f, "distance_to_line_segment() perpendicular to middle");

            // Точка ближе к началу отрезка
            float2 point3(0.0f, 2.0f);
            float distance3 = distance_to_line_segment(point3, line_start, line_end);
            // Расстояние до начальной точки (1,0): √((0-1)² + (2-0)²) = √(1 + 4) = √5 ≈ 2.236
            suite.assert_approximately_equal(distance3, std::sqrt(5.0f), "distance_to_line_segment() closest to start");
        }

        // ============================================================================
        // 15. Граничные случаи
        // ============================================================================
        suite.section("Граничные случаи");

        // Тест с очень маленькими значениями
        {
            float epsilon = 1e-30f;
            float2 tiny(epsilon, epsilon);
            suite.assert_true(tiny.approximately_zero(1e-20f), "Tiny values approximately_zero");
        }

        // Тест с очень большими значениями
        {
            float large = 1e10f;
            float2 huge(large, large);
            suite.assert_false(huge.approximately_zero(), "Huge values not approximately_zero");

            float2 normalized_huge = huge.normalize();
            float expected_length = 1.0f;
            suite.assert_approximately_equal(normalized_huge.length(), expected_length,
                "Normalize huge values", 1e-6f);
        }

        // Тест деления на ноль (векторное)
        {
            float2 a(1.0f, 2.0f);
            float2 zero_vec(0.0f, 0.0f);

            // Должно вызвать деление на ноль, но мы проверим что происходит
            // В C++ деление на ноль float дает inf или NaN
            float2 result = a / zero_vec;

            // Проверяем что результат содержит бесконечности или NaN
            suite.assert_true(!std::isfinite(result.x) || !std::isfinite(result.y),
                "Division by zero vector produces non-finite values");
        }

        // Тест деления на ноль (скалярное)
        {
            float2 v(1.0f, 2.0f);
            float2 result = v / 0.0f;

            suite.assert_true(!std::isfinite(result.x) && !std::isfinite(result.y),
                "Division by zero scalar produces non-finite values");
        }

        // Тест нормализации очень маленького вектора
        {
            float2 tiny(1e-20f, 1e-20f);
            float2 normalized = tiny.normalize();

            // Должен вернуть нулевой вектор, так как длина меньше epsilon
            suite.assert_approximately_equal(normalized, float2::zero(),
                "Normalize tiny vector returns zero");
        }

        // Тест smoothstep с edge0 = edge1
        {
            float2 v(0.5f, 1.5f);
            float2 result = v.smoothstep(1.0f, 1.0f);
            // При edge0 = edge1 используется step(edge0)
            suite.assert_approximately_equal(result, v.step(1.0f), "smoothstep() with equal edges");
        }

        // Тест вращения нулевого вектора
        {
            float2 zero(0.0f, 0.0f);
            float2 rotated = zero.rotate(1.0f);
            suite.assert_approximately_equal(rotated, zero, "Rotate zero vector");
        }

        // Тест отражения от нулевой нормали
        {
            float2 v(1.0f, 2.0f);
            float2 zero_normal(0.0f, 0.0f);

            // Нормаль должна быть нормализована, но нулевой вектор не может быть нормализован
            // В коде используется dot(normal), где normal - параметр
            // Проверим что происходит при вызове с ненормализованным нулевым вектором
            float2 reflected = v.reflect(zero_normal);
            // R = I - 2*(I·N)*N = I - 0 = I
            suite.assert_approximately_equal(reflected, v, "Reflect with zero normal returns original");
        }

        // Тест скалярного произведения с самим собой
        {
            float2 v(3.0f, 4.0f);
            float dot_self = v.dot(v);
            float expected = v.length_sq();
            suite.assert_approximately_equal(dot_self, expected, "Dot product with self equals length squared");
        }

        // Тест векторного произведения с самим собой
        {
            float2 v(2.0f, 3.0f);
            float cross_self = v.cross(v);
            suite.assert_approximately_equal(cross_self, 0.0f, "Cross product with self equals zero");
        }

        // Тест угла нулевого вектора
        {
            float2 zero(0.0f, 0.0f);
            float angle = zero.angle();
            // Угол нулевого вектора не определен, но atan2(0,0) обычно возвращает 0
            suite.assert_approximately_equal(angle, 0.0f, "Angle of zero vector");
        }

        // Тест is_normalized для нулевого вектора
        {
            float2 zero(0.0f, 0.0f);
            suite.assert_false(zero.is_normalized(), "Zero vector is not normalized");
        }

        suite.footer();
    }
}
