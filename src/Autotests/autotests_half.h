// Author: DeepSeek, NSDeathman
// Test suite for Math::half type

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunHalfTests()
    {
        TestSuite suite("half Tests", true);
        suite.header();

        using namespace AfterMath;

        // ============================================================================
        // 1. Конструкторы и преобразования
        // ============================================================================
        suite.section("Конструкторы и преобразования");

        // Тест конструктора по умолчанию
        {
            half h;
            suite.assert_approximately_equal(float(h), 0.0f, "Default constructor", 0.002f);
        }

        // Тест конструктора из float
        {
            half h(1.5f);
            suite.assert_approximately_equal(float(h), 1.5f, "Float constructor", 0.002f);
        }

        // Тест конструктора из int
        {
            half h(42);
            suite.assert_approximately_equal(float(h), 42.0f, "Int constructor", 0.002f);
        }

        // Тест конструктора из double
        {
            half h(3.14159);
            suite.assert_approximately_equal(float(h), 3.14159f, "Double constructor", 0.002f);
        }

        // Тест конструктора из битов
        {
            half h = half::from_bits(0x3C00); // 1.0 в half
            suite.assert_approximately_equal(float(h), 1.0f, "From bits constructor", 0.002f);
        }

        // Тест копирующего конструктора
        {
            half original(2.5f);
            half copy(original);
            suite.assert_approximately_equal(float(copy), 2.5f, "Copy constructor", 0.002f);
        }

        // Тест преобразования в float
        {
            half h(7.5f);
            float f = float(h);
            suite.assert_approximately_equal(f, 7.5f, "Conversion to float", 0.002f);
        }

        // Тест преобразования в double
        {
            half h(7.5f);
            double d = double(h);
            suite.assert_approximately_equal(d, 7.5, "Conversion to double", 0.002f);
        }

        // ============================================================================
        // 2. Операторы присваивания
        // ============================================================================
        suite.section("Операторы присваивания");

        // Тест присваивания float
        {
            half h;
            h = 3.14f;
            suite.assert_approximately_equal(float(h), 3.14f, "Assignment from float", 0.002f);
        }

        // Тест присваивания int
        {
            half h;
            h = 100;
            suite.assert_approximately_equal(float(h), 100.0f, "Assignment from int", 0.002f);
        }

        // Тест присваивания half
        {
            half h1(5.0f);
            half h2;
            h2 = h1;
            suite.assert_approximately_equal(float(h2), 5.0f, "Assignment from half", 0.002f);
        }

        // Тест составных операторов присваивания
        {
            half h(2.0f);

            h += half(3.0f);
            suite.assert_approximately_equal(float(h), 5.0f, "Operator +=", 0.002f);

            h -= half(1.0f);
            suite.assert_approximately_equal(float(h), 4.0f, "Operator -=", 0.002f);

            h *= half(3.0f);
            suite.assert_approximately_equal(float(h), 12.0f, "Operator *=", 0.002f);

            h /= half(4.0f);
            suite.assert_approximately_equal(float(h), 3.0f, "Operator /=", 0.002f);
        }

        // ============================================================================
        // 3. Арифметические операторы
        // ============================================================================
        suite.section("Арифметические операторы");

        // Тест сложения
        {
            half a(1.5f);
            half b(2.5f);
            half result = a + b;
            suite.assert_approximately_equal(float(result), 4.0f, "Operator +", 0.002f);
        }

        // Тест вычитания
        {
            half a(5.0f);
            half b(2.5f);
            half result = a - b;
            suite.assert_approximately_equal(float(result), 2.5f, "Operator -", 0.002f);
        }

        // Тест умножения
        {
            half a(2.0f);
            half b(3.5f);
            half result = a * b;
            suite.assert_approximately_equal(float(result), 7.0f, "Operator *", 0.002f);
        }

        // Тест деления
        {
            half a(10.0f);
            half b(4.0f);
            half result = a / b;
            suite.assert_approximately_equal(float(result), 2.5f, "Operator /", 0.002f);
        }

        // Тест унарных операторов
        {
            half a(3.0f);
            suite.assert_approximately_equal(float(+a), 3.0f, "Unary +", 0.002f);
            suite.assert_approximately_equal(float(-a), -3.0f, "Unary -", 0.002f);
        }

        // Тест инкремента/декремента
        {
            half a(5.0f);
            ++a;
            suite.assert_approximately_equal(float(a), 6.0f, "Prefix increment", 0.002f);

            half b(5.0f);
            half c = b++;
            suite.assert_approximately_equal(float(b), 6.0f, "Postfix increment result", 0.002f);
            suite.assert_approximately_equal(float(c), 5.0f, "Postfix increment return", 0.002f);

            --a;
            suite.assert_approximately_equal(float(a), 5.0f, "Prefix decrement", 0.002f);

            half d = b--;
            suite.assert_approximately_equal(float(b), 5.0f, "Postfix decrement result", 0.002f);
            suite.assert_approximately_equal(float(d), 6.0f, "Postfix decrement return", 0.002f);
        }

        // ============================================================================
        // 4. Операторы сравнения
        // ============================================================================
        suite.section("Операторы сравнения");

        {
            half a(1.0f);
            half b(1.0f);
            half c(2.0f);
            half d(-1.0f);

            suite.assert_true(a == b, "Operator == for equal values");
            suite.assert_false(a == c, "Operator == for different values");
            suite.assert_true(a != c, "Operator != for different values");
            suite.assert_false(a != b, "Operator != for equal values");
            suite.assert_true(a < c, "Operator <");
            suite.assert_true(c > a, "Operator >");
            suite.assert_true(d <= a, "Operator <=");
            suite.assert_true(a >= d, "Operator >=");
        }

        // Тест сравнения с epsilon
        {
            half a(1.0f);
            half b(1.0001f);
            half c(1.1f);

            // Используем epsilon по умолчанию (0.002f)
            suite.assert_true(a.approximately(b, 0.002f), "approximately() within epsilon");
            suite.assert_false(a.approximately(c, 0.002f), "approximately() outside epsilon");
        }

        // ============================================================================
        // 5. Специальные значения
        // ============================================================================
        suite.section("Специальные значения");

        // Тест нулей
        {
            half pos_zero = half(0.0f);
            half neg_zero = half(-0.0f);

            suite.assert_true(pos_zero.is_zero(), "is_zero() for positive zero");
            suite.assert_true(neg_zero.is_zero(), "is_zero() for negative zero");
            suite.assert_true(pos_zero.is_positive_zero(), "is_positive_zero()");
            suite.assert_true(neg_zero.is_negative_zero(), "is_negative_zero()");

            // Проверка что оператор == правильно обрабатывает нули
            suite.assert_true(pos_zero == neg_zero, "Operator == for +0 and -0");
        }

        // Тест бесконечностей
        {
            half pos_inf = half::infinity();
            half neg_inf = half::negative_infinity();

            suite.assert_true(pos_inf.is_inf(), "is_inf() for positive infinity");
            suite.assert_true(neg_inf.is_inf(), "is_inf() for negative infinity");
            suite.assert_true(pos_inf.is_positive_inf(), "is_positive_inf()");
            suite.assert_true(neg_inf.is_negative_inf(), "is_negative_inf()");

            // Создание из float
            half pos_inf_f = half(std::numeric_limits<float>::infinity());
            half neg_inf_f = half(-std::numeric_limits<float>::infinity());

            suite.assert_true(pos_inf_f.is_inf(), "Infinity from float");
            suite.assert_true(neg_inf_f.is_inf(), "Negative infinity from float");
        }

        // Тест NaN
        {
            half qnan = half::quiet_nan();
            half snan = half::signaling_nan();
            half nan_f = half(std::numeric_limits<float>::quiet_NaN());

            suite.assert_true(qnan.is_nan(), "quiet_nan().is_nan()");
            suite.assert_true(snan.is_nan(), "signaling_nan().is_nan()");
            suite.assert_true(nan_f.is_nan(), "NaN from float");

            // NaN сравнения всегда false
            suite.assert_false(qnan == qnan, "NaN == NaN is false");
            suite.assert_false(qnan == half(0.0f), "NaN == 0 is false");
            suite.assert_true(qnan != qnan, "NaN != NaN is true");
        }

        // Тест конечных значений
        {
            half normal(1.0f);
            half max_val = half::max_value();
            half min_val = half::min_value();

            suite.assert_true(normal.is_finite(), "is_finite() for normal value");
            suite.assert_true(max_val.is_finite(), "is_finite() for max value");
            suite.assert_true(min_val.is_finite(), "is_finite() for min value");
            suite.assert_true(normal.is_normal(), "is_normal() for normal value");
        }

        // ============================================================================
        // 6. Битвые операции
        // ============================================================================
        suite.section("Битвые операции");

        {
            half h(3.0f);

            suite.assert_equal(h.bits(), h.bits(), "bits() consistency");
            suite.assert_equal(h.sign_bit(), 0, "sign_bit() for positive");

            half neg = -h;
            suite.assert_equal(neg.sign_bit(), 1, "sign_bit() for negative");

            // Восстановление из битов
            half from_bits = half::from_bits(h.bits());
            suite.assert_approximately_equal(float(from_bits), float(h), "from_bits() roundtrip", 0.002f);
        }

        // ============================================================================
        // 7. Математические функции (методы)
        // ============================================================================
        suite.section("Математические функции (методы)");

        // Тест abs
        {
            half pos(3.5f);
            half neg(-3.5f);

            suite.assert_approximately_equal(float(pos.abs()), 3.5f, "abs() positive", 0.002f);
            suite.assert_approximately_equal(float(neg.abs()), 3.5f, "abs() negative", 0.002f);
        }

        // Тест reciprocal
        {
            half h(4.0f);
            half recip = h.reciprocal();
            suite.assert_approximately_equal(float(recip), 0.25f, "reciprocal()", 0.002f);
        }

        // Тест to_string
        {
            half h(2.5f);
            std::string str = h.to_string();
            suite.assert_true(str.find("2.5") != std::string::npos, "to_string() contains value");
        }

        // ============================================================================
        // 8. Статические константы
        // ============================================================================
        suite.section("Статические константы");

        {
            suite.assert_true(half::infinity().is_inf(), "infinity()");
            suite.assert_true(half::negative_infinity().is_negative_inf(), "negative_infinity()");
            suite.assert_true(half::quiet_nan().is_nan(), "quiet_nan()");
            suite.assert_true(half::signaling_nan().is_nan(), "signaling_nan()");

            // Проверка диапазонов
            half max_val = half::max_value();
            half min_val = half::min_value();
            half min_denorm = half::min_denormal_value();
            half eps = half::epsilon();
            half lowest = half::lowest();

            suite.assert_true(max_val.is_finite(), "max_value() is finite");
            suite.assert_true(min_val.is_finite(), "min_value() is finite");
            suite.assert_true(min_denorm.is_finite(), "min_denormal_value() is finite");
            suite.assert_true(eps.is_finite(), "epsilon() is finite");
            suite.assert_true(lowest.is_finite(), "lowest() is finite");
        }

        // ============================================================================
        // 9. Глобальные математические функции
        // ============================================================================
        suite.section("Глобальные математические функции");

        // Основные математические функции
        {
            half h(2.0f);

            suite.assert_approximately_equal(float(sqrt(h)), std::sqrt(2.0f), "sqrt()", 0.002f);
            suite.assert_approximately_equal(float(rsqrt(h)), 1.0f / std::sqrt(2.0f), "rsqrt()", 0.002f);
            suite.assert_approximately_equal(float(sin(half(Constants::Constants<float>::Pi / 2.0f))), 1.0f, "sin(PI/2)", 0.002f);
            suite.assert_approximately_equal(float(cos(half(0.0f))), 1.0f, "cos(0)", 0.002f);
            suite.assert_approximately_equal(float(tan(half(Constants::Constants<float>::Pi / 4.0f))), 1.0f, "tan(PI/4)", 0.002f);
            suite.assert_approximately_equal(float(exp(half(1.0f))), Constants::Constants<float>::E, "exp(1)", 0.002f);
            suite.assert_approximately_equal(float(log(half(Constants::Constants<float>::E))), 1.0f, "log(e)", 0.002f);
            suite.assert_approximately_equal(float(log10(half(100.0f))), 2.0f, "log10(100)", 0.002f);
            suite.assert_approximately_equal(float(pow(half(2.0f), half(3.0f))), 8.0f, "pow(2,3)", 0.002f);
        }

        // Округление
        {
            half h1(2.3f);
            half h2(2.7f);
            half h3(-2.3f);
            half h4(-2.7f);

            suite.assert_approximately_equal(float(floor(h1)), 2.0f, "floor(2.3)", 0.002f);
            suite.assert_approximately_equal(float(floor(h2)), 2.0f, "floor(2.7)", 0.002f);
            suite.assert_approximately_equal(float(floor(h3)), -3.0f, "floor(-2.3)", 0.002f);
            suite.assert_approximately_equal(float(floor(h4)), -3.0f, "floor(-2.7)", 0.002f);

            suite.assert_approximately_equal(float(ceil(h1)), 3.0f, "ceil(2.3)", 0.002f);
            suite.assert_approximately_equal(float(ceil(h2)), 3.0f, "ceil(2.7)", 0.002f);
            suite.assert_approximately_equal(float(ceil(h3)), -2.0f, "ceil(-2.3)", 0.002f);
            suite.assert_approximately_equal(float(ceil(h4)), -2.0f, "ceil(-2.7)", 0.002f);

            suite.assert_approximately_equal(float(round(h1)), 2.0f, "round(2.3)", 0.002f);
            suite.assert_approximately_equal(float(round(h2)), 3.0f, "round(2.7)", 0.002f);
            suite.assert_approximately_equal(float(round(h3)), -2.0f, "round(-2.3)", 0.002f);
            suite.assert_approximately_equal(float(round(h4)), -3.0f, "round(-2.7)", 0.002f);

            suite.assert_approximately_equal(float(trunc(h1)), 2.0f, "trunc(2.3)", 0.002f);
            suite.assert_approximately_equal(float(trunc(h2)), 2.0f, "trunc(2.7)", 0.002f);
            suite.assert_approximately_equal(float(trunc(h3)), -2.0f, "trunc(-2.3)", 0.002f);
            suite.assert_approximately_equal(float(trunc(h4)), -2.0f, "trunc(-2.7)", 0.002f);
        }

        // Дробная часть
        {
            half h1(3.75f);
            half h2(-3.75f);

            suite.assert_approximately_equal(float(frac(h1)), 0.75f, "frac(3.75)", 0.002f);
            suite.assert_approximately_equal(float(frac(h2)), 0.25f, "frac(-3.75)", 0.002f);
        }

        // Остаток от деления
        {
            half a(7.5f);
            half b(2.0f);

            suite.assert_approximately_equal(float(fmod(a, b)), 1.5f, "fmod(7.5, 2.0)", 0.002f);
        }

        // ============================================================================
        // 10. HLSL-функции
        // ============================================================================
        suite.section("HLSL-функции");

        // saturate
        {
            half h1(-0.5f);
            half h2(0.5f);
            half h3(1.5f);

            suite.assert_approximately_equal(float(saturate(h1)), 0.0f, "saturate(-0.5)", 0.002f);
            suite.assert_approximately_equal(float(saturate(h2)), 0.5f, "saturate(0.5)", 0.002f);
            suite.assert_approximately_equal(float(saturate(h3)), 1.0f, "saturate(1.5)", 0.002f);
        }

        // clamp
        {
            half h(2.5f);
            half min_val(1.0f);
            half max_val(2.0f);

            suite.assert_approximately_equal(float(clamp(h, min_val, max_val)), 2.0f, "clamp(2.5, 1.0, 2.0)", 0.002f);
            suite.assert_approximately_equal(float(clamp(half(0.5f), min_val, max_val)), 1.0f, "clamp(0.5, 1.0, 2.0)", 0.002f);
            suite.assert_approximately_equal(float(clamp(half(1.5f), min_val, max_val)), 1.5f, "clamp(1.5, 1.0, 2.0)", 0.002f);
        }

        // lerp
        {
            half a(0.0f);
            half b(10.0f);
            half t(0.3f);

            suite.assert_approximately_equal(float(lerp(a, b, t)), 3.0f, "lerp(0, 10, 0.3)", 0.002f);
            suite.assert_approximately_equal(float(lerp(a, b, half(0.0f))), 0.0f, "lerp at 0.0", 0.002f);
            suite.assert_approximately_equal(float(lerp(a, b, half(1.0f))), 10.0f, "lerp at 1.0", 0.002f);
        }

        // step
        {
            half edge(0.5f);
            half x1(0.3f);
            half x2(0.7f);

            suite.assert_approximately_equal(float(step(edge, x1)), 0.0f, "step(0.5, 0.3)", 0.002f);
            suite.assert_approximately_equal(float(step(edge, x2)), 1.0f, "step(0.5, 0.7)", 0.002f);
        }

        // smoothstep
        {
            half edge0(0.0f);
            half edge1(1.0f);
            half x(0.5f);

            float result = float(smoothstep(edge0, edge1, x));
            float expected = 0.5f * 0.5f * (3.0f - 2.0f * 0.5f); // t²(3-2t) при t=0.5
            suite.assert_approximately_equal(result, expected, "smoothstep(0, 1, 0.5)", 0.002f);
        }

        // sign
        {
            half h1(3.5f);
            half h2(-2.5f);
            half h3(0.0f);
            half h4(-0.0f);

            suite.assert_approximately_equal(float(sign(h1)), 1.0f, "sign(3.5)", 0.002f);
            suite.assert_approximately_equal(float(sign(h2)), -1.0f, "sign(-2.5)", 0.002f);
            suite.assert_approximately_equal(float(sign(h3)), 0.0f, "sign(0)", 0.002f);
            suite.assert_approximately_equal(float(sign(h4)), 0.0f, "sign(-0)", 0.002f);
        }

        // radians/degrees
        {
            half deg(180.0f);
            half rad = radians(deg);
            suite.assert_approximately_equal(float(rad), Constants::Constants<float>::Pi, "radians(180)", 0.002f);

            half _rad2 = half(Constants::Constants<float>::Pi);
            half deg2 = degrees(_rad2);
            suite.assert_approximately_equal(float(deg2), 180.0f, "degrees(PI)", 0.2f);
        }

        // ============================================================================
        // 11. Утилитарные функции
        // ============================================================================
        suite.section("Утилитарные функции");

        // min/max
        {
            half a(2.0f);
            half b(3.0f);

            suite.assert_approximately_equal(float(min(a, b)), 2.0f, "min(2, 3)", 0.002f);
            suite.assert_approximately_equal(float(max(a, b)), 3.0f, "max(2, 3)", 0.002f);
        }

        // copysign
        {
            half a(5.0f);
            half b(-1.0f);
            half c(1.0f);

            suite.assert_approximately_equal(float(copysign(a, b)), -5.0f, "copysign(5, -1)", 0.002f);
            suite.assert_approximately_equal(float(copysign(a, c)), 5.0f, "copysign(5, 1)", 0.002f);
        }

        // approximately (глобальная)
        {
            half a(1.0f);
            half b(1.0001f);
            half c(1.1f);

            suite.assert_true(approximately(a, b, 0.002f), "global approximately() within epsilon");
            suite.assert_false(approximately(a, c, 0.002f), "global approximately() outside epsilon");
        }

        // is_valid, is_finite, is_nan, is_inf (глобальные)
        {
            half normal(1.0f);
            half nan_val = half::quiet_nan();
            half inf_val = half::infinity();

            suite.assert_true(is_valid(normal), "global is_valid() for normal");
            suite.assert_true(is_finite(normal), "global is_finite() for normal");
            suite.assert_true(is_nan(nan_val), "global is_nan()");
            suite.assert_true(is_inf(inf_val), "global is_inf()");
        }

        // ============================================================================
        // 12. Граничные случаи
        // ============================================================================
        suite.section("Граничные случаи");

        // Очень маленькие значения
        {
            half tiny(1e-6f);
            suite.assert_true(tiny.approximately_zero(0.002f), "Tiny value approximately_zero");
        }

        // Очень большие значения (близкие к max)
        {
            half large(60000.0f); // Близко к max half (65504)
            suite.assert_false(large.approximately_zero(), "Large value not approximately_zero");
        }

        // Деление на ноль
        {
            half a(1.0f);
            half zero(0.0f);
            half result = a / zero;

            suite.assert_true(result.is_inf(), "Division by zero produces infinity");
            suite.assert_true(result.is_positive_inf(), "1.0 / 0.0 = +inf");
        }

        // Деление отрицательного числа на ноль
        {
            half a(-1.0f);
            half zero(0.0f);
            half result = a / zero;

            suite.assert_true(result.is_inf(), "Negative division by zero produces infinity");
            suite.assert_true(result.is_negative_inf(), "-1.0 / 0.0 = -inf");
        }

        // Умножение на ноль
        {
            half a(1000.0f);
            half zero(0.0f);
            half result = a * zero;

            suite.assert_approximately_equal(float(result), 0.0f, "Anything * 0 = 0", 0.002f);
        }

        // Сложение бесконечностей
        {
            half inf = half::infinity();
            half result = inf + inf;
            suite.assert_true(result.is_inf(), "inf + inf = inf");
        }

        // Вычитание бесконечностей
        {
            half inf = half::infinity();
            half neg_inf = half::negative_infinity();
            half result1 = inf - inf;
            half result2 = inf + neg_inf;

            // inf - inf должно давать NaN
            suite.assert_true(result1.is_nan(), "inf - inf = NaN");
            suite.assert_true(result2.is_nan(), "inf + (-inf) = NaN");
        }

        // ============================================================================
        // 13. Тесты совместимости с другими типами
        // ============================================================================
        suite.section("Совместимость с другими типами");

        // Операции с float
        {
            half h(2.0f);
            float f = 3.0f;

            suite.assert_approximately_equal(float(h + f), 5.0f, "half + float", 0.002f);
            suite.assert_approximately_equal(float(h - f), -1.0f, "half - float", 0.002f);
            suite.assert_approximately_equal(float(h * f), 6.0f, "half * float", 0.002f);
            suite.assert_approximately_equal(float(h / f), 2.0f / 3.0f, "half / float", 0.002f);

            suite.assert_approximately_equal(float(f + h), 5.0f, "float + half", 0.002f);
            suite.assert_approximately_equal(float(f - h), 1.0f, "float - half", 0.002f);
            suite.assert_approximately_equal(float(f * h), 6.0f, "float * half", 0.002f);
            suite.assert_approximately_equal(float(f / h), 1.5f, "float / half", 0.002f);
        }

        // Операции с double
        {
            half h(2.0f);
            double d = 3.0;

            suite.assert_approximately_equal(float(h + d), 5.0f, "half + double", 0.002f);
            suite.assert_approximately_equal(float(h - d), -1.0f, "half - double", 0.002f);
            suite.assert_approximately_equal(float(h * d), 6.0f, "half * double", 0.002f);
            suite.assert_approximately_equal(float(h / d), 2.0f / 3.0f, "half / double", 0.002f);

            suite.assert_approximately_equal(float(d + h), 5.0f, "double + half", 0.002f);
            suite.assert_approximately_equal(float(d - h), 1.0f, "double - half", 0.002f);
            suite.assert_approximately_equal(float(d * h), 6.0f, "double * half", 0.002f);
            suite.assert_approximately_equal(float(d / h), 1.5f, "double / half", 0.002f);
        }

        // Операции с int
        {
            half h(2.0f);
            int i = 3;

            suite.assert_approximately_equal(float(h + i), 5.0f, "half + int", 0.002f);
            suite.assert_approximately_equal(float(h - i), -1.0f, "half - int", 0.002f);
            suite.assert_approximately_equal(float(h * i), 6.0f, "half * int", 0.002f);
            suite.assert_approximately_equal(float(h / i), 2.0f / 3.0f, "half / int", 0.002f);

            suite.assert_approximately_equal(float(i + h), 5.0f, "int + half", 0.002f);
            suite.assert_approximately_equal(float(i - h), 1.0f, "int - half", 0.002f);
            suite.assert_approximately_equal(float(i * h), 6.0f, "int * half", 0.002f);
            suite.assert_approximately_equal(float(i / h), 1.5f, "int / half", 0.002f);
        }

        suite.footer();
    }
}
