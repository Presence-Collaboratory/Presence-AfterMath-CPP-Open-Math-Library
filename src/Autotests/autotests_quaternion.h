// autotests_quaternion.h
// Автотесты для класса кватерниона с учетом исправлений

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunQuaternionTests()
    {
        TestSuite suite("Quaternion Tests");
        suite.header();

        using namespace AfterMath;

        // ============================================================================
        // 1. Конструкторы и фабричные методы
        // ============================================================================
        suite.section("Constructors and Factory Methods");

        // Конструктор по умолчанию (единичный кватернион)
        suite.assert_approximately_equal(
            quaternion().to_matrix3x3(),
            float3x3::identity(),
            "Default constructor creates identity"
        );

        // Конструктор по компонентам
        suite.assert_equal(quaternion(1, 2, 3, 4).x, 1.0f, "Component constructor x");
        suite.assert_equal(quaternion(1, 2, 3, 4).y, 2.0f, "Component constructor y");
        suite.assert_equal(quaternion(1, 2, 3, 4).z, 3.0f, "Component constructor z");
        suite.assert_equal(quaternion(1, 2, 3, 4).w, 4.0f, "Component constructor w");

        // Конструктор ось-угол
        {
            quaternion q(float3::unit_x(), Constants::PI);
            float3 axis;
            float angle;
            q.to_axis_angle(axis, angle);
            suite.assert_approximately_equal(axis, float3::unit_x(), "Axis-angle constructor axis");
            suite.assert_approximately_equal(angle, Constants::PI, "Axis-angle constructor angle");
        }

        // Конструктор из углов Эйлера (Yaw-Pitch-Roll)
        {
            quaternion q = quaternion::from_euler(0.3f, 0.5f, 0.2f); // Yaw, Pitch, Roll
            float3 euler = q.to_euler();
            suite.assert_approximately_equal(euler.x, 0.3f, "Euler constructor yaw", 1e-4f);
            suite.assert_approximately_equal(euler.y, 0.5f, "Euler constructor pitch", 1e-4f);
            suite.assert_approximately_equal(euler.z, 0.2f, "Euler constructor roll", 1e-4f);
        }

        // Конструктор из матрицы
        {
            float3x3 rot = float3x3::rotation_x(Constants::HALF_PI);
            quaternion q(rot);
            quaternion q2 = quaternion::from_matrix(rot);
            suite.assert_approximately_equal(q, q2, "Matrix constructor consistency");
        }

        // Фабричные методы
        suite.assert_equal(quaternion::identity(), quaternion(0, 0, 0, 1), "Identity factory method");
        suite.assert_equal(quaternion::zero(), quaternion(0, 0, 0, 0), "Zero factory method");
        suite.assert_equal(quaternion::one(), quaternion(1, 1, 1, 1), "One factory method");

        // Статические методы вращения
        {
            quaternion rx = quaternion::rotation_x(Constants::HALF_PI);
            quaternion rx_axis = quaternion(float3::unit_x(), Constants::HALF_PI);
            suite.assert_approximately_equal(rx, rx_axis, "rotation_x method");
        }

        {
            quaternion ry = quaternion::rotation_y(Constants::HALF_PI);
            quaternion ry_axis = quaternion(float3::unit_y(), Constants::HALF_PI);
            suite.assert_approximately_equal(ry, ry_axis, "rotation_y method");
        }

        {
            quaternion rz = quaternion::rotation_z(Constants::HALF_PI);
            quaternion rz_axis = quaternion(float3::unit_z(), Constants::HALF_PI);
            suite.assert_approximately_equal(rz, rz_axis, "rotation_z method");
        }

        // ============================================================================
        // 2. Базовые операции
        // ============================================================================
        suite.section("Basic Operations");

        // Сложение
        {
            quaternion a(1, 2, 3, 4);
            quaternion b(5, 6, 7, 8);
            quaternion c = a + b;
            suite.assert_equal(c.x, 6.0f, "Addition x");
            suite.assert_equal(c.y, 8.0f, "Addition y");
            suite.assert_equal(c.z, 10.0f, "Addition z");
            suite.assert_equal(c.w, 12.0f, "Addition w");
        }

        // Вычитание
        {
            quaternion a(5, 6, 7, 8);
            quaternion b(1, 2, 3, 4);
            quaternion c = a - b;
            suite.assert_equal(c.x, 4.0f, "Subtraction x");
            suite.assert_equal(c.y, 4.0f, "Subtraction y");
            suite.assert_equal(c.z, 4.0f, "Subtraction z");
            suite.assert_equal(c.w, 4.0f, "Subtraction w");
        }

        // Умножение на скаляр
        {
            quaternion a(1, 2, 3, 4);
            quaternion b = a * 2.0f;
            suite.assert_equal(b.x, 2.0f, "Scalar multiplication x");
            suite.assert_equal(b.y, 4.0f, "Scalar multiplication y");
            suite.assert_equal(b.z, 6.0f, "Scalar multiplication z");
            suite.assert_equal(b.w, 8.0f, "Scalar multiplication w");
        }

        // Деление на скаляр
        {
            quaternion a(2, 4, 6, 8);
            quaternion b = a / 2.0f;
            suite.assert_equal(b.x, 1.0f, "Scalar division x");
            suite.assert_equal(b.y, 2.0f, "Scalar division y");
            suite.assert_equal(b.z, 3.0f, "Scalar division z");
            suite.assert_equal(b.w, 4.0f, "Scalar division w");
        }

        // Отрицание
        {
            quaternion a(1, 2, 3, 4);
            quaternion b = -a;
            suite.assert_equal(b.x, -1.0f, "Negation x");
            suite.assert_equal(b.y, -2.0f, "Negation y");
            suite.assert_equal(b.z, -3.0f, "Negation z");
            suite.assert_equal(b.w, -4.0f, "Negation w");
        }

        // ============================================================================
        // 3. Умножение кватернионов
        // ============================================================================
        suite.section("Quaternion Multiplication");

        // Умножение на единичный кватернион
        {
            quaternion q(1, 2, 3, 4);
            quaternion identity = quaternion::identity();
            suite.assert_approximately_equal(q * identity, q, "Multiplication with identity (right)");
            suite.assert_approximately_equal(identity * q, q, "Multiplication with identity (left)");
        }

        // Ассоциативность умножения
        {
            quaternion rx = quaternion::rotation_x(Constants::PI / 4);
            quaternion ry = quaternion::rotation_y(Constants::PI / 4);
            quaternion rz = quaternion::rotation_z(Constants::PI / 4);

            quaternion a = (rx * ry) * rz;
            quaternion b = rx * (ry * rz);
            suite.assert_approximately_equal(a, b, "Multiplication associativity", 1e-5f);
        }

        // Умножение на обратный
        {
            quaternion q(1, 2, 3, 4);
            q = q.normalize();
            quaternion inv = q.inverse();
            suite.assert_approximately_equal(q * inv, quaternion::identity(), "Quaternion * inverse = identity", 1e-5f);
        }

        // ============================================================================
        // 4. Нормализация и длина
        // ============================================================================
        suite.section("Normalization and Length");

        // Нормализация
        {
            quaternion q(1, 2, 3, 4);
            quaternion n = q.normalize();
            suite.assert_approximately_equal(n.length(), 1.0f, "Normalized length = 1", 1e-6f);
            suite.assert_true(n.is_normalized(), "is_normalized() returns true for normalized quaternion");
        }

        // Длина и квадрат длины
        {
            quaternion q(3, 0, 4, 0);
            suite.assert_approximately_equal(q.length_sq(), 25.0f, "Length squared");
            suite.assert_approximately_equal(q.length(), 5.0f, "Length");
        }

        // Быстрая нормализация
        {
            quaternion q(1, 2, 3, 4);
            quaternion n1 = q.normalize();
            quaternion n2 = q.fast_normalize();
            suite.assert_approximately_equal(n1, n2, "Fast normalization vs regular", 1e-3f);
        }

        // ============================================================================
        // 5. Сопряжение и обращение
        // ============================================================================
        suite.section("Conjugate and Inverse");

        // Сопряжение
        {
            quaternion q(1, 2, 3, 4);
            quaternion conj = q.conjugate();
            suite.assert_equal(conj.x, -1.0f, "Conjugate x");
            suite.assert_equal(conj.y, -2.0f, "Conjugate y");
            suite.assert_equal(conj.z, -3.0f, "Conjugate z");
            suite.assert_equal(conj.w, 4.0f, "Conjugate w");
        }

        // Обращение единичного кватерниона
        {
            quaternion q = quaternion(1, 2, 3, 4).normalize();
            quaternion inv = q.inverse();
            suite.assert_approximately_equal(inv, q.conjugate(), "Inverse = conjugate for unit quaternion");
        }

        // ============================================================================
        // 6. Преобразование векторов
        // ============================================================================
        suite.section("Vector Transformation");

        // Вращение единичным кватернионом
        {
            float3 v(1, 2, 3);
            quaternion q = quaternion::identity();
            float3 rotated = q * v;
            suite.assert_approximately_equal(rotated, v, "Identity rotation leaves vector unchanged");
        }

        // Вращение на 90 градусов вокруг оси X
        {
            float3 v(0, 1, 0);
            quaternion q = quaternion::rotation_x(Constants::HALF_PI);
            float3 rotated = q * v;
            suite.assert_approximately_equal(rotated, float3(0, 0, 1), "90° X rotation", 1e-5f);
        }

        // Вращение на 90 градусов вокруг оси Y
        {
            float3 v(1, 0, 0);
            quaternion q = quaternion::rotation_y(Constants::HALF_PI);
            float3 rotated = q * v;
            suite.assert_approximately_equal(rotated, float3(0, 0, -1), "90° Y rotation", 1e-5f);
        }

        // Вращение на 90 градусов вокруг оси Z
        {
            float3 v(1, 0, 0);
            quaternion q = quaternion::rotation_z(Constants::HALF_PI);
            float3 rotated = q * v;
            suite.assert_approximately_equal(rotated, float3(0, 1, 0), "90° Z rotation", 1e-5f);
        }

        // Преобразование направления
        {
            float3 dir = float3(1, 2, 3).normalize();
            quaternion q = quaternion::rotation_x(Constants::PI / 3);
            float3 transformed = q.transform_direction(dir);
            suite.assert_approximately_equal(transformed.length(), 1.0f, "transform_direction preserves length", 1e-5f);
        }

        // ============================================================================
        // 7. Конверсия в/из матриц
        // ============================================================================
        suite.section("Matrix Conversions");

        // Кватернион -> матрица 3x3 -> кватернион
        {
            quaternion q = quaternion(1, 2, 3, 4).normalize();
            float3x3 m = q.to_matrix3x3();
            quaternion q2 = quaternion::from_matrix(m);
            // q и q2 могут отличаться знаком (двойное покрытие)
            suite.assert_true(q.approximately(q2) || q.approximately(-q2),
                "Quaternion -> matrix3x3 -> quaternion round trip");
        }

        // Кватернион -> матрица 4x4 -> кватернион
        {
            quaternion q = quaternion(1, 2, 3, 4).normalize();
            float4x4 m = q.to_matrix4x4();
            quaternion q2 = quaternion::from_matrix(m);
            suite.assert_true(q.approximately(q2) || q.approximately(-q2),
                "Quaternion -> matrix4x4 -> quaternion round trip");
        }

        // Эквивалентность вращения через кватернион и матрицу
        {
            quaternion q = quaternion::rotation_y(Constants::PI / 3);
            float3 v(1, 2, 3);
            float3 rotated_q = q * v;
            float3 rotated_m = q.to_matrix3x3() * v;
            suite.assert_approximately_equal(rotated_q, rotated_m,
                "Quaternion and matrix rotation equivalence", 1e-5f);
        }

        // ============================================================================
        // 8. Конверсия в/из углов Эйлера
        // ============================================================================
        suite.section("Euler Angle Conversions");

        // Прямое создание из углов Эйлера
        {
            quaternion q = quaternion::from_euler(0.3f, 0.5f, 0.2f); // Yaw, Pitch, Roll
            float3 euler = q.to_euler();
            suite.assert_approximately_equal(euler.x, 0.3f, "Yaw conversion", 1e-4f);
            suite.assert_approximately_equal(euler.y, 0.5f, "Pitch conversion", 1e-4f);
            suite.assert_approximately_equal(euler.z, 0.2f, "Roll conversion", 1e-4f);
        }

        // Конверсия туда-обратно для разных углов
        {
            std::vector<float3> test_angles = {
                float3(0.3f, 0.2f, 0.1f),    // Yaw, Pitch, Roll
                float3(0.1f, 0.3f, 0.2f),
                float3(0.2f, 0.1f, 0.3f),
                float3(0.5f, 0.0f, 0.0f),    // только Yaw
                float3(0.0f, 0.5f, 0.0f),    // только Pitch
                float3(0.0f, 0.0f, 0.5f)     // только Roll
            };

            for (const auto& angles : test_angles) {
                quaternion q = quaternion::from_euler(angles);
                float3 euler_back = q.to_euler();
                quaternion q_back = quaternion::from_euler(euler_back);

                // q и q_back должны представлять одно и то же вращение
                float dot_val = std::abs(q.dot(q_back));
                suite.assert_true(dot_val > 0.9999f,
                    "Euler round trip for angles (y=" +
                    std::to_string(angles.x) + ", p=" +
                    std::to_string(angles.y) + ", r=" +
                    std::to_string(angles.z) + ")");
            }
        }

        // Только вращение по оси Y (Yaw)
        {
            float3 euler_yaw(0.3f, 0.0f, 0.0f);
            quaternion q_yaw = quaternion::from_euler(euler_yaw);
            float3 euler_back = q_yaw.to_euler();
            suite.assert_approximately_equal(euler_back.x, 0.3f, "Yaw only rotation", 1e-4f);
        }

        // Только вращение по оси X (Pitch)
        {
            float3 euler_pitch(0.0f, 0.5f, 0.0f);
            quaternion q_pitch = quaternion::from_euler(euler_pitch);
            float3 euler_back = q_pitch.to_euler();
            suite.assert_approximately_equal(euler_back.y, 0.5f, "Pitch only rotation", 1e-4f);
        }

        // Только вращение по оси Z (Roll)
        {
            float3 euler_roll(0.0f, 0.0f, 0.2f);
            quaternion q_roll = quaternion::from_euler(euler_roll);
            float3 euler_back = q_roll.to_euler();
            suite.assert_approximately_equal(euler_back.z, 0.2f, "Roll only rotation", 1e-4f);
        }

        // Консистентность с матрицей вращения
        {
            float3 euler(0.3f, 0.5f, 0.2f); // Yaw, Pitch, Roll
            quaternion q = quaternion::from_euler(euler);
            float3x3 m1 = q.to_matrix3x3();

            // Создаем матрицу вращения напрямую из углов Эйлера в порядке Yaw->Pitch->Roll
            float3x3 m2 = float3x3::rotation_y(euler.x) *  // yaw (Y)
                float3x3::rotation_x(euler.y) *  // pitch (X)
                float3x3::rotation_z(euler.z);   // roll (Z)

            suite.assert_approximately_equal(m1, m2, "Euler to matrix consistency", 1e-4f);
        }

        // ============================================================================
        // 9. Конверсия в/из ось-угол
        // ============================================================================
        suite.section("Axis-Angle Conversions");

        // Конверсия туда-обратно
        {
            float3 axis = float3(1, 2, 3).normalize();
            float angle = 1.5f;
            quaternion q = quaternion::from_axis_angle(axis, angle);

            float3 axis2;
            float angle2;
            q.to_axis_angle(axis2, angle2);

            // Ось может быть инвертирована с изменением угла (q и -q представляют одно вращение)
            quaternion q2 = quaternion::from_axis_angle(axis2, angle2);
            suite.assert_true(q.approximately(q2) || q.approximately(-q2),
                "Axis-angle -> quaternion -> axis-angle round trip");
        }

        // ============================================================================
        // 10. Look Rotation
        // ============================================================================
        suite.section("Look Rotation");

        // Взгляд вдоль оси Z
        {
            float3 forward = float3::unit_z();
            float3 up = float3::unit_y();
            quaternion q = quaternion::look_rotation(forward, up);

            // Должен быть identity (смотрим вдоль Z с Y вверх)
            suite.assert_approximately_equal(q, quaternion::identity(),
                "Look rotation along Z axis");
        }

        // Взгляд в произвольном направлении
        {
            float3 forward = float3(1, 0, 1).normalize();
            float3 up = float3::unit_y();
            quaternion q = quaternion::look_rotation(forward, up);

            // Вектор forward должен совпадать
            float3 transformed_forward = q * float3::unit_z();
            suite.assert_approximately_equal(transformed_forward, forward,
                "Look rotation aligns forward vector", 1e-4f);

            // Вектор up должен быть перпендикулярен forward
            float3 transformed_up = q * float3::unit_y();
            float dot_val = float3::dot(transformed_forward, transformed_up);
            suite.assert_approximately_equal(dot_val, 0.0f,
                "Look rotation keeps up perpendicular to forward", 1e-4f);
        }

        // ============================================================================
        // 11. Интерполяция
        // ============================================================================
        suite.section("Interpolation");

        // NLERP между двумя кватернионами
        {
            quaternion a = quaternion::rotation_x(0);
            quaternion b = quaternion::rotation_x(Constants::PI);
            quaternion mid = nlerp(a, b, 0.5f);

            // Результат должен быть нормализован
            suite.assert_approximately_equal(mid.length(), 1.0f, "NLERP result normalized", 1e-6f);

            // Проверка граничных значений
            suite.assert_approximately_equal(nlerp(a, b, 0.0f), a, "NLERP at t=0");
            suite.assert_approximately_equal(nlerp(a, b, 1.0f), b, "NLERP at t=1");
        }

        // SLERP между двумя кватернионами
        {
            quaternion a = quaternion::rotation_x(0);
            quaternion b = quaternion::rotation_x(Constants::PI / 2);
            quaternion mid = slerp(a, b, 0.5f);

            // Результат должен быть нормализован
            suite.assert_approximately_equal(mid.length(), 1.0f, "SLERP result normalized", 1e-6f);

            // Проверка граничных значений
            suite.assert_approximately_equal(slerp(a, b, 0.0f), a, "SLERP at t=0");
            suite.assert_approximately_equal(slerp(a, b, 1.0f), b, "SLERP at t=1");
        }

        // SLERP approx NLERP для малых углов
        {
            quaternion a = quaternion::rotation_x(0.1f);
            quaternion b = quaternion::rotation_x(0.2f);
            quaternion s = slerp(a, b, 0.5f);
            quaternion n = nlerp(a, b, 0.5f);
            suite.assert_approximately_equal(s, n, "SLERP approx NLERP for small angles", 1e-3f);
        }

        // ============================================================================
        // 12. From-To Rotation
        // ============================================================================
        suite.section("From-To Rotation");

        // Простое вращение из одного вектора в другой
        {
            float3 from = float3::unit_x();
            float3 to = float3::unit_y();
            quaternion q = quaternion::from_to_rotation(from, to);

            float3 rotated = q * from;
            suite.assert_approximately_equal(rotated.normalize(), to,
                "From-to rotation aligns vectors", 1e-5f);
        }

        // Противоположные векторы (вращение на 180 градусов)
        {
            float3 from = float3::unit_x();
            float3 to = -float3::unit_x();
            quaternion q = quaternion::from_to_rotation(from, to);

            float3 rotated = q * from;
            suite.assert_approximately_equal(rotated.normalize(), to,
                "From-to rotation for opposite vectors", 1e-5f);
        }

        // ============================================================================
        // 13. Свойства и валидация
        // ============================================================================
        suite.section("Properties and Validation");

        // Проверка на единичный кватернион
        {
            suite.assert_true(quaternion::identity().is_identity(), "Identity quaternion is_identity()");
            suite.assert_false(quaternion(1, 0, 0, 1).is_identity(), "Non-identity quaternion is not identity");
        }

        // Проверка на нулевой кватернион
        {
            quaternion zero = quaternion::zero();
            suite.assert_true(zero.approximately_zero(), "Zero quaternion approximately_zero()");
            suite.assert_false(quaternion::identity().approximately_zero(), "Identity not approximately zero");
        }

        // Проверка валидности
        {
            quaternion valid(1, 2, 3, 4);
            suite.assert_true(valid.is_valid(), "Normal quaternion is valid");
        }

        // Скалярное произведение
        {
            quaternion a(1, 2, 3, 4);
            quaternion b(5, 6, 7, 8);
            float dot_val = dot(a, b);
            float expected = 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8;
            suite.assert_approximately_equal(dot_val, expected, "Dot product calculation");
        }

        // Приблизительное равенство
        {
            quaternion a = quaternion(1, 2, 3, 4).normalize();
            quaternion b = quaternion(1.00001f, 2.00001f, 3.00001f, 4.00001f).normalize();
            suite.assert_true(a.approximately(b, 1e-4f), "Approximately equal for similar quaternions");
            suite.assert_true(a.approximately(-b, 1e-4f), "Quaternion and its negative represent the same rotation");
        }

        // ============================================================================
        // 14. Операторы
        // ============================================================================
        suite.section("Operators");

        // Составные операторы
        {
            quaternion a(1, 2, 3, 4);
            quaternion b(5, 6, 7, 8);

            quaternion a_plus_b = a;
            a_plus_b += b;
            suite.assert_approximately_equal(a_plus_b, a + b, "+= operator");

            quaternion a_minus_b = a;
            a_minus_b -= b;
            suite.assert_approximately_equal(a_minus_b, a - b, "-= operator");

            quaternion a_times_scalar = a;
            a_times_scalar *= 2.0f;
            suite.assert_approximately_equal(a_times_scalar, a * 2.0f, "*= scalar operator");

            quaternion a_div_scalar = a;
            a_div_scalar /= 2.0f;
            suite.assert_approximately_equal(a_div_scalar, a / 2.0f, "/= scalar operator");

            quaternion a_times_q = a.normalize();
            quaternion b_norm = b.normalize();
            quaternion a_times_b = a_times_q;
            a_times_b *= b_norm;
            suite.assert_approximately_equal(a_times_b, a_times_q * b_norm, "*= quaternion operator");
        }

        // Операторы равенства
        {
            quaternion a(1, 2, 3, 4);
            quaternion b(1, 2, 3, 4);
            quaternion c(5, 6, 7, 8);

            suite.assert_true(a == b, "== operator for equal quaternions");
            suite.assert_false(a == c, "== operator for different quaternions");
            suite.assert_true(a != c, "!= operator for different quaternions");
        }

        // ============================================================================
        // 15. SIMD операции
        // ============================================================================
        suite.section("SIMD Operations");

        // Получение/установка SIMD
        {
            quaternion q(1, 2, 3, 4);
            __m128 simd = q.get_simd();
            quaternion q2;
            q2.set_simd(simd);
            suite.assert_equal(q, q2, "SIMD get/set round trip");
        }

        // ============================================================================
        // 16. Строковое представление
        // ============================================================================
        suite.section("String Representation");

        {
            quaternion q(1.5f, 2.25f, 3.125f, 4.0625f);
            std::string str = q.to_string();
            suite.assert_true(str.length() > 0, "to_string() returns non-empty string");
            suite.assert_true(str.find("1.5") != std::string::npos ||
                str.find("1.500") != std::string::npos,
                "to_string() contains x value");
        }

        // ============================================================================
        // 17. Краевые случаи
        // ============================================================================
        suite.section("Edge Cases");

        // Деление на ноль
        {
            quaternion q(1, 2, 3, 4);
            quaternion divided = q / 0.0f;
            suite.assert_false(divided.is_valid(), "Division by zero produces invalid quaternion");
        }

        // Нормализация нулевого кватерниона
        {
            quaternion zero = quaternion::zero();
            quaternion normalized = zero.normalize();
            suite.assert_approximately_equal(normalized, quaternion::identity(),
                "Normalizing zero quaternion returns identity");
        }

        // Обращение нулевого кватерниона
        {
            quaternion zero = quaternion::zero();
            quaternion inv = zero.inverse();
            suite.assert_approximately_equal(inv, quaternion::identity(),
                "Inverse of zero quaternion returns identity");
        }

        // Очень маленький кватернион
        {
            quaternion tiny(1e-10f, 2e-10f, 3e-10f, 4e-10f);
            suite.assert_true(tiny.approximately_zero(1e-5f), "Tiny quaternion approximately zero");
        }

        // Нормализация почти нулевого кватерниона
        {
            quaternion tiny(1e-15f, 2e-15f, 3e-15f, 4e-15f);
            quaternion normalized = tiny.normalize();
            suite.assert_approximately_equal(normalized.length(), 1.0f,
                "Normalization of tiny quaternion", 1e-6f);
        }

        suite.footer();
    }
}
