// Author: DeepSeek
// Test suite for AfterMath::float2x2 class

#pragma once

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunFloat2x2Tests()
    {
        TestSuite suite("Float2x2 Tests", true);
        suite.header();

        using namespace AfterMath;

        // ============================================================================
        // 1. Конструкторы и базовые операции
        // ============================================================================
        suite.section("Конструкторы и базовые операции");

        // Тест конструктора по умолчанию (identity matrix)
        suite.assert_equal(float2x2::identity(), float2x2(), "Default constructor returns identity");

        // Тест конструктора с двумя строками
        suite.assert_equal(float2x2(float2(1, 2), float2(3, 4)),
            float2x2(1, 2, 3, 4),
            "Constructor with two float2 rows");

        // Тест конструктора с четырьмя значениями
        {
            float2x2 mat(1.0f, 2.0f, 3.0f, 4.0f);
            suite.assert_equal(mat(0, 0), 1.0f, "4-param constructor (0,0)");
            suite.assert_equal(mat(0, 1), 2.0f, "4-param constructor (0,1)");
            suite.assert_equal(mat(1, 0), 3.0f, "4-param constructor (1,0)");
            suite.assert_equal(mat(1, 1), 4.0f, "4-param constructor (1,1)");
        }

        // Тест конструктора из массива
        {
            float data[4] = { 1, 2, 3, 4 };
            float2x2 mat(data);
            suite.assert_equal(mat[0][0], 1.0f, "Array constructor [0][0]");
            suite.assert_equal(mat[0][1], 2.0f, "Array constructor [0][1]");
            suite.assert_equal(mat[1][0], 3.0f, "Array constructor [1][0]");
            suite.assert_equal(mat[1][1], 4.0f, "Array constructor [1][1]");
        }

        // Тест скалярного конструктора
        {
            float2x2 mat(5.0f);
            suite.assert_equal(mat(0, 0), 5.0f, "Scalar constructor (0,0)");
            suite.assert_equal(mat(1, 1), 5.0f, "Scalar constructor (1,1)");
            suite.assert_equal(mat(0, 1), 0.0f, "Scalar constructor off-diagonal (0,1)");
            suite.assert_equal(mat(1, 0), 0.0f, "Scalar constructor off-diagonal (1,0)");
        }

        // Тест конструктора из вектора диагонали
        {
            float2 diag(2.0f, 3.0f);
            float2x2 mat(diag);
            suite.assert_equal(mat(0, 0), 2.0f, "Diagonal vector constructor (0,0)");
            suite.assert_equal(mat(1, 1), 3.0f, "Diagonal vector constructor (1,1)");
            suite.assert_equal(mat(0, 1), 0.0f, "Diagonal vector constructor off-diagonal (0,1)");
            suite.assert_equal(mat(1, 0), 0.0f, "Diagonal vector constructor off-diagonal (1,0)");
        }

        // ============================================================================
        // 2. Доступ к элементам
        // ============================================================================
        suite.section("Доступ к элементам");

        float2x2 mat(1, 2, 3, 4);

        // Проверка оператора []
        suite.assert_equal(mat[0], float2(1, 2), "Operator[] row0");
        suite.assert_equal(mat[1], float2(3, 4), "Operator[] row1");

        // Проверка оператора ()
        suite.assert_equal(mat(0, 0), 1.0f, "Operator() (0,0)");
        suite.assert_equal(mat(0, 1), 2.0f, "Operator() (0,1)");
        suite.assert_equal(mat(1, 0), 3.0f, "Operator() (1,0)");
        suite.assert_equal(mat(1, 1), 4.0f, "Operator() (1,1)");

        // Проверка методов row/col
        suite.assert_equal(mat.row0(), float2(1, 2), "row0()");
        suite.assert_equal(mat.row1(), float2(3, 4), "row1()");
        suite.assert_equal(mat.col0(), float2(1, 3), "col0()");
        suite.assert_equal(mat.col1(), float2(2, 4), "col1()");

        // Проверка set_row/set_col
        {
            float2x2 m;
            m.set_row0(float2(10, 11));
            m.set_row1(float2(12, 13));
            suite.assert_equal(m.row0(), float2(10, 11), "set_row0");
            suite.assert_equal(m.row1(), float2(12, 13), "set_row1");
        }

        {
            float2x2 m;
            m.set_col0(float2(10, 11));
            m.set_col1(float2(12, 13));
            suite.assert_equal(m.col0(), float2(10, 11), "set_col0");
            suite.assert_equal(m.col1(), float2(12, 13), "set_col1");
        }

        // Проверка SSE данных
        {
            __m128 sse = _mm_setr_ps(1, 2, 3, 4);
            float2x2 mat_sse(sse);
            suite.assert_equal(mat_sse, float2x2(1, 2, 3, 4), "SSE constructor and getter");

            __m128 retrieved = mat_sse.sse_data();
            float2x2 mat_from_sse(retrieved);
            suite.assert_equal(mat_from_sse, mat_sse, "SSE data roundtrip");
        }

        // ============================================================================
        // 3. Статические методы создания матриц
        // ============================================================================
        suite.section("Статические методы создания матриц");

        // Identity и Zero
        suite.assert_true(float2x2::identity().is_identity(), "identity() creates identity matrix");
        suite.assert_true(float2x2::zero().approximately_zero(), "zero() creates zero matrix");

        // Rotation матрица
        {
            float angle = Constants::Constants<float>::Pi / 4.0f; // 45 градусов
            float2x2 rot = float2x2::rotation(angle);

            // Проверка свойств матрицы вращения
            suite.assert_true(rot.is_rotation(), "rotation() creates rotation matrix");
            suite.assert_approximately_equal(rot.determinant(), 1.0f, "rotation determinant = 1", 1e-6f);

            // Проверка конкретных значений для 45 градусов
            float sqrt2_2 = std::sqrt(2.0f) / 2.0f;
            float2x2 expected_rot(sqrt2_2, -sqrt2_2,
                sqrt2_2, sqrt2_2);
            suite.assert_approximately_equal(rot, expected_rot, "rotation 45 degrees");

            // Вектор (1,0) должен перейти в (cos, sin)
            float2 vec(1, 0);
            float2 transformed = rot * vec;
            suite.assert_approximately_equal(transformed, float2(sqrt2_2, sqrt2_2),
                "rotation transforms (1,0) correctly");
        }

        // Scaling матрицы
        {
            // Из вектора
            float2 scale_vec(2, 3);
            float2x2 scale = float2x2::scaling(scale_vec);
            suite.assert_equal(scale(0, 0), 2.0f, "scaling(vector) (0,0)");
            suite.assert_equal(scale(1, 1), 3.0f, "scaling(vector) (1,1)");
            suite.assert_equal(scale(0, 1), 0.0f, "scaling(vector) off-diagonal (0,1)");
            suite.assert_equal(scale(1, 0), 0.0f, "scaling(vector) off-diagonal (1,0)");

            // Из двух значений
            float2x2 scale2 = float2x2::scaling(2.0f, 3.0f);
            suite.assert_equal(scale2, scale, "scaling(x,y) equals scaling(vector)");

            // Uniform scaling
            float2x2 uniform_scale = float2x2::scaling(5.0f);
            suite.assert_equal(uniform_scale(0, 0), 5.0f, "uniform scaling (0,0)");
            suite.assert_equal(uniform_scale(1, 1), 5.0f, "uniform scaling (1,1)");
            suite.assert_equal(uniform_scale(0, 1), 0.0f, "uniform scaling off-diagonal (0,1)");
            suite.assert_equal(uniform_scale(1, 0), 0.0f, "uniform scaling off-diagonal (1,0)");
        }

        // Shear матрица
        {
            float2 shear_vec(0.5f, 0.3f);
            float2x2 shear_mat = float2x2::shear(shear_vec);

            suite.assert_equal(shear_mat(0, 0), 1.0f, "shear (0,0)");
            suite.assert_equal(shear_mat(0, 1), 0.5f, "shear (0,1)");
            suite.assert_equal(shear_mat(1, 0), 0.3f, "shear (1,0)");
            suite.assert_equal(shear_mat(1, 1), 1.0f, "shear (1,1)");

            // Из двух значений
            float2x2 shear_mat2 = float2x2::shear(0.5f, 0.3f);
            suite.assert_equal(shear_mat2, shear_mat, "shear(x,y) equals shear(vector)");

            // Тест преобразования вектора
            float2 vec(1, 1);
            float2 sheared = shear_mat * vec;
            suite.assert_equal(sheared, float2(1.5f, 1.3f), "shear transformation");
        }

        // ============================================================================
        // 4. Арифметические операции
        // ============================================================================
        suite.section("Арифметические операции");

        float2x2 A(1, 2, 3, 4);
        float2x2 B(5, 6, 7, 8);

        // Сложение
        {
            float2x2 sum = A + B;
            float2x2 expected(6, 8, 10, 12);
            suite.assert_equal(sum, expected, "Matrix addition");

            // Проверка оператора +=
            float2x2 A_copy = A;
            A_copy += B;
            suite.assert_equal(A_copy, expected, "Operator +=");
        }

        // Вычитание
        {
            float2x2 diff = A - B;
            float2x2 expected(-4, -4, -4, -4);
            suite.assert_equal(diff, expected, "Matrix subtraction");

            // Проверка оператора -=
            float2x2 A_copy = A;
            A_copy -= B;
            suite.assert_equal(A_copy, expected, "Operator -=");
        }

        // Умножение на скаляр
        {
            float2x2 scaled = A * 2.0f;
            float2x2 expected(2, 4, 6, 8);
            suite.assert_equal(scaled, expected, "Matrix * scalar");

            // Проверка оператора *=
            float2x2 A_copy = A;
            A_copy *= 2.0f;
            suite.assert_equal(A_copy, expected, "Operator *=");

            // Проверка скаляр * матрица
            float2x2 scaled2 = 2.0f * A;
            suite.assert_equal(scaled2, expected, "Scalar * matrix");
        }

        // Деление на скаляр
        {
            float2x2 A_copy = A;
            A_copy /= 2.0f;
            float2x2 expected(0.5f, 1.0f, 1.5f, 2.0f);
            suite.assert_equal(A_copy, expected, "Operator /=");
        }

        // Умножение матриц
        {
            float2x2 C(1, 0, 0, 2);
            float2x2 D(3, 0, 0, 4);
            float2x2 result = C * D;
            float2x2 expected(3, 0, 0, 8);
            suite.assert_equal(result, expected, "Matrix multiplication diagonal");

            // Недиагональные матрицы
            float2x2 E(1, 2, 3, 4);
            float2x2 F(2, 0, 1, 2);
            float2x2 EF = E * F;
            float2x2 expected_EF(4, 4, 10, 8);
            suite.assert_equal(EF, expected_EF, "Matrix multiplication non-diagonal");

            // Проверка оператора *=
            float2x2 C_copy = C;
            C_copy *= D;
            suite.assert_equal(C_copy, expected, "Operator *=");
        }

        // Унарные операторы
        {
            float2x2 neg = -A;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    suite.assert_equal(neg(i, j), -A(i, j), "Unary minus");
                }
            }

            float2x2 pos = +A;
            suite.assert_equal(pos, A, "Unary plus");
        }

        // ============================================================================
        // 5. Умножение на векторы
        // ============================================================================
        suite.section("Умножение на векторы");

        float2x2 M(1, 2, 3, 4);
        float2 v(2, 3);

        // Умножение матрицы на вектор (справа)
        {
            float2 result = M * v;
            float2 expected(8, 18); // (1*2 + 2*3, 3*2 + 4*3)
            suite.assert_equal(result, expected, "Matrix * vector");
        }

        // Умножение вектора на матрицу (слева) - использует transform_vector
        {
            float2 result = v * M;
            // Это должно быть эквивалентно v * M в смысле вектора-строки
            // В реализации это M.transform_vector(v), что дает тот же результат
            float2 expected = M * v;
            suite.assert_equal(result, expected, "Vector * matrix");
        }

        // Тест transform_vector
        {
            float2 result = M.transform_vector(v);
            float2 expected = M * v;
            suite.assert_equal(result, expected, "transform_vector");
        }

        // Тест transform_point (должен быть таким же как transform_vector для float2x2)
        {
            float2 result = M.transform_point(v);
            float2 expected = M * v;
            suite.assert_equal(result, expected, "transform_point");
        }

        // Тест mul свободные функции
        {
            float2 mul_vec = mul(v, M);
            suite.assert_equal(mul_vec, M * v, "mul(vector, matrix)");

            float2x2 mul_mat = mul(M, float2x2::identity());
            suite.assert_equal(mul_mat, M, "mul(matrix, identity)");
        }

        // ============================================================================
        // 6. Матричные операции
        // ============================================================================
        suite.section("Матричные операции");

        // Транспонирование
        {
            float2x2 mat(1, 2, 3, 4);
            float2x2 transposed = mat.transposed();
            float2x2 expected(1, 3, 2, 4);
            suite.assert_equal(transposed, expected, "transposed");

            // Проверка что транспонирование дважды возвращает исходную матрицу
            suite.assert_equal(transposed.transposed(), mat, "transpose twice returns original");

            // Проверка свободной функции
            suite.assert_equal(transpose(mat), transposed, "transpose() free function");
        }

        // Определитель
        {
            float2x2 identity = float2x2::identity();
            suite.assert_approximately_equal(identity.determinant(), 1.0f, "identity determinant = 1");

            float2x2 zero = float2x2::zero();
            suite.assert_approximately_equal(zero.determinant(), 0.0f, "zero determinant = 0");

            float2x2 mat(1, 2, 3, 4);
            float expected_det = 1 * 4 - 2 * 3; // -2
            suite.assert_approximately_equal(mat.determinant(), expected_det, "2x2 determinant calculation");

            // Проверка свободной функции
            suite.assert_approximately_equal(determinant(mat), expected_det, "determinant() free function");
        }

        // Присоединенная (adjugate) матрица
        {
            float2x2 mat(1, 2, 3, 4);
            float2x2 adj = mat.adjugate();
            float2x2 expected(4, -2, -3, 1);
            suite.assert_equal(adj, expected, "adjugate matrix");

            // Проверка свойства: A * adj(A) = det(A) * I
            float2x2 product = mat * adj;
            float2x2 det_times_identity = float2x2::identity() * mat.determinant();
            suite.assert_approximately_equal(product, det_times_identity, "A * adj(A) = det(A) * I");
        }

        // Обратная матрица
        {
            float2x2 scale = float2x2::scaling(2, 3);
            float2x2 invScale = scale.inverted();
            float2x2 expected(0.5f, 0, 0, 1.0f / 3.0f);
            suite.assert_approximately_equal(invScale, expected, "inverse of scaling matrix");

            // Проверка что A * A^(-1) = I
            float2x2 product = scale * invScale;
            suite.assert_true(product.is_identity(1e-5f), "A * A^(-1) = I for scaling");

            // Проверка матрицы вращения (обратная = транспонированная)
            float angle = Constants::Constants<float>::Pi / 6.0f;
            float2x2 rotation = float2x2::rotation(angle);
            float2x2 invRotation = rotation.inverted();
            float2x2 transposedRotation = rotation.transposed();
            suite.assert_approximately_equal(invRotation, transposedRotation,
                "inverse of rotation = transpose", 1e-5f);

            // Проверка свободной функции
            suite.assert_approximately_equal(inverse(scale), expected, "inverse() free function");

            // Матрица с нулевым определителем
            float2x2 singular(1, 2, 2, 4); // Строки линейно зависимы
            // Должна вернуть identity (как в реализации при нулевом определителе)
            float2x2 invSingular = singular.inverted();
            suite.assert_true(invSingular.is_identity(1e-5f), "singular matrix inverse returns identity");
        }

        // След матрицы
        {
            float2x2 mat(1, 2, 3, 4);
            float expectedTrace = 1 + 4;
            suite.assert_approximately_equal(mat.trace(), expectedTrace, "trace");

            // Проверка свободной функции
            suite.assert_approximately_equal(trace(mat), expectedTrace, "trace() free function");
        }

        // Диагональ
        {
            float2x2 mat(1, 2, 3, 4);
            float2 expectedDiag(1, 4);
            suite.assert_equal(mat.diagonal(), expectedDiag, "diagonal");

            // Проверка свободной функции
            suite.assert_equal(diagonal(mat), expectedDiag, "diagonal() free function");
        }

        // Норма Фробениуса
        {
            float2x2 mat(1, 0, 0, 2);
            float expectedNorm = std::sqrt(1.0f * 1.0f + 2.0f * 2.0f);
            suite.assert_approximately_equal(mat.frobenius_norm(), expectedNorm, "frobenius_norm");

            // Проверка свободной функции
            suite.assert_approximately_equal(frobenius_norm(mat), expectedNorm, "frobenius_norm() free function");
        }

        // ============================================================================
        // 7. Специальные функции
        // ============================================================================
        suite.section("Специальные функции");

        {
            float angle = Constants::Constants<float>::Pi / 3.0f; // 60 градусов
            float2x2 rot = float2x2::rotation(angle);
            float extracted = rot.get_rotation();
            suite.assert_approximately_equal(extracted, angle, "get_rotation from pure rotation", 1e-5f);

            // Матрица с РАВНОМЕРНЫМ масштабом и поворотом
            float2x2 rot_scale = float2x2::scaling(2.0f) * float2x2::rotation(angle);
            float extracted2 = rot_scale.get_rotation();
            suite.assert_approximately_equal(extracted2, angle, "get_rotation from uniform scaled rotation", 1e-5f);

            // Матрица с отражением (отрицательный детерминант)
            float2x2 reflect(1, 0, 0, -1); // Отражение по Y
            float extracted3 = reflect.get_rotation();
            // Для матриц с отражением get_rotation должен вернуть 0 (как в реализации)
            suite.assert_approximately_equal(extracted3, 0.0f, "get_rotation from reflection returns 0");
        }

        // Извлечение масштаба
        {
            float2 scale_vec(2, 3);
            float2x2 scale_mat = float2x2::scaling(scale_vec);
            float2 extracted = scale_mat.get_scale();
            suite.assert_approximately_equal(extracted, scale_vec, "get_scale from scaling matrix");

            // Для матрицы с поворотом и масштабом
            float angle = Constants::Constants<float>::Pi / 4.0f;
            float2x2 rot_scale = float2x2::rotation(angle) * float2x2::scaling(2, 3);
            float2 extracted2 = rot_scale.get_scale();
            // Масштаб должен извлекаться как длины столбцов
            float2 expected2(rot_scale.col0().length(),
                rot_scale.col1().length());
            suite.assert_approximately_equal(extracted2, expected2, "get_scale from rotation+scaling");
        }

        // Установка поворота
        {
            float2x2 mat = float2x2::scaling(2, 3);
            float new_angle = Constants::Constants<float>::Pi / 3.0f;

            mat.set_rotation(new_angle);
            float extracted = mat.get_rotation();
            suite.assert_approximately_equal(extracted, new_angle, "set_rotation", 1e-5f);

            // Масштаб должен сохраниться
            float2 scale = mat.get_scale();
            suite.assert_approximately_equal(scale, float2(2, 3), "set_rotation preserves scale", 1e-5f);
        }

        // Установка масштаба
        {
            float2x2 mat = float2x2::rotation(Constants::Constants<float>::Pi / 4.0f);
            float2 new_scale(3, 4);

            mat.set_scale(new_scale);
            float2 extracted = mat.get_scale();
            suite.assert_approximately_equal(extracted, new_scale, "set_scale", 1e-5f);

            // Поворот должен сохраниться
            float angle = mat.get_rotation();
            suite.assert_approximately_equal(angle, Constants::Constants<float>::Pi / 4.0f,
                "set_scale preserves rotation", 1e-5f);
        }

        // ============================================================================
        // 8. Проверки свойств
        // ============================================================================
        suite.section("Проверки свойств");

        // Проверка identity
        suite.assert_true(float2x2::identity().is_identity(), "identity().is_identity()");
        suite.assert_false(float2x2::zero().is_identity(), "zero().is_identity() returns false");
        suite.assert_true(float2x2::scaling(1.0f).is_identity(), "uniform scaling(1) is identity");

        // Проверка orthogonality
        {
            float2x2 rotation = float2x2::rotation(Constants::Constants<float>::Pi / 3.0f);
            suite.assert_true(rotation.is_orthogonal(), "rotation matrix is orthogonal");

            float2x2 scale = float2x2::scaling(2, 3);
            suite.assert_true(scale.is_orthogonal(), "scaling matrix is orthogonal");

            float2x2 non_ortho(1, 2, 3, 4);
            suite.assert_false(non_ortho.is_orthogonal(), "non-orthogonal matrix detected");

            // Проверка свободной функции
            suite.assert_true(is_orthogonal(rotation), "is_orthogonal() free function");
        }

        // Проверка rotation
        {
            float2x2 rotation = float2x2::rotation(Constants::Constants<float>::Pi / 3.0f);
            suite.assert_true(rotation.is_rotation(), "rotation matrix is rotation");

            float2x2 scale = float2x2::scaling(2, 2);
            suite.assert_false(scale.is_rotation(), "uniform scaling is not rotation (det != 1)");

            float2x2 scale_non_uniform = float2x2::scaling(2, 3);
            suite.assert_false(scale_non_uniform.is_rotation(), "non-uniform scaling is not rotation");

            // Проверка свободной функции
            suite.assert_true(is_rotation(rotation), "is_rotation() free function");
        }

        // Проверка approximately_zero
        suite.assert_true(float2x2::zero().approximately_zero(), "zero().approximately_zero()");
        suite.assert_false(float2x2::identity().approximately_zero(), "identity().approximately_zero() returns false");

        // Проверка approximately
        {
            float2x2 mat1(1, 2, 3, 4);

            // Матрица с небольшими различиями
            float2x2 mat2(1.000001f, 2.000001f, 3.000001f, 4.000001f);

            // Проверяем, что с epsilon по умолчанию они считаются равными
            suite.assert_true(mat1.approximately(mat2, Constants::Constants<float>::Epsilon),
                "matrices are approximately equal with default epsilon");

            // И поэтому оператор == должен возвращать true
            suite.assert_true(mat1 == mat2, "operator == returns true for approximately equal matrices");

            // А оператор != должен возвращать false
            suite.assert_false(mat1 != mat2, "operator != returns false for approximately equal matrices");

            // Теперь создадим матрицу, которая точно отличается
            float2x2 mat3(2, 2, 3, 4);

            suite.assert_true(mat1 != mat3, "operator != returns true for different matrices");
        }

        // ============================================================================
        // 9. Преобразования данных
        // ============================================================================
        suite.section("Преобразования данных");

        // to_row_major
        {
            float2x2 mat(1, 2, 3, 4);

            float rowMajor[4];
            mat.to_row_major(rowMajor);

            float expected[4] = { 1, 2, 3, 4 };
            for (int i = 0; i < 4; ++i) {
                suite.assert_equal(rowMajor[i], expected[i],
                    "to_row_major element " + std::to_string(i));
            }
        }

        // to_column_major
        {
            float2x2 mat(1, 2, 3, 4);

            float colMajor[4];
            mat.to_column_major(colMajor);

            float expected[4] = { 1, 3, 2, 4 };
            for (int i = 0; i < 4; ++i) {
                suite.assert_equal(colMajor[i], expected[i],
                    "to_column_major element " + std::to_string(i));
            }
        }

        // to_string
        {
            float2x2 mat(1.5f, 2.5f, 3.5f, 4.5f);

            std::string str = mat.to_string();
            // Проверяем что строка содержит ожидаемые значения
            suite.assert_true(str.find("1.5000") != std::string::npos, "to_string contains 1.5000");
            suite.assert_true(str.find("4.5000") != std::string::npos, "to_string contains 4.5000");
        }

        // isValid (неявно через assert_equal, но можем проверить с NaN)
        {
            float2x2 validMat(1, 2, 3, 4);
            // Поскольку в float2x2 нет метода isValid, пропускаем этот тест
            suite.skip_test("isValid method", "float2x2 doesn't have isValid method");
        }

        // ============================================================================
        // 10. Конструкторы из других типов
        // ============================================================================
        suite.section("Конструкторы из других типов");

        // В float2x2 нет конструкторов из других матричных типов или кватернионов
        // поэтому пропускаем этот раздел

        // ============================================================================
        // 11. Граничные случаи и особые значения
        // ============================================================================
        suite.section("Граничные случаи и особые значения");

        // Очень маленькие значения
        {
            float epsilon = Constants::Constants<float>::Epsilon;
            float2x2 tinyMat(epsilon, 0, 0, epsilon);

            suite.assert_true(tinyMat.approximately_zero(epsilon * 2),
                "approximately_zero with tiny values");
        }

        // Очень большие значения
        {
            float large = 1e10f;
            float2x2 largeMat(large, 0, 0, large);

            // Обратная матрица должна иметь маленькие значения
            float2x2 invLarge = largeMat.inverted();
            float2x2 expected(1.0f / large, 0, 0, 1.0f / large);
            suite.assert_approximately_equal(invLarge, expected,
                "inverse of large diagonal matrix", 1e-5f);
        }

        // Нулевой угол вращения
        {
            float2x2 rot = float2x2::rotation(0.0f);
            suite.assert_true(rot.is_identity(1e-5f), "rotation(0) returns identity");
        }

        // Вращение на 90 градусов
        {
            float2x2 rot90 = float2x2::rotation(Constants::Constants<float>::Pi / 2.0f);
            float2 vec(1, 0);
            float2 transformed = rot90 * vec;
            suite.assert_approximately_equal(transformed, float2(0, 1), "rotation 90 degrees");
        }

        // Вращение на 180 градусов
        {
            float2x2 rot180 = float2x2::rotation(Constants::Constants<float>::Pi);
            float2 vec(1, 0);
            float2 transformed = rot180 * vec;
            suite.assert_approximately_equal(transformed, float2(-1, 0), "rotation 180 degrees");
        }

        // Вращение на 360 градусов
        {
            float2x2 rot360 = float2x2::rotation(2.0f * Constants::Constants<float>::Pi);
            suite.assert_true(rot360.is_identity(1e-5f), "rotation(2pi) returns identity");
        }

        // Деление на ноль
        {
            float2x2 mat(1, 2, 3, 4);

            try {
                mat /= 0.0f;
                // Если не выброшено исключение, элементы должны быть inf или nan
                // В реализации используется 1.0f / scalar, что даст inf
                suite.assert_true(std::isinf(mat(0, 0)) || std::isnan(mat(0, 0)),
                    "division by zero produces inf/nan");
            }
            catch (...) {
                suite.skip_test("Division by zero", "Exception thrown - implementation dependent");
            }
        }

        // Матрица с отрицательным масштабом (отражение)
        {
            float2x2 reflect(-1, 0, 0, 1); // Отражение по X
            suite.assert_approximately_equal(reflect.determinant(), -1.0f,
                "reflection has determinant -1");

            float2 vec(1, 1);
            float2 reflected = reflect * vec;
            suite.assert_equal(reflected, float2(-1, 1), "reflection transformation");
        }

        // Комплексное преобразование: поворот + масштаб + сдвиг
        {
            float2x2 rot = float2x2::rotation(Constants::Constants<float>::Pi / 6.0f);
            float2x2 scale = float2x2::scaling(2, 3);
            float2x2 shear = float2x2::shear(0.1f, 0.2f);

            float2x2 complex = rot * scale * shear;

            // Проверка детерминанта
            float det_complex = complex.determinant();
            float det_expected = rot.determinant() * scale.determinant() * shear.determinant();
            suite.assert_approximately_equal(det_complex, det_expected,
                "determinant of complex transformation", 1e-5f);

            // Проверка инверсии
            float2x2 inv_complex = complex.inverted();
            float2x2 product = complex * inv_complex;
            suite.assert_true(product.is_identity(1e-4f),
                "complex transformation has valid inverse");
        }

        suite.footer();
    }
}
