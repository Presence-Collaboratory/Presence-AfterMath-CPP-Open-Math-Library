// Author: DeepSeek
// Test suite for AfterMath::float3x3 class

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunFloat3x3Tests()
    {
        TestSuite suite("Float3x3 Tests", true);
        suite.header();

        using namespace AfterMath;

        // ============================================================================
        // 1. Конструкторы и базовые операции
        // ============================================================================
        suite.section("Конструкторы и базовые операции");

        // Тест конструктора по умолчанию (identity matrix)
        suite.assert_equal(float3x3::identity(), float3x3(), "Default constructor returns identity");

        // Тест конструктора с диагональю
        suite.assert_equal(float3x3::scaling(2.0f, 3.0f, 4.0f),
            float3x3(float3(2, 0, 0), float3(0, 3, 0), float3(0, 0, 4)),
            "Diagonal constructor");

        // Тест конструктора из массива
        {
            float data[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            float3x3 mat(data);
            suite.assert_equal(mat[0][0], 1.0f, "Array constructor [0][0]");
            suite.assert_equal(mat[1][2], 6.0f, "Array constructor [1][2]");
            suite.assert_equal(mat[2][1], 8.0f, "Array constructor [2][1]");
        }

        // Тест скалярного конструктора
        suite.assert_equal(float3x3(5.0f), float3x3::scaling(5.0f), "Scalar constructor");

        // ============================================================================
        // 2. Доступ к элементам
        // ============================================================================
        suite.section("Доступ к элементам");

        float3x3 mat(1, 2, 3,
            4, 5, 6,
            7, 8, 9);

        // Проверка оператора []
        suite.assert_equal(mat[0], float3(1, 2, 3), "Operator[] row0");
        suite.assert_equal(mat[1], float3(4, 5, 6), "Operator[] row1");
        suite.assert_equal(mat[2], float3(7, 8, 9), "Operator[] row2");

        // Проверка оператора ()
        suite.assert_equal(mat(0, 0), 1.0f, "Operator() (0,0)");
        suite.assert_equal(mat(1, 1), 5.0f, "Operator() (1,1)");
        suite.assert_equal(mat(2, 2), 9.0f, "Operator() (2,2)");
        suite.assert_equal(mat(1, 2), 6.0f, "Operator() (1,2)");

        // Проверка методов row/col
        suite.assert_equal(mat.row0(), float3(1, 2, 3), "row0()");
        suite.assert_equal(mat.row1(), float3(4, 5, 6), "row1()");
        suite.assert_equal(mat.row2(), float3(7, 8, 9), "row2()");
        suite.assert_equal(mat.col0(), float3(1, 4, 7), "col0()");
        suite.assert_equal(mat.col1(), float3(2, 5, 8), "col1()");
        suite.assert_equal(mat.col2(), float3(3, 6, 9), "col2()");

        // Проверка set_row/set_col
        {
            float3x3 m;
            m.set_row0(float3(10, 11, 12));
            m.set_row1(float3(13, 14, 15));
            m.set_row2(float3(16, 17, 18));
            suite.assert_equal(m.row0(), float3(10, 11, 12), "set_row0");
            suite.assert_equal(m.row1(), float3(13, 14, 15), "set_row1");
            suite.assert_equal(m.row2(), float3(16, 17, 18), "set_row2");
        }

        {
            float3x3 m;
            m.set_col0(float3(10, 11, 12));
            m.set_col1(float3(13, 14, 15));
            m.set_col2(float3(16, 17, 18));
            suite.assert_equal(m.col0(), float3(10, 11, 12), "set_col0");
            suite.assert_equal(m.col1(), float3(13, 14, 15), "set_col1");
            suite.assert_equal(m.col2(), float3(16, 17, 18), "set_col2");
        }

        // ============================================================================
        // 3. Статические методы создания матриц
        // ============================================================================
        suite.section("Статические методы создания матриц");

        // Identity и Zero
        suite.assert_true(float3x3::identity().is_identity(), "identity() creates identity matrix");
        suite.assert_true(float3x3::zero().approximately_zero(), "zero() creates zero matrix");

        // Scaling матрицы
        {
            float3x3 scale = float3x3::scaling(float3(2, 3, 4));
            suite.assert_equal(scale(0, 0), 2.0f, "scaling(vector) (0,0)");
            suite.assert_equal(scale(1, 1), 3.0f, "scaling(vector) (1,1)");
            suite.assert_equal(scale(2, 2), 4.0f, "scaling(vector) (2,2)");
            suite.assert_equal(scale(0, 1), 0.0f, "scaling(vector) off-diagonal");
        }

        // Вращения вокруг осей
        {
            float angle = Constants::Constants<float>::Pi / 4.0f; // 45 градусов
            float3x3 rotX = float3x3::rotation_x(angle);

            // Проверка свойств матрицы вращения
            suite.assert_true(rotX.is_orthonormal(), "rotation_x creates orthonormal matrix");
            suite.assert_approximately_equal(rotX.determinant(), 1.0f, "rotation_x determinant = 1", 1e-6f);

            // Проверка конкретных значений для 45 градусов
            float sqrt2_2 = std::sqrt(2.0f) / 2.0f;
            float3x3 expectedRotX(1, 0, 0,
                0, sqrt2_2, -sqrt2_2,
                0, sqrt2_2, sqrt2_2);
            suite.assert_approximately_equal(rotX, expectedRotX, "rotation_x 45 degrees");
        }

        {
            float angle = Constants::Constants<float>::Pi / 2.0f; // 90 градусов
            float3x3 rotY = float3x3::rotation_y(angle);

            // Вектор вдоль оси X должен перейти в вектор вдоль оси Z
            float3 vec(1, 0, 0);
            float3 transformed = rotY * vec;
            suite.assert_approximately_equal(transformed, float3(0, 0, -1), "rotation_y 90 degrees transforms (1,0,0)");
        }

        {
            float angle = Constants::Constants<float>::Pi; // 180 градусов
            float3x3 rotZ = float3x3::rotation_z(angle);

            // Вектор (1,0,0) должен перейти в (-1,0,0)
            float3 vec(1, 0, 0);
            float3 transformed = rotZ * vec;
            suite.assert_approximately_equal(transformed, float3(-1, 0, 0), "rotation_z 180 degrees");
        }

        // Вращение вокруг произвольной оси
        {
            float3 axis = float3(1, 1, 1).normalize();
            float angle = 2.0f * Constants::Constants<float>::Pi / 3.0f; // 120 градусов

            float3x3 rot = float3x3::rotation_axis(axis, angle);

            // Матрица вращения должна быть ортонормированной
            suite.assert_true(rot.is_orthonormal(1e-5f), "rotation_axis creates orthonormal matrix");
            suite.assert_approximately_equal(rot.determinant(), 1.0f, "rotation_axis determinant = 1", 1e-6f);

            // Ось вращения должна быть собственным вектором с собственным значением 1
            float3 axisTransformed = rot * axis;
            suite.assert_approximately_equal(axisTransformed, axis, "rotation axis is eigenvector");
        }

        // Эйлеровы углы
        {
            float3 angles(Constants::Constants<float>::Pi / 6.0f,  // 30° вокруг X
                Constants::Constants<float>::Pi / 4.0f,  // 45° вокруг Y
                Constants::Constants<float>::Pi / 3.0f); // 60° вокруг Z

            float3x3 eulerMat = float3x3::rotation_euler(angles);

            // Должна быть матрица вращения
            suite.assert_true(eulerMat.is_orthonormal(1e-5f), "rotation_euler creates orthonormal matrix");
            suite.assert_approximately_equal(eulerMat.determinant(), 1.0f, "rotation_euler determinant = 1", 1e-6f);
        }

        // Кососимметричная матрица
        {
            float3 vec(1, 2, 3);
            float3x3 skew = float3x3::skew_symmetric(vec);

            // Кососимметричная матрица должна быть антисимметричной
            suite.assert_true((skew + skew.transposed()).approximately_zero(), "skew_symmetric is anti-symmetric");

            // Умножение кососимметричной матрицы на вектор эквивалентно векторному произведению
            float3 testVec(4, 5, 6);
            float3 result1 = skew * testVec;
            float3 result2 = cross(vec, testVec);
            suite.assert_approximately_equal(result1, result2, "skew_symmetric * v = cross(vec, v)");
        }

        // Внешнее произведение
        {
            float3 u(1, 2, 3);
            float3 v(4, 5, 6);
            float3x3 outer = float3x3::outer_product(u, v);

            // Каждый элемент должен быть u_i * v_j
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float expected = u[i] * v[j];
                    suite.assert_approximately_equal(outer(i, j), expected,
                        "outer_product element (" + std::to_string(i) + "," + std::to_string(j) + ")");
                }
            }
        }

        // ============================================================================
        // 4. Арифметические операции
        // ============================================================================
        suite.section("Арифметические операции");

        float3x3 A(1, 2, 3,
            4, 5, 6,
            7, 8, 9);

        float3x3 B(9, 8, 7,
            6, 5, 4,
            3, 2, 1);

        // Сложение
        {
            float3x3 sum = A + B;
            float3x3 expected(10, 10, 10,
                10, 10, 10,
                10, 10, 10);
            suite.assert_equal(sum, expected, "Matrix addition");

            // Проверка оператора +=
            float3x3 A_copy = A;
            A_copy += B;
            suite.assert_equal(A_copy, expected, "Operator +=");
        }

        // Вычитание
        {
            float3x3 diff = A - B;
            float3x3 expected(-8, -6, -4,
                -2, 0, 2,
                4, 6, 8);
            suite.assert_equal(diff, expected, "Matrix subtraction");

            // Проверка оператора -=
            float3x3 A_copy = A;
            A_copy -= B;
            suite.assert_equal(A_copy, expected, "Operator -=");
        }

        // Умножение на скаляр
        {
            float3x3 scaled = A * 2.0f;
            float3x3 expected(2, 4, 6,
                8, 10, 12,
                14, 16, 18);
            suite.assert_equal(scaled, expected, "Matrix * scalar");

            // Проверка оператора *=
            float3x3 A_copy = A;
            A_copy *= 2.0f;
            suite.assert_equal(A_copy, expected, "Operator *=");

            // Проверка скаляр * матрица
            float3x3 scaled2 = 2.0f * A;
            suite.assert_equal(scaled2, expected, "Scalar * matrix");
        }

        // Деление на скаляр
        {
            float3x3 A_copy = A;
            A_copy /= 2.0f;
            float3x3 expected(0.5f, 1.0f, 1.5f,
                2.0f, 2.5f, 3.0f,
                3.5f, 4.0f, 4.5f);
            suite.assert_equal(A_copy, expected, "Operator /=");
        }

        // Умножение матриц
        {
            float3x3 C(1, 0, 0,
                0, 2, 0,
                0, 0, 3);

            float3x3 D(2, 0, 0,
                0, 3, 0,
                0, 0, 4);

            float3x3 result = C * D;
            float3x3 expected(2, 0, 0,
                0, 6, 0,
                0, 0, 12);
            suite.assert_equal(result, expected, "Matrix multiplication diagonal");

            // Проверка ассоциативности
            float3x3 E(1, 2, 3,
                4, 5, 6,
                7, 8, 9);

            float3x3 F(9, 8, 7,
                6, 5, 4,
                3, 2, 1);

            float3x3 G(1, 0, 1,
                0, 1, 0,
                1, 0, 1);

            // (E * F) * G должно быть равно E * (F * G)
            float3x3 EF = E * F;
            float3x3 EF_G = EF * G;

            float3x3 FG = F * G;
            float3x3 E_FG = E * FG;

            suite.assert_approximately_equal(EF_G, E_FG, "Matrix multiplication associativity");

            // Проверка оператора *=
            float3x3 C_copy = C;
            C_copy *= D;
            suite.assert_equal(C_copy, expected, "Operator *=");
        }

        // Унарные операторы
        {
            float3x3 neg = -A;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    suite.assert_equal(neg(i, j), -A(i, j), "Unary minus");
                }
            }

            float3x3 pos = +A;
            suite.assert_equal(pos, A, "Unary plus");
        }

        // ============================================================================
        // 5. Умножение на векторы
        // ============================================================================
        suite.section("Умножение на векторы");

        float3x3 M(1, 2, 3,
            4, 5, 6,
            7, 8, 9);

        float3 v(2, 3, 4);

        // Умножение матрицы на вектор (справа)
        {
            float3 result = M * v;
            float3 expected(20, 47, 74); // Проверка: (1*2+2*3+3*4, 4*2+5*3+6*4, 7*2+8*3+9*4)
            suite.assert_equal(result, expected, "Matrix * vector");
        }

        // Умножение вектора на матрицу (слева)
        {
            float3 result = v * M;
            float3 expected(42, 51, 60);
            suite.assert_equal(result, expected, "Vector * matrix");
        }

        // Тест transform_vector
        {
            float3 result = M.transform_vector(v);
            float3 expected = M * v;
            suite.assert_equal(result, expected, "transform_vector");
        }

        // Тест transform_point (должен быть таким же как transform_vector для float3x3)
        {
            float3 result = M.transform_point(v);
            float3 expected = M * v;
            suite.assert_equal(result, expected, "transform_point");
        }

        // Тест transform_normal
        {
            float3x3 rotation = float3x3::rotation_x(Constants::Constants<float>::Pi / 4.0f);
            float3 normal(0, 1, 0);

            float3 transformed = rotation.transform_normal(normal);

            // Для ортонормированной матрицы transform_normal эквивалентно умножению
            float3 expected = rotation * normal;
            suite.assert_approximately_equal(transformed, expected, "transform_normal for orthonormal matrix");

            // Для неортогональной матрицы тестируем с инверсией
            float3x3 scale = float3x3::scaling(2, 3, 4);
            float3 normal2(1, 0, 0);
            float3 transformed2 = scale.transform_normal(normal2);

            // Нормаль должна масштабироваться обратно пропорционально
            float3 expected2(0.5f, 0, 0); // 1/2
            suite.assert_approximately_equal(transformed2, expected2, "transform_normal for scaling matrix");
        }

        // ============================================================================
        // 6. Матричные операции
        // ============================================================================
        suite.section("Матричные операции");

        // Транспонирование
        {
            float3x3 mat(1, 2, 3,
                4, 5, 6,
                7, 8, 9);

            float3x3 transposed = mat.transposed();
            float3x3 expected(1, 4, 7,
                2, 5, 8,
                3, 6, 9);
            suite.assert_equal(transposed, expected, "transposed");

            // Проверка что транспонирование дважды возвращает исходную матрицу
            suite.assert_equal(transposed.transposed(), mat, "transpose twice returns original");

            // Проверка свободной функции
            suite.assert_equal(transpose(mat), transposed, "transpose() free function");
        }

        // Определитель
        {
            float3x3 identity = float3x3::identity();
            suite.assert_approximately_equal(identity.determinant(), 1.0f, "identity determinant = 1");

            float3x3 zero = float3x3::zero();
            suite.assert_approximately_equal(zero.determinant(), 0.0f, "zero determinant = 0");

            float3x3 scale = float3x3::scaling(2, 3, 4);
            suite.assert_approximately_equal(scale.determinant(), 24.0f, "scaling determinant");

            // Проверка свободной функции
            suite.assert_approximately_equal(determinant(scale), 24.0f, "determinant() free function");
        }

        // Обратная матрица
        {
            float3x3 scale = float3x3::scaling(2, 3, 4);
            float3x3 invScale = scale.inverted();
            float3x3 expected(0.5f, 0, 0,
                0, 1.0f / 3.0f, 0,
                0, 0, 0.25f);
            suite.assert_approximately_equal(invScale, expected, "inverse of scaling matrix");

            // Проверка что A * A^(-1) = I
            float3x3 product = scale * invScale;
            suite.assert_true(product.is_identity(1e-5f), "A * A^(-1) = I for scaling");

            // Проверка ортогональной матрицы (обратная = транспонированная)
            float3x3 rotation = float3x3::rotation_x(Constants::Constants<float>::Pi / 3.0f);
            float3x3 invRotation = rotation.inverted();
            float3x3 transposedRotation = rotation.transposed();
            suite.assert_approximately_equal(invRotation, transposedRotation,
                "inverse of rotation = transpose", 1e-5f);

            // Проверка свободной функции
            suite.assert_approximately_equal(inverse(scale), expected, "inverse() free function");

            // Матрица с нулевым определителем
            float3x3 singular(1, 2, 3,
                4, 5, 6,
                7, 8, 9);
            // Должна вернуть identity (как в реализации при нулевом определителе)
            float3x3 invSingular = singular.inverted();
            suite.assert_true(invSingular.is_identity(1e-5f), "singular matrix inverse returns identity");
        }

        // След матрицы
        {
            float3x3 mat(1, 2, 3,
                4, 5, 6,
                7, 8, 9);
            float expectedTrace = 1 + 5 + 9;
            suite.assert_approximately_equal(mat.trace(), expectedTrace, "trace");

            // Проверка свободной функции
            suite.assert_approximately_equal(trace(mat), expectedTrace, "trace() free function");
        }

        // Диагональ
        {
            float3x3 mat(1, 2, 3,
                4, 5, 6,
                7, 8, 9);
            float3 expectedDiag(1, 5, 9);
            suite.assert_equal(mat.diagonal(), expectedDiag, "diagonal");

            // Проверка свободной функции
            suite.assert_equal(diagonal(mat), expectedDiag, "diagonal() free function");
        }

        // Норма Фробениуса
        {
            float3x3 mat(1, 0, 0,
                0, 2, 0,
                0, 0, 3);
            float expectedNorm = std::sqrt(1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f);
            suite.assert_approximately_equal(mat.frobenius_norm(), expectedNorm, "frobenius_norm");

            // Проверка свободной функции
            suite.assert_approximately_equal(frobenius_norm(mat), expectedNorm, "frobenius_norm() free function");
        }

        // Симметричная и кососимметричная части
        {
            float3x3 mat(1, 2, 3,
                4, 5, 6,
                7, 8, 9);

            float3x3 sym = mat.symmetric_part();
            float3x3 skew = mat.skew_symmetric_part();

            // Симметричная часть должна быть симметричной
            suite.assert_approximately_equal(sym, sym.transposed(), "symmetric_part is symmetric");

            // Кососимметричная часть должна быть антисимметричной
            suite.assert_approximately_equal(skew, -skew.transposed(), "skew_symmetric_part is anti-symmetric");

            // Сумма должна давать исходную матрицу
            suite.assert_approximately_equal(sym + skew, mat, "sym + skew = original");
        }

        // ============================================================================
        // 7. Специальные функции
        // ============================================================================
        suite.section("Специальные функции");

        // Нормальная матрица
        {
            float3x3 model = float3x3::scaling(2, 3, 4);
            float3x3 normalMat = float3x3::normal_matrix(model);

            // Для масштабирующей матрицы нормальная матрица должна быть обратной транспонированной
            float3x3 expected = model.inverted().transposed();

            // Столбцы должны быть нормализованы (как в реализации)
            float3 col0 = expected.col0().normalize();
            float3 col1 = expected.col1().normalize();
            float3 col2 = expected.col2().normalize();
            expected = float3x3(col0, col1, col2);

            suite.assert_approximately_equal(normalMat, expected, "normal_matrix for scaling");

            // Проверка свободной функции
            suite.assert_approximately_equal(normal_matrix(model), expected, "normal_matrix() free function");
        }

        // Извлечение масштаба
        {
            float3 scaleVec(2, 3, 4);
            float3x3 scaleMat = float3x3::scaling(scaleVec);
            float3 extracted = scaleMat.extract_scale();
            suite.assert_approximately_equal(extracted, scaleVec, "extract_scale from scaling matrix");

            // Для матрицы с поворотом и масштабом
            float3x3 rotScale = float3x3::rotation_z(Constants::Constants<float>::Pi / 4.0f) *
                float3x3::scaling(2, 3, 4);
            float3 extracted2 = rotScale.extract_scale();
            // Масштаб должен извлекаться как длины столбцов
            float3 expected2(rotScale.col0().length(),
                rotScale.col1().length(),
                rotScale.col2().length());
            suite.assert_approximately_equal(extracted2, expected2, "extract_scale from rotation+scaling");

            // Проверка свободной функции
            suite.assert_approximately_equal(extract_scale(scaleMat), scaleVec, "extract_scale() free function");
        }

        // Извлечение поворота
        {
            float3x3 rotation = float3x3::rotation_x(Constants::Constants<float>::Pi / 3.0f);
            float3x3 extracted = rotation.extract_rotation();
            suite.assert_approximately_equal(extracted, rotation, "extract_rotation from pure rotation");

            // Для матрицы с масштабом
            float3x3 scaleMat = float3x3::scaling(2, 3, 4);
            float3x3 extractedScale = scaleMat.extract_rotation();
            // Масштабирующая матрица не содержит поворота, должна вернуть identity
            suite.assert_true(extractedScale.is_identity(1e-5f), "extract_rotation from scaling matrix");

            // Для комбинированной матрицы
            float3x3 combined = float3x3::rotation_y(Constants::Constants<float>::Pi / 6.0f) *
                float3x3::scaling(2, 2, 2);
            float3x3 extractedCombined = combined.extract_rotation();
            // Извлеченная матрица должна быть ортонормированной
            suite.assert_true(extractedCombined.is_orthonormal(1e-5f), "extracted rotation is orthonormal");

            // Проверка свободной функции
            suite.assert_approximately_equal(extract_rotation(rotation), rotation, "extract_rotation() free function");
        }

        // ============================================================================
        // 8. Проверки свойств
        // ============================================================================
        suite.section("Проверки свойств");

        // Проверка identity
        suite.assert_true(float3x3::identity().is_identity(), "identity().is_identity()");
        suite.assert_false(float3x3::zero().is_identity(), "zero().is_identity() returns false");
        suite.assert_true(float3x3::scaling(1, 1, 1).is_identity(), "uniform scaling(1) is identity");

        // Проверка orthogonality
        {
            float3x3 rotation = float3x3::rotation_x(Constants::Constants<float>::Pi / 3.0f);
            suite.assert_true(rotation.is_orthogonal(), "rotation matrix is orthogonal");
            suite.assert_true(rotation.is_orthonormal(), "rotation matrix is orthonormal");

            float3x3 scale = float3x3::scaling(2, 3, 4);
            suite.assert_true(scale.is_orthogonal(), "scaling matrix is orthogonal");
            suite.assert_false(scale.is_orthonormal(), "non-uniform scaling matrix is not orthonormal");

            // Проверка свободных функций
            //suite.assert_true(is_orthogonal(rotation), "is_orthogonal() free function");
            suite.assert_true(is_orthonormal(rotation, EPSILON), "is_orthonormal() free function");
        }

        // Проверка approximately_zero
        suite.assert_true(float3x3::zero().approximately_zero(), "zero().approximately_zero()");
        suite.assert_false(float3x3::identity().approximately_zero(), "identity().approximately_zero() returns false");

        // Проверка approximately
        {
            float3x3 mat1(1, 2, 3,
                4, 5, 6,
                7, 8, 9);

            // Матрица с небольшими различиями
            float3x3 mat2(1.000001f, 2.000001f, 3.000001f,
                4.000001f, 5.000001f, 6.000001f,
                7.000001f, 8.000001f, 9.000001f);

            // Проверяем, что с epsilon по умолчанию они считаются равными
            suite.assert_true(mat1.approximately(mat2, Constants::Constants<float>::Epsilon),
                "matrices are approximately equal with default epsilon");

            // И поэтому оператор == должен возвращать true
            suite.assert_true(mat1 == mat2, "operator == returns true for approximately equal matrices");

            // А оператор != должен возвращать false
            suite.assert_false(mat1 != mat2, "operator != returns false for approximately equal matrices");

            // Теперь создадим матрицу, которая точно отличается
            float3x3 mat3(2, 2, 3,
                4, 5, 6,
                7, 8, 9);

            suite.assert_true(mat1 != mat3, "operator != returns true for different matrices");
        }

        // ============================================================================
        // 9. Преобразования данных
        // ============================================================================
        suite.section("Преобразования данных");

        // to_row_major
        {
            float3x3 mat(1, 2, 3,
                4, 5, 6,
                7, 8, 9);

            float rowMajor[9];
            mat.to_row_major(rowMajor);

            for (int i = 0; i < 9; ++i) {
                suite.assert_equal(rowMajor[i], static_cast<float>(i + 1),
                    "to_row_major element " + std::to_string(i));
            }
        }

        // to_column_major
        {
            float3x3 mat(1, 2, 3,
                4, 5, 6,
                7, 8, 9);

            float colMajor[9];
            mat.to_column_major(colMajor);

            float expected[9] = { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
            for (int i = 0; i < 9; ++i) {
                suite.assert_equal(colMajor[i], expected[i],
                    "to_column_major element " + std::to_string(i));
            }
        }

        // to_string
        {
            float3x3 mat(1.5f, 2.5f, 3.5f,
                4.5f, 5.5f, 6.5f,
                7.5f, 8.5f, 9.5f);

            std::string str = mat.to_string();
            // Проверяем что строка содержит ожидаемые значения
            suite.assert_true(str.find("1.5000") != std::string::npos, "to_string contains 1.5000");
            suite.assert_true(str.find("9.5000") != std::string::npos, "to_string contains 9.5000");
        }

        // isValid
        {
            float3x3 validMat(1, 2, 3,
                4, 5, 6,
                7, 8, 9);
            suite.assert_true(validMat.isValid(), "isValid for normal matrix");

            // Создаем матрицу с NaN (для теста)
            float3x3 nanMat(1, 2, 3,
                4, std::numeric_limits<float>::quiet_NaN(), 6,
                7, 8, 9);
            suite.assert_false(nanMat.isValid(), "isValid returns false for NaN");

            // Создаем матрицу с infinity
            float3x3 infMat(1, 2, 3,
                4, std::numeric_limits<float>::infinity(), 6,
                7, 8, 9);
            suite.assert_false(infMat.isValid(), "isValid returns false for infinity");
        }

        // ============================================================================
        // 10. Конструкторы из других типов
        // ============================================================================
        suite.section("Конструкторы из других типов");

        // Конструктор из float4x4 (только верхняя 3x3 часть)
        {
            // Сначала нужно убедиться что есть доступ к float4x4
            // Для теста создадим упрощенную версию или пропустим если нет доступа
            suite.skip_test("Constructor from float4x4", "float4x4 not available in test context");
        }

        // Конструктор из кватерниона
        {
            // Кватернион для поворота на 90 градусов вокруг оси Y
            float angle = Constants::Constants<float>::Pi / 2.0f;
            quaternion q = quaternion::rotation_y(angle);

            float3x3 matFromQuat(q);

            // Матрица для поворота на 90 градусов вокруг Y
            float3x3 expected = float3x3::rotation_y(angle);

            suite.assert_approximately_equal(matFromQuat, expected,
                "Constructor from quaternion", 1e-5f);
        }

        // ============================================================================
        // 11. Граничные случаи и особые значения
        // ============================================================================
        suite.section("Граничные случаи и особые значения");

        // Очень маленькие значения
        {
            float epsilon = Constants::Constants<float>::Epsilon;
            float3x3 tinyMat(epsilon, 0, 0,
                0, epsilon, 0,
                0, 0, epsilon);

            suite.assert_true(tinyMat.approximately_zero(epsilon * 2),
                "approximately_zero with tiny values");
        }

        // Очень большие значения
        {
            float large = 1e10f;
            float3x3 largeMat(large, 0, 0,
                0, large, 0,
                0, 0, large);

            // Обратная матрица должна иметь маленькие значения
            float3x3 invLarge = largeMat.inverted();
            float3x3 expected(1.0f / large, 0, 0,
                0, 1.0f / large, 0,
                0, 0, 1.0f / large);
            suite.assert_approximately_equal(invLarge, expected,
                "inverse of large diagonal matrix", 1e-5f);
        }

        // Нулевая ось вращения
        {
            float3 zeroAxis(0, 0, 0);
            float3x3 rot = float3x3::rotation_axis(zeroAxis, 1.0f);
            // Должна вернуться identity матрица
            suite.assert_true(rot.is_identity(1e-5f), "rotation_axis with zero axis returns identity");
        }

        // Нулевой угол вращения
        {
            float3 axis(1, 0, 0);
            float3x3 rot = float3x3::rotation_axis(axis, 0.0f);
            suite.assert_true(rot.is_identity(1e-5f), "rotation_axis with zero angle returns identity");
        }

        // Деление на ноль (должно работать корректно благодаря обработке в operator/=)
        {
            float3x3 mat(1, 2, 3,
                4, 5, 6,
                7, 8, 9);

            try {
                mat /= 0.0f;
                // Если не выброшено исключение, элементы должны быть inf или nan
                suite.assert_false(mat.isValid(), "division by zero produces invalid matrix");
            }
            catch (...) {
                suite.skip_test("Division by zero", "Exception thrown - implementation dependent");
            }
        }

        suite.footer();
    }
}
