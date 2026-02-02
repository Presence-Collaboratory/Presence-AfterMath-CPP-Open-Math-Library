// Author: DeepSeek
// Test suite for Math::float4x4 class

#include "AutotestCore.h"

namespace MathTests
{
    void RunFloat4x4Tests()
    {
        TestSuite suite("Float4x4 Tests", true);
        suite.header();

        using namespace Math;

        // ============================================================================
        // 1. Конструкторы и базовые операции
        // ============================================================================
        suite.section("Конструкторы и базовые операции");

        // Тест конструктора по умолчанию (identity matrix)
        suite.assert_equal(float4x4::identity(), float4x4(), "Default constructor returns identity");

        // Тест конструктора с 4 векторами
        {
            float4x4 mat(
                float4(1, 2, 3, 4),
                float4(5, 6, 7, 8),
                float4(9, 10, 11, 12),
                float4(13, 14, 15, 16)
            );
            suite.assert_equal(mat.row0(), float4(1, 2, 3, 4), "Constructor from 4 vectors row0");
            suite.assert_equal(mat.row3(), float4(13, 14, 15, 16), "Constructor from 4 vectors row3");
        }

        // Тест конструктора с 16 скалярами
        {
            float4x4 mat(
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            );
            suite.assert_equal(mat(0, 0), 1.0f, "Scalar constructor (0,0)");
            suite.assert_equal(mat(1, 1), 6.0f, "Scalar constructor (1,1)");
            suite.assert_equal(mat(2, 2), 11.0f, "Scalar constructor (2,2)");
            suite.assert_equal(mat(3, 3), 16.0f, "Scalar constructor (3,3)");
        }

        // Тест конструктора из массива
        {
            float data[16] = {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            };
            float4x4 mat(data);
            suite.assert_equal(mat[0][0], 1.0f, "Array constructor [0][0]");
            suite.assert_equal(mat[1][2], 7.0f, "Array constructor [1][2]");
            suite.assert_equal(mat[3][3], 16.0f, "Array constructor [3][3]");
        }

        // Тест скалярного конструктора
        float4x4 scalarMat(5.0f);
        float4x4 expectedScaling = float4x4::scaling(5.0f);
        expectedScaling(3, 3) = 5.0f; // Скалярный конструктор устанавливает и (3,3) в 5
        suite.assert_equal(scalarMat, expectedScaling, "Scalar constructor creates matrix with scalar on diagonal");

        // Тест конструктора из диагонали
        {
            float4x4 mat(float4(2, 3, 4, 5));
            suite.assert_equal(mat(0, 0), 2.0f, "Diagonal constructor (0,0)");
            suite.assert_equal(mat(1, 1), 3.0f, "Diagonal constructor (1,1)");
            suite.assert_equal(mat(2, 2), 4.0f, "Diagonal constructor (2,2)");
            suite.assert_equal(mat(3, 3), 5.0f, "Diagonal constructor (3,3)");
            suite.assert_equal(mat(0, 1), 0.0f, "Diagonal constructor off-diagonal (0,1)");
        }

        // ============================================================================
        // 2. Доступ к элементам
        // ============================================================================
        suite.section("Доступ к элементам");

        float4x4 mat(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );

        // Проверка оператора []
        suite.assert_equal(mat[0], float4(1, 2, 3, 4), "Operator[] row0");
        suite.assert_equal(mat[1], float4(5, 6, 7, 8), "Operator[] row1");
        suite.assert_equal(mat[2], float4(9, 10, 11, 12), "Operator[] row2");
        suite.assert_equal(mat[3], float4(13, 14, 15, 16), "Operator[] row3");

        // Проверка оператора ()
        suite.assert_equal(mat(0, 0), 1.0f, "Operator() (0,0)");
        suite.assert_equal(mat(1, 1), 6.0f, "Operator() (1,1)");
        suite.assert_equal(mat(2, 2), 11.0f, "Operator() (2,2)");
        suite.assert_equal(mat(3, 3), 16.0f, "Operator() (3,3)");
        suite.assert_equal(mat(1, 2), 7.0f, "Operator() (1,2)");
        suite.assert_equal(mat(3, 0), 13.0f, "Operator() (3,0)");

        // Проверка методов row()
        suite.assert_equal(mat.row0(), float4(1, 2, 3, 4), "row0()");
        suite.assert_equal(mat.row1(), float4(5, 6, 7, 8), "row1()");
        suite.assert_equal(mat.row2(), float4(9, 10, 11, 12), "row2()");
        suite.assert_equal(mat.row3(), float4(13, 14, 15, 16), "row3()");

        // Проверка методов col()
        suite.assert_equal(mat.col0(), float4(1, 5, 9, 13), "col0()");
        suite.assert_equal(mat.col1(), float4(2, 6, 10, 14), "col1()");
        suite.assert_equal(mat.col2(), float4(3, 7, 11, 15), "col2()");
        suite.assert_equal(mat.col3(), float4(4, 8, 12, 16), "col3()");

        // Проверка set_row()
        {
            float4x4 m;
            m.set_row0(float4(10, 11, 12, 13));
            m.set_row1(float4(14, 15, 16, 17));
            m.set_row2(float4(18, 19, 20, 21));
            m.set_row3(float4(22, 23, 24, 25));
            suite.assert_equal(m.row0(), float4(10, 11, 12, 13), "set_row0");
            suite.assert_equal(m.row1(), float4(14, 15, 16, 17), "set_row1");
            suite.assert_equal(m.row2(), float4(18, 19, 20, 21), "set_row2");
            suite.assert_equal(m.row3(), float4(22, 23, 24, 25), "set_row3");
        }

        // ============================================================================
        // 3. Статические методы создания матриц
        // ============================================================================
        suite.section("Статические методы создания матриц");

        // Identity и Zero
        suite.assert_true(float4x4::identity().is_identity(), "identity() creates identity matrix");
        suite.assert_true(float4x4::zero().approximately_zero(), "zero() creates zero matrix");

        // Матрица трансляции
        {
            float3 translation(2, 3, 4);
            float4x4 transMat = float4x4::translation(translation);

            // Проверка структуры матрицы трансляции
            suite.assert_true(transMat.is_affine(), "translation matrix is affine");
            suite.assert_equal(transMat.get_translation(), translation, "get_translation() returns correct translation");
            suite.assert_equal(transMat(3, 0), 2.0f, "translation matrix (3,0)");
            suite.assert_equal(transMat(3, 1), 3.0f, "translation matrix (3,1)");
            suite.assert_equal(transMat(3, 2), 4.0f, "translation matrix (3,2)");
            suite.assert_equal(transMat(3, 3), 1.0f, "translation matrix (3,3)");

            // Проверка что верхняя 3x3 часть - identity
            suite.assert_equal(transMat(0, 0), 1.0f, "translation matrix rotation part (0,0)");
            suite.assert_equal(transMat(1, 1), 1.0f, "translation matrix rotation part (1,1)");
            suite.assert_equal(transMat(2, 2), 1.0f, "translation matrix rotation part (2,2)");
        }

        // Матрица масштабирования
        {
            float3 scale(2, 3, 4);
            float4x4 scaleMat = float4x4::scaling(scale);

            suite.assert_equal(scaleMat.get_scale(), scale, "get_scale() returns correct scale");
            suite.assert_equal(scaleMat(0, 0), 2.0f, "scaling matrix (0,0)");
            suite.assert_equal(scaleMat(1, 1), 3.0f, "scaling matrix (1,1)");
            suite.assert_equal(scaleMat(2, 2), 4.0f, "scaling matrix (2,2)");
            suite.assert_equal(scaleMat(3, 3), 1.0f, "scaling matrix (3,3)");

            // Проверка равномерного масштабирования
            float4x4 uniformScale = float4x4::scaling(5.0f);
            suite.assert_equal(uniformScale.get_scale(), float3(5, 5, 5), "uniform scaling matrix");
        }

        // Матрицы вращения
        {
            float angle = Constants::Constants<float>::Pi / 4.0f; // 45 градусов

            // Вращение вокруг X
            float4x4 rotX = float4x4::rotation_x(angle);
            suite.assert_true(rotX.is_affine(), "rotation_x creates affine matrix");
            suite.assert_approximately_equal(rotX.determinant(), 1.0f, "rotation_x determinant = 1", 1e-6f);

            // Вращение вокруг Y
            float4x4 rotY = float4x4::rotation_y(angle);
            suite.assert_true(rotY.is_affine(), "rotation_y creates affine matrix");
            suite.assert_approximately_equal(rotY.determinant(), 1.0f, "rotation_y determinant = 1", 1e-6f);

            // Вращение вокруг Z
            float4x4 rotZ = float4x4::rotation_z(angle);
            suite.assert_true(rotZ.is_affine(), "rotation_z creates affine matrix");
            suite.assert_approximately_equal(rotZ.determinant(), 1.0f, "rotation_z determinant = 1", 1e-6f);

            // Проверка конкретного вращения
            float4x4 rotY90 = float4x4::rotation_y(Constants::Constants<float>::Pi / 2.0f);
            float3 point(1, 0, 0);
            float3 transformed = rotY90.transform_point(point);
            suite.assert_approximately_equal(transformed, float3(0, 0, -1), "rotation_y 90 degrees", 1e-5f);
        }

        // Вращение вокруг произвольной оси
        {
            float3 axis = float3(1, 1, 1).normalize();
            float angle = Constants::Constants<float>::Pi / 3.0f; // 60 градусов

            float4x4 rot = float4x4::rotation_axis(axis, angle);
            suite.assert_true(rot.is_affine(), "rotation_axis creates affine matrix");
            suite.assert_approximately_equal(rot.determinant(), 1.0f, "rotation_axis determinant = 1", 1e-6f);

            // Ось вращения должна быть собственным вектором
            float3 axisTransformed = rot.transform_vector(axis);
            suite.assert_approximately_equal(axisTransformed, axis, "rotation axis is eigenvector", 1e-5f);
        }

        // Эйлеровы углы
        {
            float3 angles(
                Constants::Constants<float>::Pi / 6.0f,  // 30° вокруг X
                Constants::Constants<float>::Pi / 4.0f,  // 45° вокруг Y
                Constants::Constants<float>::Pi / 3.0f   // 60° вокруг Z
            );

            float4x4 eulerMat = float4x4::rotation_euler(angles);
            suite.assert_true(eulerMat.is_affine(), "rotation_euler creates affine matrix");
            suite.assert_approximately_equal(eulerMat.determinant(), 1.0f, "rotation_euler determinant = 1", 1e-6f);
        }

        // Матрица TRS (translation, rotation, scale)
        {
            float3 translation(1, 2, 3);
            quaternion rotation = quaternion::rotation_y(Constants::Constants<float>::Pi / 4.0f);
            float3 scale(2, 3, 4);

            float4x4 trsMat = float4x4::TRS(translation, rotation, scale);

            // Проверка что компоненты извлекаются корректно
            float3 extractedTranslation = trsMat.get_translation();
            float3 extractedScale = trsMat.get_scale();

            suite.assert_approximately_equal(extractedTranslation, translation, "TRS matrix translation", 1e-5f);
            suite.assert_approximately_equal(extractedScale, scale, "TRS matrix scale", 1e-5f);
        }

        // Проекционные матрицы
        {
            float fov = Constants::Constants<float>::Pi / 3.0f; // 60 градусов
            float aspect = 16.0f / 9.0f;
            float zNear = 0.1f;
            float zFar = 100.0f;

            // Перспективная проекция
            float4x4 persp = float4x4::perspective(fov, aspect, zNear, zFar);
            suite.assert_false(persp.is_affine(), "perspective matrix is not affine");
            suite.assert_not_equal(persp.determinant(), 0.0f, "perspective matrix has non-zero determinant");

            // Ортографическая проекция
            float4x4 ortho = float4x4::orthographic(800.0f, 600.0f, 0.1f, 100.0f);
            suite.assert_true(ortho.is_affine(), "orthographic matrix is affine");
        }

        // Матрица вида (look at)
        {
            float3 eye(0, 0, 5);
            float3 target(0, 0, 0);
            float3 up(0, 1, 0);

            float4x4 view = float4x4::look_at(eye, target, up);
            suite.assert_true(view.is_affine(), "look_at matrix is affine");

            // Точка в пространстве должна преобразовываться правильно
            float3 worldPoint(1, 2, 3);
            float3 viewPoint = view.transform_point(worldPoint);

            // Мы не знаем точные координаты, но можем проверить свойства
            suite.assert_true(std::isfinite(viewPoint.x) && std::isfinite(viewPoint.y) && std::isfinite(viewPoint.z),
                "look_at transform produces finite coordinates");
        }

        // ============================================================================
        // 4. Арифметические операции
        // ============================================================================
        suite.section("Арифметические операции");

        float4x4 A(
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );

        float4x4 B(
            16, 15, 14, 13,
            12, 11, 10, 9,
            8, 7, 6, 5,
            4, 3, 2, 1
        );

        // Сложение
        {
            float4x4 sum = A + B;
            float4x4 expected(
                17, 17, 17, 17,
                17, 17, 17, 17,
                17, 17, 17, 17,
                17, 17, 17, 17
            );
            suite.assert_equal(sum, expected, "Matrix addition");

            // Проверка оператора +=
            float4x4 A_copy = A;
            A_copy += B;
            suite.assert_equal(A_copy, expected, "Operator +=");
        }

        // Вычитание
        {
            float4x4 diff = A - B;
            float4x4 expected(
                -15, -13, -11, -9,
                -7, -5, -3, -1,
                1, 3, 5, 7,
                9, 11, 13, 15
            );
            suite.assert_equal(diff, expected, "Matrix subtraction");

            // Проверка оператора -=
            float4x4 A_copy = A;
            A_copy -= B;
            suite.assert_equal(A_copy, expected, "Operator -=");
        }

        // Умножение на скаляр
        {
            float4x4 scaled = A * 2.0f;
            float4x4 expected(
                2, 4, 6, 8,
                10, 12, 14, 16,
                18, 20, 22, 24,
                26, 28, 30, 32
            );
            suite.assert_equal(scaled, expected, "Matrix * scalar");

            // Проверка оператора *=
            float4x4 A_copy = A;
            A_copy *= 2.0f;
            suite.assert_equal(A_copy, expected, "Operator *=");

            // Проверка скаляр * матрица
            float4x4 scaled2 = 2.0f * A;
            suite.assert_equal(scaled2, expected, "Scalar * matrix");
        }

        // Деление на скаляр
        {
            float4x4 A_copy = A;
            A_copy /= 2.0f;
            float4x4 expected(
                0.5f, 1.0f, 1.5f, 2.0f,
                2.5f, 3.0f, 3.5f, 4.0f,
                4.5f, 5.0f, 5.5f, 6.0f,
                6.5f, 7.0f, 7.5f, 8.0f
            );
            suite.assert_equal(A_copy, expected, "Operator /=");
        }

        // Умножение матриц
        {
            // Диагональные матрицы для простой проверки
            float4x4 C = float4x4::scaling(2, 3, 4);
            float4x4 D = float4x4::scaling(3, 4, 5);

            float4x4 result = C * D;
            float4x4 expected = float4x4::scaling(6, 12, 20);
            suite.assert_approximately_equal(result, expected, "Matrix multiplication of scaling matrices", 1e-5f);

            // Проверка ассоциативности (A*B)*C = A*(B*C)
            float4x4 E = float4x4::translation(1, 2, 3);
            float4x4 F = float4x4::rotation_x(Constants::Constants<float>::Pi / 6.0f);
            float4x4 G = float4x4::scaling(2, 2, 2);

            float4x4 EF = E * F;
            float4x4 EF_G = EF * G;

            float4x4 FG = F * G;
            float4x4 E_FG = E * FG;

            suite.assert_approximately_equal(EF_G, E_FG, "Matrix multiplication associativity", 1e-5f);

            // Проверка оператора *=
            float4x4 C_copy = C;
            C_copy *= D;
            suite.assert_approximately_equal(C_copy, expected, "Operator *=", 1e-5f);
        }

        // Унарные операторы
        {
            float4x4 neg = -A;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    suite.assert_equal(neg(i, j), -A(i, j), "Unary minus element (" + std::to_string(i) + "," + std::to_string(j) + ")");
                }
            }

            float4x4 pos = +A;
            suite.assert_equal(pos, A, "Unary plus");
        }

        // ============================================================================
        // 5. Умножение на векторы
        // ============================================================================
        suite.section("Умножение на векторы");

        float4x4 M = float4x4::scaling(2, 3, 4) * float4x4::translation(1, 2, 3);
        float3 point(1, 1, 1);
        float3 vector(1, 0, 0);
        float4 homogPoint(1, 1, 1, 1);
        float4 homogVector(1, 0, 0, 0);

        // Преобразование точки
        {
            float3 transformed = M.transform_point(point);
            // Ожидается: точка (1,1,1) масштабируется (2,3,4) => (2,3,4), затем сдвигается (1,2,3) => (3,5,7)
            float3 expected(3, 5, 7);
            suite.assert_approximately_equal(transformed, expected, "transform_point", 1e-5f);
        }

        // Преобразование вектора (без трансляции)
        {
            float3 transformed = M.transform_vector(vector);
            // Вектор не должен подвергаться трансляции, только масштабированию и вращению
            // В данном случае только масштабирование: (1,0,0) * 2 = (2,0,0)
            float3 expected(2, 0, 0);
            suite.assert_approximately_equal(transformed, expected, "transform_vector", 1e-5f);
        }

        // Преобразование направления (нормализация после преобразования)
        {
            float3 direction(1, 0, 0);
            float3 transformed = M.transform_direction(direction);
            // Направление должно быть нормализовано после масштабирования
            float3 expected(1, 0, 0); // Масштабирование в 2 раза, но нормализация возвращает единичный вектор
            suite.assert_approximately_equal(transformed, expected, "transform_direction", 1e-5f);
        }

        // Умножение однородного вектора
        {
            float4 transformed = M.transform_vector(homogPoint);
            // Ожидается: (1,1,1,1) преобразуется в (3,5,7,1)
            float4 expected(3, 5, 7, 1);
            suite.assert_approximately_equal(transformed, expected, "transform_vector with homogeneous point", 1e-5f);

            float4 transformedVec = M.transform_vector(homogVector);
            // Вектор с w=0 не должен подвергаться трансляции
            float4 expectedVec(2, 0, 0, 0);
            suite.assert_approximately_equal(transformedVec, expectedVec, "transform_vector with homogeneous vector", 1e-5f);
        }

        // Оператор * для вектора и матрицы
        {
            float3 result = point * M;  // Оператор * использует transform_point
            float3 expected = M.transform_point(point);
            suite.assert_approximately_equal(result, expected, "point * matrix operator", 1e-5f);

            float4 result4 = homogPoint * M;  // Оператор * использует transform_vector
            float4 expected4 = M.transform_vector(homogPoint);
            suite.assert_approximately_equal(result4, expected4, "float4 * matrix operator", 1e-5f);
        }

        // ============================================================================
        // 6. Матричные операции
        // ============================================================================
        suite.section("Матричные операции");

        // Транспонирование
        {
            float4x4 mat(
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            );

            float4x4 transposed = mat.transposed();
            float4x4 expected(
                1, 5, 9, 13,
                2, 6, 10, 14,
                3, 7, 11, 15,
                4, 8, 12, 16
            );
            suite.assert_equal(transposed, expected, "transposed");

            // Проверка что транспонирование дважды возвращает исходную матрицу
            suite.assert_equal(transposed.transposed(), mat, "transpose twice returns original");

            // Проверка свободной функции
            suite.assert_equal(transpose(mat), transposed, "transpose() free function");
        }

        // Определитель
        {
            float4x4 identity = float4x4::identity();
            suite.assert_approximately_equal(identity.determinant(), 1.0f, "identity determinant = 1");

            float4x4 zero = float4x4::zero();
            suite.assert_approximately_equal(zero.determinant(), 0.0f, "zero determinant = 0");

            float4x4 scale = float4x4::scaling(2, 3, 4);
            suite.assert_approximately_equal(scale.determinant(), 24.0f, "scaling determinant");

            // Проверка свободной функции
            suite.assert_approximately_equal(determinant(scale), 24.0f, "determinant() free function");
        }

        // Обратная матрица (аффинная)
        {
            float4x4 affine = float4x4::translation(1, 2, 3) *
                float4x4::rotation_x(Constants::Constants<float>::Pi / 4.0f) *
                float4x4::scaling(2, 3, 4);

            float4x4 inverseAffine = affine.inverted();

            // Проверка что A * A^(-1) = I
            float4x4 product = affine * inverseAffine;
            suite.assert_true(product.is_identity(1e-5f), "A * A^(-1) = I for affine matrix");

            // Проверка что A^(-1) * A = I
            float4x4 product2 = inverseAffine * affine;
            suite.assert_true(product2.is_identity(1e-5f), "A^(-1) * A = I for affine matrix");

            // Проверка свободной функции
            suite.assert_approximately_equal(inverse(affine), inverseAffine, "inverse() free function");
        }

        // Обратная матрица (полная)
        {
            // Создаем неаффинную матрицу (проекционную)
            float4x4 persp = float4x4::perspective(
                Constants::Constants<float>::Pi / 3.0f,
                16.0f / 9.0f,
                0.1f,
                100.0f
            );

            float4x4 inversePersp = persp.inverted();

            // Проверка что определитель произведения близок к 1
            float4x4 product = persp * inversePersp;
            suite.assert_approximately_equal(product.determinant(), 1.0f, "perspective matrix inverse check", 1e-5f);
        }

        // Обратная аффинная матрица (оптимизированная версия)
        {
            float4x4 affine = float4x4::TRS(
                float3(1, 2, 3),
                quaternion::rotation_y(Constants::Constants<float>::Pi / 6.0f),
                float3(2, 3, 4)
            );

            float4x4 inv1 = affine.inverted();
            float4x4 inv2 = affine.inverted_affine();

            // Обе версии должны давать одинаковый результат для аффинной матрицы
            suite.assert_approximately_equal(inv1, inv2, "inverted() == inverted_affine() for affine matrix", 1e-5f);
        }

        // Присоединенная матрица
        {
            float4x4 mat = float4x4::scaling(2, 3, 4);
            float4x4 adj = mat.adjugate();

            // Для диагональной матрицы присоединенная матрица тоже диагональна
            float det = mat.determinant(); // 2*3*4*1 = 24

            // Алгебраические дополнения для диагональной матрицы:
            // (0,0): 3*4*1 = 12
            // (1,1): 2*4*1 = 8
            // (2,2): 2*3*1 = 6
            // (3,3): 2*3*4 = 24

            // Проверяем диагональные элементы
            suite.assert_approximately_equal(adj(0, 0), 12.0f, "adjugate (0,0)", 1e-5f);
            suite.assert_approximately_equal(adj(1, 1), 8.0f, "adjugate (1,1)", 1e-5f);
            suite.assert_approximately_equal(adj(2, 2), 6.0f, "adjugate (2,2)", 1e-5f);
            suite.assert_approximately_equal(adj(3, 3), 24.0f, "adjugate (3,3)", 1e-5f);

            // Проверяем, что недиагональные элементы равны 0
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if (i != j) {
                        suite.assert_approximately_equal(adj(i, j), 0.0f,
                            "adjugate off-diagonal (" + std::to_string(i) + "," + std::to_string(j) + ")", 1e-5f);
                    }
                }
            }
        }

        // Нормальная матрица
        {
            float4x4 model = float4x4::TRS(
                float3(1, 2, 3),
                quaternion::rotation_x(Constants::Constants<float>::Pi / 4.0f),
                float3(2, 3, 4)
            );

            float3x3 normalMat = model.normal_matrix();

            // Нормальная матрица должна быть транспонированной обратной верхней 3x3 части
            float3x3 upper(
                float3(model(0, 0), model(0, 1), model(0, 2)),
                float3(model(1, 0), model(1, 1), model(1, 2)),
                float3(model(2, 0), model(2, 1), model(2, 2))
            );

            float3x3 expected = upper.inverted().transposed();

            // Столбцы нормальной матрицы должны быть нормализованы
            float3 col0 = expected.col0().normalize();
            float3 col1 = expected.col1().normalize();
            float3 col2 = expected.col2().normalize();
            expected = float3x3(col0, col1, col2);

            suite.assert_approximately_equal(normalMat, expected, "normal_matrix", 1e-5f);
        }

        // След матрицы
        {
            float4x4 mat(
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            );
            float expectedTrace = 1 + 6 + 11 + 16;
            suite.assert_approximately_equal(mat.trace(), expectedTrace, "trace");
        }

        // Диагональ
        {
            float4x4 mat(
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            );
            float4 expectedDiag(1, 6, 11, 16);
            suite.assert_equal(mat.diagonal(), expectedDiag, "diagonal");
        }

        // Норма Фробениуса
        {
            float4x4 mat = float4x4::scaling(2, 3, 4);
            float expectedNorm = std::sqrt(2 * 2 + 3 * 3 + 4 * 4 + 1 * 1); // Диагональные элементы: 2,3,4,1
            suite.assert_approximately_equal(mat.frobenius_norm(), expectedNorm, "frobenius_norm", 1e-5f);
        }

        // ============================================================================
        // 7. Специальные функции
        // ============================================================================
        suite.section("Специальные функции");

        // Получение и установка трансляции
        {
            float3 translation(5, 6, 7);
            float4x4 mat = float4x4::identity();
            mat.set_translation(translation);

            suite.assert_equal(mat.get_translation(), translation, "set_translation/get_translation");
            suite.assert_equal(mat(3, 0), 5.0f, "set_translation sets (3,0)");
            suite.assert_equal(mat(3, 1), 6.0f, "set_translation sets (3,1)");
            suite.assert_equal(mat(3, 2), 7.0f, "set_translation sets (3,2)");

            // Убедимся что остальная часть матрицы не изменилась
            suite.assert_true(mat.is_affine(), "matrix remains affine after set_translation");
            suite.assert_equal(mat(0, 0), 1.0f, "set_translation doesn't affect (0,0)");
        }

        // Получение и установка масштаба
        {
            float3 scale(2, 3, 4);
            float4x4 mat = float4x4::rotation_z(Constants::Constants<float>::Pi / 4.0f);
            mat.set_scale(scale);

            float3 extractedScale = mat.get_scale();
            suite.assert_approximately_equal(extractedScale, scale, "set_scale/get_scale", 1e-5f);

            // Проверка что матрица сохраняет вращение
            suite.assert_true(mat.is_affine(), "matrix remains affine after set_scale");
        }

        // Получение вращения
        {
            quaternion rot = quaternion::rotation_y(Constants::Constants<float>::Pi / 3.0f);
            float4x4 mat = float4x4::TRS(float3(1, 2, 3), rot, float3(2, 3, 4));

            quaternion extractedRot = mat.get_rotation();

            // Вращения могут отличаться на знак или на 360 градусов, проверяем что они представляют одно и то же вращение
            float4x4 rotMat1 = float4x4(rot);
            float4x4 rotMat2 = float4x4(extractedRot);

            // Убираем масштаб и трансляцию для сравнения
            rotMat1.set_scale(float3(1, 1, 1));
            rotMat2.set_scale(float3(1, 1, 1));

            suite.assert_approximately_equal(rotMat1, rotMat2, "get_rotation returns equivalent rotation", 1e-5f);
        }

        // ============================================================================
        // 8. Проверки свойств
        // ============================================================================
        suite.section("Проверки свойств");

        // Проверка identity
        suite.assert_true(float4x4::identity().is_identity(), "identity().is_identity()");
        suite.assert_false(float4x4::zero().is_identity(), "zero().is_identity() returns false");
        suite.assert_true(float4x4::scaling(1, 1, 1).is_identity(), "uniform scaling(1) is identity");

        // Проверка аффинности
        {
            float4x4 affine = float4x4::translation(1, 2, 3);
            suite.assert_true(affine.is_affine(), "translation matrix is affine");

            float4x4 nonAffine = float4x4::perspective(Constants::Constants<float>::Pi / 3, 1.0f, 0.1f, 100.0f);
            suite.assert_false(nonAffine.is_affine(), "perspective matrix is not affine");
        }

        // Проверка ортогональности
        {
            float4x4 rotation = float4x4::rotation_x(Constants::Constants<float>::Pi / 3.0f);
            suite.assert_true(rotation.is_orthogonal(), "rotation matrix is orthogonal");

            float4x4 scale = float4x4::scaling(2, 3, 4);
            suite.assert_false(scale.is_orthogonal(), "non-uniform scaling matrix is not orthogonal");

            float4x4 uniformScale = float4x4::scaling(2);
            suite.assert_false(uniformScale.is_orthogonal(), "uniform scaling matrix is not orthogonal (length != 1)");
        }

        // Проверка approximately_zero
        suite.assert_true(float4x4::zero().approximately_zero(), "zero().approximately_zero()");
        suite.assert_false(float4x4::identity().approximately_zero(), "identity().approximately_zero() returns false");

        // Проверка approximately
        {
            float4x4 mat1 = float4x4::identity();
            float4x4 mat2 = float4x4::identity();
            mat2(0, 0) = 1.000001f;

            suite.assert_true(mat1.approximately(mat2, 1e-4f), "approximately with epsilon");
            suite.assert_false(mat1.approximately(mat2, 1e-7f), "approximately with strict epsilon fails");

            // Проверка операторов == и !=
            suite.assert_true(mat1 == mat1, "operator == for identical matrices");

            // Для небольших различий оператор == должен возвращать true (использует epsilon по умолчанию)
            if (mat1.approximately(mat2)) {
                suite.assert_true(mat1 == mat2, "operator == returns true for approximately equal matrices");
                suite.assert_false(mat1 != mat2, "operator != returns false for approximately equal matrices");
            }

            // Для явно разных матриц
            float4x4 mat3 = float4x4::scaling(2);
            suite.assert_true(mat1 != mat3, "operator != returns true for different matrices");
        }

        // ============================================================================
        // 9. Преобразования данных
        // ============================================================================
        suite.section("Преобразования данных");

        // to_row_major
        {
            float4x4 mat(
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            );

            float rowMajor[16];
            mat.to_row_major(rowMajor);

            for (int i = 0; i < 16; ++i) {
                suite.assert_equal(rowMajor[i], static_cast<float>(i + 1),
                    "to_row_major element " + std::to_string(i));
            }
        }

        // to_column_major
        {
            float4x4 mat(
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            );

            float colMajor[16];
            mat.to_column_major(colMajor);

            // Column-major order: сначала все элементы первого столбца, затем второго и т.д.
            float expected[16] = { 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 };
            for (int i = 0; i < 16; ++i) {
                suite.assert_equal(colMajor[i], expected[i],
                    "to_column_major element " + std::to_string(i));
            }
        }

        // to_string
        {
            float4x4 mat = float4x4::identity();
            std::string str = mat.to_string();

            // Проверяем что строка содержит ожидаемые значения
            suite.assert_true(str.find("1.0000") != std::string::npos, "to_string contains 1.0000");
            suite.assert_true(str.find("0.0000") != std::string::npos, "to_string contains 0.0000");
        }

        // ============================================================================
        // 10. Конструкторы из других типов
        // ============================================================================
        suite.section("Конструкторы из других типов");

        // Конструктор из float3x3
        {
            float3x3 mat3x3(
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            );

            float4x4 mat4x4(mat3x3);

            // Верхняя 3x3 часть должна совпадать
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    suite.assert_equal(mat4x4(i, j), mat3x3(i, j),
                        "float3x3 to float4x4 conversion element (" + std::to_string(i) + "," + std::to_string(j) + ")");
                }
            }

            // Последняя строка и столбец должны быть (0,0,0,1)
            suite.assert_equal(mat4x4.row3(), float4(0, 0, 0, 1), "float3x3 to float4x4 row3");
            suite.assert_equal(mat4x4.col3(), float4(0, 0, 0, 1), "float3x3 to float4x4 col3");
        }

        // Конструктор из кватерниона
        {
            quaternion q = quaternion::rotation_y(Constants::Constants<float>::Pi / 2.0f);
            float4x4 mat(q);

            // Матрица из кватерниона должна быть ортогональной
            suite.assert_true(mat.is_orthogonal(), "quaternion constructor creates orthogonal matrix");
            suite.assert_approximately_equal(mat.determinant(), 1.0f, "quaternion constructor determinant = 1", 1e-6f);

            // Проверка конкретного вращения
            float3 point(1, 0, 0);
            float3 transformed = mat.transform_point(point);
            float3 expected(0, 0, -1); // Вращение на 90 градусов вокруг Y
            suite.assert_approximately_equal(transformed, expected, "quaternion rotation", 1e-5f);
        }

        // ============================================================================
        // 11. Граничные случаи и особые значения
        // ============================================================================
        suite.section("Граничные случаи и особые значения");

        // Очень маленькие значения
        {
            float epsilon = Constants::Constants<float>::Epsilon;
            float4x4 tinyMat(epsilon);

            suite.assert_true(tinyMat.approximately_zero(epsilon * 2),
                "approximately_zero with tiny values");
        }

        // Очень большие значения
        {
            float large = 1e10f;
            float4x4 largeMat = float4x4::scaling(large);

            // Обратная матрица должна иметь маленькие значения
            float4x4 invLarge = largeMat.inverted();
            float4x4 expected = float4x4::scaling(1.0f / large);
            suite.assert_approximately_equal(invLarge, expected,
                "inverse of large scaling matrix", 1e-5f);
        }

        // Вырожденная матрица (нулевой масштаб)
        {
            float4x4 degenerate = float4x4::scaling(0, 1, 1);

            // Определитель должен быть 0
            suite.assert_approximately_equal(degenerate.determinant(), 0.0f, "degenerate matrix has zero determinant");

            // Обратная матрица должна вернуть identity (как в реализации при нулевом определителе)
            float4x4 invDegenerate = degenerate.inverted();
            suite.assert_true(invDegenerate.is_identity(1e-5f), "degenerate matrix inverse returns identity");
        }

        // Нулевая ось вращения
        {
            float3 zeroAxis(0, 0, 0);
            float4x4 rot = float4x4::rotation_axis(zeroAxis, 1.0f);
            // Должна вернуться identity матрица
            suite.assert_true(rot.is_identity(1e-5f), "rotation_axis with zero axis returns identity");
        }

        // Нулевой угол вращения
        {
            float3 axis(1, 0, 0);
            float4x4 rot = float4x4::rotation_axis(axis, 0.0f);
            suite.assert_true(rot.is_identity(1e-5f), "rotation_axis with zero angle returns identity");
        }

        // Матрица с бесконечностями и NaN
        {
            float4x4 validMat = float4x4::identity();
            // Для этих тестов нам нужен доступ к приватным полям, так что пропускаем
            suite.skip_test("Infinity and NaN tests", "Requires direct field access");
        }

        // Деление на ноль
        {
            float4x4 mat = float4x4::identity();

            try {
                mat /= 0.0f;
                // Если не выброшено исключение, элементы должны быть inf или nan
                // Проверяем что матрица невалидна
                bool hasInfOrNaN = false;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        if (!std::isfinite(mat(i, j))) {
                            hasInfOrNaN = true;
                            break;
                        }
                    }
                    if (hasInfOrNaN) break;
                }
                suite.assert_true(hasInfOrNaN, "division by zero produces inf/nan values");
            }
            catch (...) {
                suite.skip_test("Division by zero", "Exception thrown - implementation dependent");
            }
        }

        // ============================================================================
        // 12. Проекционные матрицы (детальные тесты)
        // ============================================================================
        suite.section("Проекционные матрицы");

        // Перспективная проекция (левый/правый, нулевая/обратная глубина)
        {
            float fov = Constants::Constants<float>::Pi / 3.0f;
            float aspect = 16.0f / 9.0f;
            float zNear = 0.1f;
            float zFar = 100.0f;

            // LH ZO (левый, нулевая глубина)
            float4x4 perspLHZO = float4x4::perspective_lh_zo(fov, aspect, zNear, zFar);

            // Проверка что ближняя плоскость проецируется в 0
            float4 nearPoint(0, 0, zNear, 1);
            float4 projectedNear = perspLHZO.transform_vector(nearPoint);
            projectedNear /= projectedNear.w;
            suite.assert_approximately_equal(projectedNear.z, 0.0f, "LH ZO: near plane projects to 0", 1e-5f);

            // Проверка что дальняя плоскость проецируется в 1
            float4 farPoint(0, 0, zFar, 1);
            float4 projectedFar = perspLHZO.transform_vector(farPoint);
            projectedFar /= projectedFar.w;
            suite.assert_approximately_equal(projectedFar.z, 1.0f, "LH ZO: far plane projects to 1", 1e-5f);

            // RH ZO (правый, нулевая глубина)
            float4x4 perspRHZO = float4x4::perspective_rh_zo(fov, aspect, zNear, zFar);

            // Для RH ZO Z меняется в другую сторону
            nearPoint = float4(0, 0, zNear, 1);
            projectedNear = perspRHZO.transform_vector(nearPoint);
            projectedNear /= projectedNear.w;
            suite.assert_approximately_equal(projectedNear.z, 0.0f, "RH ZO: near plane projects to 0", 1e-5f);
        }

        // Ортографическая проекция
        {
            float width = 800.0f;
            float height = 600.0f;
            float zNear = 0.1f;
            float zFar = 100.0f;

            float4x4 ortho = float4x4::orthographic(width, height, zNear, zFar);

            // Точки на ближней и дальней плоскости должны иметь одинаковые координаты X,Y
            float4 nearPoint(-width / 2, -height / 2, zNear, 1);
            float4 farPoint(-width / 2, -height / 2, zFar, 1);

            float4 projectedNear = ortho.transform_vector(nearPoint);
            float4 projectedFar = ortho.transform_vector(farPoint);

            suite.assert_approximately_equal(projectedNear.x, -1.0f, "ortho: left edge projects to -1", 1e-5f);
            suite.assert_approximately_equal(projectedNear.y, -1.0f, "ortho: bottom edge projects to -1", 1e-5f);
            suite.assert_approximately_equal(projectedFar.z, 1.0f, "ortho: far plane projects to 1", 1e-5f);
        }

        suite.footer();
    }
}
